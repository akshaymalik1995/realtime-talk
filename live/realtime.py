"""Thin helpers over the raw OpenAI Realtime WebSocket. Events are plain dicts."""
import asyncio
import collections
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from live import audio, config, tools as toolkit


async def connect():
    return await websockets.connect(
        config.URL,
        additional_headers={"Authorization": f"Bearer {config.api_key()}"},
        max_size=None,  # ponytail: audio events can be large; no client-side cap
    )


async def send(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


async def recv(ws) -> dict:
    event = json.loads(await ws.recv())
    if event["type"] == "error":
        raise RuntimeError(event["error"])
    return event


async def collect(ws, until: str, timeout: float = 30) -> list[dict]:
    """Receive events until one of type `until`. Each event gets `_t` = seconds since the call.
    Raises TimeoutError if the server goes quiet for `timeout` s (a stall), instead of hanging."""
    t0, events = time.perf_counter(), []
    while True:
        try:
            e = await asyncio.wait_for(recv(ws), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"stall: no server event for {timeout}s while waiting for {until} "
                               f"(last: {events[-1]['type'] if events else 'none'})") from None
        e["_t"] = round(time.perf_counter() - t0, 3)
        events.append(e)
        if e["type"] == until:
            return events


# server_vad = "silence for N ms ends your turn": predictable. semantic_vad = model judges if
# you sounded finished: fewer cut-offs, but it waited ~5 s after a complete question in our test.
SERVER_VAD = {"type": "server_vad", "silence_duration_ms": 500}
SEMANTIC_VAD = {"type": "semantic_vad"}


def session_config(instructions: str, voice: str = "marin", turn_detection: dict | None = SERVER_VAD,
                   transcribe: str | None = "gpt-4o-mini-transcribe", tools: list | None = None,
                   max_output_tokens: int | None = None) -> dict:
    """`transcribe`: side model that turns YOUR audio into text (the realtime model itself
    never produces text of what you said). None = no user transcripts.
    `tools`: list of tool specs (live.tools.spec) the model may propose calls to.
    `max_output_tokens`: cap per response (bounds paid usage); None = server default."""
    return {
        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        **({"max_output_tokens": max_output_tokens} if max_output_tokens else {}),
        "type": "realtime",
        "instructions": instructions,
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": turn_detection,
                      "transcription": {"model": transcribe} if transcribe else None},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": voice},
        },
    }


async def start_session(instructions: str, **kw):
    """connect → session.created → session.update → session.updated. Returns the open socket."""
    ws = await connect()
    await collect(ws, until="session.created")
    await send(ws, {"type": "session.update", "session": session_config(instructions, **kw)})
    await collect(ws, until="session.updated")
    return ws


LOGGED = {"input_audio_buffer.speech_stopped", "input_audio_buffer.committed",
          "response.created", "conversation.item.truncated"}


async def answer_tool_calls(ws, response: dict, registry: dict) -> list[tuple]:
    """For each function_call in a finished response: run it (in a thread, so a slow tool
    can't freeze the audio loop), send its output, then ask the model to continue.
    Returns [(name, arguments, output)]."""
    calls = [o for o in response.get("output", []) if o["type"] == "function_call"]
    if response["status"] != "completed" or not calls:
        return []  # cancelled by barge-in → the call never happened
    done = []
    for c in calls:
        out = await asyncio.to_thread(toolkit.run, registry, c["name"], c["arguments"])
        await send(ws, {"type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": c["call_id"], "output": json.dumps(out)}})
        done.append((c["name"], c["arguments"], out))
    await send(ws, {"type": "response.create"})
    return done


STALL_S = 5  # seconds without any server event, mid-response, before the watchdog reports a stall


class Stalled(ConnectionError):
    """The Realtime connection went silent mid-response and stopped answering pings."""
FILLER_TEXT = "Let me check."
ASSETS = Path(__file__).resolve().parent.parent / "assets"


async def load_filler(voice: str = "marin") -> np.ndarray:
    """"Let me check." in the model's own voice. Generated once via the Realtime API, verified
    by its transcript, cached in assets/. Falls back to Windows TTS if the model won't comply."""
    path = ASSETS / f"filler_{voice}.npy"
    if path.exists():
        return np.load(path)
    ws = await start_session("Repeat the user's message word for word, exactly once. Say nothing else.",
                             voice=voice, turn_detection=None, transcribe=None)
    try:
        await send(ws, user_text(FILLER_TEXT))
        await send(ws, {"type": "response.create"})
        ev = await collect(ws, until="response.done")
    finally:
        await ws.close()
    said = next((e["transcript"] for e in ev if e["type"] == "response.output_audio_transcript.done"), "")
    if said.strip().lower().rstrip(".!") != FILLER_TEXT.lower().rstrip("."):
        return audio.say(FILLER_TEXT)  # don't cache a wrong take
    pcm = audio.from_bytes(b"".join(audio.from_b64(e["delta"]) for e in ev if e["type"] == "response.output_audio.delta"))
    ASSETS.mkdir(exist_ok=True)
    np.save(path, pcm)
    return pcm


async def converse(ws, chunks, speaker, seconds: float, barge_in: bool = True, tools: dict | None = None,
                   filler: bool = True, on_note=None, stop: asyncio.Event | None = None) -> list[tuple]:
    """Full duplex for `seconds`: stream `chunks` (async iterator of PCM16 bytes) up while
    playing model audio down. Returns a log of (t, event, detail); also prints it live.

    barge_in: when you start talking over the model, (1) drop its queued audio locally and
    (2) truncate its message server-side to what you actually heard.
    tools: registry {name: (spec, fn)}; proposed calls are validated + executed by live.tools.
    filler: say "Let me check." locally the instant a tool call starts (needs `tools`).
    on_note: optional callback(t, event, detail) for every log line (e.g. to stream it to a UI).
    stop: optional event; setting it ends the conversation before `seconds` (e.g. a scripted user is done)."""
    t0, log, responding = time.perf_counter(), [], False
    in_response = False  # a response was created and is not done yet (audio or not)
    last_event = t0  # watchdog: when the server last sent anything
    item_id, item_start, item_len = None, 0, 0  # model message now playing: stream start position + bytes
    filler_pcm = audio.to_bytes(await load_filler()) if tools and filler else None
    filled = set()  # response ids that already got a filler

    def note(event, detail=""):
        log.append((round(time.perf_counter() - t0, 2), event, detail))
        if on_note:
            on_note(*log[-1])
        line = " ".join(map(str, log[-1]))
        try:
            print(line)
        except UnicodeEncodeError:  # e.g. Windows cp1252 console + emoji in a transcript
            print(line.encode("ascii", "backslashreplace").decode())

    async def send_loop():  # mic → server, forever
        async for c in chunks:
            await send(ws, {"type": "input_audio_buffer.append", "audio": audio.to_b64(c)})

    async def recv_loop():  # server → speaker, forever
        nonlocal responding, in_response, item_id, item_start, item_len, last_event
        async for raw in ws:
            last_event = time.perf_counter()
            e = json.loads(raw)
            if e["type"] == "error":
                raise RuntimeError(e["error"])
            if e["type"] == "response.created":
                in_response = True
            elif e["type"] == "response.done":
                in_response = False
            if e["type"] == "response.output_audio.delta":
                if not responding:
                    responding = True
                    note("first_audio_delta")
                if e["item_id"] != item_id:
                    item_id, item_start, item_len = e["item_id"], speaker.position, 0
                b = audio.from_b64(e["delta"])
                speaker.write(b)
                item_len += len(b)
            elif (e["type"] == "response.output_item.added" and e["item"]["type"] == "function_call"
                  and filler_pcm and e["response_id"] not in filled):
                filled.add(e["response_id"])  # once per response, even if it makes several calls
                speaker.write(filler_pcm)
                note("filler", f'"{FILLER_TEXT}" while {e["item"]["name"]} runs')
            elif e["type"] == "input_audio_buffer.speech_started":
                note(e["type"], f"{speaker.buffered_seconds():.2f}s of audio still queued")
                if barge_in and (dropped := speaker.clear()):
                    heard = min(max(speaker.played - item_start, 0), item_len)  # of the model message
                    detail = f"dropped {dropped / 2 / audio.RATE:.2f}s"
                    if item_id and heard < item_len:  # fully-heard messages (or only filler cut) need no truncate
                        heard_ms = heard * 1000 // 2 // audio.RATE
                        await send(ws, {"type": "conversation.item.truncate", "item_id": item_id,
                                        "content_index": 0, "audio_end_ms": heard_ms})
                        detail += f", truncate at {heard_ms} ms"
                    note("barge_in", detail)
            elif e["type"] == "conversation.item.input_audio_transcription.completed":
                note("user_said", e["transcript"].strip())
            elif e["type"] == "response.output_audio_transcript.done":
                note("model_said", e["transcript"])
            elif e["type"] == "response.done":
                responding = False
                note(e["type"], e["response"]["status"])
                for name, args, out in await answer_tool_calls(ws, e["response"], tools or {}):
                    note("tool_call", f"{name}({' '.join(args.split())}) -> {json.dumps(out)}")
            elif e["type"] in LOGGED:
                note(e["type"], e.get("response", {}).get("status") or e.get("audio_end_ms", ""))

    async def watchdog():  # a stall must be visible, not a silent hang
        lags = collections.deque(maxlen=15)  # how late each 1 s tick woke up: our own event loop's health
        while True:
            tick = time.perf_counter()
            await asyncio.sleep(1)
            lags.append(time.perf_counter() - tick - 1)
            quiet = time.perf_counter() - last_event
            if in_response and quiet > STALL_S:
                t = time.perf_counter()
                try:  # pong back = connection fine, server went quiet; no pong = this connection is dead
                    await asyncio.wait_for(await ws.ping(), timeout=5)
                    alive = f"connection alive (pong {(time.perf_counter() - t) * 1000:.0f} ms) -> server-side"
                except Exception as exc:
                    alive = None
                    dead = f"no pong ({type(exc).__name__})"
                # a frozen loop also starves pings/pongs: rule our side in or out
                lag = f"our loop max lag {max(lags):.2f}s over the last {len(lags)}s"
                note("STALL", f"no server event for {quiet:.1f}s mid-response; {alive or dead}; {lag}")
                if not alive:  # measured 2026-11-23: OpenAI-side, per connection; it never recovers
                    raise Stalled(f"{dead} after {quiet:.0f}s without events; {lag}")
                while in_response and time.perf_counter() - last_event > STALL_S:
                    await asyncio.sleep(1)  # report each stall once

    tasks = [asyncio.create_task(f()) for f in (send_loop, recv_loop, watchdog)]
    if stop:
        tasks.append(asyncio.create_task(stop.wait()))
    # the loops only end by error (or the server closing): the first task to finish ends the conversation
    done, pending = await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for t in done:
        t.result()  # re-raise a server error instead of hiding it
    return log


DROPPED = (ConnectionClosed, OSError)  # dead link: Stalled, ConnectionClosedError, TimeoutError (stall in collect), ...


def _message(role: str, text: str) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "conversation.item.create",
            "item": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}}


async def replay(ws, history: list[tuple[str, str]], context: str = "") -> None:
    """Rebuild a dropped conversation on a new connection: the turns so far as text, then a system note."""
    for role, text in history:
        if text:
            await send(ws, _message(role, text))
    await send(ws, _message("system", "The connection dropped and was restored. Continue from where the conversation "
                                      "stands; do not greet again or repeat what you already said. " + context))


async def converse_resilient(start, chunks, speaker, seconds: float, max_resumes: int = 2, on_note=None,
                             stop: asyncio.Event | None = None, context=None, **kw) -> list[tuple]:
    """converse() that survives OpenAI's mid-response stalls (measured: per connection, server-side).

    start: async () -> new configured session (e.g. lambda: start_session(...)).
    On a dead link it opens a new session, replays the turns so far (+ context(), e.g. our own state: who is
    verified, which code is pending) and asks the model to carry on. At most `max_resumes` reconnects.
    The caller's audio is pumped through a queue, so nothing they say during a reconnect is lost."""
    q: asyncio.Queue = asyncio.Queue()
    history, log, t0 = [], [], time.perf_counter()

    async def pump():  # the one reader of `chunks`; an end or an error is queued for whichever attempt is current
        try:
            async for c in chunks:
                await q.put(c)
            await q.put(StopAsyncIteration())
        except Exception as e:  # e.g. the browser left: must end the conversation, not hang it
            await q.put(e)

    async def from_queue():
        while True:
            c = await q.get()
            if isinstance(c, BaseException):
                q.put_nowait(c)  # keep it: a later attempt must see it too
                if isinstance(c, StopAsyncIteration):
                    return
                raise c
            yield c

    def note(t, event, detail):
        log.append((t, event, detail))
        if event in ("user_said", "model_said"):
            history.append(("user" if event == "user_said" else "assistant", detail))
        elif event == "tool_call":  # replay what already happened, or the model redoes it (e.g. reuses a spent code)
            history.append(("system", f"Already done: {detail}"))
        if on_note:
            on_note(t, event, detail)

    pumper = asyncio.create_task(pump())
    deadline = t0 + seconds
    ws = await start()
    try:
        for attempt in range(max_resumes + 1):
            try:
                await converse(ws, from_queue(), speaker, max(deadline - time.perf_counter(), 0), on_note=note,
                               stop=stop, **kw)
                return log
            except DROPPED as e:
                if attempt == max_resumes or deadline - time.perf_counter() < 5:
                    raise
                detail = f"{type(e).__name__}: reconnecting ({attempt + 1}/{max_resumes}), replaying {len(history)} turns"
                note(round(time.perf_counter() - t0, 2), "RESUME", detail)
                print(f"RESUME {detail}")
                with contextlib.suppress(Exception):
                    ws.transport.abort()  # a dead link never answers a polite close: don't wait 10 s for it
                ws = await start()
                await replay(ws, history, context() if context else "")
                await send(ws, {"type": "response.create"})
        return log
    finally:
        pumper.cancel()
        with contextlib.suppress(Exception):
            await ws.close()


def user_text(text: str) -> dict:
    return {"type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}
