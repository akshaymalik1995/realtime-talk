"""OpenAI Realtime over a raw websocket: one full-duplex conversation, with tool calls, barge-in, a filler
while tools run, and recovery when OpenAI lets a connection go silent mid-response (measured: per connection,
server-side, about 1 long reply in 6; the network and our event loop stay healthy).

Only depends on `websockets`: audio is PCM16 bytes, base64 on the wire. No numpy, no sound card.

The `speaker` is anything with: write(bytes), clear() -> bytes dropped, position, played, buffered_seconds().
"""
import asyncio
import base64
import collections
import contextlib
import json
import logging
import time

import websockets
from websockets.exceptions import ConnectionClosed

from app.metrics import METRICS
from app.settings import Settings

log = logging.getLogger(__name__)
RATE = 24000          # Realtime API: 24 kHz PCM16 mono, in and out
BYTES_PER_S = RATE * 2
FILLER_TEXT = "Let me check."


class Stalled(ConnectionError):
    """The Realtime connection went silent mid-response and stopped answering pings."""


DROPPED = (ConnectionClosed, OSError)  # a dead link: Stalled, ConnectionClosedError, TimeoutError, ...


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


async def connect(s: Settings, api_key: str):
    return await websockets.connect(f"{s.openai.url}?model={s.openai.model}",
                                    additional_headers={"Authorization": f"Bearer {api_key}"},
                                    max_size=None)  # audio events can be large; the server is trusted


async def send(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


async def recv(ws) -> dict:
    event = json.loads(await ws.recv())
    if event["type"] == "error":
        raise RuntimeError(event["error"])
    return event


async def collect(ws, until: str, timeout: float = 15) -> list[dict]:
    """Receive events until one of type `until`. TimeoutError if the server goes quiet for `timeout` s."""
    events = []
    while True:
        try:
            e = await asyncio.wait_for(recv(ws), timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"no server event for {timeout}s while waiting for {until}") from None
        events.append(e)
        if e["type"] == until:
            return events


def session_config(s: Settings, instructions: str, tools: list[dict], vad: dict | None) -> dict:
    return {
        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        "type": "realtime",
        "instructions": instructions,
        "max_output_tokens": s.openai.max_output_tokens,
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": RATE}, "turn_detection": vad,
                      "transcription": s.openai.transcription(), "noise_reduction": s.openai.noise()},
            "output": {"format": {"type": "audio/pcm", "rate": RATE}, "voice": s.openai.voice},
        },
    }


async def start_session(s: Settings, api_key: str, instructions: str, tools: list[dict], vad: dict | None):
    """connect -> session.created -> session.update -> session.updated. Returns the open socket."""
    ws = await connect(s, api_key)
    try:
        await collect(ws, until="session.created")
        await send(ws, {"type": "session.update", "session": session_config(s, instructions, tools, vad)})
        await collect(ws, until="session.updated")
    except BaseException:
        await ws.close()
        raise
    return ws


async def answer_tool_calls(ws, response: dict, run_tool) -> list[tuple]:
    """For each function call in a finished response: run it, send its output, then ask the model to go on.
    A response cancelled by barge-in has its calls dropped: they never happened."""
    calls = [o for o in response.get("output", []) if o["type"] == "function_call"]
    if response["status"] != "completed" or not calls:
        return []
    done = []
    for c in calls:
        out = await run_tool(c["name"], c["arguments"])
        await send(ws, {"type": "conversation.item.create",
                        "item": {"type": "function_call_output", "call_id": c["call_id"], "output": json.dumps(out)}})
        done.append((c["name"], c["arguments"], out))
    await send(ws, {"type": "response.create"})
    return done


def _message(role: str, text: str) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "conversation.item.create",
            "item": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}}


async def replay(ws, history: list[tuple[str, str]], context: str = "") -> None:
    """Rebuild a dropped conversation on a new connection: the turns and tool calls so far, then a system note."""
    for role, text in history:
        if text:
            await send(ws, _message(role, text))
    await send(ws, _message("system", "The connection dropped and was restored. Continue from where the conversation "
                                      "stands; do not greet again or repeat what you already said. " + context))


async def converse(ws, chunks, speaker, seconds: float, run_tool=None, filler_pcm: bytes | None = None,
                   barge_in: bool = True, on_note=None, stop: asyncio.Event | None = None,
                   stall_s: float = 5, ping_timeout_s: float = 5) -> list[tuple]:
    """Full duplex for up to `seconds`: stream `chunks` (async iterator of PCM16 bytes) up while playing model
    audio down. Returns a log of (t, event, detail); on_note(t, event, detail) sees every line as it happens.

    barge_in: when the caller talks over the model, drop its queued audio and truncate its message server-side
              to what was actually heard. run_tool: async (name, arguments) -> dict. filler_pcm: played the
              instant a tool call starts. stop: an event that ends the conversation early.
    Raises Stalled when the server goes quiet mid-response and stops answering pings."""
    t0, log_, responding, in_response = time.perf_counter(), [], False, False
    last_event = t0
    item_id, item_start, item_len = None, 0, 0  # model message now playing: stream start position + bytes
    filled = set()
    said_so_far = []  # the reply being generated, as text: if the link dies mid-reply, the next one finishes it

    def note(event, detail=""):
        log_.append((round(time.perf_counter() - t0, 2), event, detail))
        log.debug("%s %s %s", *log_[-1])
        if on_note:
            on_note(*log_[-1])

    async def send_loop():  # caller -> server
        async for c in chunks:
            await send(ws, {"type": "input_audio_buffer.append", "audio": b64(c)})

    async def recv_loop():  # server -> speaker
        nonlocal responding, in_response, item_id, item_start, item_len, last_event
        async for raw in ws:
            last_event = time.perf_counter()
            e = json.loads(raw)
            kind = e["type"]
            if kind == "error":
                raise RuntimeError(e["error"])
            if kind == "response.created":
                in_response = True
                said_so_far.clear()
                note(kind, e.get("response", {}).get("status", ""))
            elif kind == "response.output_audio.delta":
                if not responding:
                    responding = True
                    note("first_audio_delta")
                if e["item_id"] != item_id:
                    item_id, item_start, item_len = e["item_id"], speaker.position, 0
                b = base64.b64decode(e["delta"])
                speaker.write(b)
                item_len += len(b)
            elif (kind == "response.output_item.added" and e["item"]["type"] == "function_call"
                  and filler_pcm and e["response_id"] not in filled):
                filled.add(e["response_id"])  # once per response, even if it makes several calls
                speaker.write(filler_pcm)
                note("filler", f'"{FILLER_TEXT}" while {e["item"]["name"]} runs')
            elif kind == "input_audio_buffer.speech_started":
                note(kind, f"{speaker.buffered_seconds():.2f}s of audio still queued")
                if barge_in and (dropped := speaker.clear()):
                    heard = min(max(speaker.played - item_start, 0), item_len)
                    detail = f"dropped {dropped / BYTES_PER_S:.2f}s"
                    if item_id and heard < item_len:  # fully heard (or only filler cut): nothing to truncate
                        heard_ms = heard * 1000 // BYTES_PER_S
                        await send(ws, {"type": "conversation.item.truncate", "item_id": item_id,
                                        "content_index": 0, "audio_end_ms": heard_ms})
                        detail += f", truncate at {heard_ms} ms"
                    note("barge_in", detail)
            elif kind == "input_audio_buffer.speech_stopped":
                note(kind)
            elif kind == "conversation.item.input_audio_transcription.delta":
                note("user_partial", e.get("delta", ""))  # live captions (streaming transcription models)
            elif kind == "conversation.item.input_audio_transcription.completed":
                note("user_said", e["transcript"].strip())
            elif kind == "conversation.item.input_audio_transcription.failed":  # the caller's words were not caught
                note("user_unheard", e.get("error", {}).get("message", ""))
            elif kind == "response.output_audio_transcript.delta":
                said_so_far.append(e.get("delta", ""))
            elif kind == "response.output_audio_transcript.done":
                said_so_far.clear()
                note("model_said", e["transcript"])
            elif kind == "response.done":
                responding = in_response = False
                note(kind, e["response"]["status"])
                if run_tool:
                    for name, args, out in await answer_tool_calls(ws, e["response"], run_tool):
                        note("tool_call", f"{name}({' '.join(args.split())}) -> {json.dumps(out)}")
            elif kind == "conversation.item.truncated":
                note(kind, e.get("audio_end_ms", ""))

    async def watchdog():  # a stall must be visible and recoverable, not a silent hang
        lags = collections.deque(maxlen=15)  # how late each 1 s tick woke: our own event loop's health
        while True:
            tick = time.perf_counter()
            await asyncio.sleep(1)
            lags.append(time.perf_counter() - tick - 1)
            quiet = time.perf_counter() - last_event
            if in_response and quiet > stall_s:
                METRICS.inc("stalls_total")
                t = time.perf_counter()
                try:  # pong = connection fine, the server is just slow; no pong = this connection is dead
                    await asyncio.wait_for(await ws.ping(), timeout=ping_timeout_s)
                    alive = f"connection alive (pong {(time.perf_counter() - t) * 1000:.0f} ms) -> server-side"
                except Exception as exc:
                    alive, dead = None, f"no pong ({type(exc).__name__})"
                lag = f"our loop max lag {max(lags):.2f}s over the last {len(lags)}s"
                note("STALL", f"no server event for {quiet:.1f}s mid-response; {alive or dead}; {lag}")
                if not alive:
                    if said_so_far:
                        note("model_cut", "".join(said_so_far))
                    raise Stalled(f"{dead} after {quiet:.0f}s without events; {lag}")
                while in_response and time.perf_counter() - last_event > stall_s:
                    await asyncio.sleep(1)  # report each slow stretch once

    tasks = [asyncio.create_task(f()) for f in (send_loop, recv_loop, watchdog)]
    if stop:
        tasks.append(asyncio.create_task(stop.wait()))
    # the loops only end by error or the server closing: the first task to finish ends the conversation
    done, pending = await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for t in done:
        t.result()  # re-raise a server error or a stall instead of hiding it
    return log_


async def converse_resilient(start, chunks, speaker, seconds: float, max_resumes: int = 2, on_note=None,
                             stop: asyncio.Event | None = None, context=None, **kw) -> list[tuple]:
    """converse() that survives a dead OpenAI connection.

    start: async () -> a new configured session. On a dead link: open a new one, replay the turns and tool calls
    so far (+ context(), our own state) and ask the model to carry on. At most `max_resumes` reconnects.
    The caller's audio is pumped through a queue, so nothing said during a reconnect is lost."""
    q: asyncio.Queue = asyncio.Queue()
    history, log_, t0 = [], [], time.perf_counter()

    async def pump():  # the one reader of `chunks`; its end or error is queued for whichever attempt is current
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

    cut = []  # a reply the dead link cut off mid-way

    def note(t, event, detail):
        log_.append((t, event, detail))
        if event in ("user_said", "model_said", "model_cut"):
            history.append(("user" if event == "user_said" else "assistant", detail))
            cut[:] = [detail] if event == "model_cut" else []
        elif event == "tool_call":  # replay what already happened, or the model redoes it (e.g. a spent code)
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
                return log_
            except DROPPED as e:
                if attempt == max_resumes or deadline - time.perf_counter() < 5:
                    raise
                METRICS.inc("resumes_total")
                detail = f"{type(e).__name__}: reconnecting ({attempt + 1}/{max_resumes}), replaying {len(history)} items"
                log.warning("RESUME %s", detail)
                note(round(time.perf_counter() - t0, 2), "RESUME", detail)
                with contextlib.suppress(Exception):
                    ws.transport.abort()  # a dead link never answers a polite close: don't wait 10 s for it
                ws = await start()
                finish = ("Your last reply was cut off by a network drop after the words above. Finish it from "
                          "exactly where it stopped, without repeating any of it. " if cut else "")
                await replay(ws, history, finish + (context() if context else ""))
                await send(ws, {"type": "response.create"})
        return log_
    finally:
        pumper.cancel()
        with contextlib.suppress(Exception):
            await ws.close()
