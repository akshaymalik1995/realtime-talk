"""The WebRTC engine: the browser's audio goes straight to OpenAI over WebRTC; our server keeps the control.

    browser ──WebRTC audio (UDP, loss-tolerant)──► OpenAI Realtime
       │ offer / answer, page events                 ▲
       ▼                                             │ side connection: tools, guards, notes
    our server ──────────────────────────────────────┘

Why: over a websocket, audio rides one TCP stream; one lost packet stalls everything behind it while TCP backs off
(measured: 3 in 10 long replies froze, two came back after exactly 61.5 s). WebRTC carries audio as small UDP
packets: a lost one is a 20 ms glitch, not a frozen reply. The side connection carries only small events.

Flow: the page sends its SDP offer over our /ws; we create the call with OUR key (POST /v1/realtime/calls with
the session config: prompt, tools, voice, turn detection), hand back the answer, then attach to the same call
(wss .../realtime?call_id=...) to run tools and report events. The key never reaches the browser. Barge-in and
truncation happen inside OpenAI (it knows what was played). The filler clip is skipped here: the model's own
"let me check" covers tool calls.
"""
import asyncio
import json
import logging
import time
import urllib.request
import uuid

import websockets

from app import realtime

log = logging.getLogger(__name__)
MAX_SDP = 20_000  # bytes; a real offer is ~3-6 KB


def rest_base(s) -> str:  # wss://api.openai.com/v1/realtime -> https://api.openai.com/v1
    return s.openai.url.replace("wss://", "https://", 1).replace("ws://", "http://", 1).rsplit("/realtime", 1)[0]


def call_config(s, instructions: str, tools: list, vad: dict | None) -> dict:
    """The session, as sent with the call (audio format is WebRTC's own)."""
    return {
        "type": "realtime", "model": s.openai.model, "instructions": instructions,
        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        "max_output_tokens": s.openai.max_output_tokens,
        "audio": {"input": {"turn_detection": vad,
                            "transcription": {"model": s.openai.transcribe_model} if s.openai.transcribe_model else None},
                  "output": {"voice": s.openai.voice}},
    }


def _post_call(s, key, offer_sdp: str, session: dict) -> tuple[str, str]:
    boundary = uuid.uuid4().hex
    part = lambda name, ctype, body: (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n"
                                      f"Content-Type: {ctype}\r\n\r\n{body}\r\n")
    body = (part("sdp", "application/sdp", offer_sdp) + part("session", "application/json", json.dumps(session))
            + f"--{boundary}--\r\n").encode()
    req = urllib.request.Request(f"{rest_base(s)}/realtime/calls", data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode(), r.headers["Location"].rstrip("/").rsplit("/", 1)[-1]


async def create_call(s, key, offer_sdp: str, session: dict) -> tuple[str, str]:
    """POST /v1/realtime/calls -> (answer SDP, call id)."""
    return await asyncio.to_thread(_post_call, s, key, offer_sdp, session)


def _hangup(s, key, call_id):
    req = urllib.request.Request(f"{rest_base(s)}/realtime/calls/{call_id}/hangup", data=b"", method="POST",
                                 headers={"Authorization": f"Bearer {key}"})
    urllib.request.urlopen(req, timeout=10).close()


async def attach(s, key, call_id):
    """The side connection to a running call."""
    return await websockets.connect(f"{s.openai.url}?call_id={call_id}", max_size=None,
                                    additional_headers={"Authorization": f"Bearer {key}"})


async def converse(s, key, offers: asyncio.Queue, send_answer, chunks, seconds: float, instructions: str,
                   tools: list, run_tool, vad: dict | None = None, on_note=None, on_ready=None,
                   stop: asyncio.Event | None = None, max_resumes: int = 2, create=None, connect=None,
                   hangup=None) -> list[tuple]:
    """One WebRTC conversation. offers: the page's SDP offer arrives here; send_answer(sdp) returns OpenAI's.
    chunks: the page's message stream, drained only to notice it leaving. create/connect: injectable for tests."""
    create, connect = create or create_call, connect or attach
    hangup = hangup or (lambda call_id: asyncio.to_thread(_hangup, s, key, call_id))
    t0, log_ = time.perf_counter(), []

    def note(event, detail=""):
        log_.append((round(time.perf_counter() - t0, 2), event, detail))
        log.debug("%s %s %s", *log_[-1])
        if on_note:
            on_note(*log_[-1])

    async def drain():  # ends (or raises) when the page leaves
        async for _ in chunks:
            pass

    leave = asyncio.create_task(drain())
    call_id = None
    try:
        got = asyncio.create_task(asyncio.wait_for(offers.get(), 30))
        await asyncio.wait([got, leave], return_when=asyncio.FIRST_COMPLETED)
        if not got.done():
            got.cancel()
            leave.result()  # the page left before offering: BrowserGone
            return log_
        offer = got.result()
        answer, call_id = await create(s, key, offer, call_config(s, instructions, tools, vad))
        await send_answer(answer)
        note("call", call_id)

        speaking, resumes = False, 0

        async def events():
            nonlocal speaking, resumes
            ws = await connect(s, key, call_id)
            if on_ready:
                on_ready()
            while True:
                try:
                    async for raw in ws:
                        e = json.loads(raw)
                        kind = e["type"]
                        if kind == "error":
                            raise RuntimeError(e["error"])
                        if kind == "input_audio_buffer.speech_started":
                            note(kind, "")
                            if speaking:
                                note("barge_in", "OpenAI stops and truncates the reply")
                        elif kind == "input_audio_buffer.speech_stopped":
                            note(kind)
                        elif kind == "conversation.item.input_audio_transcription.completed":
                            note("user_said", e["transcript"].strip())
                        elif kind == "conversation.item.input_audio_transcription.failed":
                            note("user_unheard", e.get("error", {}).get("message", ""))
                        elif kind == "output_audio_buffer.started":
                            speaking = True
                            note("first_audio_delta")
                        elif kind in ("output_audio_buffer.stopped", "output_audio_buffer.cleared"):
                            speaking = False
                            note("speaking_done")
                        elif kind == "response.output_audio_transcript.done":
                            note("model_said", e["transcript"])
                        elif kind == "response.done":
                            note(kind, e["response"]["status"])
                            for name, args, out in await realtime.answer_tool_calls(ws, e["response"], run_tool):
                                note("tool_call", f"{name}({' '.join(args.split())}) -> {json.dumps(out)}")
                    return
                except realtime.DROPPED as exc:  # the side link died; the call (audio) carries on: re-attach
                    if resumes == max_resumes:
                        raise
                    resumes += 1
                    note("RESUME", f"control link dropped ({type(exc).__name__}): re-attaching ({resumes}/{max_resumes})")
                    ws.transport.abort()
                    ws = await connect(s, key, call_id)

        runner = asyncio.create_task(events())
        waits = [runner, leave] + ([asyncio.create_task(stop.wait())] if stop else [])
        done, pending = await asyncio.wait(waits, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            t.result()
        return log_
    finally:
        leave.cancel()
        if call_id:
            try:  # end the call at OpenAI too: stops billing at once instead of at its own timeout
                await hangup(call_id)
            except Exception as e:
                log.warning("hangup %s failed: %r", call_id, e)
