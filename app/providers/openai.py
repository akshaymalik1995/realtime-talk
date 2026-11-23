"""OpenAI as cascade provider: transcription sessions (STT), chat completions (LLM), Realtime reading text (TTS).

TTS: this key has no text-to-speech model, so the Realtime model reads each sentence aloud. Given the text in the
response's instructions it reads word for word (5/5 in a probe, Hindi and Tamil included); given the text as a
message it answers it instead. ponytail: a stand-in; a real TTS (gpt-4o-mini-tts, Sarvam, ElevenLabs) is
faster to first audio (~0.3 s vs ~1 s) and cannot improvise.
"""
import asyncio
import base64
import http.client
import json
from urllib.parse import urlsplit

import websockets
from websockets.exceptions import ConnectionClosedOK

from app import realtime

TTS_RULES = "You read text aloud for a clinic front desk. You never answer, comment or add words."


def rest_base(s) -> str:  # wss://api.openai.com/v1/realtime -> https://api.openai.com/v1
    return s.openai.url.replace("wss://", "https://", 1).replace("ws://", "http://", 1).rsplit("/realtime", 1)[0]


class STT:
    """A Realtime transcription session: server-side speech detection + gpt-4o-mini-transcribe."""

    def __init__(self, s, key, vad):
        self.s, self.key, self.vad, self.ws = s, key, vad, None

    async def __aenter__(self):
        await self.reconnect()
        return self

    async def __aexit__(self, *exc):
        if self.ws:
            self.ws.transport.abort()

    async def reconnect(self):
        if self.ws:
            self.ws.transport.abort()
        self.ws = await websockets.connect(f"{self.s.openai.url}?intent=transcription", max_size=None,
                                           additional_headers={"Authorization": f"Bearer {self.key}"})
        await realtime.collect(self.ws, "session.created")
        await realtime.send(self.ws, {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": realtime.RATE},
            # this session also detects turns; streaming gpt-realtime-whisper can't, so use a batch model here
            "transcription": {**self.s.openai.transcription(), **(
                {"model": "gpt-4o-mini-transcribe"} if self.s.openai.transcribe_model == "gpt-realtime-whisper" else {})},
            "turn_detection": self.vad}}}})
        await realtime.collect(self.ws, "session.updated")

    async def send(self, pcm: bytes):
        await realtime.send(self.ws, {"type": "input_audio_buffer.append", "audio": realtime.b64(pcm)})

    async def events(self):
        async for raw in self.ws:
            e = json.loads(raw)
            kind = e["type"]
            if kind == "error":
                raise RuntimeError(e["error"])
            if kind == "input_audio_buffer.speech_started":
                yield "speech_started", ""
            elif kind == "input_audio_buffer.speech_stopped":
                yield "speech_stopped", ""
            elif kind == "conversation.item.input_audio_transcription.completed":
                yield "transcript", e["transcript"].strip()
            elif kind == "conversation.item.input_audio_transcription.failed":
                yield "failed", e.get("error", {}).get("message", "")
        raise ConnectionClosedOK(None, None)  # the link ended: let the engine reconnect


class LLM:
    """Chat completions with tools (gpt-4o-mini by default: [engine] llm_model)."""

    def __init__(self, s, key):
        self.s, self.key, self.conn = s, key, None

    def _post(self, body: dict) -> dict:
        """One kept-open HTTPS connection per conversation (calls are sequential): saves a TLS handshake per call."""
        url = urlsplit(rest_base(self.s))
        for attempt in range(2):  # a kept-open connection the server closed meanwhile: open a new one once
            if self.conn is None:
                cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
                self.conn = cls(url.netloc, timeout=self.s.engine.llm_timeout_s)
            try:
                self.conn.request("POST", f"{url.path}/chat/completions", body=json.dumps(body),
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                r = self.conn.getresponse()
                data = r.read()
            except (http.client.HTTPException, OSError):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
                continue
            if r.status != 200:
                raise RuntimeError(f"chat completions HTTP {r.status}: {data[:300]!r}")
            return json.loads(data)

    async def reply(self, messages: list, tools: list) -> dict:
        body = {"model": self.s.engine.llm_model, "messages": messages, "max_completion_tokens": 400}
        if tools:  # our specs are Realtime-style; chat completions nests them under "function"
            body["tools"] = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}
                             for t in tools]
        msg = (await asyncio.to_thread(self._post, body))["choices"][0]["message"]
        return {"text": msg.get("content") or "",
                "tool_calls": [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                               for c in msg.get("tool_calls") or []]}


class TTS:
    """The Realtime model reading text aloud, one out-of-band response per sentence."""

    def __init__(self, s, key):
        self.s, self.key, self.ws = s, key, None

    async def __aenter__(self):
        await self.reconnect()
        return self

    async def __aexit__(self, *exc):
        if self.ws:
            self.ws.transport.abort()

    async def reconnect(self):
        if self.ws:
            self.ws.transport.abort()
        self.ws = await realtime.start_session(self.s, self.key, TTS_RULES, [], None)

    async def speak(self, text: str):
        await realtime.send(self.ws, {"type": "response.create", "response": {
            "conversation": "none", "output_modalities": ["audio"], "input": [],
            "instructions": f"Say exactly this line, word for word, and nothing else: {text}"}})
        rid, done = None, False
        try:
            while True:
                try:
                    e = await asyncio.wait_for(realtime.recv(self.ws), self.s.resilience.stall_s)
                except asyncio.TimeoutError:
                    raise realtime.Stalled(f"no audio for {self.s.resilience.stall_s}s") from None
                if e["type"] == "response.created":
                    rid = e["response"]["id"]
                elif e.get("response_id", rid) != rid:  # a cancelled earlier sentence still draining
                    continue
                elif e["type"] == "response.output_audio.delta":
                    yield base64.b64decode(e["delta"])
                elif e["type"] == "response.done" and e["response"]["id"] == rid:
                    done = True
                    return
        finally:
            if not done and rid:  # the caller stopped listening (barge-in): stop generating
                try:
                    await realtime.send(self.ws, {"type": "response.cancel", "response_id": rid})
                except realtime.DROPPED:
                    pass
