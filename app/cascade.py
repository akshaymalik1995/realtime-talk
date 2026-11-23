"""The cascade engine: speech-to-text -> a text model with our tools -> text-to-speech, each a swappable provider.

Same contract as realtime.converse_resilient: caller audio in (`chunks`), audio out (`speaker`), skills through
`run_tool`, and the same notes for the page (speech_started, user_said, model_said, first_audio_delta, filler,
tool_call, barge_in, response.done, STALL, RESUME). The conversation history is ours, so a dropped provider link
is just reopened: nothing to replay.

Turns: each final transcript is a user message. One worker answers them in order; if the caller speaks again
before a reply is spoken, the reply is dropped and the model answers everything at once. Tool calls always run
to the end (a booking must not be half-done). The caller talking over a reply stops its audio at once, and the
history keeps only the part they heard.
"""
import asyncio
import contextlib
import json
import logging
import re
import time

from app.realtime import BYTES_PER_S, DROPPED, FILLER_TEXT, Stalled

log = logging.getLogger(__name__)
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # । = Devanagari full stop

# Spoken digits, 0-9, as callers say them (formal and everyday forms). ponytail: single digits only ("बारह" = 12 stays)
DIGIT_WORDS = [
    "zero oh शून्य शुन्य जीरो ज़ीरो பூஜ்ஜியம் பூஜ்யம் சைபர் ஜீரோ",
    "one एक ஒன்று ஒன்னு",
    "two दो இரண்டு ரெண்டு",
    "three तीन மூன்று மூணு",
    "four चार நான்கு நாலு",
    "five पाँच पांच ஐந்து அஞ்சு",
    "six छह छः छै छे ஆறு",
    "seven सात ஏழு",
    "eight आठ எட்டு",
    "nine नौ ஒன்பது ஒம்பது",
]
DIGIT = {w: str(d) for d, words in enumerate(DIGIT_WORDS) for w in words.split()}
REPEAT = {"double": 2, "डबल": 2, "டபுள்": 2, "triple": 3, "ट्रिपल": 3, "ட்ரிபிள்": 3}
_d = "|".join(sorted(DIGIT, key=len, reverse=True)) + r"|\d+"
_unit = rf"(?:(?:{'|'.join(REPEAT)})\s*)?(?:{_d})"
_edge = r"\wऀ-ॣ०-෿"  # letters + Indic letters and vowel signs (not the । ॥ full stops)
DIGIT_RUN = re.compile(rf"(?<![{_edge}]){_unit}(?:[\s,–-]+{_unit})*(?![{_edge}])", re.IGNORECASE)
DIGIT_UNIT = re.compile(rf"(?:({'|'.join(REPEAT)})\s*)?({_d})", re.IGNORECASE)


def spoken_digits(text: str) -> str:
    """Numbers said digit by digit, in any language, become digits before the model reads them:
    "नौ आठ चार पाँच" -> "9845", "nine double eight" -> "988". A text model turning number words into digits
    itself gets them wrong (measured: Hindi words -> one digit wrong). Runs shorter than 3 digits stay words."""
    def join(m):
        units = DIGIT_UNIT.findall(m[0])
        if not any(w.lower() in DIGIT for _, w in units):  # already digits ("10, 11"): leave as said
            return m[0]
        digits = "".join(DIGIT.get(w.lower(), w) * REPEAT.get(r.lower(), 1) for r, w in units)
        return digits if len(digits) >= 3 else m[0]
    return DIGIT_RUN.sub(join, text)


def sentences(text: str) -> list[str]:
    return [p for p in SENTENCE_END.split(text.strip()) if p]


async def converse(parts, s, key, chunks, speaker, seconds: float, instructions: str, tools: list, run_tool,
                   vad: dict | None = None, filler_pcm: bytes | None = None, on_note=None, on_ready=None,
                   stop: asyncio.Event | None = None, max_resumes: int = 2) -> list[tuple]:
    """Run one conversation for up to `seconds`. `parts`: providers.Parts (classes; instantiated here)."""
    t0, log_ = time.perf_counter(), []
    messages = [{"role": "system", "content": instructions}]
    waiting: list[str] = []         # transcripts not answered yet
    wake = asyncio.Event()
    turn = 0                        # bumped by every transcript: a reply for an older turn is not spoken
    cut = 0                         # bumped by barge-in: the reply being spoken stops
    speaking = False
    resumes = 0

    def note(event, detail=""):
        log_.append((round(time.perf_counter() - t0, 2), event, detail))
        log.debug("%s %s %s", *log_[-1])
        if on_note:
            on_note(*log_[-1])

    def resume(what, e):
        nonlocal resumes
        if resumes == max_resumes:
            raise e
        resumes += 1
        note("RESUME", f"{what} link dropped ({type(e).__name__}): reconnecting ({resumes}/{max_resumes})")

    llm = parts.llm(s, key)
    async with parts.stt(s, key, vad) as stt, parts.tts(s, key) as tts:

        async def send_loop():  # caller -> speech-to-text
            async for c in chunks:
                with contextlib.suppress(*DROPPED):  # a dropped link loses this 100 ms; listen() reconnects it
                    await stt.send(c)

        async def listen():  # speech-to-text events -> turns
            nonlocal turn, cut
            while True:
                try:
                    async for kind, text in stt.events():
                        if kind == "speech_started":
                            note("input_audio_buffer.speech_started", f"{speaker.buffered_seconds():.2f}s of audio still queued")
                            if speaking or speaker.buffered_seconds() > 0:
                                cut += 1
                                dropped = speaker.clear()
                                note("barge_in", f"dropped {dropped / BYTES_PER_S:.2f}s")
                        elif kind == "speech_stopped":
                            note("input_audio_buffer.speech_stopped")
                        elif kind == "transcript":
                            note("user_said", text)
                            if text:
                                waiting.append(text)
                                turn += 1
                                wake.set()
                        elif kind == "failed":
                            note("user_unheard", text)
                    return
                except DROPPED as e:
                    resume("speech-to-text", e)
                    await stt.reconnect()

        async def think(my_turn) -> str | None:
            """The model's answer (running any tools it asks for), or None if the caller spoke again meanwhile."""
            filled = False
            for _ in range(5):  # tool rounds per answer: a model looping on tools must not hang the call
                for attempt in range(2):  # one retry: a single failed HTTP call should not end the call
                    try:
                        out = await llm.reply(messages, tools)
                        break
                    except Exception as e:
                        if attempt:
                            raise
                        note("STALL", f"text model: {type(e).__name__} {e}")
                if not out["tool_calls"]:
                    return out["text"].strip() if my_turn == turn else None
                messages.append({"role": "assistant", "content": out["text"] or None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in out["tool_calls"]]})
                if filler_pcm and not filled:
                    filled = True
                    speaker.write(filler_pcm)
                    note("filler", f'"{FILLER_TEXT}" while {out["tool_calls"][0]["name"]} runs')
                for c in out["tool_calls"]:  # always run to the end, even if the caller talks meanwhile
                    result = await run_tool(c["name"], c["arguments"])
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
                    note("tool_call", f"{c['name']}({' '.join(c['arguments'].split())}) -> {json.dumps(result)}")
            return "Sorry, I could not finish that. Could you say it again?"

        async def say(text, my_cut) -> str:
            """Speak `text` sentence by sentence; returns the part the caller heard."""
            nonlocal speaking
            start, spoken, heard_text = speaker.position, 0, []
            speaking = True
            try:
                for sentence in sentences(text):
                    for attempt in range(2):  # a stalled link loses one sentence at most: say it again
                        try:
                            async with contextlib.aclosing(tts.speak(sentence)) as audio:  # closing = cancel it
                                async for pcm in audio:
                                    if my_cut != cut:
                                        break
                                    if not spoken:  # the page shows the reply as its sound starts
                                        note("first_audio_delta")
                                        note("model_said", text)
                                    speaker.write(pcm)
                                    spoken += len(pcm)
                            break
                        except (Stalled, *DROPPED) as e:
                            note("STALL", f"text-to-speech: {type(e).__name__} {e}")
                            resume("text-to-speech", e)
                            await tts.reconnect()
                    if my_cut != cut:
                        heard = min(max(speaker.played - start, 0), spoken)
                        part = sentence[:len(sentence) * heard // max(spoken, 1)]  # ponytail: by audio share
                        return " ".join(heard_text + [part]).strip() + "…"
                    heard_text.append(sentence)
                return text
            finally:
                speaking = False

        async def worker():  # answers turns in order, one at a time
            while True:
                await wake.wait()
                wake.clear()
                while waiting:
                    messages.append({"role": "user", "content": spoken_digits(" ".join(waiting))})
                    waiting.clear()
                    answer = await think(turn)
                    if answer:
                        heard = await say(answer, cut)
                        messages.append({"role": "assistant", "content": heard})
                    note("response.done", "superseded" if answer is None else "completed")

        if on_ready:
            on_ready()
        tasks = [asyncio.create_task(f()) for f in (send_loop, listen, worker)]
        if stop:
            tasks.append(asyncio.create_task(stop.wait()))
        done, pending = await asyncio.wait(tasks, timeout=seconds, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in done:
            t.result()  # re-raise a provider error instead of hiding it
    return log_
