"""Mic/speaker I/O and the PCM16 <-> base64 conversions the Realtime API needs."""
import asyncio
import base64
import os
import subprocess
import tempfile
import threading
import wave

import numpy as np
import sounddevice as sd

RATE = 24000      # OpenAI Realtime: 24 kHz for both input and output
CHANNELS = 1      # mono
DTYPE = "int16"   # PCM16 = signed 16-bit little-endian samples


def record(seconds: float, rate: int = RATE) -> np.ndarray:
    """Block until `seconds` of mic audio is captured. Returns int16 array, shape (n,)."""
    rec = sd.rec(int(seconds * rate), samplerate=rate, channels=CHANNELS, dtype=DTYPE)
    sd.wait()
    return rec[:, 0]


def play(samples: np.ndarray, rate: int = RATE) -> None:
    sd.play(samples, samplerate=rate)
    sd.wait()


def rms(samples: np.ndarray) -> float:
    """Loudness: root-mean-square on the [-1, 1] scale. Silence ~0.00x, speech ~0.02-0.2."""
    return float(np.sqrt(np.mean(pcm16_to_float(samples) ** 2)))


def float_to_pcm16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def pcm16_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) / 32768


def to_bytes(samples: np.ndarray) -> bytes:
    return samples.astype("<i2").tobytes()  # '<i2' = little-endian int16, what the API expects


def from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype="<i2")


def to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def from_b64(s: str) -> bytes:
    return base64.b64decode(s)


# ---- streaming (notebook 03+) ----

async def mic_chunks(chunk_ms: int = 100):
    """Live mic as an async stream of PCM16 byte chunks. The audio thread hands each
    chunk to the asyncio loop via a queue; the stream closes when the generator does."""
    loop, q = asyncio.get_running_loop(), asyncio.Queue()

    def cb(indata, frames, t, status):  # runs on PortAudio's thread, not the event loop
        loop.call_soon_threadsafe(q.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE,
                           blocksize=RATE * chunk_ms // 1000, callback=cb):
        while True:
            yield await q.get()


def say(text: str) -> np.ndarray:
    """Deterministic text → PCM16 at 24 kHz with Windows' built-in synth (a fake user voice).
    ponytail: Windows-only (System.Speech); swap for any TTS on other OSes."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "say.wav")
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, "
              "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
              "$s.SetOutputToWaveFile($env:SAY_OUT, $f); $s.Speak($env:SAY_TEXT); $s.Dispose()")
        # text goes via env var, never into the command string (no injection)
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                       env={**os.environ, "SAY_OUT": path, "SAY_TEXT": text})
        with wave.open(path) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


async def file_chunks(samples: np.ndarray, chunk_ms: int = 100):
    """Fake mic: plays `samples` in real-time-paced chunks, then silence forever.
    The server can't tell this from a real mic: it only ever sees chunks."""
    n = RATE * chunk_ms // 1000
    for i in range(0, len(samples), n):
        yield to_bytes(samples[i:i + n])
        await asyncio.sleep(chunk_ms / 1000)
    while True:
        yield bytes(n * 2)
        await asyncio.sleep(chunk_ms / 1000)


class ScriptedUser:
    """Fake user for multi-turn tests (notebook 06+). Says each line, then waits until the model has
    answered (a response that spoke and completed) and its audio has played out and stayed quiet for `quiet_s`
    (a tool call's second response may follow the first), then says the next line.
    A line may be a function, called when it is time to speak (e.g. to read out a code just texted).
    After the last answer `done` is set: pass it as converse(stop=...). Use: converse(ws, u.chunks(), spk,
    ..., on_note=u.on_note, stop=u.done)."""

    def __init__(self, lines, speaker, quiet_s: float = 1.5, chunk_ms: int = 100):
        self.lines, self.speaker, self.quiet_s, self.chunk_ms = list(lines), speaker, quiet_s, chunk_ms
        self.answered, self.done, self.said = asyncio.Event(), asyncio.Event(), []
        self._spoke = False  # the response in progress has said something

    def on_note(self, t, event, detail):
        """An answer = a response that spoke AND completed. A response cut off by our own next
        sentence (server VAD ends a turn at each pause) still reports what it said: not an answer."""
        if event == "model_said":
            self._spoke = True
        elif event == "response.done":
            if detail != "cancelled" and self._spoke:  # "incomplete" (token cap) also ends the model's turn
                self.answered.set()
            self._spoke = False

    async def chunks(self):
        n, step = RATE * self.chunk_ms // 1000, self.chunk_ms / 1000
        silence = bytes(n * 2)
        for line in self.lines:
            text = line() if callable(line) else line
            self.said.append(text)
            self.answered.clear()
            pcm = await asyncio.to_thread(say, text)  # TTS takes ~1 s; don't freeze the audio loop
            for i in range(0, len(pcm), n):
                yield to_bytes(pcm[i:i + n])
                await asyncio.sleep(step)
            last, quiet = None, 0.0
            while not (self.answered.is_set() and quiet >= self.quiet_s):
                yield silence
                await asyncio.sleep(step)
                pos = self.speaker.position
                quiet = quiet + step if pos == last and self.speaker.buffered_seconds() < 0.05 else 0.0
                last = pos
        self.done.set()
        while True:
            yield silence
            await asyncio.sleep(step)


class Speaker:
    """Playback buffer drained by the sound card's callback. write() appends model audio;
    clear() drops everything not yet played (needed for barge-in)."""

    def __init__(self):
        self.buf, self.lock = bytearray(), threading.Lock()
        self.played = 0  # bytes actually sent to the sound card (silence padding excluded)
        self.stream = sd.OutputStream(samplerate=RATE, channels=CHANNELS, dtype=DTYPE, callback=self._cb)

    def _cb(self, outdata, frames, t, status):
        n = frames * 2
        with self.lock:
            chunk = bytes(self.buf[:n])
            del self.buf[:n]
            self.played += len(chunk)
        outdata[:, 0] = from_bytes(chunk.ljust(n, b"\0"))  # pad with silence if buffer runs dry

    # Stream positions (bytes). Everything written forms one linear stream, so
    # "how much of message X was heard" = played - X's start position, clamped.
    @property
    def position(self) -> int:
        """Stream position where the next write() will land."""
        with self.lock:
            return self.played + len(self.buf)

    def write(self, b: bytes) -> None:
        with self.lock:
            self.buf.extend(b)

    def clear(self) -> int:
        """Drop unplayed audio; returns how many bytes were dropped. Afterwards `played`
        is exact: nothing more is consumed until new audio is written."""
        with self.lock:
            n = len(self.buf)
            self.buf.clear()
            return n

    def buffered_seconds(self) -> float:
        return len(self.buf) / 2 / RATE

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
