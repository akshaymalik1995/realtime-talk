"""Providers for the cascade engine (app/cascade.py): speech-to-text, a text model, text-to-speech.

A provider is a module here (app/providers/<name>.py) defining any of three classes, picked by name in config:

    [engine]  kind = "cascade"   stt = "openai"   llm = "openai"   tts = "openai"

    STT(settings, key, vad)    async context manager. send(pcm16_24k) feeds audio; events() yields
                               ("speech_started" | "speech_stopped" | "transcript" | "failed", text).
                               reconnect() replaces a dropped link.
    LLM(settings, key)         async reply(messages, tools) -> {"text": str, "tool_calls": [{"id", "name", "arguments"}]}
                               messages use the OpenAI chat format (most providers accept it); tools are our specs.
    TTS(settings, key)         async context manager. speak(text) yields PCM16 24 kHz chunks; reconnect().

A link that dies raises one of realtime.DROPPED (or realtime.Stalled); the engine reconnects and carries on.
Adding Sarvam or ElevenLabs = a new module with the classes it offers, then its name in config.
"""
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Parts:
    stt: type
    llm: type
    tts: type


def load(s) -> Parts:
    """The classes named in [engine]. Fails at startup on a missing provider or class."""
    def get(part):
        name = getattr(s.engine, part)
        try:
            return getattr(importlib.import_module(f"app.providers.{name}"), part.upper())
        except (ImportError, AttributeError):
            raise LookupError(f"engine.{part} = '{name}': no {part.upper()} in app/providers/{name}.py") from None
    return Parts(get("stt"), get("llm"), get("tts"))
