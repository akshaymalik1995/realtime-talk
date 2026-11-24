"""Settings: one TOML file + environment overrides, checked once at startup.

Every tunable lives in config, not code: model, voice, turn detection, token caps, limits, prompts, agents.

- File: $APP_CONFIG, else config/app.toml.
- Override any value with APP__<SECTION>__<KEY> (values parsed as JSON when they can be), e.g.
  APP__SERVER__PORT=9000  APP__OPENAI__MODEL=gpt-realtime  APP__AGENTS__FRONT_DESK__TOOLSETS='["identity"]'
- Secrets only from the environment (OPENAI_API_KEY). `openai.dev_key_ini` is a development convenience
  that production config leaves empty.
- Unknown keys and wrong types are errors: a typo must stop the server, not be silently ignored.
"""
import configparser
import json
import os
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent


class SettingsError(ValueError):
    pass


@dataclass(frozen=True)
class Server:
    host: str = "127.0.0.1"
    port: int = 8000
    origins: tuple = ("http://127.0.0.1:8000", "http://localhost:8000")  # browsers allowed to open /ws
    max_sessions: int = 20          # concurrent conversations; more get a "busy" answer
    session_limit_s: int = 300      # hard cap per conversation (bounds paid usage)
    max_message_bytes: int = 65536  # largest browser frame accepted
    drain_s: int = 10               # on shutdown, how long running conversations may finish
    log_format: str = "text"        # text | json
    log_level: str = "INFO"


@dataclass(frozen=True)
class OpenAI:
    url: str = "wss://api.openai.com/v1/realtime"
    model: str = "gpt-realtime-mini"
    voice: str = "marin"
    transcribe_model: str = "gpt-4o-mini-transcribe"  # captions only: the model hears the audio itself
    transcribe_language: str = ""   # ISO code ("hi", "ta"); "" = guess per turn (short phrases often land in the
                                    # wrong script: Hindi "हाँ" came back in Korean)
    transcribe_prompt: str = ""     # a hint, for models that take one (gpt-realtime-whisper does not)
    noise_reduction: str = ""       # "near_field" (headset), "far_field" (phone on speaker, laptop); "" = off
    max_output_tokens: int = 1000   # per response; audio counts (~300 tokens = ~10 s of speech)
    dev_key_ini: str = ""           # dev only: an ini file with [dev] openai_key

    def transcription(self) -> dict | None:
        """The input transcription config (the page's captions), or None."""
        if not self.transcribe_model:
            return None
        return {"model": self.transcribe_model, **({"language": self.transcribe_language} if self.transcribe_language else {}),
                **({"prompt": self.transcribe_prompt} if self.transcribe_prompt else {})}

    def noise(self) -> dict | None:
        return {"type": self.noise_reduction} if self.noise_reduction else None


@dataclass(frozen=True)
class Engine:
    kind: str = "realtime"          # realtime: speech-to-speech over our websocket | webrtc: the same model, audio
                                    # straight from the browser (app/webrtc.py) | cascade: stt -> llm -> tts
    stt: str = "openai"             # cascade parts: provider modules in app/providers/
    llm: str = "openai"
    tts: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 20.0
    transcript_first: bool = True   # speech-to-speech engines: reply only once the caller's transcript is in the
                                    # model's context (exact digits; ~0.4 s later; no reply to wordless noise)


@dataclass(frozen=True)
class Resilience:
    stall_s: float = 2.0            # quiet this long mid-response -> ping (replies stream ~8x faster than they play)
    ping_timeout_s: float = 2.0     # no pong in this long -> the connection is dead: reconnect
    max_resumes: int = 2            # reconnects per conversation


@dataclass(frozen=True)
class Clinic:
    name: str = "City Care Clinic"


@dataclass(frozen=True)
class Store:
    backend: str = "sqlite"         # sqlite: the demo clinic in `path` | docassist: DocAssist's MySQL ([docassist])
    path: str = "data/app.db"
    seed_demo: bool = True          # demo doctors/patients on an empty database (never in production)


@dataclass(frozen=True)
class DocAssist:
    config_ini: str = ""            # dev: DocAssist's config.ini (sql_* keys); else DOCASSIST_SQL_* environment variables
    section: str = "dev"            # which config.ini section
    hospital_id: int = 999          # every doctor, patient and appointment is this hospital's


@dataclass(frozen=True)
class Otp:
    ttl_s: int = 300
    max_tries: int = 3
    max_sends: int = 3              # codes texted per conversation (stops SMS pumping)


@dataclass(frozen=True)
class Skills:
    timeout_s: float = 10.0         # a skill slower than this returns an error instead of hanging the call
    filler_audio: str = "assets/filler_marin.pcm"  # played the instant a tool call starts; "" = none


@dataclass(frozen=True)
class Agent:
    prompt: str                     # template file, relative to the project root
    toolsets: tuple                 # skill modules (app/skills/<name>.py) this agent may use; "*" = all
    vad: dict = field(default_factory=lambda: {"type": "server_vad", "silence_duration_ms": 500})
    filler: bool = True
    greeting: str = ""              # said first, once the call connects ({clinic} filled in); "" = wait for the caller


@dataclass(frozen=True)
class Settings:
    server: Server
    openai: OpenAI
    engine: Engine
    resilience: Resilience
    clinic: Clinic
    store: Store
    docassist: DocAssist
    otp: Otp
    skills: Skills
    agents: dict
    root: Path = ROOT

    def path(self, p: str) -> Path:
        """A configured path, relative to the project root unless absolute."""
        return Path(p) if Path(p).is_absolute() else self.root / p


SECTIONS = {"server": Server, "openai": OpenAI, "engine": Engine, "resilience": Resilience, "clinic": Clinic, "store": Store,
            "docassist": DocAssist, "otp": Otp, "skills": Skills}


def _value(v, kind, where):
    if kind is tuple and isinstance(v, list):
        v = tuple(v)
    ok = (isinstance(v, (int, float)) and not isinstance(v, bool)) if kind is float else \
        (type(v) is int) if kind is int else isinstance(v, kind)  # bool is an int: check exactly
    if not ok:
        raise SettingsError(f"{where}: expected {kind.__name__}, got {type(v).__name__}")
    return float(v) if kind is float else v


def _build(cls, data, where):
    if not isinstance(data, dict):
        raise SettingsError(f"{where}: expected a table")
    known = {f.name: f for f in fields(cls)}
    if unknown := sorted(set(data) - set(known)):
        raise SettingsError(f"{where}: unknown key(s) {unknown}")
    kw = {}
    for name, f in known.items():
        if name in data:
            kw[name] = _value(data[name], f.type, f"{where}.{name}")
        elif f.default is MISSING and f.default_factory is MISSING:
            raise SettingsError(f"{where}: missing required key '{name}'")
    return cls(**kw)


def _apply_env(data: dict, env: Mapping[str, str]) -> None:
    for key, raw in env.items():
        if not key.startswith("APP__"):
            continue
        path = [p.lower() for p in key[5:].split("__")]
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        node = data
        for p in path[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise SettingsError(f"{key}: {p} is not a table")
        node[path[-1]] = value


def load(path: str | Path | None = None, env: Mapping[str, str] = os.environ) -> Settings:
    path = Path(path or env.get("APP_CONFIG") or ROOT / "config" / "app.toml")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SettingsError(f"cannot read settings {path}: {e}") from None
    _apply_env(data, env)
    if unknown := sorted(set(data) - set(SECTIONS) - {"agents"}):
        raise SettingsError(f"unknown section(s) {unknown}")
    sections = {name: _build(cls, data.get(name, {}), name) for name, cls in SECTIONS.items()}
    agents = {name: _build(Agent, spec, f"agents.{name}") for name, spec in data.get("agents", {}).items()}
    if not agents:
        raise SettingsError("no [agents.<name>] defined")
    s = Settings(**sections, agents=agents)
    for name, a in agents.items():
        if not s.path(a.prompt).is_file():
            raise SettingsError(f"agents.{name}.prompt: no file {s.path(a.prompt)}")
        if not a.toolsets:
            raise SettingsError(f"agents.{name}.toolsets: empty")
    if s.store.backend not in ("sqlite", "docassist"):
        raise SettingsError("store.backend: sqlite or docassist")
    if s.engine.kind not in ("realtime", "webrtc", "cascade"):
        raise SettingsError("engine.kind: realtime, webrtc or cascade")
    if s.server.log_format not in ("text", "json"):
        raise SettingsError("server.log_format: text or json")
    return s


def api_key(s: Settings, env: Mapping[str, str] = os.environ) -> str:
    """OPENAI_API_KEY from the environment; in development optionally from openai.dev_key_ini. Never logged."""
    if key := env.get("OPENAI_API_KEY"):
        return key
    if s.openai.dev_key_ini:
        c = configparser.ConfigParser(interpolation=None)
        if c.read(s.openai.dev_key_ini) and c.has_option("dev", "openai_key"):
            return c["dev"]["openai_key"].strip()
    raise SettingsError("no OPENAI_API_KEY in the environment (and no usable openai.dev_key_ini)")
