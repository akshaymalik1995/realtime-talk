"""A Session is one conversation. Everything the agent knows lives here, in our code, not in the model:
which tools this agent may use, which patient is verified, the one pending code, the shared-phone choice.
That is also what makes reconnects cheap: a new model connection only needs the words so far.
"""
import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from app import registry, store
from app.domain import numbers
from app.metrics import METRICS
from app.settings import Settings, SettingsError
from app.verification import Otp, SimulatedSms

log = logging.getLogger(__name__)


class _Strict(dict):
    def __missing__(self, key):
        raise SettingsError(f"prompt uses unknown placeholder {{{key}}} (literal braces: write {{{{ and }}}})")


@dataclass
class Session:
    settings: Settings
    agent: str
    emit: Callable[[dict], None] = lambda m: None   # UI hook: every tool call, every SMS
    now: Callable[[], datetime] = datetime.now
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    patient: dict | None = None                     # verified caller: {"id", "name"}
    choices: list | None = None                     # patients on a just-verified shared phone
    sms: object = None
    otp: Otp | None = None
    said: list = field(default_factory=list)        # the caller's latest words (transcripts), newest last

    def __post_init__(self):
        if self.agent not in self.settings.agents:
            raise SettingsError(f"unknown agent '{self.agent}'")
        self.config = self.settings.agents[self.agent]
        self.store = store.open_store(self.settings)  # where patients and appointments live ([store] backend)
        self.sms = self.sms or SimulatedSms(on_send=lambda m: self.emit({"type": "sms", **m}))
        o = self.settings.otp
        self.otp = self.otp or Otp(self.sms, self.settings.clinic.name, o.ttl_s, o.max_tries, o.max_sends)
        self.tools = registry.select(self.config.toolsets)  # this agent's allowlist
        self._unheard = 0                       # turns the caller finished whose transcript has not arrived yet
        self._heard = asyncio.Event()

    def hear(self, event: str, detail: str) -> None:
        """The engine's notes about the caller's speech: tool arguments are checked against their own words."""
        if event == "input_audio_buffer.speech_stopped":
            self._unheard += 1
        elif event in ("user_said", "user_unheard"):
            self._unheard = max(self._unheard - 1, 0)
            if detail:
                self.said = (self.said + [detail])[-3:]
            self._heard.set()

    async def caller_words(self) -> str:
        """The caller's latest words. Their transcript often lands just after the model's tool call: wait briefly."""
        if self._unheard:
            self._heard.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._heard.wait(), 1.5)
        return self.said[-1] if self.said else ""

    def state(self) -> dict:
        """What the UI (and a reconnect) needs to know. Never the code itself."""
        return {"patient": self.patient,
                "code_pending_for": self.otp.pending["purpose"] if self.otp.pending else None,
                "choosing_between": [c["name"] for c in self.choices] if self.choices else None}

    def specs(self) -> list[dict]:
        return [s.spec for s in self.tools.values()]

    def instructions(self) -> str:
        """The agent's prompt template (config) filled with today's date, the clinic and one line per skill."""
        template = self.settings.path(self.config.prompt).read_text(encoding="utf-8")
        today = self.now()
        lines = "\n".join(f"- {s.name}: {s.when}" for s in self.tools.values() if s.when)
        return template.format_map(_Strict(clinic=self.settings.clinic.name, skills=lines,
                                           today=f"{today:%A} {today.day} {today:%B %Y}")).strip()

    async def run_tool(self, name: str, arguments: str) -> dict:
        """The one door from the model to our code: allowlist, the caller's own digits, then registry.call
        (validate, guard, timeout)."""
        s = self.tools.get(name)
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            args = {}
        if isinstance(args, dict) and numbers.VALID.keys() & args.keys():
            if fixed := numbers.fix_digits(args, await self.caller_words()):
                log.info("%s: digits from the caller's words, not the model's hearing: %s", name, fixed)
                METRICS.inc("digits_corrected_total", skill=name)
                arguments = json.dumps(args)
        out = await registry.call(s, self, arguments, self.settings.skills.timeout_s) if s else \
            {"error": f"unknown tool {name!r}"}
        status = "error" if "error" in out else str(out.get("status", "ok"))
        METRICS.inc("tool_calls_total", skill=name if s else "unknown", status=status)
        self.emit({"type": "desk", "skill": name, "args": args if isinstance(args, dict) else {}, "result": out,
                   "state": self.state()})
        return out

    def context(self) -> str:
        """Our side of the conversation, for replay after a reconnect."""
        return f"Our state: {json.dumps(self.state())}"
