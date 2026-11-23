"""One-time codes and the SMS that carries them.

The code never reaches the model: Otp texts it and checks it; tools only ever see "code sent to phone ending 2367".
Production: implement Sms.send with a real provider (Twilio, MSG91, ...) and pick it in app.session.
"""
import hmac
import logging
import re
import secrets
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)


class Sms(Protocol):
    def send(self, to: str, text: str) -> None: ...


class SimulatedSms:
    """Development/demo SMS: keeps an outbox and hands every message to `on_send` (the page's phone panel)."""

    def __init__(self, on_send: Callable[[dict], None] = lambda m: None):
        self.outbox: list[dict] = []
        self.on_send = on_send

    def send(self, to: str, text: str) -> None:
        msg = {"to": to, "text": text}
        self.outbox.append(msg)
        self.on_send(msg)

    def last_code(self) -> str:
        """Test helper: the digits of the latest SMS (what a patient reads off their phone)."""
        return re.sub(r"\D", "", self.outbox[-1]["text"]) if self.outbox else ""


class Otp:
    """One pending code per conversation. A new code replaces the old; codes are single-use, expire after `ttl`
    seconds, lock after `max_tries` wrong answers, and at most `max_sends` are texted per conversation."""

    def __init__(self, sms: Sms, sender: str = "the clinic", ttl: float = 300, max_tries: int = 3, max_sends: int = 3,
                 gen: Callable[[], str] = lambda: f"{secrets.randbelow(10 ** 6):06d}", clock=time.monotonic):
        self.sms, self.sender, self.ttl, self.max_tries, self.max_sends = sms, sender, ttl, max_tries, max_sends
        self.gen, self.clock = gen, clock
        self.sent = 0
        self.pending = None  # {"purpose", "payload", "code", "at", "tries"}

    def send(self, phone: str, purpose: str, payload) -> dict | None:
        """Text a code. Returns a problem dict instead when the per-conversation limit is reached."""
        if self.sent >= self.max_sends:
            log.warning("otp send limit reached (%d)", self.max_sends)
            return {"status": "too_many_codes", "problem": "too many codes were sent in this call",
                    "next": "Apologise and suggest calling again later."}
        code = self.gen()
        self.pending = {"purpose": purpose, "payload": payload, "code": code, "at": self.clock(), "tries": 0}
        self.sent += 1
        self.sms.send("+" + phone, f"Your {self.sender} {purpose} code is {code}.")
        return None

    def check(self, code) -> tuple[str | None, object, dict | None]:
        """(purpose, payload, None) on a match, else (None, None, problem-for-the-model)."""
        p = self.pending
        if not p:
            return None, None, {"status": "no_code_pending", "problem": "no code was sent in this conversation"}
        if self.clock() - p["at"] > self.ttl:
            self.pending = None
            return None, None, {"status": "expired", "problem": "the code expired; start again"}
        if not hmac.compare_digest(re.sub(r"\D", "", str(code)), p["code"]):
            p["tries"] += 1
            if p["tries"] >= self.max_tries:
                self.pending = None
                return None, None, {"status": "locked", "problem": "too many wrong codes; start again"}
            return None, None, {"status": "wrong_code", "tries_left": self.max_tries - p["tries"]}
        self.pending = None
        return p["purpose"], p["payload"], None
