"""Logging: plain text for development, one JSON object per line for production log pipelines.
Every line carries the conversation id (a contextvar set per connection; tasks and worker threads inherit it)."""
import contextvars
import json
import logging
import sys
import time

session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")


class _SessionId(logging.Filter):
    def filter(self, record):
        record.session = session_id.get()
        return True


class _Json(logging.Formatter):
    def format(self, r):
        out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(r.created)) + f".{int(r.msecs):03d}Z",
               "level": r.levelname, "logger": r.name, "session": getattr(r, "session", "-"), "msg": r.getMessage()}
        if r.exc_info:
            out["exc"] = self.formatException(r.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup(level: str = "INFO", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_SessionId())
    handler.setFormatter(_Json() if fmt == "json" else
                         logging.Formatter("%(asctime)s %(levelname)s [%(session)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("websockets").setLevel(logging.WARNING)  # every page load is a "rejected" handshake
