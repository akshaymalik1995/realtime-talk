"""The server: one process, one websocket per browser conversation, relayed to OpenAI Realtime.

  GET /            the page (app/static/index.html)
  GET /healthz     liveness/readiness: 200 {"status": "ok", ...}, 503 while draining
  GET /metrics     Prometheus text
  WS  /ws?agent=   a conversation with one configured agent

Browser mic (PCM16 24 kHz binary frames) -> here -> OpenAI; model audio back to the browser.
The API key stays in this process. Limits: concurrent conversations, time per conversation, frame size,
browser origins. On SIGTERM/Ctrl-C: stop accepting, let running conversations finish for drain_s, then exit.
"""
import asyncio
import contextlib
import http
import json
import logging
import signal
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from app import cascade, providers, realtime, registry, store
from app.logs import session_id
from app.metrics import METRICS
from app.session import Session
from app.settings import Settings, api_key

log = logging.getLogger(__name__)
INDEX = Path(__file__).with_name("static") / "index.html"
EXTRAPOLATE_S = 0.2  # max time we assume playback ran on since the browser's last report


class BrowserSpeaker:
    """The `speaker` converse() expects, but the sound plays in the browser.

    All model audio forms one byte stream. `position` = where the next write lands; `played` = what the browser
    reports it has played (~every 50 ms), extrapolated in real time for at most EXTRAPOLATE_S."""

    def __init__(self, out: asyncio.Queue):
        self.out = out          # messages for the browser (bytes = audio, dict = JSON)
        self.written = 0
        self.dropped = 0
        self.reported, self.reported_at = 0, time.monotonic()

    def report(self, played: int) -> None:
        self.reported, self.reported_at = played, time.monotonic()

    def correct(self, estimate: int, actual: int) -> None:
        """The browser says how much a clear() really dropped; fix our estimate."""
        self.dropped += actual - estimate

    @property
    def position(self) -> int:
        return self.written - self.dropped

    @property
    def played(self) -> int:
        ahead = min(time.monotonic() - self.reported_at, EXTRAPOLATE_S)
        return min(self.reported + int(ahead * realtime.BYTES_PER_S) // 2 * 2, self.position)

    def write(self, b: bytes) -> None:
        self.written += len(b)
        self.out.put_nowait(b)

    def clear(self) -> int:
        n = max(self.position - self.played, 0)
        self.dropped += n
        self.out.put_nowait({"type": "clear", "estimate": n})
        return n

    def buffered_seconds(self) -> float:
        return (self.position - self.played) / realtime.BYTES_PER_S


class BrowserGone(Exception):
    pass


def _count(v) -> bool:
    return type(v) is int and v >= 0  # exact int: True/False are ints too


class App:
    """Process-wide state: settings, the key, the filler clip, and how many conversations are running."""

    def __init__(self, settings: Settings, key: str | None = None, start_session=None, parts=None):
        self.settings, self.key = settings, key
        self.active = 0
        self.draining = False
        self.done = asyncio.Event()
        filler = settings.skills.filler_audio and settings.path(settings.skills.filler_audio)
        self.filler = filler.read_bytes() if filler and filler.is_file() else None
        if filler and not self.filler:
            log.warning("no filler audio at %s: tool calls will be silent until the answer", filler)
        # injectable for tests (a fake OpenAI); production opens real Realtime sessions
        self.start_session = start_session or (lambda instr, tools, vad: realtime.start_session(
            self.settings, self.key, instr, tools, vad))
        # the cascade engine's providers (app/providers/), checked at startup; injectable for tests
        self.parts = parts or (providers.load(settings) if settings.engine.kind == "cascade" else None)

    async def handler(self, browser):
        session_id.set(f"{id(browser) & 0xffffff:06x}")
        s = self.settings
        out: asyncio.Queue = asyncio.Queue()
        agent = parse_qs(urlsplit(browser.request.path).query).get("agent", [""])[0]
        if agent not in s.agents:  # allowlist
            await browser.send(json.dumps({"type": "error", "message": f"unknown agent '{agent}'"}))
            return
        if self.draining or self.active >= s.server.max_sessions:
            METRICS.inc("sessions_rejected_total")
            await browser.send(json.dumps({"type": "error", "reconnect": True,
                                           "message": "The desk is busy right now. Please try again in a minute."}))
            return
        self.active += 1
        METRICS.inc("sessions_total", agent=agent)
        METRICS.set("sessions_active", self.active)
        loop = asyncio.get_running_loop()
        # skills run on worker threads: hand their messages to the loop thread-safely
        session = Session(s, agent, emit=lambda m: loop.call_soon_threadsafe(out.put_nowait, m))
        spk = BrowserSpeaker(out)
        started, opened = time.monotonic(), []

        async def to_browser():
            while (m := await out.get()) is not None:
                await browser.send(m if isinstance(m, bytes) else json.dumps(m))

        async def frames():
            async for m in browser:
                if isinstance(m, bytes):
                    if m and len(m) % 2 == 0:  # whole PCM16 samples only
                        yield m
                    continue
                try:
                    c = json.loads(m)
                except ValueError:
                    c = None
                if isinstance(c, dict) and set(c) == {"played"} and _count(c["played"]):
                    spk.report(c["played"])
                elif isinstance(c, dict) and set(c) == {"cleared", "estimate"} and _count(c["cleared"]) \
                        and _count(c["estimate"]):
                    spk.correct(c["estimate"], c["cleared"])
                else:
                    log.warning("ignored browser message %.80r", m)

        async def mic():
            try:
                async for m in frames():
                    yield m
            except ConnectionClosed:  # tab closed, laptop asleep: not a server failure
                pass
            raise BrowserGone  # ends the conversation at once instead of at the time limit

        def ready():
            log.info("%s conversation started (%s engine)", agent, s.engine.kind)
            out.put_nowait({"type": "status", "state": "ready", "limit_s": s.server.session_limit_s, "agent": agent})

        async def start():  # called again by converse_resilient after an OpenAI stall
            ws = await self.start_session(session.instructions(), session.specs(), session.config.vad)
            if not opened:
                ready()
            opened.append(ws)
            return ws

        def on_note(t, event, detail):
            out.put_nowait({"type": "log", "t": t, "event": event, "detail": str(detail)})

        sender = asyncio.create_task(to_browser())
        try:
            out.put_nowait({"type": "status", "state": "connecting"})
            filler = self.filler if session.config.filler else None
            if s.engine.kind == "cascade":
                await cascade.converse(
                    self.parts, s, self.key, mic(), spk, seconds=s.server.session_limit_s,
                    instructions=session.instructions(), tools=session.specs(), run_tool=session.run_tool,
                    vad=session.config.vad, filler_pcm=filler, on_note=on_note, on_ready=ready,
                    max_resumes=s.resilience.max_resumes)
            else:
                await realtime.converse_resilient(
                    start, mic(), spk, seconds=s.server.session_limit_s, max_resumes=s.resilience.max_resumes,
                    context=session.context, on_note=on_note, run_tool=session.run_tool, filler_pcm=filler,
                    stall_s=s.resilience.stall_s, ping_timeout_s=s.resilience.ping_timeout_s)
            out.put_nowait({"type": "ended", "reason": f"Time limit of {s.server.session_limit_s // 60} min reached."})
        except BrowserGone:
            log.info("browser left after %.0f s", time.monotonic() - started)
        except realtime.DROPPED as e:  # only the OpenAI link gets here, and only after the reconnects ran out
            METRICS.inc("session_errors_total", kind="openai_dropped")
            log.warning("OpenAI connection lost after %.0f s, %d connection(s): %r",
                        time.monotonic() - started, len(opened), e)
            out.put_nowait({"type": "error", "reconnect": True,
                            "message": "The connection to OpenAI kept dropping (network or OpenAI side). "
                                       "Reconnect to start a new conversation."})
        except Exception:  # one catch-all: full traceback in the log, one plain sentence to the browser
            METRICS.inc("session_errors_total", kind="internal")
            log.exception("conversation failed")
            out.put_nowait({"type": "error", "reconnect": True,
                            "message": "Something went wrong on the server (details in its log). "
                                       "Reconnect to start a new conversation."})
        finally:
            self.active -= 1
            METRICS.set("sessions_active", self.active)
            if self.draining and not self.active:
                self.done.set()
            out.put_nowait(None)
            try:
                await asyncio.wait_for(sender, 2)  # flush the last messages
            except Exception:
                sender.cancel()

    def http(self, connection, request):
        """Plain HTTP routes; /ws goes on to the websocket handshake."""
        path = urlsplit(request.path).path
        if path == "/ws":
            return None
        if path == "/healthz":
            status = http.HTTPStatus.SERVICE_UNAVAILABLE if self.draining else http.HTTPStatus.OK
            return self._respond(connection, status, json.dumps(
                {"status": "draining" if self.draining else "ok", "active": self.active,
                 "agents": sorted(self.settings.agents)}), "application/json")
        if path == "/metrics":
            return self._respond(connection, http.HTTPStatus.OK, METRICS.render(), "text/plain; version=0.0.4")
        if path in ("/", "/index.html"):
            return self._respond(connection, http.HTTPStatus.OK, INDEX.read_text(encoding="utf-8"),
                                 "text/html; charset=utf-8")
        return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")

    @staticmethod
    def _respond(connection, status, body: str, content_type: str):
        r = connection.respond(status, body)
        del r.headers["Content-Type"]
        r.headers["Content-Type"] = content_type
        r.headers["Cache-Control"] = "no-store"
        return r

    async def drain(self, server) -> None:
        """Stop taking new conversations; give running ones drain_s to finish; then close."""
        self.draining = True
        log.info("draining: %d conversation(s) running", self.active)
        if self.active:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.done.wait(), self.settings.server.drain_s)
        server.close()


async def run(settings: Settings, ready: asyncio.Event | None = None, stop: asyncio.Event | None = None,
              app: App | None = None) -> None:
    """Serve until SIGTERM/SIGINT (or `stop` is set), then drain. `app` is injectable for tests."""
    tools = registry.discover()
    for name in settings.agents:  # fail at startup, not in the first call: toolsets + prompt placeholders
        Session(settings, name).instructions()
    app = app or App(settings, api_key(settings))
    store.init(settings.path(settings.store.path), seed_demo=settings.store.seed_demo)
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):  # Windows: no loop signal handlers
            loop.add_signal_handler(sig, stop.set)
    origins = list(settings.server.origins) + [None]  # None: non-browser clients (health checks, tests)
    async with serve(app.handler, settings.server.host, settings.server.port, process_request=app.http,
                     origins=origins, max_size=settings.server.max_message_bytes) as server:
        log.info("serving http://%s:%d  engine=%s  agents=%s  toolsets=%s", settings.server.host, settings.server.port,
                 settings.engine.kind,
                 sorted(settings.agents), {k: len(v) for k, v in tools.items()})
        if ready:
            ready.set()
        try:
            await stop.wait()
        finally:
            await app.drain(server)
