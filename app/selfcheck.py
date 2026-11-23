"""Free checks for the whole backend (no OpenAI, no key, no sound card): python -m app.selfcheck

Each check is a function, so notebooks can run them one at a time. A fake OpenAI (a local websocket server
that speaks the Realtime protocol from a script) stands in for the real one.
"""
import asyncio
import concurrent.futures
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from app import cascade, providers, realtime, registry, server, settings, store
from app.metrics import METRICS
from app.session import Session
from app.verification import Otp, SimulatedSms

TMP = Path(tempfile.mkdtemp(prefix="app-selfcheck-"))


def _sync(coro):
    """Run a coroutine to completion from sync code, inside or outside a running loop (e.g. a notebook)."""
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def test_settings(env=None) -> settings.Settings:
    """Settings for tests: a fresh temp database, plus any overrides."""
    base = {"APP__STORE__PATH": str(TMP / f"t{time.monotonic_ns()}.db"), "APP__OPENAI__DEV_KEY_INI": '""'}
    return settings.load(env={**base, **(env or {})})


def check_settings():
    s = test_settings({"APP__SERVER__PORT": "9001", "APP__OPENAI__MODEL": "gpt-x"})
    assert s.server.port == 9001 and s.openai.model == "gpt-x" and s.resilience.stall_s == 2.0
    assert s.agents["front_desk"].vad["silence_duration_ms"] == 900
    for env, msg in [({"APP__SERVER__PROT": "1"}, "unknown key"), ({"APP__SERVER__PORT": '"80"'}, "expected int"),
                     ({"APP__STORE__SEED_DEMO": "1"}, "expected bool"), ({"APP__NOPE__X": "1"}, "unknown section"),
                     ({"APP__ENGINE__KIND": '"magic"'}, "engine.kind"),
                     ({"APP__AGENTS__X__PROMPT": '"prompts/missing.md"', "APP__AGENTS__X__TOOLSETS": '["orders"]'},
                      "no file")]:
        try:
            test_settings(env)
            raise AssertionError(f"accepted {env}")
        except settings.SettingsError as e:
            assert msg in str(e), (env, e)
    try:
        settings.api_key(s, env={})
        raise AssertionError("no key accepted")
    except settings.SettingsError:
        pass
    assert settings.api_key(s, env={"OPENAI_API_KEY": "k"}) == "k"
    print("settings ok")


async def check_registry():
    from typing import Literal
    sets = registry.discover()
    assert {"identity", "registration", "appointments", "clinic_info", "orders"} <= set(sets)

    def doc(f, d):
        f.__doc__ = d
        return f

    bad = [(doc(lambda x: {}, "D."), "first parameter"),
           (doc(lambda ctx, x: {}, "D."), "type hint")]

    def no_doc(ctx):
        return {}

    def weird(ctx, x: list):
        """D."""

    def ghost(ctx, x: int):
        """D.

        Args:
            y: not a parameter
        """
    for fn, msg in [*bad, (no_doc, "docstring"), (weird, "unsupported type"), (ghost, "unknown argument")]:
        try:
            registry.describe(fn, "t")
            raise AssertionError(f"accepted {fn}")
        except registry.SkillDefinitionError as e:
            assert msg in str(e), e

    def demo(ctx, n: int, kind: Literal["a", "b"], note: str = "") -> dict:
        """Does a thing.

        When: when testing.
        Args:
            n: how many
        """
    sk = registry.describe(demo, "demo")
    assert sk.when == "when testing." and sk.description == "Does a thing."
    assert sk.parameters == {"type": "object", "additionalProperties": False, "required": ["n", "kind"],
                             "properties": {"n": {"type": "integer", "description": "how many"},
                                            "kind": {"type": "string", "enum": ["a", "b"]}, "note": {"type": "string"}}}
    p = sk.parameters
    assert registry.validate({"n": 1, "kind": "a"}, p) is None
    for args, msg in [([], "JSON object"), ({"n": 1}, "missing"), ({"n": 1, "kind": "a", "x": 0}, "unknown"),
                      ({"n": True, "kind": "a"}, "boolean"), ({"n": 1.5, "kind": "a"}, "expected integer"),
                      ({"n": 1, "kind": "c"}, "one of")]:
        assert msg in registry.validate(args, p), args

    class Ctx:
        patient = None

    async def calls():
        guarded = registry.Skill("g", "t", "d", "", {"type": "object", "properties": {}, "required": []},
                                 lambda ctx: {"ok": 1}, True)
        assert (await registry.call(guarded, Ctx(), "{}", 1))["status"] == "needs_verification"
        slow = registry.Skill("s", "t", "d", "", guarded.parameters, lambda ctx: time.sleep(2) or {}, False)
        assert "too long" in (await registry.call(slow, Ctx(), "{}", 0.2))["error"]
        boom = registry.Skill("b", "t", "d", "", guarded.parameters, lambda ctx: 1 / 0, False)
        assert "ZeroDivisionError" in (await registry.call(boom, Ctx(), "{}", 1))["error"]
        notdict = registry.Skill("n", "t", "d", "", guarded.parameters, lambda ctx: [1], False)
        assert "invalid result" in (await registry.call(notdict, Ctx(), "{}", 1))["error"]
        assert "too large" in (await registry.call(notdict, Ctx(), " " * 20_000, 1))["error"]
        assert "not valid JSON" in (await registry.call(notdict, Ctx(), "{", 1))["error"]
    await calls()
    print("registry ok")


async def check_session():
    """The whole front-desk flow through Session.run_tool: the exact path a model's proposal takes."""
    s = test_settings()
    store.init(s.path(s.store.path), seed_demo=True)
    registry.discover()
    events, codes = [], iter(["111111", "222222", "333333", "444444"])
    fixed = datetime.combine(datetime.now().date(), datetime.min.time()).replace(minute=1)
    sms = SimulatedSms(on_send=lambda m: events.append({"type": "sms", **m}))
    ses = Session(s, "front_desk", emit=events.append, now=lambda: fixed, sms=sms,
                  otp=Otp(sms, "City Care Clinic", gen=lambda: next(codes)))
    call = lambda tool, **a: _sync(ses.run_tool(tool, json.dumps(a)))  # noqa: E731
    tue, wed = store.next_weekday("Tuesday"), store.next_weekday("Wednesday")

    assert call("my_appointments")["status"] == "needs_verification"                    # guard, in code
    assert "unknown tool" in call("get_order_status", order_id="A123")["error"]        # not this agent's tool
    assert call("verify_patient", phone="9000000000")["status"] == "not_found"
    out = call("verify_patient", phone="98123 45678")
    assert out["status"] == "code_sent" and "111111" not in json.dumps(out) and "Asha" not in json.dumps(out)
    assert call("confirm_code", code="Please read it")["status"] == "need_digits" and ses.otp.pending["tries"] == 0
    assert call("confirm_code", code="000000")["status"] == "wrong_code"
    assert call("confirm_code", code="1 1 1 1 1 1")["status"] == "choose_patient"
    assert call("choose_patient", name="asha")["status"] == "verified" and ses.patient["id"] == 1
    assert [a["time"] for a in call("my_appointments")["appointments"]] == ["10:00 AM"]
    assert call("cancel_appointment", appointment_id=2)["status"] == "not_found"        # Rohan's, not hers
    assert call("book_appointment", doctor="Dr Rao", date=f"{wed}", time="10:40 AM", reason="fever")["status"] == "booked"
    assert call("cancel_appointment", appointment_id=1)["status"] == "cancelled"
    assert call("start_registration", first_name="Meera", last_name="Iyer", phone="9444123456", sex="Female",
                dob="1995-04-03")["status"] == "code_sent"                              # optional args omitted
    assert call("confirm_code", code="222222")["status"] == "registered" and ses.patient["name"] == "Meera Iyer"
    assert call("confirm_code", code="222222")["status"] == "already_verified"
    assert call("verify_patient", phone="9845012367")["status"] == "code_sent"           # 3rd code this call
    assert call("verify_patient", phone="9845012367")["status"] == "too_many_codes"     # 4th: SMS pumping cap
    desk = [e for e in events if e["type"] == "desk"]
    assert not any(c in json.dumps(e["result"]) for e in desk for c in ("111111", "222222", "333333"))
    assert sum(e["type"] == "sms" for e in events) == 3 and desk[-1]["state"]["code_pending_for"] == "verification"
    assert "Today is" in ses.instructions() and "- book_appointment:" in ses.instructions()
    assert METRICS.get("tool_calls_total", skill="book_appointment", status="booked") >= 1
    print("session ok")


async def check_resume():
    """An upstream that goes silent mid-response and stops answering pings is detected, and
    converse_resilient reconnects, replays the turns so far, and the reply completes."""
    conns, replayed = [], []
    delta = realtime.b64(bytes(4800))

    async def fake(ws):
        n = len(conns)
        conns.append(ws)
        try:
            if n == 0:
                for _ in range(3):
                    await ws.recv()
                for e in ({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
                          {"type": "response.created", "response": {"status": "in_progress"}},
                          {"type": "response.output_audio_transcript.delta", "delta": "We open at"},
                          {"type": "response.output_audio.delta", "item_id": "i1", "delta": delta}):
                    await ws.send(json.dumps(e))
                ws.transport.pause_reading()  # stop reading: pings go unanswered, like the real dead link
                await asyncio.sleep(8)
                return
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "conversation.item.create":
                    replayed.append((m["item"]["role"], m["item"]["content"][0]["text"]))
                elif m["type"] == "response.create":
                    for e in ({"type": "response.created", "response": {"status": "in_progress"}},
                              {"type": "response.output_audio_transcript.done", "transcript": "carrying on"},
                              {"type": "response.done", "response": {"status": "completed", "output": []}}):
                        await ws.send(json.dumps(e))
        except ConnectionClosed:
            pass

    async def silence():
        while True:
            yield bytes(4800)
            await asyncio.sleep(0.1)

    done, answered = asyncio.Event(), []
    t = time.monotonic()

    def on_note(_t, ev, d):
        if ev == "model_said":
            answered.append(time.monotonic() - t)
            done.set()

    async with serve(fake, "127.0.0.1", 8768):
        log = await realtime.converse_resilient(
            lambda: connect("ws://127.0.0.1:8768"), silence(), server.BrowserSpeaker(asyncio.Queue()), seconds=30,
            context=lambda: "STATE", on_note=on_note, stop=done, stall_s=1, ping_timeout_s=2)
    events = [e for _, e, _ in log]
    assert events.index("STALL") < events.index("RESUME") and len(conns) == 2, events
    assert replayed[0] == ("user", "hello") and replayed[-1][1].endswith("STATE"), replayed
    assert replayed[1] == ("assistant", "We open at") and "Finish it" in replayed[-1][1], replayed  # cut reply finished
    assert answered and answered[0] < 8, answered
    print(f"resume ok: stall detected, reconnected, cut reply finished, reply {answered[0]:.1f}s after the start")


class FakeOpenAI:
    """A scripted Realtime server: on the caller's first audio it 'hears' a phone number and proposes
    verify_patient; after the tool output it asks for the code. Records what our server sent back."""

    def __init__(self):
        self.tool_outputs = []

    async def handler(self, ws):
        try:
            await ws.send(json.dumps({"type": "session.created"}))
            cfg = json.loads(await ws.recv())
            assert cfg["type"] == "session.update" and cfg["session"]["tools"]
            await ws.send(json.dumps({"type": "session.updated"}))
            heard = False
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "input_audio_buffer.append" and not heard:
                    heard = True
                    call = {"type": "function_call", "name": "verify_patient", "call_id": "c1",
                            "arguments": json.dumps({"phone": "9845012367"})}
                    for e in ({"type": "input_audio_buffer.speech_started"},
                              {"type": "conversation.item.input_audio_transcription.completed",
                               "transcript": "My number is 9845012367."},
                              {"type": "response.created", "response": {"status": "in_progress"}},
                              {"type": "response.output_item.added", "response_id": "r1", "item": call},
                              {"type": "response.done", "response": {"status": "completed", "output": [call]}}):
                        await ws.send(json.dumps(e))
                elif m["type"] == "conversation.item.create" and m["item"]["type"] == "function_call_output":
                    self.tool_outputs.append(m["item"]["output"])
                elif m["type"] == "response.create":
                    for e in ({"type": "response.created", "response": {"status": "in_progress"}},
                              {"type": "response.output_audio.delta", "item_id": "a1", "delta": realtime.b64(bytes(9600))},
                              {"type": "response.output_audio_transcript.done", "transcript": "Please read me the code."},
                              {"type": "response.done", "response": {"status": "completed", "output": []}}):
                        await ws.send(json.dumps(e))
        except ConnectionClosed:
            pass


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


async def check_server():
    """The real server, end to end, against the fake OpenAI: page, health, metrics, agent allowlist,
    a conversation with a tool call + SMS, and the concurrency limit."""
    port, up = 8781, 8782
    s = test_settings({"APP__SERVER__PORT": str(port), "APP__SERVER__MAX_SESSIONS": "1", "APP__ENGINE__KIND": '"realtime"',
                       "APP__OPENAI__URL": f'"ws://127.0.0.1:{up}"'})
    fake = FakeOpenAI()
    app = server.App(s, key="test-key-not-real")
    calls0 = METRICS.get("tool_calls_total", skill="verify_patient", status="code_sent")
    rejected0 = METRICS.get("sessions_rejected_total")
    ready, stop = asyncio.Event(), asyncio.Event()
    async with serve(fake.handler, "127.0.0.1", up):
        srv = asyncio.create_task(server.run(s, ready=ready, stop=stop, app=app))
        await asyncio.wait_for(ready.wait(), 10)
        base = f"http://127.0.0.1:{port}"
        status, page = await asyncio.to_thread(_get, base + "/")
        assert status == 200 and "/ws?agent=" in page
        status, body = await asyncio.to_thread(_get, base + "/healthz")
        assert status == 200 and json.loads(body)["status"] == "ok"
        assert (await asyncio.to_thread(_get, base + "/nope"))[0] == 404

        async with connect(f"ws://127.0.0.1:{port}/ws?agent=nope") as ws:
            assert "unknown agent" in json.loads(await ws.recv())["message"]

        msgs, audio = [], 0
        async with connect(f"ws://127.0.0.1:{port}/ws?agent=front_desk", max_size=2 ** 22) as ws:
            while json.loads(await ws.recv()).get("state") != "ready":
                pass
            async with connect(f"ws://127.0.0.1:{port}/ws?agent=front_desk") as second:  # max_sessions = 1
                assert "busy" in json.loads(await second.recv())["message"]
            await ws.send(bytes(4800))
            await ws.send(json.dumps({"played": 0}))
            while True:
                m = await asyncio.wait_for(ws.recv(), 10)
                if isinstance(m, bytes):
                    audio += len(m)
                    continue
                m = json.loads(m)
                msgs.append(m)
                if m.get("event") == "model_said":
                    break
        kinds = [m["type"] for m in msgs]
        desk = next(m for m in msgs if m["type"] == "desk")
        sms = next(m for m in msgs if m["type"] == "sms")
        code = re.sub(r"\D", "", sms["text"])
        assert desk["skill"] == "verify_patient" and desk["result"]["status"] == "code_sent", desk
        assert desk["state"]["code_pending_for"] == "verification" and "log" in kinds
        assert fake.tool_outputs and code not in fake.tool_outputs[0], fake.tool_outputs  # the model never saw it
        assert audio >= len(app.filler or b"") + 9600, audio                                 # filler + the reply
        await asyncio.sleep(0.3)
        status, metrics = await asyncio.to_thread(_get, base + "/metrics")
        assert status == 200 and 'tool_calls_total{skill="verify_patient",status="code_sent"}' in metrics
        assert METRICS.get("tool_calls_total", skill="verify_patient", status="code_sent") == calls0 + 1
        assert METRICS.get("sessions_rejected_total") == rejected0 + 1 and "sessions_active 0" in metrics, metrics
        stop.set()
        await asyncio.wait_for(srv, 15)
    print("server ok: page, health, metrics, agent allowlist, tool call + SMS end to end, busy limit")


def fake_parts(script, replies, tts_chunks=2, tts_delay=0.0, tts_fail_once=False):
    """Scripted cascade providers (no network). script: (delay, kind, text) STT events; replies: LLM answers in
    order. Returns (Parts, record of what each part was given)."""
    rec = {"llm": [], "tts": [], "sent": 0, "tts_reconnects": 0}

    class STT:
        def __init__(self, s, key, vad):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def reconnect(self):
            pass

        async def send(self, pcm):
            rec["sent"] += len(pcm)

        async def events(self):
            for delay, kind, text in script:
                await asyncio.sleep(delay)
                yield kind, text
            await asyncio.sleep(3600)

    class LLM:
        def __init__(self, s, key):
            pass

        async def reply(self, messages, tools):
            rec["llm"].append(json.loads(json.dumps(messages)))
            return replies.pop(0)

    class TTS(STT):
        def __init__(self, s, key):
            self.fail = tts_fail_once

        async def reconnect(self):
            rec["tts_reconnects"] += 1

        async def speak(self, text):
            if self.fail:
                self.fail = False
                raise realtime.Stalled("scripted stall")
            rec["tts"].append(text)
            for _ in range(tts_chunks):
                await asyncio.sleep(tts_delay)
                yield bytes(4800)

    return providers.Parts(STT, LLM, TTS), rec


async def _run_cascade(parts, until, s=None, session=None):
    """Run the cascade engine until `until(events)` is true; returns (events [(event, detail)], speaker)."""
    s = s or test_settings({"APP__ENGINE__KIND": '"cascade"'})
    session = session or Session(s, "front_desk")
    spk, done, seen = server.BrowserSpeaker(asyncio.Queue()), asyncio.Event(), []

    def on_note(_t, ev, d):
        seen.append((ev, d))
        if until(seen):
            done.set()

    async def silence():
        while True:
            yield bytes(4800)
            await asyncio.sleep(0.05)

    await cascade.converse(parts, s, "test-key", silence(), spk, seconds=10, instructions=session.instructions(),
                           tools=session.specs(), run_tool=session.run_tool, filler_pcm=bytes(2400),
                           on_note=on_note, stop=done)
    return seen, spk


async def check_cascade():
    """The cascade engine with scripted providers: the page's events in order, a tool call through the real
    Session (the code never reaches the model), the filler, a stalled text-to-speech link retried, barge-in that
    keeps only the heard part, a reply dropped when the caller speaks again, and the server end to end."""
    registry.discover()
    # 0. spoken digits become digits before the model reads them, in any language; other words stay
    for said, want in [("मेरा नंबर है नौ आठ चार पाँच, शून्य एक दो, तीन छः सात।", "मेरा नंबर है 9845012367।"),
                       ("நம்பர் ஒன்பது எட்டு நான்கு ஐந்து பூஜ்ஜியம் ஒன்று இரண்டு மூன்று ஆறு ஏழு.", "நம்பர் 9845012367."),
                       ("nine eight four five zero one two three double seven", "9845012377"),
                       ("मुझे एक अपॉइंटमेंट चाहिए", "मुझे एक अपॉइंटमेंट चाहिए"), ("slots at 10, 11", "slots at 10, 11"),
                       ("one two", "one two"), ("एकदम सही", "एकदम सही")]:
        assert cascade.spoken_digits(said) == want, (said, cascade.spoken_digits(said))
    # 1. a tool call, the filler, and a text-to-speech stall that is retried
    call = {"id": "c1", "name": "verify_patient", "arguments": json.dumps({"phone": "9845012367"})}
    parts, rec = fake_parts([(0.05, "speech_started", ""), (0.05, "speech_stopped", ""),
                             (0.05, "transcript", "My number is 9845012367.")],
                            [{"text": "", "tool_calls": [call]}, {"text": "I sent a code. Please read it out.", "tool_calls": []}],
                            tts_fail_once=True)
    s = test_settings({"APP__ENGINE__KIND": '"cascade"'})
    store.init(s.path(s.store.path), seed_demo=True)
    session = Session(s, "front_desk")
    seen, spk = await _run_cascade(parts, lambda e: e[-1][0] == "response.done", s, session)
    order = [e for e, _ in seen]
    want = ["input_audio_buffer.speech_started", "user_said", "filler", "tool_call", "STALL", "RESUME",
            "first_audio_delta", "model_said", "response.done"]
    assert [e for e in order if e in want] == want, order
    tool_msg = rec["llm"][1][-1]
    code = session.sms.last_code()
    assert tool_msg["role"] == "tool" and "code_sent" in tool_msg["content"] and code not in tool_msg["content"], tool_msg
    assert session.state()["code_pending_for"] == "verification" and rec["tts_reconnects"] == 1
    assert rec["tts"] == ["I sent a code.", "Please read it out."] and spk.position == 2400 + 4 * 4800, (rec, spk.position)

    # 2. barge-in mid-reply, then a reply superseded by more speech: one answer to both
    parts, rec = fake_parts([(0.0, "transcript", "What are your hours?"), (0.3, "speech_started", ""),
                             (0.05, "transcript", "Actually, which doctors?")],
                            [{"text": "We open at nine. We close at five.", "tool_calls": []},
                             {"text": "Doctor Kapoor and Doctor Rao.", "tool_calls": []}], tts_chunks=10, tts_delay=0.05)
    seen, _ = await _run_cascade(parts, lambda e: sum(1 for x, _ in e if x == "response.done") == 2)
    order = [e for e, _ in seen]
    assert "barge_in" in order and order.index("barge_in") < order.index("user_said", 1), order
    history = rec["llm"][1]
    assert history[-2]["role"] == "assistant" and history[-2]["content"].endswith("…"), history[-2:]
    assert history[-1] == {"role": "user", "content": "Actually, which doctors?"}, history[-1]
    assert rec["tts"][-1] == "Doctor Kapoor and Doctor Rao.", rec["tts"]

    # 3. the server end to end with engine = cascade
    port = 8783
    s = test_settings({"APP__ENGINE__KIND": '"cascade"', "APP__SERVER__PORT": str(port)})
    parts, rec = fake_parts([(0.2, "transcript", "My number is 9845012367.")],
                            [{"text": "", "tool_calls": [call]}, {"text": "Please read me the code.", "tool_calls": []}])
    app = server.App(s, key="test-key-not-real", parts=parts)
    ready, stop = asyncio.Event(), asyncio.Event()
    srv = asyncio.create_task(server.run(s, ready=ready, stop=stop, app=app))
    await asyncio.wait_for(ready.wait(), 10)
    msgs = []
    async with connect(f"ws://127.0.0.1:{port}/ws?agent=front_desk", max_size=2 ** 22) as ws:
        await ws.send(bytes(4800))
        while True:
            m = await asyncio.wait_for(ws.recv(), 10)
            if isinstance(m, bytes):
                continue
            msgs.append(json.loads(m))
            if any(m.get("event") == "model_said" for m in msgs) and any(m["type"] == "desk" for m in msgs):
                break
    kinds = [m["type"] for m in msgs]
    assert "status" in kinds and "sms" in kinds and any(m["type"] == "desk" and m["result"]["status"] == "code_sent"
                                                       for m in msgs), kinds
    stop.set()
    await asyncio.wait_for(srv, 15)
    print("cascade ok: spoken digits, event order, tool call + filler, stall retried, barge-in keeps the heard part, "
          "superseded reply, server end to end")


async def check_webrtc():
    """The WebRTC engine through the real server, with OpenAI faked: the page's offer goes to call creation with
    our prompt and tools (never the key), the answer comes back, the side connection runs the tool call, the page
    gets its events and the SMS, and the call is hung up when the page leaves."""
    port, up = 8785, 8786
    s = test_settings({"APP__ENGINE__KIND": '"webrtc"', "APP__SERVER__PORT": str(port)})
    calls, hung, outputs = [], [], []
    call = {"type": "function_call", "name": "verify_patient", "call_id": "c1",
            "arguments": json.dumps({"phone": "9845012367"})}

    async def side(ws):  # the fake side connection: hears a number, proposes verify_patient, then speaks
        try:
            for e in ({"type": "input_audio_buffer.speech_started"}, {"type": "input_audio_buffer.speech_stopped"},
                      {"type": "conversation.item.input_audio_transcription.completed", "transcript": "My number is 9845012367."},
                      {"type": "response.done", "response": {"status": "completed", "output": [call]}}):
                await ws.send(json.dumps(e))
            async for raw in ws:
                m = json.loads(raw)
                if m["type"] == "conversation.item.create":
                    outputs.append(m["item"]["output"])
                elif m["type"] == "response.create":
                    for e in ({"type": "output_audio_buffer.started"},
                              {"type": "response.output_audio_transcript.done", "transcript": "Please read me the code."},
                              {"type": "response.done", "response": {"status": "completed", "output": []}},
                              {"type": "output_audio_buffer.stopped"}):
                        await ws.send(json.dumps(e))
        except ConnectionClosed:
            pass

    async def create(_s, key, offer, session):
        calls.append((key, offer, session))
        return "v=0 answer", "rtc_test"

    async def hangup(call_id):
        hung.append(call_id)

    hooks = {"create": create, "connect": lambda _s, _k, _c: connect(f"ws://127.0.0.1:{up}"), "hangup": hangup}
    app = server.App(s, key="test-key-not-real", rtc=hooks)
    ready, stop = asyncio.Event(), asyncio.Event()
    async with serve(side, "127.0.0.1", up):
        srv = asyncio.create_task(server.run(s, ready=ready, stop=stop, app=app))
        await asyncio.wait_for(ready.wait(), 10)
        msgs = []
        async with connect(f"ws://127.0.0.1:{port}/ws?agent=front_desk") as ws:
            first = json.loads(await ws.recv())
            assert first == {"type": "status", "state": "connecting", "engine": "webrtc"}, first
            await ws.send(json.dumps({"sdp": "v=0 offer"}))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 10))
                msgs.append(m)
                if any(x.get("event") == "speaking_done" for x in msgs) and any(x["type"] == "desk" for x in msgs):
                    break
        for _ in range(50):
            if hung:
                break
            await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(srv, 15)
    kinds = [m["type"] for m in msgs]
    key, offer, session = calls[0]
    assert offer == "v=0 offer" and session["tools"] and "front desk" in session["instructions"], session
    assert {"type": "rtc_answer", "sdp": "v=0 answer"} in msgs and "sms" in kinds, kinds
    code = re.sub(r"\D", "", next(m for m in msgs if m["type"] == "sms")["text"])
    assert outputs and "code_sent" in outputs[0] and code not in outputs[0], outputs  # the model never saw it
    events = [m.get("event") for m in msgs if m["type"] == "log"]
    assert events.index("user_said") < events.index("tool_call") < events.index("first_audio_delta") \
        < events.index("model_said"), events
    assert hung == ["rtc_test"], hung
    print("webrtc ok: offer -> call with prompt + tools, answer back, tool call on the side connection, "
          "page events + SMS, hung up on leave")


async def check_all():
    check_settings()
    await check_registry()
    await check_session()
    await check_resume()
    await check_server()
    await check_cascade()
    await check_webrtc()
    print("ALL OK")


def main():
    asyncio.run(check_all())


if __name__ == "__main__":
    main()
