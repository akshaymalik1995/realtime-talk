"""The front desk: one voice agent, one conversation state, every function a skill (notebook 07, web app).

Adding a function = one decorated function (here, or in any module imported before the session starts):

    @skill("What it does (the model reads this).", guide="When to use it.", needs_patient=True,
           params={"x": {"type": "string"}})
    def my_skill(s: Session, x):
        return {"status": "ok", ...}

The Session is everything one conversation knows: which patient is verified, the one pending code and what it is
for, the patients waiting to be chosen on a shared phone, and the SMS outbox. Rules live in code, not the prompt:
a needs_patient skill never runs before verification, whatever the model asks; one confirm_code serves every code.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from live import appointments, patients, tools
from live.clinic import CLINIC, DB_PATH, Otp, db

VAD = patients.VAD  # dictation-friendly turn end (900 ms of silence)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    params: dict
    fn: Callable
    needs_patient: bool
    guide: str


SKILLS: dict[str, Skill] = {}


def skill(description: str, params: dict | None = None, needs_patient: bool = False, guide: str = ""):
    def register(fn):
        SKILLS[fn.__name__] = Skill(fn.__name__, description, params or {}, fn, needs_patient, guide)
        return fn
    return register


@dataclass
class Session:
    db: Path = DB_PATH
    otp: Otp = field(default_factory=Otp)
    patient: dict | None = None                         # verified caller: {"id", "name"}
    choices: list | None = None                         # patients on a just-verified shared phone
    emit: Callable[[dict], None] = lambda m: None       # UI hook: every skill call, every SMS
    now: Callable[[], datetime] = datetime.now

    def state(self) -> dict:
        """What the UI (and a debugger) needs to see. Never the code itself."""
        return {"patient": self.patient,
                "code_pending_for": self.otp.pending["purpose"] if self.otp.pending else None,
                "choosing_between": [c["name"] for c in self.choices] if self.choices else None}

    def registry(self) -> dict:
        """{name: (spec, fn)} for live.tools.run: every skill, bound to this session, behind the guards."""
        return {sk.name: (tools.spec(sk.name, sk.description, sk.params), self._runner(sk)) for sk in SKILLS.values()}

    def _runner(self, sk: Skill):
        def run(**args):
            sent = len(self.otp.outbox)
            if sk.needs_patient and not self.patient:
                out = {"status": "needs_verification", "problem": "the caller is not verified yet",
                       "next": "Ask for their mobile number and call verify_patient."}
            else:
                out = sk.fn(self, **args)
            self.emit({"type": "desk", "skill": sk.name, "args": args, "result": out, "state": self.state()})
            for sms in self.otp.outbox[sent:]:
                self.emit({"type": "sms", **sms})
            return out
        return run

    def instructions(self) -> str:
        today = self.now()
        lines = [f"- {sk.name}: {sk.guide}" for sk in SKILLS.values() if sk.guide]
        return "\n".join([
            f"You are the voice front desk of {CLINIC}. Today is {today:%A} {today.day} {today:%B %Y}.",
            "Be brief and warm. Ask one thing at a time.",
            "Use the tools for every fact: never invent doctors, times, codes, appointments or patient details.",
            'When a tool result has a "next", do what it says.',
            "When you have what a tool needs, call it in that same turn. Never announce an action "
            "(\"I'll send a code\", \"I'll book that\") instead of calling the tool.",
            "Anything about an existing patient (their appointments, booking, cancelling) needs a verified caller: "
            "ask for their mobile number, call verify_patient, then confirm_code with the code they read from the SMS. "
            "A new patient becomes verified by registering.",
            "The moment the caller gives a mobile number, call verify_patient in that same turn. Codes are sent only by "
            "verify_patient or start_registration: never say a code was or will be sent unless one returned code_sent.",
            "Copy phone numbers and codes digit for digit as the caller said them.",
            'Give tools dates as YYYY-MM-DD, worked out from today\'s date ("tomorrow", "next Tuesday"), and times like 9:15 AM.',
            "Before booking or cancelling, say what you are about to do (doctor, day, time) and get a yes. "
            "Once they say yes, call book_appointment (or cancel_appointment) right away: do not check slots again "
            "or ask again. If the time they asked for is not free, offer the nearest free times instead.",
            "Tools:", *lines])


def verified(s: Session, p: dict) -> dict:
    s.patient, s.choices = {"id": p["id"], "name": p["name"]}, None
    return {"status": "verified", "name": p["name"],
            "next": f"Greet {p['name'].split()[0]}, then carry on with what they already asked for "
                    "(for example call my_appointments if they wanted their appointments). Ask only if they asked nothing yet."}


# ---- identity ----

@skill("Look up an existing patient by mobile number and text them a 6-digit code to prove it is their phone.",
       {"phone": {"type": "string", "description": "10-digit mobile number, digits only"}},
       guide="first step for any existing patient.")
def verify_patient(s: Session, phone):
    number = patients.normalize_phone(phone)
    if not number:
        return {"status": "bad_phone", "problem": "need a 10-digit mobile number"}
    found = patients.by_phone(s.db, number)
    if not found:
        s.otp.pending = None
        return {"status": "not_found", "problem": "no patient has this number",
                "next": "Check the number, or offer to register them as a new patient."}
    # who is on this phone is only revealed after the code proves the caller holds it
    return {"status": "code_sent", "sent_to": s.otp.send(number, "verification", found),
            "next": "Ask for the 6-digit code that was just texted."}


@skill("Check the 6-digit code the caller read from the SMS (for verification or registration).",
       {"code": {"type": "string", "description": "exactly the 6 digits the caller read out"}},
       guide="only after the caller has read out the 6 digits of a code.")
def confirm_code(s: Session, code):
    if len(re.sub(r"\D", "", str(code))) != 6:  # a placeholder or a half-heard code must not burn one of the 3 tries
        return {"status": "need_digits", "problem": "that is not a 6-digit code",
                "next": "Ask the caller to read the 6 digits from the SMS." if s.otp.pending else
                        "No code has been sent yet: call verify_patient (or start_registration) first."}
    if not s.otp.pending and s.patient:  # e.g. the model repeats a call after a reconnect: nothing to redo
        return {"status": "already_verified", "name": s.patient["name"],
                "next": "They are verified: carry on with what they asked for."}
    purpose, payload, problem = s.otp.check(code)
    if problem:
        return problem
    if purpose == "registration":
        out = patients.insert(s.db, payload)
        if out["status"] == "registered":
            verified(s, {"id": out["patient_id"], "name": out["name"]})
            out["next"] = "Give them their patient ID, then ask if they would like to book an appointment."
        return out
    if len(payload) == 1:
        return verified(s, payload[0])
    s.choices = payload
    return {"status": "choose_patient", "names": [p["name"] for p in payload],
            "next": "Several patients share this phone: ask which of them they are calling about."}


@skill("Pick which patient on a shared phone the caller means, after confirm_code asked to choose.",
       {"name": {"type": "string", "description": "the name the caller said"}},
       guide="after confirm_code returns choose_patient.")
def choose_patient(s: Session, name):
    if not s.choices:
        return {"status": "nothing_to_choose", "problem": "verify the phone first"}
    said = name.lower().split()
    match = [p for p in s.choices if said and all(w in p["name"].lower().split() for w in said)]
    if len(match) != 1:
        return {"status": "which_one", "names": [p["name"] for p in s.choices]}
    return verified(s, match[0])


# ---- registration (notebook 06's rules) ----

@skill("Validate a NEW patient's details and text a 6-digit code to their phone. Call again with corrections.",
       {"first_name": {"type": "string"}, "last_name": {"type": "string"},
        "phone": {"type": "string", "description": "10-digit mobile number, digits only"},
        "sex": {"type": "string", "enum": ["Male", "Female"]},
        "dob": {"type": "string", "description": "YYYY-MM-DD, or empty if only age is known"},
        "age": {"type": "string", "description": "age in years, or empty if dob is given"},
        "email": {"type": "string", "description": "empty if not given"}},
       guide="for new patients: call it the moment you have first and last name, mobile, sex and date of birth "
             "(or age), in that same turn; its result is what you read back. Do not ask for email.")
def start_registration(s: Session, first_name, last_name, phone, sex, dob="", age="", email=""):
    return patients.start(s.db, s.otp, first_name, last_name, phone, sex, dob, age, email)


# ---- doctors and appointments ----

@skill("List the clinic's doctors, their specialties and working days.", guide="when asked which doctors there are.")
def list_doctors(s: Session):
    return {"doctors": appointments.doctors(s.db)}


@skill("Free appointment times for one doctor on one date.",
       {"doctor": {"type": "string", "description": "doctor's name or specialty, as the caller said it"},
        "date": {"type": "string", "description": "YYYY-MM-DD"}},
       guide="before booking, to offer times.")
def find_slots(s: Session, doctor, date):
    return appointments.find_slots(s.db, doctor, date, s.now())


@skill("Book an appointment for the verified patient.",
       {"doctor": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
        "time": {"type": "string", "description": "a free time from find_slots, like 9:15 AM"},
        "reason": {"type": "string", "description": "what the visit is for, in a few words"}},
       needs_patient=True, guide="after the caller said yes to a doctor, day, time and reason.")
def book_appointment(s: Session, doctor, date, time, reason):
    return appointments.book(s.db, s.patient["id"], doctor, date, time, reason, s.now())


@skill("The verified patient's upcoming appointments.", needs_patient=True,
       guide="when the caller asks about their appointments.")
def my_appointments(s: Session):
    found = appointments.upcoming(s.db, s.patient["id"], s.now())
    return {"patient": s.patient["name"], "appointments": found} if found else \
        {"patient": s.patient["name"], "appointments": [], "next": "Say there are none, and offer to book one."}


@skill("Cancel one of the verified patient's appointments.",
       {"appointment_id": {"type": "integer", "description": "from my_appointments"}},
       needs_patient=True, guide="after the caller said yes to cancelling a specific appointment.")
def cancel_appointment(s: Session, appointment_id):
    return appointments.cancel(s.db, s.patient["id"], appointment_id, s.now())


if __name__ == "__main__":  # self-check: python -m live.frontdesk   (whole flow, no model)
    import json
    import tempfile
    from live.clinic import init_db, next_weekday

    with tempfile.TemporaryDirectory() as d:
        path = init_db(Path(d) / "t.db")
        events, codes = [], iter(["111111", "222222", "333333"])
        fixed = datetime.combine(datetime.now().date(), datetime.min.time()).replace(minute=1)
        s = Session(path, Otp(gen=lambda: next(codes)), emit=events.append, now=lambda: fixed)
        reg = s.registry()
        call = lambda tool_, **a: tools.run(reg, tool_, json.dumps(a))  # noqa: E731  (the same path the model takes)
        tue, wed = next_weekday("Tuesday"), next_weekday("Wednesday")

        assert call("my_appointments")["result"]["status"] == "needs_verification"          # guard, in code
        assert call("book_appointment", doctor="Kapoor", date=f"{tue}", time="9:00 AM", reason="x")["result"][
            "status"] == "needs_verification"
        assert call("verify_patient", phone="9000000000")["result"]["status"] == "not_found"
        out = call("verify_patient", phone="98123 45678")["result"]
        assert out["status"] == "code_sent" and "111111" not in json.dumps(out) and "Asha" not in json.dumps(out)
        assert call("confirm_code", code="Please read the code")["result"]["status"] == "need_digits"
        assert s.otp.pending["tries"] == 0                                                 # ...and no try was used
        assert call("confirm_code", code="000000")["result"]["status"] == "wrong_code"
        out = call("confirm_code", code="1 1 1 1 1 1")["result"]
        assert out == {"status": "choose_patient", "names": ["Asha Sharma", "Rohan Sharma"], "next": out["next"]}
        assert call("my_appointments")["result"]["status"] == "needs_verification"          # phone ok, person not yet
        assert call("choose_patient", name="Priya")["result"]["status"] == "which_one"
        assert call("choose_patient", name="asha")["result"]["status"] == "verified" and s.patient["id"] == 1
        mine = call("my_appointments")["result"]["appointments"]
        assert [(a["doctor"], a["time"]) for a in mine] == [("Dr. Anil Kapoor", "10:00 AM")]
        assert call("cancel_appointment", appointment_id=2)["result"]["status"] == "not_found"  # Rohan's: not hers
        assert call("find_slots", doctor="pediatrician", date=f"{wed}")["result"]["free"][:2] == ["10:00 AM", "10:40 AM"]
        b = call("book_appointment", doctor="Dr Rao", date=f"{wed}", time="10:40 AM", reason="fever")["result"]
        assert b["status"] == "booked"
        assert call("cancel_appointment", appointment_id=1)["result"]["status"] == "cancelled"
        assert [a["doctor"] for a in call("my_appointments")["result"]["appointments"]] == ["Dr. Sunita Rao"]

        out = call("start_registration", first_name="Meera", last_name="Iyer", phone="9444123456", sex="Female",
                   dob="1995-04-03", age="", email="")["result"]
        assert out["status"] == "code_sent" and s.state()["code_pending_for"] == "registration"
        out = call("confirm_code", code="222222")["result"]
        assert out["status"] == "registered" and s.patient["name"] == "Meera Iyer"           # registering verifies
        assert call("my_appointments")["result"]["appointments"] == []
        assert call("confirm_code", code="222222")["result"]["status"] == "already_verified"  # repeat after a resume

        assert [e["type"] for e in events].count("sms") == 2
        # the model only ever sees results: no result may contain a code (the caller saying it, in args, is fine)
        assert not any(c in json.dumps(e["result"]) for e in events if e["type"] == "desk" for c in ("111111", "222222"))
        text = s.instructions()
        assert all(name in text for name in SKILLS) and "Today is" in text
    print("ok")
