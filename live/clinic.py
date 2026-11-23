"""The simulated clinic everything else runs on (notebooks 06-07, web app): one SQLite test database,
a simulated SMS outbox, and one-time codes. Nothing here talks to DocAssist; table shapes and rules
follow DocAssist's (pyDocAssist: table/Patient.py, blueprints/appointment_api.py).
"""
import contextlib
import hmac
import re
import secrets
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "test.db"
CLINIC = "City Care Clinic"

SCHEMA = """
CREATE TABLE IF NOT EXISTS Patient (
    Patient_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT NOT NULL, Last_Name TEXT NOT NULL,
    DOB TEXT NOT NULL, age INTEGER, sex TEXT,
    Phone_Number TEXT NOT NULL, Email TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS Doctor (
    Doctor_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT NOT NULL, Last_Name TEXT NOT NULL, Specialty TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS Availability (          -- DocAssist: doctor_availability
    doctor_id INTEGER NOT NULL REFERENCES Doctor(Doctor_ID),
    day TEXT NOT NULL,                              -- Monday .. Sunday
    start_time TEXT NOT NULL, end_time TEXT NOT NULL,   -- HH:MM, 24 h
    duration_minutes INTEGER NOT NULL DEFAULT 15);
CREATE TABLE IF NOT EXISTS Appointment (
    Appointment_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER REFERENCES Patient(Patient_ID),  -- NULL for a doctor's leave block
    doctor_id INTEGER NOT NULL REFERENCES Doctor(Doctor_ID),
    appointment_date TEXT NOT NULL,                 -- YYYY-MM-DD HH:MM:00
    status TEXT NOT NULL DEFAULT 'scheduled',       -- scheduled | cancelled | completed | unavailable
    reason TEXT, notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""
PATIENTS = [("asha", "sharma", "1985-03-02", 41, "Female", "919812345678"),
            ("rohan", "sharma", "2015-07-19", 11, "Male", "919812345678"),  # a child shares his mother's phone
            ("vikram", "nair", "1979-11-30", 46, "Male", "919845012367")]  # few repeated digits: see notebook 07
DOCTORS = [("anil", "kapoor", "General Medicine", [(d, "09:00", "13:00", 15) for d in
            ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]),
           ("sunita", "rao", "Pediatrics", [(d, "10:00", "12:00", 20) for d in ("Monday", "Wednesday", "Friday")]),
           ("vikas", "shah", "Cardiology", [(d, "14:00", "17:00", 30) for d in ("Tuesday", "Thursday")])]


@contextlib.contextmanager
def db(path=DB_PATH):
    """One connection per operation (tools run on worker threads; sqlite connections don't cross threads).
    Commits on success, rolls back on error, always closes."""
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    try:
        with c:
            yield c
    finally:
        c.close()


def next_weekday(name: str, after: date | None = None) -> date:
    """The next `name` day strictly after `after` (default today)."""
    d = (after or date.today()) + timedelta(days=1)
    while d.strftime("%A") != name:
        d += timedelta(days=1)
    return d


def init_db(path=DB_PATH, reset=False) -> Path:
    """Create the tables; seed patients, doctors with weekly hours, and a few appointments relative to today."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.unlink(missing_ok=True)
    empty = lambda table: not c.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()  # noqa: E731
    with db(path) as c:
        c.executescript(SCHEMA)
        # each table seeds on its own, so an older test.db (patients only, notebook 06) gains doctors too
        if empty("Patient"):
            c.executemany("INSERT INTO Patient (First_Name, Last_Name, DOB, age, sex, Phone_Number) "
                          "VALUES (?,?,?,?,?,?)", PATIENTS)
        if empty("Doctor"):
            for first, last, specialty, hours in DOCTORS:
                doc = c.execute("INSERT INTO Doctor (First_Name, Last_Name, Specialty) VALUES (?,?,?)",
                                (first, last, specialty)).lastrowid
                c.executemany("INSERT INTO Availability VALUES (?,?,?,?,?)", [(doc, *h) for h in hours])
        if not empty("Appointment"):
            return path
        tue, wed = next_weekday("Tuesday"), next_weekday("Wednesday")
        c.executemany("INSERT INTO Appointment (patient_id, doctor_id, appointment_date, status, reason, notes) "
                      "VALUES (?,?,?,?,?,?)", [
                          (1, 1, f"{tue} 10:00:00", "scheduled", "fever and cough", None),       # Asha, Dr Kapoor
                          (2, 2, f"{wed} 10:20:00", "scheduled", "vaccination", None),           # Rohan, Dr Rao
                          (3, 1, f"{tue} 09:00:00", "cancelled", "back pain", None),             # Vikram, cancelled
                          (None, 1, f"{wed} 00:00:00", "unavailable", None, "full-day leave")])  # Dr Kapoor away
    return path


class Otp:
    """One pending one-time code per conversation, texted to a phone (the simulated outbox).
    The code never leaves this object except as the SMS text; tools only ever see "code sent to ...".
    A new code replaces the old one; a code is single-use, expires after `ttl` s, and locks after `max_tries`."""

    def __init__(self, gen=lambda: f"{secrets.randbelow(10 ** 6):06d}", clock=time.monotonic, ttl=300, max_tries=3):
        self.gen, self.clock, self.ttl, self.max_tries = gen, clock, ttl, max_tries
        self.outbox: list[dict] = []   # {"to", "text"}: what the patient's phone received
        self.pending = None            # {"purpose", "payload", "code", "at", "tries"}

    def send(self, phone: str, purpose: str, payload) -> str:
        code = self.gen()
        self.pending = {"purpose": purpose, "payload": payload, "code": code, "at": self.clock(), "tries": 0}
        self.outbox.append({"to": "+" + phone, "text": f"Your {CLINIC} {purpose} code is {code}."})
        return f"phone ending {phone[-4:]}"

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

    def last_code(self) -> str:
        """Test helper: the digits of the latest SMS (what a patient reads off their phone)."""
        return re.sub(r"\D", "", self.outbox[-1]["text"]) if self.outbox else ""
