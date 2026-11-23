"""Storage: SQLite, one connection per operation (skills run on worker threads; connections don't cross threads).

Production notes: WAL mode lets readers and one writer work at once; busy_timeout makes a second writer wait
instead of failing. Writes that check-then-insert use BEGIN IMMEDIATE so two callers can't take one slot.
Swapping in Postgres means re-implementing db() and the SQL dialect bits; the rules never open connections
themselves. Table shapes follow DocAssist (pyDocAssist table/Patient.py, appointment_api.py).
"""
import contextlib
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS Meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS Patient (
    Patient_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT NOT NULL, Last_Name TEXT NOT NULL,
    DOB TEXT NOT NULL, age INTEGER, sex TEXT,
    Phone_Number TEXT NOT NULL, Email TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS patient_phone ON Patient(Phone_Number);
CREATE TABLE IF NOT EXISTS Doctor (
    Doctor_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    First_Name TEXT NOT NULL, Last_Name TEXT NOT NULL, Specialty TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS Availability (
    doctor_id INTEGER NOT NULL REFERENCES Doctor(Doctor_ID),
    day TEXT NOT NULL, start_time TEXT NOT NULL, end_time TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL DEFAULT 15);
CREATE TABLE IF NOT EXISTS Appointment (
    Appointment_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER REFERENCES Patient(Patient_ID),      -- NULL for a doctor's leave block
    doctor_id INTEGER NOT NULL REFERENCES Doctor(Doctor_ID),
    appointment_date TEXT NOT NULL,                           -- YYYY-MM-DD HH:MM:00
    status TEXT NOT NULL DEFAULT 'scheduled',                 -- scheduled | cancelled | completed | unavailable
    reason TEXT, notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS appt_doctor_day ON Appointment(doctor_id, appointment_date);
CREATE INDEX IF NOT EXISTS appt_patient ON Appointment(patient_id, status);
"""

DEMO_PATIENTS = [("asha", "sharma", "1985-03-02", 41, "Female", "919812345678"),
                 ("rohan", "sharma", "2015-07-19", 11, "Male", "919812345678"),  # a child shares his mother's phone
                 ("vikram", "nair", "1979-11-30", 46, "Male", "919845012367")]
DEMO_DOCTORS = [("anil", "kapoor", "General Medicine",
                 [(d, "09:00", "13:00", 15) for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]),
                ("sunita", "rao", "Pediatrics", [(d, "10:00", "12:00", 20) for d in ("Monday", "Wednesday", "Friday")]),
                ("vikas", "shah", "Cardiology", [(d, "14:00", "17:00", 30) for d in ("Tuesday", "Thursday")])]


@contextlib.contextmanager
def db(path):
    """One connection per operation. Commits on success, rolls back on error, always closes."""
    c = sqlite3.connect(path, timeout=5)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
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


def init(path, seed_demo: bool = False, reset: bool = False) -> Path:
    """Create the schema (idempotent) and, on an empty database and only if asked, demo data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{path}{suffix}").unlink(missing_ok=True)
    with db(path) as c:
        c.execute("PRAGMA journal_mode = WAL")  # persists in the file
        c.executescript(SCHEMA)
        row = c.execute("SELECT value FROM Meta WHERE key='schema_version'").fetchone()
        if row and int(row[0]) > SCHEMA_VERSION:
            raise RuntimeError(f"database schema v{row[0]} is newer than this code (v{SCHEMA_VERSION})")
        c.execute("INSERT OR REPLACE INTO Meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        if seed_demo and not c.execute("SELECT 1 FROM Doctor LIMIT 1").fetchone():
            _seed(c)
    return path


def _seed(c) -> None:
    c.executemany("INSERT INTO Patient (First_Name, Last_Name, DOB, age, sex, Phone_Number) VALUES (?,?,?,?,?,?)",
                  DEMO_PATIENTS)
    for first, last, specialty, hours in DEMO_DOCTORS:
        doc = c.execute("INSERT INTO Doctor (First_Name, Last_Name, Specialty) VALUES (?,?,?)",
                        (first, last, specialty)).lastrowid
        c.executemany("INSERT INTO Availability VALUES (?,?,?,?,?)", [(doc, *h) for h in hours])
    tue, wed = next_weekday("Tuesday"), next_weekday("Wednesday")
    c.executemany("INSERT INTO Appointment (patient_id, doctor_id, appointment_date, status, reason, notes) "
                  "VALUES (?,?,?,?,?,?)", [
                      (1, 1, f"{tue} 10:00:00", "scheduled", "fever and cough", None),
                      (2, 2, f"{wed} 10:20:00", "scheduled", "vaccination", None),
                      (3, 1, f"{tue} 09:00:00", "cancelled", "back pain", None),
                      (None, 1, f"{wed} 00:00:00", "unavailable", None, "full-day leave")])
