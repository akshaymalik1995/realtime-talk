"""Storage: SQLite, one connection per operation (skills run on worker threads; connections don't cross threads).

Production notes: WAL mode lets readers and one writer work at once; busy_timeout makes a second writer wait
instead of failing. Writes that check-then-insert use BEGIN IMMEDIATE so two callers can't take one slot.
Another database is another store: connect() plus the same named SQL statements (app/docassist.py: DocAssist's
MySQL). The rules never open connections themselves. Table shapes follow DocAssist (pyDocAssist table/Patient.py, appointment_api.py).
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


class Sqlite:
    """The demo store: a SQLite file. The rules (app/domain) open connections with connect() and run these named
    statements; another store (app/docassist.py) supplies the same names for its own database."""
    SQL = {
        "doctors": "SELECT d.Doctor_ID, d.First_Name, d.Last_Name, d.Specialty, GROUP_CONCAT(DISTINCT a.day) "
                   "FROM Doctor d LEFT JOIN Availability a ON a.doctor_id = d.Doctor_ID "
                   "GROUP BY d.Doctor_ID ORDER BY d.Doctor_ID",
        "hours": "SELECT start_time, end_time, duration_minutes FROM Availability WHERE doctor_id=? AND day=? "
                 "ORDER BY start_time",
        "day_appointments": "SELECT appointment_date, status, notes FROM Appointment WHERE doctor_id=? "
                            "AND status!='cancelled' AND substr(appointment_date, 1, 10)=?",
        "same_day": "SELECT appointment_date FROM Appointment WHERE patient_id=? AND doctor_id=? AND status='scheduled' "
                    "AND substr(appointment_date, 1, 10)=?",
        "book": "INSERT INTO Appointment (patient_id, doctor_id, appointment_date, status, reason) "
                "VALUES (?, ?, ?, 'scheduled', ?)",
        "upcoming": "SELECT a.Appointment_ID, a.appointment_date, a.reason, d.First_Name, d.Last_Name, d.Specialty "
                    "FROM Appointment a JOIN Doctor d ON d.Doctor_ID = a.doctor_id WHERE a.patient_id=? "
                    "AND a.status='scheduled' AND a.appointment_date>=? ORDER BY a.appointment_date",
        "appointment": "SELECT status, appointment_date FROM Appointment WHERE Appointment_ID=? AND patient_id=?",
        "cancel": "UPDATE Appointment SET status='cancelled' WHERE Appointment_ID=?",
        "adult_on_phone": "SELECT First_Name, Last_Name FROM Patient WHERE Phone_Number=? AND is_active=1 AND age>=18",
        "email_taken": "SELECT 1 FROM Patient WHERE Email=? AND is_active=1",
        "register": "INSERT INTO Patient (First_Name, Last_Name, DOB, age, sex, Phone_Number, Email) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
        "on_phone": "SELECT Patient_ID, First_Name, Last_Name FROM Patient WHERE Phone_Number=? AND is_active=1 "
                    "ORDER BY Patient_ID",
    }

    def __init__(self, path):
        self.path = path

    @contextlib.contextmanager
    def connect(self, write: bool = False):
        """write: one IMMEDIATE transaction, so a check-then-insert can't race another caller."""
        with db(self.path) as c:
            if write:
                c.execute("BEGIN IMMEDIATE")
            yield c

    def check(self) -> list:
        with self.connect() as c:
            return c.execute(self.SQL["doctors"]).fetchall()


def open_store(settings):
    """The store named by [store] backend."""
    if settings.store.backend == "docassist":
        from app.docassist import DocAssistStore  # needs mysql-connector-python; only loaded when used
        return DocAssistStore(settings.docassist)
    return Sqlite(settings.path(settings.store.path))


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
