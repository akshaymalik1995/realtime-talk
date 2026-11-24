"""DocAssist's MySQL database as the store ([store] backend = "docassist"): the same rules as the demo clinic, on
DocAssist's own tables, scoped to one hospital ([docassist] hospital_id).

Tables (pyDocAssist sql/hospital_db_*.sql): Patient, Patient_Hospital, Doctor, Doctor_Hospital, doctor_availability,
Appointments. Writes follow DocAssist's own inserts (routes/hospital_admin_routes.py, table/Appointments.py): a new
patient is a Patient row plus a Patient_Hospital link; an appointment carries the hospital id.

Its data is less uniform than the demo's, so the SQL meets it where it is: phones are matched on their last 10 digits
(stored as 10, 12 or +91... digits), adulthood comes from the date of birth (age is often empty), an empty
specialization reads "General".

Credentials: DOCASSIST_SQL_HOST / _USER / _PASSWORD / _DATABASE in the environment, else the sql_* keys of
[docassist] config_ini's section (development). Each operation takes its own connection from a small pool (never
two threads on one): a remote MySQL costs ~1.9 s to connect, which a caller would hear on every lookup.
ponytail: no row locks on booking; two callers taking one slot in the same instant would both succeed.
Add SELECT ... FOR UPDATE on the doctor's day if that ever matters.
"""
import configparser
import contextlib
import logging
import os
import threading

import mysql.connector

log = logging.getLogger(__name__)

PHONE = "RIGHT(REGEXP_REPLACE(COALESCE(p.Phone_Number, ''), '[^0-9]', ''), 10) = RIGHT(%s, 10)"
WHEN = "CAST({}appointment_date AS CHAR)"  # 'YYYY-MM-DD HH:MM:SS' text, like the demo store (no % patterns:
                                           # the driver reads any %s inside SQL as a parameter, even escaped)
SPECIALTY = "COALESCE(NULLIF(d.Specialization, ''), 'General')"


def statements(h: int) -> dict:
    """The named statements, for hospital `h` (an int: safe to place in the SQL)."""
    h = int(h)
    return {
        "doctors": f"SELECT d.Doctor_ID, d.First_Name, d.Last_Name, {SPECIALTY}, GROUP_CONCAT(DISTINCT da.day) "
                   f"FROM Doctor d JOIN Doctor_Hospital dh ON dh.doctor_id = d.Doctor_ID "
                   f"LEFT JOIN doctor_availability da ON da.doctor_id = d.Doctor_ID AND da.duration_minutes > 0 "
                   f"WHERE dh.hospital_id = {h} AND dh.association_ended IS NULL AND COALESCE(d.is_active, 1) = 1 "
                   f"GROUP BY d.Doctor_ID ORDER BY d.Doctor_ID",
        "hours": "SELECT start_time, end_time, duration_minutes FROM doctor_availability WHERE doctor_id=%s AND day=%s",
        "day_appointments": f"SELECT {WHEN.format('')}, appointment_status, notes FROM Appointments WHERE doctor_id=%s "
                            f"AND appointment_status!='cancelled' AND DATE(appointment_date)=%s",
        "same_day": f"SELECT {WHEN.format('')} FROM Appointments WHERE patient_id=%s AND doctor_id=%s "
                    f"AND appointment_status='scheduled' AND DATE(appointment_date)=%s",
        "book": f"INSERT INTO Appointments (patient_id, doctor_id, hospital_id, appointment_date, appointment_status, "
                f"appointment_reason) VALUES (%s, %s, {h}, %s, 'scheduled', %s)",
        "upcoming": f"SELECT a.appointment_id, {WHEN.format('a.')}, a.appointment_reason, d.First_Name, d.Last_Name, "
                    f"{SPECIALTY} FROM Appointments a JOIN Doctor d ON d.Doctor_ID = a.doctor_id WHERE a.patient_id=%s "
                    f"AND a.hospital_id = {h} AND a.appointment_status='scheduled' AND a.appointment_date>=%s "
                    f"ORDER BY a.appointment_date",
        "appointment": f"SELECT appointment_status, {WHEN.format('')} FROM Appointments WHERE appointment_id=%s "
                       f"AND patient_id=%s AND hospital_id = {h}",
        "cancel": "UPDATE Appointments SET appointment_status='cancelled' WHERE appointment_id=%s",
        "adult_on_phone": f"SELECT p.First_Name, p.Last_Name FROM Patient p JOIN Patient_Hospital ph "
                          f"ON ph.patient_id = p.Patient_ID AND ph.hospital_id = {h} AND ph.relationship_ended IS NULL "
                          f"WHERE {PHONE} AND COALESCE(p.is_active, 1) = 1 "
                          f"AND TIMESTAMPDIFF(YEAR, p.DOB, CURDATE()) >= 18",
        "email_taken": "SELECT 1 FROM Patient WHERE Email=%s AND COALESCE(is_active, 1) = 1",
        "register": "INSERT INTO Patient (First_Name, Last_Name, DOB, age, sex, Phone_Number, Email, is_active) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, 1)",
        "link_patient": f"INSERT INTO Patient_Hospital (patient_id, hospital_id, relationship_started) "
                        f"VALUES (%s, {h}, CURDATE())",
        "on_phone": f"SELECT p.Patient_ID, p.First_Name, p.Last_Name FROM Patient p JOIN Patient_Hospital ph "
                    f"ON ph.patient_id = p.Patient_ID AND ph.hospital_id = {h} AND ph.relationship_ended IS NULL "
                    f"WHERE {PHONE} AND COALESCE(p.is_active, 1) = 1 ORDER BY p.Patient_ID",
    }


def credentials(cfg, env=os.environ) -> dict:
    """DOCASSIST_SQL_* from the environment, else the config.ini section. Never logged."""
    keys = ("host", "user", "password", "database")
    if all(env.get(f"DOCASSIST_SQL_{k.upper()}") for k in keys):
        return {k: env[f"DOCASSIST_SQL_{k.upper()}"] for k in keys}
    c = configparser.ConfigParser(interpolation=None)
    if not (cfg.config_ini and c.read(cfg.config_ini) and c.has_section(cfg.section)):
        raise RuntimeError("DocAssist store: set DOCASSIST_SQL_HOST/_USER/_PASSWORD/_DATABASE or [docassist] config_ini")
    s = c[cfg.section]
    return {"host": s["sql_host"], "user": s["sql_username"], "password": s["sql_password"], "database": s["sql_database"]}


class _Conn:
    """What the rules expect of a connection: execute(sql, params) -> a cursor with fetchone/fetchall/lastrowid."""

    def __init__(self, cn):
        self.cn = cn

    def execute(self, sql, params=()):
        cur = self.cn.cursor(buffered=True)  # buffered: the next statement never trips over unread rows
        cur.execute(sql, params)
        return cur


class _Idle:
    """Connections kept open between operations, opened only when needed (the driver's own pool opens all of them
    up front: ~19 s for five, longer than a skill may take). Each operation holds one connection alone."""

    def __init__(self, creds: dict, keep: int = 4):
        self.creds, self.keep, self.idle, self.lock = creds, keep, [], threading.Lock()

    def get(self):
        with self.lock:
            cn = self.idle.pop() if self.idle else None
        if cn is not None:
            try:
                cn.ping(reconnect=True, attempts=2, delay=0)  # the server may have dropped it while idle
                return cn
            except mysql.connector.Error:
                pass
        return mysql.connector.connect(**self.creds, connection_timeout=10, autocommit=False)

    def put(self, cn):
        with self.lock:
            if len(self.idle) < self.keep:
                self.idle.append(cn)
                return
        cn.close()


_POOLS: dict = {}
_POOLS_LOCK = threading.Lock()


def _pool(creds: dict) -> _Idle:
    """One per database for the whole process: conversations come and go, connections stay warm."""
    key = (creds["host"], creds["database"], creds["user"])
    with _POOLS_LOCK:
        return _POOLS.setdefault(key, _Idle(creds))


class DocAssistStore:
    def __init__(self, cfg, commit: bool = True):
        self.SQL, self.creds, self.commit = statements(cfg.hospital_id), credentials(cfg), commit
        # a store opens with each call: ready a connection now, while the caller is still saying hello. Opening one
        # to the remote MySQL takes 4-7 s, which the first lookup (the phone number) would otherwise make them hear.
        threading.Thread(target=self.warm, daemon=True).start()

    def warm(self):
        pool = _pool(self.creds)
        try:
            pool.put(pool.get())  # get() pings a kept connection (reopening it if dead) or opens a new one
        except mysql.connector.Error as e:
            log.warning("docassist warm-up failed (the first lookup will connect): %s", e)

    @contextlib.contextmanager
    def connect(self, write: bool = False):
        """A pooled connection for one operation. Writes commit at the end (commit=False: always roll back)."""
        pool = _pool(self.creds)
        cn = pool.get()
        try:
            yield _Conn(cn)
            if write and self.commit:
                cn.commit()
            else:
                cn.rollback()
        except BaseException:
            try:
                cn.rollback()
            finally:
                cn.close()  # its state is unknown: don't hand it to the next operation
            raise
        pool.put(cn)

    def check(self) -> list:
        with self.connect() as c:
            return c.execute(self.SQL["doctors"]).fetchall()


if __name__ == "__main__":  # live check on the configured DocAssist database, ROLLED BACK: python -m app.docassist
    import json
    import sys
    from datetime import date, datetime, timedelta

    from app import settings
    from app.domain import appointments as rules
    from app.domain import patients

    class OneTransaction(DocAssistStore):  # every step on one connection, undone at the end: nothing is saved
        def __init__(self, cfg):
            super().__init__(cfg, commit=False)
            self.cn = mysql.connector.connect(**self.creds, connection_timeout=10, autocommit=False)

        @contextlib.contextmanager
        def connect(self, write=False):
            yield _Conn(self.cn)

    s = settings.load(sys.argv[1] if len(sys.argv) > 1 else None)
    st = OneTransaction(s.docassist)
    try:
        docs = [d for d in rules.doctors(st) if d["days"]]
        assert docs, f"no doctor with hours at hospital {s.docassist.hospital_id}"
        rec, errs = patients.validate("Voicetest", "Patient", "9000071234", "Female", dob="1990-05-17")
        assert not errs and not patients.check_new(st, rec), errs
        pid = patients.insert(st, rec)["patient_id"]
        assert [p["id"] for p in patients.by_phone(st, rec["phone"])] == [pid]
        doc = docs[0]
        day = next(date.today() + timedelta(days=i) for i in range(1, 8)
                   if f"{date.today() + timedelta(days=i):%A}" in doc["days"])
        name = doc["name"].split()[-1]
        free = rules.find_slots(st, name, f"{day}")["free"]
        b = rules.book(st, pid, name, f"{day}", free[0], "voice desk check")
        assert b["status"] == "booked" and free[0] not in rules.find_slots(st, name, f"{day}")["free"], b
        assert [a["appointment_id"] for a in rules.upcoming(st, pid, datetime.now())] == [b["appointment_id"]]
        assert rules.cancel(st, pid, b["appointment_id"], datetime.now())["status"] == "cancelled"
        print(json.dumps({"hospital": s.docassist.hospital_id, "doctors_with_hours": len(docs), "registered": True,
                          "found_by_phone": True, "booked": b["time"], "cancelled": True}))
    finally:
        st.cn.rollback()
        st.cn.close()
    print("docassist ok (rolled back: nothing saved)")
