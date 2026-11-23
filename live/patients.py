"""Patient rules: registration (notebook 06) and lookup by phone (notebook 07).

Rules copied from DocAssist (pyDocAssist: PatientService.register_patient, routes/patient_routes.py,
Utility.validate_*): names required, DOB as DD/MM/YYYY or YYYY-MM-DD and not in the future, age must
match DOB within 1 year, sex Male/Female, optional email must be valid, an adult's phone must be unique,
phone stored with the 91 prefix. Storage and SMS are the simulated clinic (live/clinic.py).

Notebook 06's two tools (Registrar):
  start_registration(...)    validate + duplicate check, then text a 6-digit code to the patient's phone.
                             The model gets a read-back, never the code.
  confirm_registration(code) the patient reads the code back; only a match writes the record.
Notebook 07 folds the same rules into the one front desk (live/frontdesk.py).
"""
import re
import time
from datetime import date, datetime
from pathlib import Path

from live.clinic import DB_PATH, Otp, db as _db, init_db  # noqa: F401 (init_db, DB_PATH: notebook 06's API)

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,49}$")
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")  # DocAssist's regex

INSTRUCTIONS = """You are the front desk of a hospital, registering new patients by voice. Be brief and warm.
Collect: first name, last name, 10-digit mobile number, sex (Male or Female), and date of birth (or age if they do not know it).
Ask only for what is missing. Do not ask for email.
As soon as you have all of them, call start_registration right away. Do not read the details back before calling it:
its result is what you read back. Spell the names and say the phone number digit by digit.
Then ask for the 6-digit code that was just texted to their phone.
If they correct anything, call start_registration again with the fixed details; a new code is sent.
When they say the code, call confirm_registration with the digits.
Only say the patient is registered after confirm_registration returns status registered, then give the patient ID."""
# Dictation has pauses: end the turn after 900 ms of silence, not 500, so a patient isn't cut off mid-details.
VAD = {"type": "server_vad", "silence_duration_ms": 900}



# ---- validation (pure functions) ----

def normalize_phone(phone: str) -> str | None:
    """Indian mobile: 10 digits -> '91' + digits; 12 digits starting 91 kept. Anything else: None."""
    d = re.sub(r"\D", "", phone or "")
    if len(d) == 10:
        return "91" + d
    if len(d) == 12 and d.startswith("91"):
        return d
    return None


def parse_dob(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s.strip(), fmt).date()
            break
        except ValueError:
            continue
    else:
        raise ValueError("date of birth must be DD/MM/YYYY or YYYY-MM-DD")
    if d > date.today():
        raise ValueError("date of birth cannot be in the future")
    if d.year < 1900:
        raise ValueError("date of birth is before 1900")
    return d


def age_on(dob: date, today: date | None = None) -> int:
    t = today or date.today()
    return t.year - dob.year - ((t.month, t.day) < (dob.month, dob.day))


def dob_from_age(age: int, today: date | None = None) -> date:
    """DocAssist's estimate (same day, `age` years ago); 29 Feb falls back to 28 Feb."""
    t = today or date.today()
    try:
        return t.replace(year=t.year - age)
    except ValueError:
        return t.replace(year=t.year - age, day=28)


def validate(first_name, last_name, phone, sex, dob="", age="", email="") -> tuple[dict | None, list[str]]:
    """Allowlist check of one registration. Returns (clean record, []) or (None, [problems])."""
    errs, rec = [], {}
    for key, v in (("first_name", first_name), ("last_name", last_name)):
        v = " ".join(str(v).split())
        if NAME_RE.fullmatch(v):
            rec[key] = v.lower()  # DocAssist stores names lower-case
        else:
            errs.append(f"{key.replace('_', ' ')}: letters only, 1-50 characters")
    rec["phone"] = normalize_phone(str(phone))
    if not rec["phone"]:
        errs.append("phone: must be a 10-digit mobile number (or 12 digits starting with 91)")
    if sex not in ("Male", "Female"):
        errs.append("sex: must be Male or Female")
    rec["sex"] = sex
    dob, age = str(dob).strip(), str(age).strip()
    born = years = None
    if dob:
        try:
            born = parse_dob(dob)
        except ValueError as e:
            errs.append(f"dob: {e}")
    if age:
        if age.isdigit() and int(age) <= 120:
            years = int(age)
        else:
            errs.append("age: a whole number from 0 to 120")
    if not dob and not age:
        errs.append("need a date of birth or an age")
    if born and years is not None and abs(age_on(born) - years) > 1:
        errs.append(f"age {years} doesn't match date of birth {born:%d %B %Y} (that is {age_on(born)})")
    if born:
        rec["dob"], rec["age"] = born, age_on(born)
    elif years is not None:
        rec["dob"], rec["age"] = dob_from_age(years), years
    email = str(email).strip()
    if email and not EMAIL_RE.fullmatch(email):
        errs.append("email: not a valid address")
    rec["email"] = email or None
    return (None, errs) if errs else (rec, [])


def duplicates(c, rec) -> list[str]:
    """DocAssist: an adult's phone must be unique; an email must be unique."""
    errs = []
    if rec["age"] >= 18:
        row = c.execute("SELECT First_Name, Last_Name FROM Patient WHERE Phone_Number=? AND is_active=1 AND age>=18",
                        (rec["phone"],)).fetchone()
        if row:
            errs.append(f"this phone is already registered to an adult patient ({row[0].title()} {row[1][0].upper()}.)")
    if rec["email"] and c.execute("SELECT 1 FROM Patient WHERE Email=? AND is_active=1", (rec["email"],)).fetchone():
        errs.append("this email is already registered to another patient")
    return errs


def spell(word: str) -> str:
    return "-".join(ch.upper() for ch in word if ch.isalpha())


def read_back(rec) -> dict:
    """What the model should read out: spelled names, phone digit by digit."""
    local = rec["phone"][2:]
    return {"name": f"{rec['first_name'].title()} {rec['last_name'].title()}",
            "spelled": f"{spell(rec['first_name'])}  {spell(rec['last_name'])}",
            "phone_digits": " ".join(local),
            "sex": rec["sex"],
            "date_of_birth": f"{rec['dob']:%d %B %Y}", "age": rec["age"],
            **({"email": rec["email"]} if rec["email"] else {})}


def start(path, otp: Otp, first_name, last_name, phone, sex, dob="", age="", email="") -> dict:
    """Validate + duplicate-check a registration, then text a code (purpose "registration").
    The model's view: a read-back, never the code."""
    rec, errs = validate(first_name, last_name, phone, sex, dob, age, email)
    if not errs:
        with _db(path) as c:
            errs = duplicates(c, rec)
    if errs:
        otp.pending = None  # a failed attempt also cancels an older pending code
        return {"status": "invalid", "problems": errs}
    sent_to = otp.send(rec["phone"], "registration", rec)
    return {"status": "code_sent", "sent_to": sent_to, "read_back": read_back(rec),
            "next": "First read back every detail in read_back: spell both names letter by letter and say "
                    "the phone digits one by one. Then ask for the 6-digit code that was just texted."}


def insert(path, rec) -> dict:
    """Write a validated registration, re-checking duplicates inside the write transaction."""
    with _db(path) as c:
        c.execute("BEGIN IMMEDIATE")
        if errs := duplicates(c, rec):
            return {"status": "invalid", "problems": errs}
        cur = c.execute("INSERT INTO Patient (First_Name, Last_Name, DOB, age, sex, Phone_Number, Email) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (rec["first_name"], rec["last_name"], rec["dob"].isoformat(), rec["age"], rec["sex"],
                         rec["phone"], rec["email"]))
    return {"status": "registered", "patient_id": cur.lastrowid, "name": read_back(rec)["name"]}


def by_phone(path, phone: str) -> list[dict]:
    """Active patients on a phone (a parent and children may share one)."""
    with _db(path) as c:
        rows = c.execute("SELECT Patient_ID, First_Name, Last_Name FROM Patient WHERE Phone_Number=? AND is_active=1 "
                         "ORDER BY Patient_ID", (phone,)).fetchall()
    return [{"id": r[0], "name": f"{r[1].title()} {r[2].title()}"} for r in rows]


class Registrar:
    """Notebook 06: one conversation that can only register. (Notebook 07's front desk does this and more.)"""

    def __init__(self, path=DB_PATH, otp=None, clock=time.monotonic):
        self.path = Path(path)
        self.codes = Otp(clock=clock, **({"gen": otp} if otp else {}))

    outbox = property(lambda self: self.codes.outbox)
    pending = property(lambda self: self.codes.pending)

    def start_registration(self, first_name, last_name, phone, sex, dob="", age="", email=""):
        return start(self.path, self.codes, first_name, last_name, phone, sex, dob, age, email)

    def confirm_registration(self, code):
        purpose, rec, problem = self.codes.check(code)
        return problem or insert(self.path, rec)

    def tools(self, spec) -> dict:
        """Registry for live.tools.run. Optional fields are strings; "" means not given."""
        s = {"type": "string"}
        return {
            "start_registration": (spec(
                "start_registration",
                "Validate a new patient's details and text a 6-digit confirmation code to their phone. "
                "Call again with corrected details if the patient fixes anything.",
                {"first_name": s, "last_name": s,
                 "phone": {"type": "string", "description": "10-digit mobile number, digits only"},
                 "sex": {"type": "string", "enum": ["Male", "Female"]},
                 "dob": {"type": "string", "description": "YYYY-MM-DD, or empty if only age is known"},
                 "age": {"type": "string", "description": "age in years, or empty if dob is given"},
                 "email": {"type": "string", "description": "empty if not given"}}),
                self.start_registration),
            "confirm_registration": (spec(
                "confirm_registration",
                "Register the patient once they read back the 6-digit code from the SMS.",
                {"code": {"type": "string", "description": "the 6 digits the patient read out"}}),
                self.confirm_registration),
        }


if __name__ == "__main__":  # self-check: python -m live.patients
    import tempfile
    from live import tools

    assert normalize_phone("98765 43210") == "919876543210" and normalize_phone("+91-98765-43210") == "919876543210"
    assert normalize_phone("12345") is None and normalize_phone("449876543210") is None
    assert parse_dob("15/08/1990") == parse_dob("1990-08-15") == date(1990, 8, 15)
    assert dob_from_age(4, date(2028, 2, 29)) == date(2024, 2, 29) and dob_from_age(1, date(2028, 2, 29)) == date(2027, 2, 28)
    assert age_on(date(1990, 8, 15), date(2026, 8, 14)) == 35
    ok, e = validate("Ravi", "Kumar", "9876543210", "Male", dob="1990-08-15", age="35")
    assert not e and ok["phone"] == "919876543210" and ok["first_name"] == "ravi"
    for bad in (dict(first_name=""), dict(first_name="R2D2"), dict(phone="12345"), dict(sex="M"),
                dict(dob="2999-01-01"), dict(dob="31/02/1990"), dict(dob="", age=""), dict(age="35.5"),
                dict(dob="1990-08-15", age="50"), dict(email="not-an-email")):
        args = dict(first_name="Ravi", last_name="Kumar", phone="9876543210", sex="Male", dob="1990-08-15", age="",
                    email="") | bad
        assert validate(**args)[1], bad

    with tempfile.TemporaryDirectory() as d:
        db = init_db(Path(d) / "t.db")
        codes = iter(["111111", "222222", "333333"])
        t = [0.0]
        r = Registrar(db, otp=lambda: next(codes), clock=lambda: t[0])
        assert r.confirm_registration("1")["status"] == "no_code_pending"
        assert r.start_registration("Asha", "Rao", "98123 45678", "Female", age="30")["status"] == "invalid"  # adult dup
        assert r.start_registration("Mini", "Sharma", "9812345678", "Female", age="6")["status"] == "code_sent"  # child ok
        out = r.start_registration("Ravi", "Kumar", "9876543210", "Male", dob="1990-08-15")
        assert out["status"] == "code_sent" and "222222" not in str(out) and r.outbox[-1]["text"].endswith("222222.")
        assert r.confirm_registration("000000") == {"status": "wrong_code", "tries_left": 2}
        assert r.confirm_registration("2 2 2 2 2 2")["status"] == "registered"         # spoken digits with spaces
        assert r.confirm_registration("222222")["status"] == "no_code_pending"         # code is single-use
        r.start_registration("Ravi", "Kumar", "9000000001", "Male", age="40")
        t[0] = r.codes.ttl + 1
        assert r.confirm_registration("333333")["status"] == "expired"
        reg = r.tools(tools.spec)
        assert tools.run(reg, "confirm_registration", '{"code": "1", "admin": true}')["error"].startswith("unknown arg")
        assert [p["name"] for p in by_phone(db, "919812345678")] == ["Asha Sharma", "Rohan Sharma"]  # Mini never confirmed
        with _db(db) as c:
            assert c.execute("SELECT COUNT(*) FROM Patient").fetchone()[0] == 4        # 3 seeded + Ravi
    print("ok")
