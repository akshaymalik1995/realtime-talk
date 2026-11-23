"""Patient rules, after DocAssist (pyDocAssist: PatientService.register_patient, routes/patient_routes.py,
Utility.validate_*): names required, DOB as DD/MM/YYYY or YYYY-MM-DD and not in the future, age must match DOB
within 1 year, sex Male/Female, optional email must be valid, an adult's phone must be unique, phone stored with
the 91 prefix, names stored lower-case.
"""
import re
from datetime import date, datetime

from app.store import db

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,49}$")
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")  # DocAssist's regex


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
            rec[key] = v.lower()
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


def check_new(path, rec) -> list[str]:
    with db(path) as c:
        return duplicates(c, rec)


def spell(word: str) -> str:
    return "-".join(ch.upper() for ch in word if ch.isalpha())


def read_back(rec) -> dict:
    """What the model should read out: spelled names, phone digit by digit."""
    return {"name": f"{rec['first_name'].title()} {rec['last_name'].title()}",
            "spelled": f"{spell(rec['first_name'])}  {spell(rec['last_name'])}",
            "phone_digits": " ".join(rec["phone"][2:]),
            "sex": rec["sex"],
            "date_of_birth": f"{rec['dob']:%d %B %Y}", "age": rec["age"],
            **({"email": rec["email"]} if rec["email"] else {})}


def insert(path, rec) -> dict:
    """Write a validated registration, re-checking duplicates inside the write transaction."""
    with db(path) as c:
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
    with db(path) as c:
        rows = c.execute("SELECT Patient_ID, First_Name, Last_Name FROM Patient WHERE Phone_Number=? AND is_active=1 "
                         "ORDER BY Patient_ID", (phone,)).fetchall()
    return [{"id": r[0], "name": f"{r[1].title()} {r[2].title()}"} for r in rows]
