"""Appointment rules, after DocAssist (pyDocAssist: blueprints/appointment_api.py get_available_timeslots /
schedule_appointment / cancel_appointment, services/appointment_service.py):

- each doctor has weekly hours split into fixed-length slots (Availability.duration_minutes);
- a slot is taken by any non-cancelled appointment overlapping it;
- a doctor's "unavailable" entry closes the whole day ("full-day" in notes) or a range ("unavailable from HH:MM to HH:MM");
- booking: a doctor, a date that isn't past, a free slot start (not already over today), a reason;
  one active booking per patient per doctor per day (DocAssist flags this; a voice desk refuses it);
- cancelling: only the patient's own, only while "scheduled".
All functions take a store (app/store.py Sqlite, app/docassist.py DocAssistStore): its connect() and its named SQL
statements. The rules are the same for every store; only the SQL differs. They return plain dicts for the model.
"""
import re
from datetime import date, datetime

STOP_WORDS = {"dr", "doctor", "the", "a", "with", "for", "and", "specialist"}


def fmt_time(minutes: int) -> str:
    h, m = divmod(minutes, 60)
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def fmt_day(d: date) -> str:
    return f"{d:%A} {d.day} {d:%B %Y}"


def parse_time(s: str) -> int | None:
    """'9:15 AM', '9 am', '09:15', '14:30', '2.30 pm' -> minutes after midnight; None if unreadable."""
    m = re.fullmatch(r"\s*(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?\s*m?\.?\s*", s.lower())
    if not m:
        return None
    h, mins, ap = int(m[1]), int(m[2] or 0), m[3]
    if mins > 59 or h > 23 or (ap and not 1 <= h <= 12):
        return None
    if ap:
        h = h % 12 + (12 if ap == "p" else 0)
    return h * 60 + mins


def parse_day(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _doctors(c, sql) -> list[dict]:
    out = []
    for r in c.execute(sql["doctors"], ()).fetchall():  # id, first, last, specialty, working days (comma list)
        days = (r[4] or "").split(",")
        week = [d for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday") if d in days]
        out.append({"id": r[0], "name": f"Dr. {r[1].title()} {r[2].title()}", "specialty": r[3], "days": week,
                    "_words": set(re.findall(r"[a-z0-9]+", f"{r[1]} {r[2]} {r[3]}".lower()))})
    return out


def doctors(store) -> list[dict]:
    with store.connect() as c:
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in _doctors(c, store.SQL)]


def find_doctor(c, sql, query: str) -> tuple[dict | None, dict | None]:
    """Resolve what the caller said ("Dr Kapoor", "the pediatrician", "cardiology") to exactly one doctor."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in STOP_WORDS and len(w) >= 3]
    # prefixes, so "pediatrician" matches "pediatrics" and "cardio" matches "cardiology"
    hits = lambda d: sum(any(dw.startswith(w[:6]) for dw in d["_words"]) for w in words)  # noqa: E731
    docs = _doctors(c, sql)
    best = max((hits(d) for d in docs), default=0)
    found = [d for d in docs if best and hits(d) == best]  # the most words matched wins: "Meera Iyer" over "Meera"
    if len(found) == 1:
        return found[0], None
    names = [f"{d['name']} ({d['specialty']})" for d in docs]
    if not found:
        return None, {"status": "unknown_doctor", "problem": f"no doctor matches '{query}'", "doctors": names}
    return None, {"status": "which_doctor", "problem": f"'{query}' matches several doctors",
                  "doctors": [f"{d['name']} ({d['specialty']})" for d in found]}


def _slots(c, sql, doc: dict, day: date, now: datetime) -> dict:
    """Free and taken slot starts for one doctor and day, or why the day is closed."""
    if day < now.date():
        return {"closed": "that date is in the past"}
    hours = c.execute(sql["hours"], (doc["id"], f"{day:%A}")).fetchall()
    if not hours:
        return {"closed": f"{doc['name']} does not work on {day:%A}s (works {', '.join(doc['days'])})"}
    rows = c.execute(sql["day_appointments"], (doc["id"], day.isoformat())).fetchall()
    busy, blocks = [], []
    for when, status, notes in rows:
        start = parse_time(when[11:16])
        notes = (notes or "").lower()
        if status == "unavailable" and "full-day" in notes:
            return {"closed": f"{doc['name']} is on leave that day"}
        if status == "unavailable" and (m := re.search(r"from\s(\d{2}:\d{2})\s*to\s*(\d{2}:\d{2})", notes)):
            blocks.append((parse_time(m[1]), parse_time(m[2])))
        elif status != "unavailable":
            busy.append(start)
    free, taken = [], []
    for start_s, end_s, dur in hours:
        m, end = parse_time(start_s), parse_time(end_s)
        if m is None or end is None or not dur:  # unreadable hours in someone else's data: skip, don't crash
            continue
        while m + dur <= end:
            clash = any(m < b + dur and b < m + dur for b in busy) or any(a <= m < b for a, b in blocks)
            past = day == now.date() and m <= now.hour * 60 + now.minute
            (taken if clash or past else free).append(m)
            m += dur
    return {"free": free, "taken": taken}


def find_slots(store, doctor: str, day_s: str, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    day = parse_day(day_s)
    if not day:
        return {"status": "bad_date", "problem": "date must be YYYY-MM-DD"}
    with store.connect() as c:
        doc, problem = find_doctor(c, store.SQL, doctor)
        if problem:
            return problem
        s = _slots(c, store.SQL, doc, day, now)
    if "closed" in s:
        return {"status": "closed", "doctor": doc["name"], "date": fmt_day(day), "problem": s["closed"]}
    return {"status": "ok", "doctor": doc["name"], "specialty": doc["specialty"], "date": fmt_day(day),
            "free": [fmt_time(m) for m in s["free"]], "free_count": len(s["free"]),
            "next": "If the caller already asked for a time that is in free, confirm that one; otherwise offer two "
                    "or three free times. Never read the whole list."}


def book(store, patient_id: int, doctor: str, day_s: str, time_s: str, reason: str,
         now: datetime | None = None) -> dict:
    now = now or datetime.now()
    day, minutes, reason = parse_day(day_s), parse_time(time_s), " ".join(str(reason).split())[:200]
    if not day:
        return {"status": "bad_date", "problem": "date must be YYYY-MM-DD"}
    if minutes is None:
        return {"status": "bad_time", "problem": "time must look like 9:15 AM or 14:30"}
    if not reason:
        return {"status": "need_reason", "problem": "ask what the visit is for"}
    with store.connect(write=True) as c:  # check + insert in one write transaction
        doc, problem = find_doctor(c, store.SQL, doctor)
        if problem:
            return problem
        s = _slots(c, store.SQL, doc, day, now)
        if "closed" in s:
            return {"status": "closed", "problem": s["closed"]}
        if minutes not in s["free"]:
            return {"status": "slot_unavailable", "problem": f"{fmt_time(minutes)} is not free",
                    "free": [fmt_time(m) for m in s["free"]][:6]}
        same = c.execute(store.SQL["same_day"], (patient_id, doc["id"], day.isoformat())).fetchone()
        if same:
            return {"status": "already_booked",
                    "problem": f"already booked with {doc['name']} that day at {fmt_time(parse_time(same[0][11:16]))}"}
        h, m = divmod(minutes, 60)
        appt = c.execute(store.SQL["book"], (patient_id, doc["id"], f"{day} {h:02d}:{m:02d}:00", reason)).lastrowid
    return {"status": "booked", "appointment_id": appt, "doctor": doc["name"], "date": fmt_day(day),
            "time": fmt_time(minutes), "reason": reason}


def upcoming(store, patient_id: int, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now()
    with store.connect() as c:
        rows = c.execute(store.SQL["upcoming"], (patient_id, f"{now:%Y-%m-%d %H:%M:00}")).fetchall()
    return [{"appointment_id": r[0], "doctor": f"Dr. {r[3].title()} {r[4].title()}", "specialty": r[5],
             "date": fmt_day(parse_day(r[1][:10])), "time": fmt_time(parse_time(r[1][11:16])), "reason": r[2]}
            for r in rows]


def cancel(store, patient_id: int, appointment_id: int, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    with store.connect(write=True) as c:
        r = c.execute(store.SQL["appointment"], (appointment_id, patient_id)).fetchone()
        if not r:  # someone else's appointment looks exactly like a missing one
            return {"status": "not_found", "problem": f"no appointment {appointment_id} for this patient"}
        if r[0] != "scheduled" or r[1] < f"{now:%Y-%m-%d %H:%M:00}":
            return {"status": "not_cancellable", "problem": f"appointment {appointment_id} is {r[0]} or already past"}
        c.execute(store.SQL["cancel"], (appointment_id,))
    return {"status": "cancelled", "appointment_id": appointment_id}


if __name__ == "__main__":  # self-check: python -m app.domain.appointments
    import tempfile
    from pathlib import Path
    from app.store import Sqlite, init, next_weekday

    assert [parse_time(s) for s in ("9:15 AM", "9 am", "09:15", "14:30", "2.30 pm", "12:00 PM", "12 am")] == \
        [555, 540, 555, 870, 870, 720, 0]
    assert parse_time("25:00") is None and parse_time("13 pm") is None and parse_time("noon") is None
    assert fmt_time(555) == "9:15 AM" and fmt_time(870) == "2:30 PM" and fmt_time(0) == "12:00 AM"

    with tempfile.TemporaryDirectory() as d:
        db_ = Sqlite(init(Path(d) / "t.db", seed_demo=True))
        now = datetime.combine(date.today(), datetime.min.time()).replace(minute=1)  # the seed is relative to today
        tue, wed, thu = next_weekday("Tuesday"), next_weekday("Wednesday"), next_weekday("Thursday")
        with db_.connect() as c:
            assert find_doctor(c, db_.SQL, "Dr. Kapoor")[0]["name"] == "Dr. Anil Kapoor"
            assert find_doctor(c, db_.SQL, "the pediatrician")[0]["name"] == "Dr. Sunita Rao"
            assert find_doctor(c, db_.SQL, "cardiology")[0]["name"] == "Dr. Vikas Shah"
            assert find_doctor(c, db_.SQL, "Dr. Smith")[1]["status"] == "unknown_doctor"
        s = find_slots(db_, "Kapoor", f"{tue}", now)  # 09:00-13:00 in 15 min = 16 slots, Asha holds 10:00
        assert s["status"] == "ok" and s["free_count"] == 15 and "10:00 AM" not in s["free"] and s["free"][0] == "9:00 AM"
        assert find_slots(db_, "Kapoor", f"{wed}", now)["status"] == "closed"            # full-day leave
        assert "does not work" in find_slots(db_, "Shah", f"{wed}", now)["problem"]       # cardiology: Tue/Thu only
        assert find_slots(db_, "Kapoor", "2000-01-03", now)["problem"] == "that date is in the past"
        late = datetime.combine(tue, datetime.min.time()).replace(hour=11, minute=5)       # same day: past slots taken
        assert find_slots(db_, "Kapoor", f"{tue}", late)["free"][0] == "11:15 AM"
        assert book(db_, 3, "Kapoor", f"{tue}", "10:00 AM", "checkup", now)["status"] == "slot_unavailable"
        assert book(db_, 3, "Kapoor", f"{tue}", "10:05 AM", "checkup", now)["status"] == "slot_unavailable"  # not a slot start
        assert book(db_, 3, "Kapoor", f"{tue}", "10:15", "", now)["status"] == "need_reason"
        ok = book(db_, 3, "Kapoor", f"{tue}", "10:15", "checkup", now)
        assert ok["status"] == "booked" and ok["time"] == "10:15 AM"
        assert book(db_, 3, "Kapoor", f"{tue}", "11:00 AM", "again", now)["status"] == "already_booked"
        assert book(db_, 1, "Kapoor", f"{tue}", "10:15 AM", "x", now)["status"] == "slot_unavailable"  # Vikram has it
        assert [a["time"] for a in upcoming(db_, 3, now)] == ["10:15 AM"]              # his cancelled seed isn't listed
        assert cancel(db_, 1, ok["appointment_id"], now)["status"] == "not_found"       # Asha can't cancel Vikram's
        assert cancel(db_, 3, ok["appointment_id"], now)["status"] == "cancelled"
        assert cancel(db_, 3, ok["appointment_id"], now)["status"] == "not_cancellable"
        assert book(db_, 1, "Kapoor", f"{tue}", "10:15 AM", "x", now)["status"] == "already_booked"  # Asha: 10:00 that day
        assert book(db_, 2, "Kapoor", f"{tue}", "10:15 AM", "x", now)["status"] == "booked"  # freed slot is bookable
        assert find_slots(db_, "Rao", f"{thu}", now)["status"] == "closed"
    print("ok")
