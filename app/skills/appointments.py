"""Appointments for the verified patient: free slots, book, list, cancel."""
from app.domain import appointments as rules
from app.registry import skill


@skill
def find_slots(ctx, doctor: str, date: str) -> dict:
    """Free appointment times for one doctor on one date.

    When: before booking, to offer times.
    Args:
        doctor: doctor's name or specialty in English letters, like Kapoor or pediatrician
        date: YYYY-MM-DD
    """
    return rules.find_slots(ctx.db, doctor, date, ctx.now())


@skill(needs_patient=True)
def book_appointment(ctx, doctor: str, date: str, time: str, reason: str) -> dict:
    """Book an appointment for the verified patient.

    When: after the caller said yes to a doctor, day, time and reason.
    Args:
        doctor: doctor's name or specialty in English letters, like Kapoor or pediatrician
        date: YYYY-MM-DD
        time: a free time from find_slots, like 9:15 AM
        reason: what the visit is for, in a few words
    """
    return rules.book(ctx.db, ctx.patient["id"], doctor, date, time, reason, ctx.now())


@skill(needs_patient=True)
def my_appointments(ctx) -> dict:
    """The verified patient's upcoming appointments.

    When: when the caller asks about their appointments.
    """
    found = rules.upcoming(ctx.db, ctx.patient["id"], ctx.now())
    if found:
        return {"patient": ctx.patient["name"], "appointments": found}
    return {"patient": ctx.patient["name"], "appointments": [], "next": "Say there are none, and offer to book one."}


@skill(needs_patient=True)
def cancel_appointment(ctx, appointment_id: int) -> dict:
    """Cancel one of the verified patient's appointments.

    When: after the caller said yes to cancelling a specific appointment.
    Args:
        appointment_id: from my_appointments
    """
    return rules.cancel(ctx.db, ctx.patient["id"], appointment_id, ctx.now())
