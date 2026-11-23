"""New patients: validate, check duplicates, text a code. confirm_code (identity) writes the record."""
from typing import Literal

from app.domain import patients
from app.registry import skill


@skill
def start_registration(ctx, first_name: str, last_name: str, phone: str, sex: Literal["Male", "Female"],
                       dob: str = "", age: str = "", email: str = "") -> dict:
    """Validate a NEW patient's details and text a 6-digit code to their phone. Call again with corrections.

    When: for new patients: call it the moment you have first and last name, mobile, sex and date of birth (or age), in that same turn; its result is what you read back. Do not ask for email.
    Args:
        phone: 10-digit mobile number, digits only
        dob: YYYY-MM-DD, or empty if only age is known
        age: age in years, or empty if dob is given
        email: empty if not given
    """
    rec, errs = patients.validate(first_name, last_name, phone, sex, dob, age, email)
    if not errs:
        errs = patients.check_new(ctx.db, rec)
    if errs:
        ctx.otp.pending = None  # a failed attempt also cancels an older pending code
        return {"status": "invalid", "problems": errs}
    if problem := ctx.otp.send(rec["phone"], "registration", rec):
        return problem
    return {"status": "code_sent", "sent_to": f"phone ending {rec['phone'][-4:]}", "read_back": patients.read_back(rec),
            "next": "First read back every detail in read_back: spell both names letter by letter and say "
                    "the phone digits one by one. Then ask for the 6-digit code that was just texted."}
