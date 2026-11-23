"""Who is calling: verify a phone with a texted code, then pick the patient if several share it."""
import re

from app.domain import patients
from app.registry import skill


def verified(ctx, p: dict) -> dict:
    ctx.patient, ctx.choices = {"id": p["id"], "name": p["name"]}, None
    return {"status": "verified", "name": p["name"],
            "next": f"Greet {p['name'].split()[0]}, then carry on with what they already asked for "
                    "(for example call my_appointments if they wanted their appointments). "
                    "Ask only if they asked nothing yet."}


@skill
def verify_patient(ctx, phone: str) -> dict:
    """Look up an existing patient by mobile number and text them a 6-digit code to prove it is their phone.

    When: first step for any existing patient; call it the moment they give a mobile number.
    Args:
        phone: 10-digit mobile number, digit for digit as the caller said it
    """
    number = patients.normalize_phone(phone)
    if not number:
        return {"status": "bad_phone", "problem": "need a 10-digit mobile number"}
    found = patients.by_phone(ctx.db, number)
    if not found:
        ctx.otp.pending = None
        return {"status": "not_found", "problem": "no patient has this number",
                "next": "Check the number, or offer to register them as a new patient."}
    # who is on this phone is only revealed after the code proves the caller holds it
    if problem := ctx.otp.send(number, "verification", found):
        return problem
    return {"status": "code_sent", "sent_to": f"phone ending {number[-4:]}",
            "next": "Ask for the 6-digit code that was just texted."}


@skill
def confirm_code(ctx, code: str) -> dict:
    """Check the 6-digit code the caller read from the SMS (for verification or registration).

    When: only after the caller has read out the 6 digits of a code.
    Args:
        code: exactly the 6 digits the caller read out
    """
    if len(re.sub(r"\D", "", code)) != 6:  # a placeholder or a half-heard code must not burn a try
        return {"status": "need_digits", "problem": "that is not a 6-digit code",
                "next": "Ask the caller to read the 6 digits from the SMS." if ctx.otp.pending else
                        "No code has been sent yet: call verify_patient (or start_registration) first."}
    if not ctx.otp.pending and ctx.patient:  # e.g. the model repeats a call after a reconnect
        return {"status": "already_verified", "name": ctx.patient["name"],
                "next": "They are verified: carry on with what they asked for."}
    purpose, payload, problem = ctx.otp.check(code)
    if problem:
        return problem
    if purpose == "registration":
        out = patients.insert(ctx.db, payload)
        if out["status"] == "registered":
            verified(ctx, {"id": out["patient_id"], "name": out["name"]})
            out["next"] = "Give them their patient ID, then ask if they would like to book an appointment."
        return out
    if len(payload) == 1:
        return verified(ctx, payload[0])
    ctx.choices = payload
    return {"status": "choose_patient", "names": [p["name"] for p in payload],
            "next": "Several patients share this phone: ask which of them they are calling about."}


@skill
def choose_patient(ctx, name: str) -> dict:
    """Pick which patient on a shared phone the caller means, after confirm_code asked to choose.

    When: after confirm_code returns choose_patient.
    Args:
        name: the name the caller said
    """
    if not ctx.choices:
        return {"status": "nothing_to_choose", "problem": "verify the phone first"}
    said = name.lower().split()
    match = [p for p in ctx.choices if said and all(w in p["name"].lower().split() for w in said)]
    if len(match) != 1:
        return {"status": "which_one", "names": [p["name"] for p in ctx.choices]}
    return verified(ctx, match[0])
