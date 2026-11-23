"""Public clinic information: no patient needed."""
from app.domain import appointments as rules
from app.registry import skill


@skill
def list_doctors(ctx) -> dict:
    """List the clinic's doctors, their specialties and working days.

    When: when asked which doctors there are.
    """
    return {"doctors": rules.doctors(ctx.db)}


@skill
def clinic_hours(ctx) -> dict:
    """The clinic's opening hours.

    When: when asked when the clinic is open.
    """
    return {"hours": "Monday to Friday, 9 AM to 5 PM. Closed on weekends."}
