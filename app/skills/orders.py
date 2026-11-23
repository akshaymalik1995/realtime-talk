"""Support demo (notebook 05): order status and a unit conversion. Used by the "support" agent."""
from typing import Literal

from app.registry import skill

ORDERS = {"A123": {"status": "shipped", "arrives": "Thursday"},
          "B456": {"status": "processing", "arrives": "next week"}}


@skill
def get_order_status(ctx, order_id: str) -> dict:
    """Look up a customer's order by its ID, e.g. A123.

    When: when asked about an order.
    Args:
        order_id: Order ID like A123
    """
    key = "".join(ch for ch in order_id.upper() if ch.isalnum())
    if key not in ORDERS:
        return {"status": "not_found", "problem": f"no order {key}"}
    return {"status": "ok", **ORDERS[key]}


@skill
def convert_temperature(ctx, value: float, to: Literal["C", "F"]) -> dict:
    """Convert a temperature to Celsius (C) or Fahrenheit (F).

    When: when asked to convert a temperature.
    """
    result = round(value * 9 / 5 + 32, 1) if to == "F" else round((value - 32) * 5 / 9, 1)
    return {"status": "ok", "result": result, "unit": to}
