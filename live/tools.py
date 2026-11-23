"""Deterministic tool calling: the model only PROPOSES a call (name + JSON args).
This code decides: validate against the schema, run real Python, return a result.

registry = {name: (spec, fn)}"""
import json
import logging

MAX_ARGS_CHARS = 10_000  # bound size before parsing (trust boundary)
TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


def spec(name: str, description: str, properties: dict) -> dict:
    """Function tool spec in the Realtime API's shape. All params required, no extras allowed."""
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(properties), "additionalProperties": False}}


def validate(args, params: dict) -> str | None:
    """Allowlist check of `args` against a flat JSON-schema subset. Returns an error message or None."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = params["properties"]
    if extra := sorted(set(args) - set(props)):
        return f"unknown argument(s): {extra}"
    if missing := [k for k in params["required"] if k not in args]:
        return f"missing argument(s): {missing}"
    for k, v in args.items():
        t = props[k]["type"]
        if isinstance(v, bool) and t != "boolean":  # bool is an int subclass in Python
            return f"{k}: expected {t}, got boolean"
        if not isinstance(v, TYPES[t]):
            return f"{k}: expected {t}, got {type(v).__name__}"
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"{k}: must be one of {props[k]['enum']}"
    return None


def run(registry: dict, name: str, arguments: str) -> dict:
    """Execute one proposed call. Never raises: every failure becomes {'error': ...} so the
    model can tell the user (or retry) instead of the voice loop crashing."""
    if name not in registry:
        return {"error": f"unknown tool {name!r}"}
    if len(arguments) > MAX_ARGS_CHARS:
        return {"error": "arguments too large"}
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as e:
        return {"error": f"arguments are not valid JSON: {e}"}
    tool_spec, fn = registry[name]
    if err := validate(args, tool_spec["parameters"]):
        return {"error": err}
    try:
        return {"result": fn(**args)}
    except Exception as e:
        logging.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":  # self-check: python -m live.tools
    s = spec("f", "d", {"n": {"type": "integer"}, "u": {"type": "string", "enum": ["C", "F"]}})
    reg = {"f": (s, lambda n, u: n * 2)}
    assert run(reg, "f", '{"n": 2, "u": "C"}') == {"result": 4}
    assert "unknown tool" in run(reg, "g", "{}")["error"]
    assert "not valid JSON" in run(reg, "f", "{")["error"]
    assert "must be a JSON object" in run(reg, "f", "[1]")["error"]
    assert "unknown argument" in run(reg, "f", '{"n": 1, "u": "C", "x": 0}')["error"]
    assert "missing" in run(reg, "f", '{"n": 1}')["error"]
    assert "got boolean" in run(reg, "f", '{"n": true, "u": "C"}')["error"]
    assert "expected integer" in run(reg, "f", '{"n": 1.5, "u": "C"}')["error"]
    assert "one of" in run(reg, "f", '{"n": 1, "u": "K"}')["error"]
    assert "too large" in run(reg, "f", " " * (MAX_ARGS_CHARS + 1))["error"]
    reg_boom = {"f": (s, lambda n, u: 1 / 0)}
    assert "ZeroDivisionError" in run(reg_boom, "f", '{"n": 1, "u": "C"}')["error"]
    print("ok")
