"""Skill registry: plain typed Python functions the model may call. Adding one = writing it in any module under
app/skills/. discover() imports those modules; the tool schema comes from the signature, the description,
the "When:" line and the per-argument help from the docstring. The module name is the skill's toolset;
agents choose toolsets in config.

    @skill(needs_patient=True)
    def cancel_appointment(ctx, appointment_id: int) -> dict:
        '''Cancel one of the verified patient's appointments.

        When: after the caller said yes to cancelling a specific appointment.
        Args:
            appointment_id: from my_appointments
        '''

Supported argument types: str, int, float, bool, typing.Literal[...] (an enum). An argument with a default
is optional. The first parameter is always `ctx`: the conversation (app.session.Session).

The model only PROPOSES a call. call() decides: bound the size, parse, validate against the schema
(allowlist), check the guard, run with a timeout. Every failure becomes {"error": ...}, never a crash.
"""
import asyncio
import importlib
import inspect
import json
import logging
import pkgutil
import re
import typing
from dataclasses import dataclass

log = logging.getLogger(__name__)

JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
PY_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
MAX_ARGS_CHARS = 10_000  # bound size before parsing (trust boundary)


@dataclass(frozen=True)
class Skill:
    name: str
    toolset: str          # the module it lives in
    description: str      # what it does (sent to the model)
    when: str             # one line for the prompt: when to use it
    parameters: dict      # JSON schema of the arguments
    fn: typing.Callable
    needs_patient: bool   # guard: refuse unless the conversation has a verified patient

    @property
    def spec(self) -> dict:
        """The function-tool definition the Realtime API expects."""
        return {"type": "function", "name": self.name, "description": self.description, "parameters": self.parameters}


REGISTRY: dict[str, Skill] = {}


class SkillDefinitionError(TypeError):
    pass


def skill(fn=None, *, needs_patient: bool = False, name: str | None = None):
    """Register a function as a skill. Usable bare (@skill) or with options (@skill(needs_patient=True))."""
    def register(f):
        s = describe(f, name or f.__name__, needs_patient)
        old = REGISTRY.get(s.name)
        if old and (old.fn.__module__, old.fn.__qualname__) != (f.__module__, f.__qualname__):
            raise SkillDefinitionError(f"skill '{s.name}' is defined twice: {old.fn.__module__} and {f.__module__}")
        REGISTRY[s.name] = s  # same function re-imported (e.g. a notebook re-run) simply replaces itself
        return f
    return register(fn) if fn else register


def _schema(t, where) -> dict:
    if typing.get_origin(t) is typing.Literal:
        values = typing.get_args(t)
        kinds = {type(v) for v in values}
        if len(kinds) != 1 or next(iter(kinds)) not in JSON_TYPES:
            raise SkillDefinitionError(f"{where}: Literal values must share one simple type")
        return {"type": JSON_TYPES[kinds.pop()], "enum": list(values)}
    if t in JSON_TYPES:
        return {"type": JSON_TYPES[t]}
    raise SkillDefinitionError(f"{where}: unsupported type {t!r} (use str, int, float, bool or Literal)")


def describe(f, name: str, needs_patient: bool = False) -> Skill:
    """Read a function into a Skill: schema from type hints, texts from the docstring."""
    doc = inspect.getdoc(f)
    if not doc:
        raise SkillDefinitionError(f"{name}: needs a docstring (the model reads it)")
    description = doc.split("\n\n")[0].replace("\n", " ").strip()
    when = (m[1].strip() if (m := re.search(r"^When:\s*(.+)$", doc, re.M)) else "")
    args_help = {}
    if m := re.search(r"^Args:\s*\n((?:[ \t]+.+\n?)+)", doc, re.M):
        for line in m[1].splitlines():
            if am := re.match(r"\s+(\w+):\s*(.+)", line):
                args_help[am[1]] = am[2].strip()
    params = list(inspect.signature(f).parameters.values())
    if not params or params[0].name != "ctx":
        raise SkillDefinitionError(f"{name}: the first parameter must be `ctx`")
    hints = typing.get_type_hints(f)
    props, required = {}, []
    for p in params[1:]:
        if p.name not in hints:
            raise SkillDefinitionError(f"{name}.{p.name}: needs a type hint")
        props[p.name] = _schema(hints[p.name], f"{name}.{p.name}")
        if p.name in args_help:
            props[p.name]["description"] = args_help[p.name]
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    unknown_help = set(args_help) - set(props)
    if unknown_help:
        raise SkillDefinitionError(f"{name}: docstring describes unknown argument(s) {sorted(unknown_help)}")
    parameters = {"type": "object", "properties": props, "required": required, "additionalProperties": False}
    return Skill(name, f.__module__.rsplit(".", 1)[-1], description, when, parameters, f, needs_patient)


def discover(package: str = "app.skills") -> dict[str, list[str]]:
    """Import every module of `package` (their @skill decorators register them). Returns {toolset: [skills]}."""
    pkg = importlib.import_module(package)
    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        if not m.name.startswith("_"):
            importlib.import_module(f"{package}.{m.name}")
    sets: dict[str, list[str]] = {}
    for s in REGISTRY.values():
        sets.setdefault(s.toolset, []).append(s.name)
    return sets


def select(toolsets) -> dict[str, Skill]:
    """The skills of some toolsets ("*" = all), in a stable order."""
    known = {s.toolset for s in REGISTRY.values()}
    if missing := [t for t in toolsets if t != "*" and t not in known]:
        raise SkillDefinitionError(f"unknown toolset(s) {missing}; known: {sorted(known)}")
    return {s.name: s for s in sorted(REGISTRY.values(), key=lambda s: (s.toolset, s.name))
            if "*" in toolsets or s.toolset in toolsets}


def validate(args, params: dict) -> str | None:
    """Allowlist check of parsed arguments against a flat JSON schema. Returns an error message or None."""
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
        if not isinstance(v, PY_TYPES[t]):
            return f"{k}: expected {t}, got {type(v).__name__}"
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"{k}: must be one of {props[k]['enum']}"
    return None


async def call(s: Skill, ctx, arguments: str, timeout: float) -> dict:
    """Run one proposed call. Never raises: every failure is returned as {"error": ...} (and logged)."""
    if len(arguments) > MAX_ARGS_CHARS:
        return {"error": "arguments too large"}
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"error": f"arguments are not valid JSON: {e}"}
    if err := validate(args, s.parameters):
        return {"error": err}
    if s.needs_patient and not getattr(ctx, "patient", None):
        return {"status": "needs_verification", "problem": "the caller is not verified yet",
                "next": "Ask for their mobile number and call verify_patient."}
    try:
        if inspect.iscoroutinefunction(s.fn):
            out = await asyncio.wait_for(s.fn(ctx, **args), timeout)
        else:  # sync skills run on a worker thread, so a slow one can't freeze the audio loop
            out = await asyncio.wait_for(asyncio.to_thread(s.fn, ctx, **args), timeout)
    except asyncio.TimeoutError:
        log.error("skill %s timed out after %ss", s.name, timeout)
        return {"error": f"{s.name} took too long; tell the caller and offer to try again"}
    except Exception as e:
        log.exception("skill %s failed", s.name)
        return {"error": f"{type(e).__name__}: {e}"}
    if not isinstance(out, dict):
        log.error("skill %s returned %s, not a dict", s.name, type(out).__name__)
        return {"error": f"{s.name} returned an invalid result"}
    return out
