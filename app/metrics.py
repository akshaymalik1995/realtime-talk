"""In-process counters and gauges, served at /metrics in the Prometheus text format.

Enough for one process (dashboards, alerts on stalls/errors). Several processes behind a load balancer:
scrape each one, or swap this for prometheus_client; callers only use inc()/set().
"""
import threading
from collections import defaultdict

HELP = {
    "sessions_total": "Conversations started.",
    "sessions_active": "Conversations in progress.",
    "sessions_rejected_total": "Conversations refused because the server was full or shutting down.",
    "tool_calls_total": "Tool calls by skill and result status.",
    "stalls_total": "OpenAI connections that went silent mid-response.",
    "resumes_total": "Reconnects after a stall.",
    "session_errors_total": "Conversations that ended with an error, by kind.",
}


def _escape(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()  # skills (and so inc()) also run on worker threads
        self._values: dict[tuple, float] = defaultdict(float)

    def inc(self, name: str, by: float = 1, **labels) -> None:
        with self._lock:
            self._values[(name, tuple(sorted(labels.items())))] += by

    def set(self, name: str, value: float, **labels) -> None:
        with self._lock:
            self._values[(name, tuple(sorted(labels.items())))] = value

    def get(self, name: str, **labels) -> float:
        with self._lock:
            return self._values.get((name, tuple(sorted(labels.items()))), 0)

    def render(self) -> str:
        with self._lock:
            items = sorted(self._values.items())
        out, seen = [], set()
        for (name, labels), value in items:
            if name not in seen:
                seen.add(name)
                kind = "gauge" if name.endswith("_active") else "counter"
                out += [f"# HELP {name} {HELP.get(name, name)}", f"# TYPE {name} {kind}"]
            label = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
            out.append(f"{name}{{{label}}} {value:g}" if label else f"{name} {value:g}")
        return "\n".join(out) + "\n"


METRICS = Metrics()
