"""OpenTelemetry instrumentation and in-process accounting.

Two distinct concerns live here:

* **Tracing** — :func:`alm_span` wraps any block in an OTel span.  When
  ``ALM_TELEMETRY_ENABLED`` is false the tracer is a no-op provider, so the
  runtime pays nothing for instrumentation it does not export.
* **Accounting** — :class:`UsageMeter` accumulates tokens, cost and latency per
  model and per tier.  The evaluation harness (§11) reads it to compute *cost
  per query* and *p95 latency*, which are federation-level metrics that no
  single backend can report on its own.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_initialised = False


def init_telemetry(enabled: bool = True, console_export: bool = False) -> None:
    """Install a tracer provider.  Idempotent."""
    global _initialised
    if _initialised or not enabled:
        return
    provider = TracerProvider()
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialised = True


def get_tracer(name: str = "alm") -> trace.Tracer:
    return trace.get_tracer(name)


@contextmanager
def alm_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    """Start a span named ``name`` with optional attributes."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in (attributes or {}).items():
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, str(value))
        yield span


def current_trace_ids() -> tuple[str, str]:
    """Return ``(trace_id, span_id)`` of the active span as hex strings."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return "", ""
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CallRecord:
    """A single inference call, as accounted by the federation."""

    model_id: str
    tier: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    component: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageMeter:
    """Accumulates :class:`CallRecord` entries for one federation session."""

    records: list[CallRecord] = field(default_factory=list)

    def record(self, call: CallRecord) -> None:
        self.records.append(call)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.cost_usd for r in self.records), 8)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    @property
    def total_latency_ms(self) -> float:
        """Wall-clock is measured elsewhere; this is summed inference time."""
        return round(sum(r.latency_ms for r in self.records), 3)

    @property
    def call_count(self) -> int:
        return len(self.records)

    def by_tier(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            bucket = out.setdefault(
                r.tier, {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0, "latency_ms": 0.0}
            )
            bucket["calls"] += 1
            bucket["tokens"] += r.total_tokens
            bucket["cost_usd"] += r.cost_usd
            bucket["latency_ms"] += r.latency_ms
        return {k: {m: round(v, 8) for m, v in b.items()} for k, b in out.items()}

    def by_model(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for r in self.records:
            bucket = out.setdefault(
                r.model_id, {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0, "latency_ms": 0.0}
            )
            bucket["calls"] += 1
            bucket["tokens"] += r.total_tokens
            bucket["cost_usd"] += r.cost_usd
            bucket["latency_ms"] += r.latency_ms
        return {k: {m: round(v, 8) for m, v in b.items()} for k, b in out.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "inference_latency_ms": self.total_latency_ms,
            "by_tier": self.by_tier(),
            "by_model": self.by_model(),
        }

    def merge(self, other: UsageMeter) -> None:
        self.records.extend(other.records)


class Stopwatch:
    """Monotonic wall-clock timer in milliseconds."""

    __slots__ = ("_start", "_elapsed")

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._elapsed: float | None = None

    def stop(self) -> float:
        if self._elapsed is None:
            self._elapsed = (time.perf_counter() - self._start) * 1000.0
        return self._elapsed

    @property
    def elapsed_ms(self) -> float:
        if self._elapsed is not None:
            return self._elapsed
        return (time.perf_counter() - self._start) * 1000.0
