"""
Structured logging for all services, plus a per-request trace collector.

Two things live here:
1. `configure_logging()` / `log_step()` — same JSON structured logging
   pattern as before: every node/tool call emits start/end/error records
   tagged with a trace_id, for grep-by-trace_id debugging.
2. `TraceCollector` — an in-memory, per-request list of step records
   (agent, node, latency, token usage, cost) that the orchestrator
   assembles and returns to the client. This is what powers the
   Streamlit "thinking" dropdown — it's a superset view of what already
   gets logged, just also handed back over the wire for that one request.
"""
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from shared.core.config import get_settings
from shared.core.pricing import estimate_cost_usd

_trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(trace_id: str) -> None:
    _trace_id_ctx.set(trace_id)


def get_trace_id() -> str:
    return _trace_id_ctx.get()


@dataclass
class StepRecord:
    node: str
    agent: str | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    cost_usd: float = 0.0
    note: str | None = None


@dataclass
class TraceCollector:
    """Accumulates StepRecords for a single request. Not thread-safe by
    design — one collector per request/coroutine, held via contextvar."""
    steps: list[StepRecord] = field(default_factory=list)

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    def total_cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.steps), 6)

    def total_tokens(self) -> dict:
        return {
            "input_tokens": sum(s.input_tokens for s in self.steps),
            "output_tokens": sum(s.output_tokens for s in self.steps),
        }

    def to_list(self) -> list[dict]:
        return [
            {
                "node": s.node,
                "agent": s.agent,
                "latency_ms": s.latency_ms,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "model": s.model,
                "cost_usd": s.cost_usd,
                "note": s.note,
            }
            for s in self.steps
        ]


_trace_collector_ctx: ContextVar[TraceCollector | None] = ContextVar(
    "trace_collector", default=None
)


def start_trace_collector() -> TraceCollector:
    collector = TraceCollector()
    _trace_collector_ctx.set(collector)
    return collector


def get_trace_collector() -> TraceCollector | None:
    return _trace_collector_ctx.get()


def record_llm_usage(node: str, agent: str, latency_ms: float, model: str,
                      input_tokens: int, output_tokens: int, note: str | None = None) -> None:
    """Call this right after any LLM invocation to feed the trace collector."""
    collector = get_trace_collector()
    if collector is None:
        return
    cost = estimate_cost_usd(model, input_tokens, output_tokens)
    collector.add(StepRecord(
        node=node, agent=agent, latency_ms=latency_ms, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cost_usd=cost, note=note,
    ))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": get_trace_id(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "message", "levelname", "levelno", "name",
                       "pathname", "filename", "module", "exc_info", "exc_text",
                       "stack_info", "lineno", "funcName", "created", "msecs",
                       "relativeCreated", "thread", "threadName", "processName",
                       "process"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | trace=%(trace_id)s | %(message)s"
        ))
    root.handlers = [handler]


_logger = logging.getLogger("agentic.steps")


def _preview(value: Any, max_len: int = 400) -> str:
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "...<truncated>"


@contextmanager
def log_step(node: str, agent: str | None = None, **input_fields: Any) -> Iterator[dict]:
    start = time.perf_counter()
    ctx: dict[str, Any] = {}
    _logger.info(
        "step.start",
        extra={"node": node, "agent": agent, "phase": "start", "input": _preview(input_fields)},
    )
    try:
        yield ctx
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _logger.error(
            "step.error",
            extra={"node": node, "agent": agent, "phase": "error",
                   "latency_ms": latency_ms, "error": str(exc)},
            exc_info=True,
        )
        raise
    else:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _logger.info(
            "step.end",
            extra={"node": node, "agent": agent, "phase": "end",
                   "latency_ms": latency_ms, "output": _preview(ctx.get("output", "<no output set>"))},
        )
        collector = get_trace_collector()
        if collector is not None and "usage" not in ctx:
            # Steps that don't report LLM usage still get a bare timing record
            collector.add(StepRecord(node=node, agent=agent, latency_ms=latency_ms,
                                      note=_preview(ctx.get("output"), 120)))
