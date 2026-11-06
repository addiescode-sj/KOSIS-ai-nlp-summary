"""The spans an answer carries (decision_metadata["spans"], T1-22), made from the question's records (T1-28 S3b).

The same dicts the pipeline's _Calls.span wrote before S3b: name, start_ms (from the question's start), duration_ms
(waits left out, as in latency_seconds), status, attributes (None values dropped). A model step's attributes are the
OpenTelemetry GenAI names first (gen_ai.request.model, gen_ai.usage.input_tokens / output_tokens, llm.attempts,
llm.status), then the step's own. Phase 2 (evals/runners/stat.py, evals/trace.py) and eval/grade_run.py read them, so
names, keys and order stay as they were.
"""

from __future__ import annotations

import time

from kosis_chat.llm.model import STATUS_OK
from kosis_chat.schemas.state import LLMCall, Record, StepSpan

# Every span name a stat answer can carry. A question's spans come in the order its steps ended (compose and verify
# repeat on a retry; event_lookup comes while a written plan is computed, after llm_plan or llm_plan_retry).
SPAN_NAMES = ("judge", "analyze.rules", "analyze.llm_fill", "analyze.llm_fill_retry", "analyze.llm_plan", "event_lookup",
              "analyze.llm_plan_retry", "compute", "compose", "verify", "template", "fallback_chatbot")


def llm_attributes(model: str, usage: dict, attempts: int, status: str) -> dict:
    """OpenTelemetry GenAI attribute names for one model step (HCX reports promptTokens/completionTokens)."""
    return {"gen_ai.request.model": model, "gen_ai.usage.input_tokens": usage.get("promptTokens"),
            "gen_ai.usage.output_tokens": usage.get("completionTokens"), "llm.attempts": attempts, "llm.status": status}


def step_span(name: str, started: float, status: str = "ok", **attributes) -> StepSpan:
    """A step without a model call, ended now."""
    return StepSpan(name=name, started=started, ended=time.perf_counter(), status=status, attributes=attributes)


def span_of(record: Record, t0: float) -> dict:
    """One record as an OpenTelemetry-shaped span dict."""
    if isinstance(record, LLMCall):
        status = "ok" if record.status == STATUS_OK else "error"
        waited = record.slept + record.retry_wait_seconds
        attributes = {**llm_attributes(record.model, record.usage, record.attempts, record.status), **record.attributes}
    else:
        status, waited, attributes = record.status, 0.0, record.attributes
    return {
        "name": record.name, "start_ms": round((record.started - t0) * 1000, 1),
        "duration_ms": round(max(0.0, record.ended - record.started - waited) * 1000, 1), "status": status,
        "attributes": {k: v for k, v in attributes.items() if v is not None},
    }


def spans_of(records: list[Record], t0: float) -> list[dict]:
    """The spans of a question's records, in record order (the order the steps ended).

    In the graph (T1-28 S4) only the main path writes records, one node after another, so this is the fixed order of
    the old router whatever the chart branch (which records nothing) does; no sorting is needed. Sorting by node
    name would break it anyway: compose and verify repeat, event_lookup comes between llm_plan and llm_plan_retry."""
    return [span_of(r, t0) for r in records]
