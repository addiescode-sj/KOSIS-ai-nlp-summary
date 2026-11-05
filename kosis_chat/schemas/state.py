"""What one question records as it goes through the steps (T1-28 S3b): its model calls and its step spans.

The pipeline used to sum its model calls into a mutable _Calls object while it went. Now each LLM step returns the
calls it made as LLMCall records, the router records the other steps as StepSpan records, both in one list in the
order they ended, and the totals an Answer reports (graph/legacy.py) and the spans it carries (graph/spans.py
spans_of) are read from that list. StatState (the LangGraph state) comes with S4.

Times are time.perf_counter() values: spans_of turns them into offsets from the question's start. They are kept
unvalidated (SkipValidation), like every number a record carries, so an int stays an int (shared/boundary.py).
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, SkipValidation

from kosis_chat.shared.boundary import BoundaryModel


class LLMCall(BoundaryModel):
    """One model call of a step (ask_structured: the first request and its format retries), or the fallback chatbot.

    name: the span name (analyze.llm_fill, analyze.llm_fill_retry, analyze.llm_plan, analyze.llm_plan_retry,
    event_lookup, compose, fallback_chatbot). slept: the wait before the call (a second call for the same question,
    StatPipeline retry_interval); retry_wait_seconds: the waits between format retries. Both are left out of the span
    duration and added to Answer.retry_wait_seconds. status: the call's status (llm.model STATUS_*). attributes: the
    step's own span attributes (event=, attempt=), written after the gen_ai ones.
    summed: False for the fallback chatbot, whose Answer reports its own calls (they are added to it, not summed here).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    started: SkipValidation[float]
    ended: SkipValidation[float]
    model: SkipValidation[str | None]  # as the client names itself, unchecked (the span drops a None, as before S3b)
    status: SkipValidation[str]
    attempts: SkipValidation[int] = 0
    usage: SkipValidation[dict[str, Any]] = Field(default_factory=dict)
    raw: SkipValidation[list[str]] = Field(default_factory=list)
    errors: SkipValidation[list[str]] = Field(default_factory=list)
    slept: SkipValidation[float] = 0.0
    retry_wait_seconds: SkipValidation[float] = 0.0
    prompt_chars: SkipValidation[int] = 0  # len(system) + len(user), Answer.input_tokens_estimate
    attributes: SkipValidation[dict[str, Any]] = Field(default_factory=dict)
    summed: bool = True


class StepSpan(BoundaryModel):
    """A step that calls no model (judge, analyze.rules, compute, verify, template): its span as recorded."""

    model_config = ConfigDict(frozen=True)

    name: str
    started: SkipValidation[float]
    ended: SkipValidation[float]
    status: SkipValidation[str] = "ok"
    attributes: SkipValidation[dict[str, Any]] = Field(default_factory=dict)


Record = LLMCall | StepSpan


def called_before(records: list[Record]) -> bool:
    """A model answer came back already for this question (then a further call waits retry_interval first)."""
    return any(r.raw for r in records if isinstance(r, LLMCall) and r.summed)


__all__ = ["LLMCall", "Record", "StepSpan", "called_before"]
