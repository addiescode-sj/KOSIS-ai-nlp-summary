"""What one question records as it goes through the steps (T1-28 S3b): its model calls and its step spans.

The pipeline used to sum its model calls into a mutable _Calls object while it went. Now each LLM step returns the
calls it made as LLMCall records, the router records the other steps as StepSpan records, both in one list in the
order they ended, and the totals an Answer reports (graph/legacy.py) and the spans it carries (graph/spans.py
spans_of) are read from that list. StatState (T1-28 S4) is the LangGraph state one question carries through the graph
(graph/build.py): the records and decision_metadata are its two reducer fields (append, merge).

Times are time.perf_counter() values: spans_of turns them into offsets from the question's start. They are kept
unvalidated (SkipValidation), like every number a record carries, so an int stays an int (shared/boundary.py).
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

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


def append(old: list, new: list) -> list:
    """StatState.records reducer: a node's records go after the ones before it (a new list; neither is changed)."""
    return [*old, *new]


def merge(old: dict, new: dict) -> dict:
    """StatState.meta reducer: what a node wrote, as dict.update would (a key already there keeps its place, a new key
    goes last), into a new dict."""
    return {**old, **new}


class StatState(BaseModel):
    """What one question carries through the graph (T1-28 S4, graph/build.py).

    LangGraph builds this model from the channels before every node (`StatState(**channels)`) and a node returns a
    dict of the fields it changes. records and meta are reducer fields (Annotated[..., append] / [..., merge]): a
    node returns only its own records / entries. Every other field is the last value written; only one node writes
    each in a step (chart runs beside compose_llm and writes only `chart_out`). No field has a node's name (chart_out,
    caveats_out: kept apart so a state key never reads as a node name). Objects are stored as passed
    (SkipValidation, Any), never validated or copied: the step In/Out models check them at the step boundary. Their
    types live in the stage packages, which this package does not import (named in the comments).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    golden_scope: SkipValidation[dict[str, Any] | None] = None  # run_eval's golden scope, for the fallback chatbot
    started: SkipValidation[float] = 0.0  # time.perf_counter() at the question's start (span offsets, judge's start)
    records: Annotated[SkipValidation[list[Any]], append] = Field(default_factory=list)  # Record, in the order made
    meta: Annotated[SkipValidation[dict[str, Any]], merge] = Field(default_factory=dict)  # decision_metadata, no spans
    step_route: str | None = None  # the route the last node on the main path chose (the conditional edges read it)
    verdict: Any = None  # judge.schemas.Verdict
    rule: Any = None  # analyze.rules.Plan: the rules' plan
    plan: Any = None  # analyze.rules.Plan: the plan computed (the rules', the fill's, or the model's query_plan)
    built: Any = None  # analyze.schemas.Built, None until a plan is placed
    unresolved: SkipValidation[list[str] | None] = None
    computed: Any = None  # compute.schemas.Computed (set early by plan_write_llm, which computes its own plan)
    clarify: Any = None  # compute.resolve.NeedsClarify (plan_write_llm's ask-back)
    query: Any = None  # kosis_ai.schemas.StatQuery
    source: Any = None  # kosis_ai.schemas.Source
    refusal: Any = None  # kosis_ai.schemas.Refusal (compute's)
    alternatives: SkipValidation[list[Any]] = Field(default_factory=list)
    caveats_out: SkipValidation[list[Any]] = Field(default_factory=list)  # compose.caveats Caveat (node "caveats")
    shown: SkipValidation[list[str]] = Field(default_factory=list)
    context_ids: SkipValidation[list[str]] = Field(default_factory=list)
    compose_attempt: int = 0  # compose_llm <-> verify loop (at most COMPOSE_MAX_ATTEMPTS attempts)
    compose_problems: SkipValidation[list[str]] = Field(default_factory=list)
    compose_prompt: str | None = None
    parts: SkipValidation[list[Any] | None] = None
    verified: bool = False
    chart_out: Any = None  # kosis_ai.schemas.Chart (written by the chart branch, node "chart")
    stat: Any = None  # kosis_ai.schemas.StatAnswer (render / answer_final)
    fallback: Any = None  # chatbot.Answer of the fallback chatbot


__all__ = ["LLMCall", "Record", "StatState", "StepSpan", "append", "called_before", "merge"]
