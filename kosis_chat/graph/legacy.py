"""The legacy Answer a stat question returns (T1-28 S3b): what StatPipeline.ask returned, built from its records.

eval/run_eval.py and Phase 2 read the chatbot.Answer of StatPipeline.ask (attempts, usage, raw_responses,
format_errors, retry_wait_seconds, latency_seconds, input_tokens_estimate, decision_metadata with its spans), so the
totals are summed from the LLMCall records exactly as the pipeline's _Calls summed them, in call order.
This file is the graph's one door to kosis_chat/legacy/ (T1-28 S6): the B0 chatbot's Answer and judge_record, and the
fallback node (legacy/fallback.py) that graph/build.py and graph/adapters.py take from here. No other graph or stage
file imports legacy/ (tests/test_layout.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from kosis_chat.graph.spans import spans_of
from kosis_chat.judge.schemas import Verdict
from kosis_chat.legacy.chatbot import Answer, judge_record
from kosis_chat.legacy.fallback import FallbackIn, step_fallback_chatbot  # noqa: F401  for graph/build.py, adapters.py
from kosis_chat.llm.model import STATUS_OK
from kosis_chat.prepare.schemas import TableMeta
from kosis_chat.schemas.state import LLMCall, Record, StatState

from kosis_ai.schemas import StatAnswer  # isort: skip

# The decision_metadata keys a stat answer itself can write (beside "spans", and beside the keys of the fallback
# chatbot's own record, which it merges in). tests/test_nodes_llm.py keeps the set.
DECISION_KEYS = frozenset({
    "pipeline", "outcome_origin", "numbers_origin", "defaults_origin", "route",  # every answer
    "unresolved", "fill", "fill_status", "fill_retry", "fill_retry_status", "unresolved_after_fill",  # plan_fill_llm
    "event_lookup", "plan", "plan_status", "plan_problems", "plan_retry", "plan_retry_status",  # plan_write_llm
    "plan_retry_problems",
    "type", "pattern",  # the plan computed ("plan" too: the v3 plan code computed)
    "compose", "compose_status", "verify_problems", "template_problems", "label_digits", "numbers_verified",  # compose
    "chart", "caveats",
    "fallback_prompt",  # fallback_chatbot
})


def add_usage(total: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Add the API's numeric token counts of one call into a running total (in place; returned for convenience)."""
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    return total


@dataclass
class Totals:
    """The model calls of one question summed the way Answer reports them (was pipeline._Calls)."""

    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    waited: float = 0.0
    prompt_chars: int = 0


def totals_of(records: list[Record]) -> Totals:
    """Sum the summed LLMCall records in order (the fallback chatbot's call is reported by its own Answer)."""
    total = Totals()
    for call in records:
        if not isinstance(call, LLMCall) or not call.summed:
            continue
        if call.slept:
            total.waited += call.slept
        total.attempts += call.attempts
        add_usage(total.usage, call.usage)
        total.raw += call.raw
        total.errors += call.errors
        total.waited += call.retry_wait_seconds
        total.prompt_chars += call.prompt_chars
    return total


def to_answer(stat: StatAnswer, verdict: Verdict, meta: dict, records: list[Record], started: float, *,
              table_meta: TableMeta, prompt_version: str) -> Answer:
    """The Answer of a question the stat pipeline answered itself (answered, refused, asked back)."""
    total = totals_of(records)
    return Answer(
        text=stat.text, status=STATUS_OK, table_meta=table_meta, prompt_version=prompt_version,
        input_tokens_estimate=total.prompt_chars, attempts=total.attempts, usage=total.usage,
        latency_seconds=max(0.0, time.perf_counter() - started - total.waited), retry_wait_seconds=total.waited,
        scope={"mode": "cells", "cells": len(stat.result.facts)}, format_errors=total.errors, raw_responses=total.raw,
        answer_format="json", decision_metadata=meta | {"spans": spans_of(records, started)}, judge=judge_record(verdict),
        stat_answer=stat,
    )


def fallback_answer(answer: Answer, meta: dict, records: list[Record], started: float, *, fallback_prompt: str) -> Answer:
    """The fallback chatbot's Answer with the analysis calls made before it added, not lost.

    The chatbot's own record wins (its model wrote the numbers; its outcome_origin says whether a rule or the model
    decided)."""
    total = totals_of(records)
    return replace(
        answer, attempts=answer.attempts + total.attempts, raw_responses=total.raw + answer.raw_responses,
        format_errors=total.errors + answer.format_errors, retry_wait_seconds=answer.retry_wait_seconds + total.waited,
        latency_seconds=max(0.0, time.perf_counter() - started - answer.retry_wait_seconds - total.waited),
        usage=add_usage(dict(total.usage), answer.usage),
        input_tokens_estimate=total.prompt_chars + answer.input_tokens_estimate,
        decision_metadata={**meta, **answer.decision_metadata, "pipeline": "stat", "route": "fallback_chatbot",
                           "fallback_prompt": fallback_prompt, "numbers_origin": "model", "numbers_verified": False,
                           "spans": spans_of(records, started)},
    )


def answer_of(state: StatState, *, table_meta: TableMeta, prompt_version: str, fallback_prompt: str) -> Answer:
    """The Answer of a question the graph ran (T1-28 S4): the fallback chatbot's, or the stat answer's with its chart.

    The same Answer StatPipeline.ask built at each of its exits before S4 (to_answer / fallback_answer; an answered
    question carries its chart in Answer.chart: a ChartSpec since T1-28 S5, the Vega-Lite spec before)."""
    if state.fallback is not None:
        return fallback_answer(state.fallback, state.meta, state.records, state.started, fallback_prompt=fallback_prompt)
    answer = to_answer(state.stat, state.verdict, state.meta, state.records, state.started, table_meta=table_meta,
                       prompt_version=prompt_version)
    return replace(answer, chart=state.chart_out) if state.chart_out is not None else answer
