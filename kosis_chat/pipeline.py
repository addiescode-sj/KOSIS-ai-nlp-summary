"""C pipeline: numbers by code, sentences by the model (docs/plans/20261024-c-compute-pipeline/PLAN.md).

judge (rules) -> analyze (rules, the model fills what they leave open) -> compute -> compose (model, {f1} citations)
-> verify (one retry, then a code template) -> render. StatPipeline.ask returns the same Answer as Chatbot.ask, with
the finished StatAnswer in Answer.stat_answer. A question the analysis cannot place goes to the compact-v2 chatbot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from kosis_chat.answer.chart import chart_spec
from kosis_chat.answer.render import final_text, render_parts
from kosis_chat.answer.template import template_parts
from kosis_chat.answer.verify import label_digits, verify
from kosis_chat.chatbot import Answer, Chatbot, judge_record
from kosis_chat.compute import CalcRefused, compute
from kosis_chat.compute.analyze import Plan, Unresolved, build_request, rule_plan, to_query
from kosis_chat.config import Settings, load_settings
from kosis_chat.contract import to_source
from kosis_chat.data import preprocessing
from kosis_chat.data.schema import TableInput
from kosis_chat.judge.answerability import Verdict, judge
from kosis_chat.judge.question import build_index
from kosis_chat.llm import analyze_fill, compose
from kosis_chat.llm.model import STATUS_OK, LLMClient, build_client
from kosis_chat.llm.structured import StructuredResult, ask_structured

from kosis_ai.schemas import Refusal, StatAnswer, StatQuery  # isort: skip

VERSION = compose.VERSION
FALLBACK_PROMPT = "compact-v2"


def add_usage(total: dict[str, Any], usage: dict[str, Any]) -> dict[str, Any]:
    """Add the API's numeric token counts of one call into a running total (in place; returned for convenience)."""
    for key, value in usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    return total


@dataclass
class _Calls:
    """Model calls made for one question, summed the way Answer reports them."""

    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    waited: float = 0.0
    prompt_chars: int = 0

    def add(self, result: StructuredResult, system: str, user: str) -> None:
        self.attempts += result.attempts
        add_usage(self.usage, result.usage)
        self.raw += result.raw
        self.errors += result.errors
        self.waited += result.retry_wait_seconds
        self.prompt_chars += len(system) + len(user)


class StatPipeline:
    """Answers questions about one table; same attributes run_eval reads from a Chatbot."""

    def __init__(
        self,
        table: TableInput,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        org_id: str | None = None,
        retry_interval: float = 0.0,
        fallback_scope: str = "auto",
    ) -> None:
        self.settings = settings or load_settings()
        self._client = client or build_client(self.settings)
        self.table = preprocessing.clean(table)
        self.index = build_index(self.table)
        self.org_id = org_id
        self.retry_interval = retry_interval
        self.prompt_version = VERSION
        self.system_prompt = compose.SYSTEM_PROMPT
        self.answer_model = compose.ComposeOutput
        self.decision_metadata = {"pipeline": "stat", "outcome_origin": "rules", "numbers_origin": "code",
                                  "defaults_origin": "rules"}
        self.fallback_scope = fallback_scope
        self._fallback: Chatbot | None = None

    @property
    def client(self) -> LLMClient:
        return self._client

    @client.setter
    def client(self, value: LLMClient) -> None:  # run_eval swaps in its recording client
        self._client = value
        if self._fallback is not None:
            self._fallback.client = value

    @property
    def fallback(self) -> Chatbot:
        """compact-v2 with the same rules, for the questions the analysis cannot place (built on first use)."""
        if self._fallback is None:
            self._fallback = Chatbot(
                self.table, client=self._client, settings=replace(self.settings, prompt_version=FALLBACK_PROMPT),
                prompt_version=FALLBACK_PROMPT, scope_mode=self.fallback_scope, answer_format="json",
                retry_interval=self.retry_interval, judge_mode="rules",
            )
        return self._fallback

    # ------------------------------------------------------------ steps

    def _ask_model(self, calls: _Calls, system: str, user: str, model) -> StructuredResult:
        if calls.raw and self.retry_interval:  # a second call for the same question: stay under the rate limit
            started = time.perf_counter()
            time.sleep(self.retry_interval)
            calls.waited += time.perf_counter() - started
        result = ask_structured(self._client, system, user, model, retry_interval=self.retry_interval,
                                max_input_tokens=self.settings.max_input_tokens)
        calls.add(result, system, user)
        return result

    def _request(self, plan: Plan, calls: _Calls, meta: dict):
        try:
            return plan, build_request(plan, self.index, self.table)
        except Unresolved as exc:
            meta["unresolved"] = exc.what
        plan = replace(plan, unresolved=plan.unresolved or meta["unresolved"])
        user = analyze_fill.user_prompt(plan.question, self.index, plan)
        meta["route"] = "llm_fill"
        filled, problems, kind = self._fill(plan, user, calls, meta, "fill")
        if filled is None:
            return plan, None
        if (built := self._build(filled, kind, meta)) is not None:
            return built
        if problems:  # the model's answer had a correctable fault: say it back once (analyze-fill-v2)
            again, _, kind = self._fill(plan, analyze_fill.retry_prompt(user, problems), calls, meta, "fill_retry")
            if again is not None:
                if (built := self._build(again, kind, meta)) is not None:
                    return built
                filled = again
        return filled, None

    def _fill(self, plan: Plan, user: str, calls: _Calls, meta: dict, key: str):
        """(plan filled by the model, problems to say back, the type the model named), or (None, [], None)."""
        result = self._ask_model(calls, analyze_fill.SYSTEM_PROMPT, user, analyze_fill.AnalyzeFill)
        if result.status != STATUS_OK:
            meta[f"{key}_status"] = result.status
            return None, [], None
        filled, ignored, problems = analyze_fill.apply(plan, result.value, self.index)
        meta[key] = result.value.model_dump(exclude_defaults=True) | ({"ignored": ignored} if ignored else {})
        return filled, problems, result.value.type

    def _build(self, filled: Plan, kind, meta: dict):
        tries = [filled]
        if kind and kind != filled.type and not filled.op:  # an operation fixes its own type (value/trend/share)
            tries.append(replace(filled, type=kind))
        for candidate in tries:
            try:
                return candidate, build_request(candidate, self.index, self.table)
            except Unresolved as exc:
                meta["unresolved_after_fill"] = exc.what
        return None

    def _compose(self, question: str, computed, note: str | None, calls: _Calls, meta: dict) -> tuple[list, bool]:
        """(parts, verified): the model's sentence once it passes verification, else a code template.

        verified is what StatAnswer.numbers_verified records: the parts the answer shows passed verify()."""
        user = compose.user_prompt(question, computed, note)
        problems: list[str] = []
        for attempt in range(2):
            if problems:
                user = compose.retry_prompt(user, problems)
            result = self._ask_model(calls, compose.SYSTEM_PROMPT, user, compose.ComposeOutput)
            if result.status != STATUS_OK:
                meta["compose_status"] = result.status
                break
            parts = compose.to_parts(result.value.answer, computed)
            problems = verify(parts, computed)
            meta.setdefault("verify_problems", []).append(problems)
            if not problems:
                meta["compose"] = "model" if attempt == 0 else "model_retry"
                return parts, True
        meta["compose"] = "template"
        parts = template_parts(meta["type"], computed)
        problems = verify(parts, computed)  # a template cites every main fact by construction; recorded, not assumed
        if problems:
            meta["template_problems"] = problems
        return parts, not problems

    # ------------------------------------------------------------ entry point

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        started = time.perf_counter()
        verdict = judge(question, self.table, self.index)
        meta: dict[str, Any] = {**self.decision_metadata, "route": "rules"}
        if verdict.final:
            return self._done(self._verdict_answer(question, verdict), verdict, meta | {"route": "rules_final"}, _Calls(), started)
        calls = _Calls()
        plan = rule_plan(question, self.index, verdict.default_applied)
        plan, request = self._request(plan, calls, meta)
        if request is None:  # neither the rules nor the model placed the question: the compact-v2 chatbot answers
            answer = self.fallback.ask(question, golden_scope)
            # The chatbot's own record wins (its model wrote the numbers; its outcome_origin says whether a rule or
            # the model decided); the analysis calls made before it are added, not lost.
            return replace(
                answer, attempts=answer.attempts + calls.attempts, raw_responses=calls.raw + answer.raw_responses,
                format_errors=calls.errors + answer.format_errors, retry_wait_seconds=answer.retry_wait_seconds + calls.waited,
                latency_seconds=max(0.0, time.perf_counter() - started - answer.retry_wait_seconds - calls.waited),
                usage=add_usage(dict(calls.usage), answer.usage),
                input_tokens_estimate=calls.prompt_chars + answer.input_tokens_estimate,
                decision_metadata={**meta, **answer.decision_metadata, "pipeline": "stat", "route": "fallback_chatbot",
                                   "fallback_prompt": FALLBACK_PROMPT, "numbers_origin": "model", "numbers_verified": False},
            )
        source = to_source(self.table.meta, org_id=self.org_id)
        query = to_query(plan, self.index, request)
        meta["type"] = plan.type
        try:
            computed = compute(self.table, request)
        except CalcRefused as exc:
            stat = StatAnswer(outcome="refused", query=query, refusal=Refusal(reason=exc.reason, message=exc.message),
                              text=final_text(exc.message, source), source=source, numbers_origin="none")
            return self._done(stat, verdict, meta | {"route": f"{meta['route']}+calc_refused"}, calls, started)
        parts, verified = self._compose(question, computed, computed.note, calls, meta)
        meta["label_digits"] = label_digits(parts, computed)
        meta["numbers_verified"] = verified
        body = render_parts(parts, computed.result)
        chart = chart_spec(plan.type, computed)
        stat = StatAnswer(
            outcome="answered", query=query, result=computed.result, parts=parts,
            text=final_text(body, source, verdict.default_applied, [computed.note]),
            default_applied=verdict.default_applied, source=source,
            numbers_origin="code", numbers_verified=verified, defaults_verified=True,  # rule_plan computed with them
            chart=chart,
        )
        meta["chart"] = chart.kind if chart else None
        return replace(self._done(stat, verdict, meta, calls, started), chart=chart.spec if chart else None)

    def _verdict_answer(self, question: str, verdict: Verdict) -> StatAnswer:
        query = StatQuery(question=question)
        if verdict.outcome == "refused":
            r = verdict.refusal
            source = None if r.reason == "out_of_scope" else to_source(self.table.meta, org_id=self.org_id)
            return StatAnswer(outcome="refused", query=query, refusal=Refusal(reason=r.reason, message=r.message, available=r.available),
                              text=final_text(r.message, source), source=source, numbers_origin="none")
        options = list(verdict.clarify_options)
        body = "\n".join([verdict.clarify_question, *(f"- {o}" for o in options)])
        source = to_source(self.table.meta, org_id=self.org_id)
        return StatAnswer(outcome="clarify", query=query, clarify_options=options, text=final_text(body, source), source=source,
                          numbers_origin="none")

    def _done(self, stat: StatAnswer, verdict: Verdict, meta: dict, calls: _Calls, started: float) -> Answer:
        return Answer(
            text=stat.text, status=STATUS_OK, table_meta=self.table.meta, prompt_version=self.prompt_version,
            input_tokens_estimate=calls.prompt_chars, attempts=calls.attempts, usage=calls.usage,
            latency_seconds=max(0.0, time.perf_counter() - started - calls.waited), retry_wait_seconds=calls.waited,
            scope={"mode": "cells", "cells": len(stat.result.facts)}, format_errors=calls.errors, raw_responses=calls.raw,
            answer_format="json", decision_metadata=meta, judge=judge_record(verdict), stat_answer=stat,
        )
