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


@dataclass
class _Calls:
    """Model calls made for one question, summed the way Answer reports them."""

    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    raw: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    waited: float = 0.0
    prompt_chars: int = 0
    spans: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def add(self, result: StructuredResult, system: str, user: str) -> None:
        self.attempts += result.attempts
        for key, value in result.usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.usage[key] = self.usage.get(key, 0) + value
        self.raw += result.raw
        self.errors += result.errors
        self.waited += result.retry_wait_seconds
        self.prompt_chars += len(system) + len(user)

    def span(self, name: str, started: float, status: str = "ok", waited: float = 0.0, **attributes) -> None:
        """One step as an OpenTelemetry-shaped span (T1-22); waits are left out of the duration, as in latency_seconds."""
        self.spans.append({
            "name": name, "start_ms": round((started - self.t0) * 1000, 1),
            "duration_ms": round(max(0.0, time.perf_counter() - started - waited) * 1000, 1), "status": status,
            "attributes": {k: v for k, v in attributes.items() if v is not None},
        })


def _llm_attributes(model: str, usage: dict, attempts: int, status: str) -> dict:
    """OpenTelemetry GenAI attribute names for one model step (HCX reports promptTokens/completionTokens)."""
    return {"gen_ai.request.model": model, "gen_ai.usage.input_tokens": usage.get("promptTokens"),
            "gen_ai.usage.output_tokens": usage.get("completionTokens"), "llm.attempts": attempts, "llm.status": status}


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

    @property
    def model_name(self) -> str:
        return self.settings.model_name if self._client.name == "hcx" else self._client.name

    def _ask_model(self, calls: _Calls, stage: str, system: str, user: str, model, **attributes) -> StructuredResult:
        started = time.perf_counter()
        slept = 0.0
        if calls.raw and self.retry_interval:  # a second call for the same question: stay under the rate limit
            time.sleep(self.retry_interval)
            slept = time.perf_counter() - started
            calls.waited += slept
        result = ask_structured(self._client, system, user, model, retry_interval=self.retry_interval,
                                max_input_tokens=self.settings.max_input_tokens)
        calls.add(result, system, user)
        calls.span(stage, started, "ok" if result.status == STATUS_OK else "error", slept + result.retry_wait_seconds,
                   **_llm_attributes(self.model_name, result.usage, result.attempts, result.status), **attributes)
        return result

    def _request(self, question: str, verdict: Verdict, calls: _Calls, meta: dict):
        started = time.perf_counter()
        plan = rule_plan(question, self.index, verdict.default_applied)
        try:
            request = build_request(plan, self.index, self.table)
            calls.span("analyze.rules", started, type=plan.type)
            return plan, request
        except Unresolved as exc:
            meta["unresolved"] = exc.what
            calls.span("analyze.rules", started, "error", type=plan.type, unresolved=exc.what)
        user = analyze_fill.user_prompt(plan.question, self.index, replace(plan, unresolved=plan.unresolved or meta["unresolved"]))
        result = self._ask_model(calls, "analyze.llm_fill", analyze_fill.SYSTEM_PROMPT, user, analyze_fill.AnalyzeFill)
        meta["route"] = "llm_fill"
        if result.status != STATUS_OK:
            meta["fill_status"] = result.status
            return plan, None
        filled, ignored = analyze_fill.apply(plan, result.value, self.index)
        meta["fill"] = result.value.model_dump(exclude_defaults=True) | ({"ignored": ignored} if ignored else {})
        tries = [filled]
        if result.value.type and result.value.type != filled.type:
            tries.append(replace(filled, type=result.value.type))
        for candidate in tries:
            try:
                return candidate, build_request(candidate, self.index, self.table)
            except Unresolved as exc:
                meta["unresolved_after_fill"] = exc.what
        return filled, None

    def _compose(self, question: str, computed, note: str | None, calls: _Calls, meta: dict) -> list:
        user = compose.user_prompt(question, computed, note)
        problems: list[str] = []
        for attempt in range(2):
            if problems:
                user = compose.retry_prompt(user, problems)
            result = self._ask_model(calls, "compose", compose.SYSTEM_PROMPT, user, compose.ComposeOutput, attempt=attempt)
            if result.status != STATUS_OK:
                meta["compose_status"] = result.status
                break
            started = time.perf_counter()
            parts = compose.to_parts(result.value.answer, computed)
            problems = verify(parts, computed)
            calls.span("verify", started, "error" if problems else "ok", attempt=attempt, problems=len(problems))
            meta.setdefault("verify_problems", []).append(problems)
            if not problems:
                meta["compose"] = "model" if attempt == 0 else "model_retry"
                return parts
        started = time.perf_counter()
        parts = template_parts(meta["type"], computed)
        meta["compose"] = "template"
        calls.span("template", started)
        return parts

    # ------------------------------------------------------------ entry point

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        started = time.perf_counter()
        calls = _Calls(t0=started)
        verdict = judge(question, self.table, self.index)
        calls.span("judge", started, outcome=verdict.outcome, rule=verdict.rule, final=verdict.final)
        meta: dict[str, Any] = {**self.decision_metadata, "route": "rules"}
        if verdict.final:
            return self._done(self._verdict_answer(question, verdict), verdict, meta | {"route": "rules_final"}, calls, started)
        plan, request = self._request(question, verdict, calls, meta)
        if request is None:  # neither the rules nor the model placed the question: the compact-v2 chatbot answers
            fallback_started = time.perf_counter()
            answer = self.fallback.ask(question, golden_scope)
            # ponytail: the chatbot's own steps stay one span; split it if the fallback route gets tuned
            calls.span("fallback_chatbot", fallback_started, "ok" if answer.status == STATUS_OK else "error",
                       answer.retry_wait_seconds, **_llm_attributes(self.model_name, answer.usage, answer.attempts, answer.status))
            usage = dict(calls.usage)
            for key, value in answer.usage.items():  # keep the analyze.llm_fill tokens spent before the fallback
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            return replace(
                answer, attempts=answer.attempts + calls.attempts, raw_responses=calls.raw + answer.raw_responses, usage=usage,
                format_errors=calls.errors + answer.format_errors, retry_wait_seconds=answer.retry_wait_seconds + calls.waited,
                latency_seconds=max(0.0, time.perf_counter() - started - answer.retry_wait_seconds - calls.waited),
                decision_metadata={**answer.decision_metadata, **meta, "route": "fallback_chatbot", "fallback_prompt": FALLBACK_PROMPT,
                                   "spans": calls.spans},
            )
        source = to_source(self.table.meta, org_id=self.org_id)
        query = to_query(plan, self.index, request)
        meta["type"] = plan.type
        compute_started = time.perf_counter()
        try:
            computed = compute(self.table, request)
            calls.span("compute", compute_started, facts=len(computed.result.facts))
        except CalcRefused as exc:
            calls.span("compute", compute_started, outcome="refused", reason=exc.reason)  # a refusal is an answer, not a failure
            stat = StatAnswer(outcome="refused", query=query, refusal=Refusal(reason=exc.reason, message=exc.message),
                              text=final_text(exc.message, source), source=source)
            return self._done(stat, verdict, meta | {"route": f"{meta['route']}+calc_refused"}, calls, started)
        parts = self._compose(question, computed, computed.note, calls, meta)
        meta["label_digits"] = label_digits(parts, computed)
        body = render_parts(parts, computed.result)
        stat = StatAnswer(
            outcome="answered", query=query, result=computed.result, parts=parts,
            text=final_text(body, source, verdict.default_applied, [computed.note]),
            default_applied=verdict.default_applied, source=source,
        )
        chart = chart_spec(plan.type, computed)
        meta["chart"] = chart is not None
        return replace(self._done(stat, verdict, meta, calls, started), chart=chart)

    def _verdict_answer(self, question: str, verdict: Verdict) -> StatAnswer:
        query = StatQuery(question=question)
        if verdict.outcome == "refused":
            r = verdict.refusal
            source = None if r.reason == "out_of_scope" else to_source(self.table.meta, org_id=self.org_id)
            return StatAnswer(outcome="refused", query=query, refusal=Refusal(reason=r.reason, message=r.message, available=r.available),
                              text=final_text(r.message, source), source=source)
        options = list(verdict.clarify_options)
        body = "\n".join([verdict.clarify_question, *(f"- {o}" for o in options)])
        source = to_source(self.table.meta, org_id=self.org_id)
        return StatAnswer(outcome="clarify", query=query, clarify_options=options, text=final_text(body, source), source=source)

    def _done(self, stat: StatAnswer, verdict: Verdict, meta: dict, calls: _Calls, started: float) -> Answer:
        return Answer(
            text=stat.text, status=STATUS_OK, table_meta=self.table.meta, prompt_version=self.prompt_version,
            input_tokens_estimate=calls.prompt_chars, attempts=calls.attempts, usage=calls.usage,
            latency_seconds=max(0.0, time.perf_counter() - started - calls.waited), retry_wait_seconds=calls.waited,
            scope={"mode": "cells", "cells": len(stat.result.facts)}, format_errors=calls.errors, raw_responses=calls.raw,
            answer_format="json", decision_metadata=meta | {"spans": calls.spans}, judge=judge_record(verdict), stat_answer=stat,
        )
