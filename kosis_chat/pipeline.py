"""C pipeline: numbers by code, sentences by the model (docs/plans/20261024-c-compute-pipeline/PLAN.md).

judge (rules) -> analyze (rules, the model fills what they leave open) -> compute -> compose (model, {f1} citations)
-> verify (one retry, then a code template) -> render. StatPipeline.ask returns the same Answer as Chatbot.ask, with
the finished StatAnswer in Answer.stat_answer. A question neither the rules nor the fill can place is planned by the
model as v3 blocks (analyze_plan, S5b; code still computes), and only when that fails too goes to the compact-v2
chatbot. A plan that names something several codes could be asks back (outcome clarify).
S6: the answer also records its caveats (shown as '참고:' lines), the facts shown beside it (context_fact_ids: the
other scenarios of a projection the question did not choose, a newer period), a table of targets by periods, and an
ask-back's options with their codes (clarify_choices).
T1-28 S3a: the pure steps are the stage nodes (judge/ analyze/ compute/ compose/ chart/ render/ node.py: step_<name>(In,
StatResources) -> Out), called here in the same order; the model steps (fill, plan, event lookup, compose, fallback)
stay in this class until S3b.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

import kosis_chat.shared.patterns  # noqa: F401  first: puts src/ on the path when kosis_ai is not installed (S6 removes)
from kosis_chat.analyze.node import step_plan_rules
from kosis_chat.analyze.rules import Plan, Unresolved, build_plan
from kosis_chat.analyze.schemas import PlanRulesIn, PlanRulesOut
from kosis_chat.chart.node import step_chart
from kosis_chat.chart.schemas import ChartIn
from kosis_chat.chatbot import Answer, Chatbot, judge_record
from kosis_chat.compose.node import step_caveats, step_template, step_verify
from kosis_chat.compose.schemas import CaveatsIn, TemplateIn, VerifyIn
from kosis_chat.compose.verify import label_digits
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.resolve import NeedsClarify
from kosis_chat.compute.schemas import ComputeIn
from kosis_chat.config import Settings, load_settings, settings_from_llm
from kosis_chat.judge.node import step_judge
from kosis_chat.judge.schemas import JudgeIn, Verdict
from kosis_chat.llm import analyze_fill, analyze_plan, compose, event_lookup
from kosis_chat.llm.model import STATUS_OK, LLMClient
from kosis_chat.llm.providers import build_client
from kosis_chat.llm.structured import StructuredResult, ask_structured
from kosis_chat.prepare.schemas import TableInput
from kosis_chat.render.node import causal_note, step_answer_final, step_render
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn
from kosis_chat.render.source import to_source
from kosis_chat.schemas.resources import LLMConfig, StatResources, load_resources

from kosis_ai.schemas import PlanV3, StatAnswer  # isort: skip

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
    spans: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def add(self, result: StructuredResult, system: str, user: str) -> None:
        self.attempts += result.attempts
        add_usage(self.usage, result.usage)
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


_causal_note = causal_note  # T1-28 S3a: moved to render/node.py (the same function)


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
        llm: LLMConfig | None = None,
    ) -> None:
        # llm (T1-28 S2c): the injected model; a Settings alone is the Phase 2 shim (its key from the shell env).
        self.settings = settings or (settings_from_llm(llm) if llm is not None else load_settings())
        self.llm = llm
        # Built on first use by an LLM step (LLMKeyMissing then, not here): rule-only answers need no key.
        self._client: LLMClient | None = client
        # The cleaned table, its index and lookups (T1-28 S2): the index is the one compute's checks use too.
        self.resources: StatResources = load_resources(table, org_id)
        self.table = self.resources.table
        self.index = self.resources.index
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
        if self._client is None:
            self._client = build_client(self.llm if self.llm is not None else self.settings)
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
                self.table, client=self.client, settings=replace(self.settings, prompt_version=FALLBACK_PROMPT),
                prompt_version=FALLBACK_PROMPT, scope_mode=self.fallback_scope, answer_format="json",
                retry_interval=self.retry_interval, judge_mode="rules",
            )
        return self._fallback

    # ------------------------------------------------------------ steps

    @property
    def model_name(self) -> str:
        return self.settings.model_name if self.client.name == "hcx" else self.client.name

    def _ask_model(self, calls: _Calls, stage: str, system: str, user: str, model, **attributes) -> StructuredResult:
        started = time.perf_counter()
        slept = 0.0
        if calls.raw and self.retry_interval:  # a second call for the same question: stay under the rate limit
            time.sleep(self.retry_interval)
            slept = time.perf_counter() - started
            calls.waited += slept
        result = ask_structured(self.client, system, user, model, retry_interval=self.retry_interval,
                                max_input_tokens=self.settings.max_input_tokens)
        calls.add(result, system, user)
        calls.span(stage, started, "ok" if result.status == STATUS_OK else "error", slept + result.retry_wait_seconds,
                   **_llm_attributes(self.model_name, result.usage, result.attempts, result.status), **attributes)
        return result

    def _place(self, rules: PlanRulesOut, calls: _Calls, meta: dict, started: float):
        """The rules' plan when it was built (step_plan_rules), else the model's fill of what it left open."""
        plan = rules.rule
        if rules.route == "built":
            calls.span("analyze.rules", started, type=plan.type)
            return plan, rules.built
        meta["unresolved"] = rules.unresolved
        calls.span("analyze.rules", started, "error", type=plan.type, unresolved=rules.unresolved)
        plan = plan.model_copy(update={"unresolved": plan.unresolved or meta["unresolved"]})
        user = analyze_fill.user_prompt(plan.question, self.index, plan)
        meta["route"] = "llm_fill"
        filled, problems, kind = self._fill(plan, user, calls, meta, "fill")
        if filled is None:
            return plan, None
        if (placed := self._build(filled, kind, meta)) is not None:
            return placed
        if problems:  # the model's answer had a correctable fault: say it back once (analyze-fill-v2)
            again, _, kind = self._fill(plan, analyze_fill.retry_prompt(user, problems), calls, meta, "fill_retry")
            if again is not None:
                if (placed := self._build(again, kind, meta)) is not None:
                    return placed
                filled = again
        return filled, None

    def _fill(self, plan: Plan, user: str, calls: _Calls, meta: dict, key: str):
        """(plan filled by the model, problems to say back, the type the model named), or (None, [], None)."""
        stage = "analyze.llm_fill" if key == "fill" else "analyze.llm_fill_retry"
        result = self._ask_model(calls, stage, analyze_fill.SYSTEM_PROMPT, user, analyze_fill.AnalyzeFill)
        if result.status != STATUS_OK:
            meta[f"{key}_status"] = result.status
            return None, [], None
        filled, ignored, problems = analyze_fill.apply(plan, result.value, self.index)
        meta[key] = result.value.model_dump(exclude_defaults=True) | ({"ignored": ignored} if ignored else {})
        return filled, problems, result.value.type

    def _build(self, filled: Plan, kind, meta: dict):
        tries = [filled]
        if kind and kind != filled.type and not filled.pattern:  # a period pattern fixes its own type (value/trend/share)
            tries.append(filled.model_copy(update={"type": kind}))
        for candidate in tries:
            try:
                return candidate, build_plan(candidate, self.index, self.table)
            except Unresolved as exc:
                meta["unresolved_after_fill"] = exc.what
        return None

    def _written(self, question: str, rule: Plan, calls: _Calls, meta: dict) -> analyze_plan.Placed | NeedsClarify | None:
        """The model's own plan (analyze-plan-v1), said back once when it fails; None when it still fails."""
        lookup = self._lookup(calls, meta)
        user = analyze_plan.user_prompt(question, self.index, rule)
        meta["route"] = "llm_plan"
        clarify = None
        for key in ("plan", "plan_retry"):
            result = self._ask_model(calls, f"analyze.llm_{key}", analyze_plan.SYSTEM_PROMPT, user, PlanV3)
            if result.status != STATUS_OK:
                meta[f"{key}_status"] = result.status
                return None
            meta[key] = result.value.model_dump(exclude_defaults=True)
            placed = analyze_plan.place(question, result.value, self.table, self.index, lookup)
            if not placed.problems:
                return placed
            meta[f"{key}_problems"] = placed.problems
            clarify = placed.clarify
            user = analyze_plan.retry_prompt(user, placed.problems)
        # what several codes could be is asked back; an event nobody knows, or a plan that is still wrong, is not
        return clarify if clarify is not None and len(clarify.options) >= 2 else None

    def _lookup(self, calls: _Calls, meta: dict):
        """When an event the calendar lacks began, asked of the model once per question (event-lookup-v1)."""
        known: dict[str, str | None] = meta.setdefault("event_lookup", {})

        def lookup(name: str) -> str | None:
            if name not in known:
                result = self._ask_model(calls, "event_lookup", event_lookup.SYSTEM_PROMPT, event_lookup.user_prompt(name),
                                         event_lookup.EventStart, event=name)
                known[name] = result.value.start if result.status == STATUS_OK else None
            return known[name]

        return lookup

    def _compose(self, question: str, computed, cautions: list[str], context_ids: list[str], calls: _Calls,
                 meta: dict) -> tuple[list, bool]:
        """(parts, verified): the model's sentence once it passes verification, else a code template.

        verified is what StatAnswer.numbers_verified records: the parts the answer shows passed verify()."""
        user = compose.user_prompt(question, computed, cautions, context_ids)
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
            problems = step_verify(VerifyIn(parts=parts, computed=computed, attempt=attempt), self.resources).problems
            calls.span("verify", started, "error" if problems else "ok", attempt=attempt, problems=len(problems))
            meta.setdefault("verify_problems", []).append(problems)
            if not problems:
                meta["compose"] = "model" if attempt == 0 else "model_retry"
                return parts, True
        started = time.perf_counter()
        meta["compose"] = "template"
        # a template cites every main fact by construction; its problems are recorded, not assumed
        written = step_template(TemplateIn(type=meta["type"], computed=computed), self.resources)
        if written.problems:
            meta["template_problems"] = written.problems
        calls.span("template", started, "error" if written.problems else "ok")
        return written.parts, written.verified

    # ------------------------------------------------------------ entry point

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        """The steps in order (T1-28 S3a: the pure ones are the stage nodes' step_* functions; a manual router)."""
        res = self.resources
        started = time.perf_counter()
        calls = _Calls(t0=started)
        judged = step_judge(JudgeIn(question=question), res)
        verdict = judged.verdict
        calls.span("judge", started, outcome=verdict.outcome, rule=verdict.rule, final=verdict.final)
        meta: dict[str, Any] = {**self.decision_metadata, "route": "rules"}
        if judged.route == "final":
            return self._done(self._verdict_answer(question, verdict), verdict, meta | {"route": "rules_final"}, calls, started)
        rules_started = time.perf_counter()
        rules = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res,
                                build=build_plan)  # this module's name: tests swap it
        rule = rules.rule
        plan, built = self._place(rules, calls, meta, rules_started)
        computed = None
        if built is None:  # neither the rules nor the fill placed it: the model writes the plan
            written = self._written(question, rule, calls, meta)
            if isinstance(written, NeedsClarify):
                return self._done(self._clarify_answer(question, written), verdict, meta | {"route": "llm_plan+clarify"},
                                  calls, started)
            if written is not None:
                plan, built, computed = written.query_plan, written.built, written.computed
        if built is None:  # no plan from the rules or the model: the compact-v2 chatbot answers
            fallback_started = time.perf_counter()
            answer = self.fallback.ask(question, golden_scope)
            # ponytail: the chatbot's own steps stay one span; split it if the fallback route gets tuned
            calls.span("fallback_chatbot", fallback_started, "ok" if answer.status == STATUS_OK else "error",
                       answer.retry_wait_seconds, **_llm_attributes(self.model_name, answer.usage, answer.attempts, answer.status))
            # The chatbot's own record wins (its model wrote the numbers; its outcome_origin says whether a rule or
            # the model decided); the analysis calls made before it are added, not lost.
            return replace(
                answer, attempts=answer.attempts + calls.attempts, raw_responses=calls.raw + answer.raw_responses,
                format_errors=calls.errors + answer.format_errors, retry_wait_seconds=answer.retry_wait_seconds + calls.waited,
                latency_seconds=max(0.0, time.perf_counter() - started - answer.retry_wait_seconds - calls.waited),
                usage=add_usage(dict(calls.usage), answer.usage),
                input_tokens_estimate=calls.prompt_chars + answer.input_tokens_estimate,
                decision_metadata={**meta, **answer.decision_metadata, "pipeline": "stat", "route": "fallback_chatbot",
                                   "fallback_prompt": FALLBACK_PROMPT, "numbers_origin": "model", "numbers_verified": False,
                                   "spans": calls.spans},
            )
        source = to_source(self.table.meta, org_id=self.org_id)
        meta["type"] = plan.type
        meta["pattern"] = built.pattern
        meta["plan"] = built.plan.model_dump(mode="json", exclude_none=True)  # the plan code computed (S9)
        compute_started = time.perf_counter()
        done = step_compute(ComputeIn(plan=plan, built=built, default_applied=verdict.default_applied, computed=computed), res)
        query, built = done.query, done.built
        if done.route == "refused":  # a refusal is an answer, not a failure
            calls.span("compute", compute_started, outcome="refused", reason=done.refusal.reason)
            stat = step_answer_final(AnswerFinalIn(kind="refused", question=question, refusal=done.refusal, query=query,
                                                   source=source), res).stat
            return self._done(stat, verdict, meta | {"route": f"{meta['route']}+calc_refused"}, calls, started)
        computed = done.computed
        calls.span("compute", compute_started, facts=len(computed.result.facts))
        noted = step_caveats(CaveatsIn(computed=computed, built=built, alternatives=done.alternatives,
                                       default_applied=verdict.default_applied), res)
        caveats, shown, context_ids = noted.caveats, noted.shown, noted.context_ids
        parts, verified = self._compose(question, computed, shown, context_ids, calls, meta)
        meta["label_digits"] = label_digits(parts, computed)
        meta["numbers_verified"] = verified
        chart = step_chart(ChartIn(type=plan.type, computed=computed), res).chart
        stat = step_render(RenderIn(question=question, query=query, computed=computed, parts=parts, verified=verified,
                                    source=source, default_applied=verdict.default_applied, caveats=caveats, shown=shown,
                                    context_ids=context_ids, chart=chart), res).stat
        meta["chart"] = chart.kind if chart else None
        meta["caveats"] = [c.kind for c in caveats]
        return replace(self._done(stat, verdict, meta, calls, started), chart=chart.spec if chart else None)

    def _verdict_answer(self, question: str, verdict: Verdict) -> StatAnswer:
        return step_answer_final(AnswerFinalIn(kind="verdict", question=question, verdict=verdict), self.resources).stat

    def _clarify_answer(self, question: str, asked: NeedsClarify) -> StatAnswer:
        return step_answer_final(AnswerFinalIn(kind="clarify", question=question, clarify=asked), self.resources).stat

    def _done(self, stat: StatAnswer, verdict: Verdict, meta: dict, calls: _Calls, started: float) -> Answer:
        return Answer(
            text=stat.text, status=STATUS_OK, table_meta=self.table.meta, prompt_version=self.prompt_version,
            input_tokens_estimate=calls.prompt_chars, attempts=calls.attempts, usage=calls.usage,
            latency_seconds=max(0.0, time.perf_counter() - started - calls.waited), retry_wait_seconds=calls.waited,
            scope={"mode": "cells", "cells": len(stat.result.facts)}, format_errors=calls.errors, raw_responses=calls.raw,
            answer_format="json", decision_metadata=meta | {"spans": calls.spans}, judge=judge_record(verdict), stat_answer=stat,
        )
