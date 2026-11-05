"""C pipeline: numbers by code, sentences by the model (docs/plans/20261024-c-compute-pipeline/PLAN.md).

judge (rules) -> analyze (rules, the model fills what they leave open) -> compute -> compose (model, {f1} citations)
-> verify (one retry, then a code template) -> render. StatPipeline.ask returns the same Answer as Chatbot.ask, with
the finished StatAnswer in Answer.stat_answer. A question neither the rules nor the fill can place is planned by the
model as v3 blocks (analyze_plan, S5b; code still computes), and only when that fails too goes to the compact-v2
chatbot. A plan that names something several codes could be asks back (outcome clarify).
S6: the answer also records its caveats (shown as '참고:' lines), the facts shown beside it (context_fact_ids: the
other scenarios of a projection the question did not choose, a newer period), a table of targets by periods, and an
ask-back's options with their codes (clarify_choices).
T1-28 S3a/S3b: every step is a stage node (judge/ analyze/ compute/ compose/ chart/ render/ node.py, legacy/fallback.py:
step_<name>(In, StatResources) -> Out), called here in the same order. The model steps get a ModelAccess (llm=) and
return their calls as LLMCall records; the Answer and its spans are made from the records (graph/legacy.py,
graph/spans.py).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import kosis_chat.shared.patterns  # noqa: F401  first: puts src/ on the path when kosis_ai is not installed (S6 removes)
from kosis_chat.analyze.node import step_plan_fill_llm, step_plan_rules, step_plan_write_llm
from kosis_chat.analyze.rules import build_plan
from kosis_chat.analyze.schemas import PlanFillIn, PlanRulesIn, PlanWriteIn
from kosis_chat.chart.node import step_chart
from kosis_chat.chart.schemas import ChartIn
from kosis_chat.chatbot import Answer, Chatbot
from kosis_chat.compose.node import step_caveats, step_compose_llm, step_template, step_verify
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS, CaveatsIn, ComposeLLMIn, TemplateIn, VerifyIn
from kosis_chat.compose.verify import label_digits
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.resolve import NeedsClarify
from kosis_chat.compute.schemas import ComputeIn
from kosis_chat.config import Settings, load_settings, settings_from_llm
from kosis_chat.graph.legacy import add_usage, fallback_answer, to_answer  # noqa: F401  add_usage: old name
from kosis_chat.graph.spans import llm_attributes, step_span
from kosis_chat.judge.node import step_judge
from kosis_chat.judge.schemas import JudgeIn, Verdict
from kosis_chat.legacy.fallback import FALLBACK_PROMPT, FallbackIn, step_fallback_chatbot
from kosis_chat.llm import compose  # the old path kept: pipeline.VERSION is kosis_chat.llm.compose.VERSION (Phase 2)
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import LLMClient
from kosis_chat.llm.providers import build_client
from kosis_chat.prepare.schemas import TableInput
from kosis_chat.render.node import causal_note, step_answer_final, step_render
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn
from kosis_chat.render.source import to_source
from kosis_chat.schemas.resources import LLMConfig, StatResources, load_resources
from kosis_chat.schemas.state import Record, called_before

from kosis_ai.schemas import StatAnswer  # isort: skip

VERSION = compose.VERSION
# Kept by name for callers of the old pipeline module (T1-28 S3b): FALLBACK_PROMPT is legacy/fallback.py's,
# add_usage graph/legacy.py's, _llm_attributes graph/spans.py llm_attributes.
_llm_attributes = llm_attributes
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

    def _model_access(self) -> ModelAccess:
        """How this question's model steps reach the model: the client is read at the first call (built then)."""
        return ModelAccess(client=lambda: self.client, model_name=lambda: self.model_name,
                           retry_interval=self.retry_interval, max_input_tokens=self.settings.max_input_tokens)

    def _compose(self, question: str, computed, cautions: list[str], context_ids: list[str], records: list[Record],
                 meta: dict, llm: ModelAccess) -> tuple[list, bool]:
        """(parts, verified): the model's sentence once it passes verification, else a code template.

        At most COMPOSE_MAX_ATTEMPTS model attempts (step_compose_llm, each checked by step_verify), then the template.
        verified is what StatAnswer.numbers_verified records: the parts the answer shows passed verify()."""
        res = self.resources
        prompt: str | None = None
        problems: list[str] = []
        for attempt in range(COMPOSE_MAX_ATTEMPTS):
            written = step_compose_llm(ComposeLLMIn(question=question, computed=computed, cautions=cautions,
                                                    context_ids=context_ids, attempt=attempt, problems=problems,
                                                    prompt=prompt, called=called_before(records)), res, llm=llm)
            records.append(written.call)
            meta.update(written.meta)
            prompt = written.prompt
            if written.route == "template":
                break
            started = time.perf_counter()
            checked = step_verify(VerifyIn(parts=written.parts, computed=computed, attempt=attempt), res)
            problems = checked.problems
            records.append(step_span("verify", started, "error" if problems else "ok", attempt=attempt,
                                     problems=len(problems)))
            meta.setdefault("verify_problems", []).append(problems)
            if checked.route == "ok":
                meta["compose"] = "model" if attempt == 0 else "model_retry"
                return written.parts, True
            if checked.route == "template":
                break
        started = time.perf_counter()
        meta["compose"] = "template"
        # a template cites every main fact by construction; its problems are recorded, not assumed
        templated = step_template(TemplateIn(type=meta["type"], computed=computed), res)
        if templated.problems:
            meta["template_problems"] = templated.problems
        records.append(step_span("template", started, "error" if templated.problems else "ok"))
        return templated.parts, templated.verified

    # ------------------------------------------------------------ entry point

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        """The steps in order (T1-28 S3a/S3b: each is a stage node's step_* function; a manual router)."""
        res = self.resources
        llm = self._model_access()
        started = time.perf_counter()
        records: list[Record] = []
        judged = step_judge(JudgeIn(question=question), res)
        verdict = judged.verdict
        records.append(step_span("judge", started, outcome=verdict.outcome, rule=verdict.rule, final=verdict.final))
        meta: dict[str, Any] = {**self.decision_metadata, "route": "rules"}
        if judged.route == "final":
            return self._done(self._verdict_answer(question, verdict), verdict, meta | {"route": "rules_final"}, records,
                              started)
        rules_started = time.perf_counter()
        rules = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res,
                                build=build_plan)  # this module's name: tests swap it
        rule = plan = rules.rule
        built = rules.built
        if rules.route == "built":
            records.append(step_span("analyze.rules", rules_started, type=plan.type))
        else:  # the model fills what the rules left open (analyze-fill-v2)
            meta["unresolved"] = rules.unresolved
            records.append(step_span("analyze.rules", rules_started, "error", type=plan.type, unresolved=rules.unresolved))
            filled = step_plan_fill_llm(PlanFillIn(rule=rule, unresolved=rules.unresolved, called=called_before(records)),
                                        res, llm=llm, build=build_plan)
            records += filled.calls
            meta.update(filled.meta)
            plan, built = filled.plan, filled.built
        computed = None
        if built is None:  # neither the rules nor the fill placed it: the model writes the plan
            written = step_plan_write_llm(PlanWriteIn(question=question, rule=rule, called=called_before(records)), res,
                                          llm=llm)
            records += written.calls
            meta.update(written.meta)
            if written.route == "clarify":
                return self._done(self._clarify_answer(question, written.clarify), verdict,
                                  meta | {"route": "llm_plan+clarify"}, records, started)
            if written.route == "placed":
                plan, built, computed = written.placed.query_plan, written.placed.built, written.placed.computed
        if built is None:  # no plan from the rules or the model: the compact-v2 chatbot answers
            fell = step_fallback_chatbot(FallbackIn(question=question, golden_scope=golden_scope), res,
                                         chatbot=lambda: self.fallback, llm=llm)
            records.append(fell.call)
            return fallback_answer(fell.answer, meta, records, started, fallback_prompt=FALLBACK_PROMPT)
        source = to_source(self.table.meta, org_id=self.org_id)
        meta["type"] = plan.type
        meta["pattern"] = built.pattern
        meta["plan"] = built.plan.model_dump(mode="json", exclude_none=True)  # the plan code computed (S9)
        compute_started = time.perf_counter()
        done = step_compute(ComputeIn(plan=plan, built=built, default_applied=verdict.default_applied, computed=computed), res)
        query, built = done.query, done.built
        if done.route == "refused":  # a refusal is an answer, not a failure
            records.append(step_span("compute", compute_started, outcome="refused", reason=done.refusal.reason))
            stat = step_answer_final(AnswerFinalIn(kind="refused", question=question, refusal=done.refusal, query=query,
                                                   source=source), res).stat
            return self._done(stat, verdict, meta | {"route": f"{meta['route']}+calc_refused"}, records, started)
        computed = done.computed
        records.append(step_span("compute", compute_started, facts=len(computed.result.facts)))
        noted = step_caveats(CaveatsIn(computed=computed, built=built, alternatives=done.alternatives,
                                       default_applied=verdict.default_applied), res)
        caveats, shown, context_ids = noted.caveats, noted.shown, noted.context_ids
        parts, verified = self._compose(question, computed, shown, context_ids, records, meta, llm)
        meta["label_digits"] = label_digits(parts, computed)
        meta["numbers_verified"] = verified
        chart = step_chart(ChartIn(type=plan.type, computed=computed), res).chart
        stat = step_render(RenderIn(question=question, query=query, computed=computed, parts=parts, verified=verified,
                                    source=source, default_applied=verdict.default_applied, caveats=caveats, shown=shown,
                                    context_ids=context_ids, chart=chart), res).stat
        meta["chart"] = chart.kind if chart else None
        meta["caveats"] = [c.kind for c in caveats]
        return replace(self._done(stat, verdict, meta, records, started), chart=chart.spec if chart else None)

    def _verdict_answer(self, question: str, verdict: Verdict) -> StatAnswer:
        return step_answer_final(AnswerFinalIn(kind="verdict", question=question, verdict=verdict), self.resources).stat

    def _clarify_answer(self, question: str, asked: NeedsClarify) -> StatAnswer:
        return step_answer_final(AnswerFinalIn(kind="clarify", question=question, clarify=asked), self.resources).stat

    def _done(self, stat: StatAnswer, verdict: Verdict, meta: dict, records: list[Record], started: float) -> Answer:
        return to_answer(stat, verdict, meta, records, started, table_meta=self.table.meta,
                         prompt_version=self.prompt_version)
