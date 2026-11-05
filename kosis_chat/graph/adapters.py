"""The graph's node functions (T1-28 S4): state -> XxxIn -> step -> XxxOut -> dict of the state fields it changes.

Each adapter reads what its step needs from the StatState, calls the step from the graph's node table (graph/build.py
NODE_TABLE, or a node_overrides entry, E4) and returns a dict: LangGraph applies it to the state (records and meta
through their reducers). They do what StatPipeline.ask did around each step before S4, in the same order: the same
spans (graph/spans.py step_span; a model step's LLMCall records as it returned them), the same decision_metadata
entries in the order the router wrote them, the same In values. The per-question things that are not resources (the
model access, the fallback chatbot, the block builder StatPipeline passes) come in the run's context (StatContext).

Only the main path writes records, meta and step_route. The chart branch (chart_node) runs beside compose_llm and
writes only `chart_out`, so the records stay in the order the main path made them (the old _Calls order) however the
branches finish, and nothing two threads share is changed (chart_spec only reads the computed facts).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kosis_chat.analyze.rules import build_plan
from kosis_chat.analyze.schemas import PlanFillIn, PlanRulesIn, PlanWriteIn
from kosis_chat.chart.schemas import ChartIn
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS, CaveatsIn, ComposeLLMIn, TemplateIn, VerifyIn
from kosis_chat.compose.verify import label_digits
from kosis_chat.compute.schemas import ComputeIn
from kosis_chat.graph.spans import step_span
from kosis_chat.judge.schemas import JudgeIn
from kosis_chat.legacy.fallback import FallbackIn
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn
from kosis_chat.render.source import to_source
from kosis_chat.schemas.resources import StatResources
from kosis_chat.schemas.state import StatState, called_before

Step = Callable[..., Any]


@dataclass(frozen=True)
class StatContext:
    """The run's context (LangGraph context_schema): what a question's model steps reach beside the resources.

    llm: the owner's model access (its client read at the first model call: LLMKeyMissing then, S2c); chatbot: the
    fallback Chatbot, built on first use; build: the block builder plan_rules and plan_fill_llm use (StatPipeline
    passes its module's build_plan at every ask, which tests replace)."""

    llm: ModelAccess
    chatbot: Callable[[], Any] = field(default=lambda: None)
    build: Callable[..., Any] = build_plan


def timed(step: Step, *args, **kwargs) -> tuple[Any, float]:
    """(the step's output, time.perf_counter() before it ran): the start of the span recorded for it."""
    started = time.perf_counter()
    return step(*args, **kwargs), started


def _ctx(runtime) -> StatContext:
    return runtime.context


# ---------------------------------------------------------------- judge, analyze


def judge_node(step: Step, res: StatResources) -> Callable:
    def judge(state: StatState) -> dict:
        judged = step(JudgeIn(question=state.question), res)
        verdict = judged.verdict
        # the judge's span starts with the question (StatPipeline.ask's start), as before S4
        span = step_span("judge", state.started, outcome=verdict.outcome, rule=verdict.rule, final=verdict.final)
        return {"verdict": verdict, "records": [span], "step_route": judged.route}
    return judge


def plan_rules_node(step: Step, res: StatResources) -> Callable:
    def plan_rules(state: StatState, runtime) -> dict:
        rules, started = timed(step, PlanRulesIn(question=state.question, default_applied=state.verdict.default_applied),
                               res, build=_ctx(runtime).build)
        rule = rules.rule
        out = {"rule": rule, "plan": rule, "built": rules.built, "step_route": rules.route}
        if rules.route == "built":
            return out | {"records": [step_span("analyze.rules", started, type=rule.type)]}
        return out | {"unresolved": rules.unresolved, "meta": {"unresolved": rules.unresolved},
                      "records": [step_span("analyze.rules", started, "error", type=rule.type, unresolved=rules.unresolved)]}
    return plan_rules


def plan_fill_llm_node(step: Step, res: StatResources) -> Callable:
    def plan_fill_llm(state: StatState, runtime) -> dict:
        ctx = _ctx(runtime)
        filled = step(PlanFillIn(rule=state.rule, unresolved=state.unresolved, called=called_before(state.records)), res,
                      llm=ctx.llm, build=ctx.build)
        return {"records": list(filled.calls), "meta": filled.meta, "plan": filled.plan, "built": filled.built,
                "step_route": filled.route}
    return plan_fill_llm


def plan_write_llm_node(step: Step, res: StatResources) -> Callable:
    def plan_write_llm(state: StatState, runtime) -> dict:
        written = step(PlanWriteIn(question=state.question, rule=state.rule, called=called_before(state.records)), res,
                       llm=_ctx(runtime).llm)
        out = {"records": list(written.calls), "meta": written.meta, "step_route": written.route}
        if written.route == "clarify":
            return out | {"clarify": written.clarify, "meta": written.meta | {"route": "llm_plan+clarify"}}
        if written.route == "placed":  # computed by code already (D12 a): compute only adds the query and checks
            placed = written.placed
            out |= {"plan": placed.query_plan, "built": placed.built, "computed": placed.computed}
        return out
    return plan_write_llm


def fallback_node(step: Step, res: StatResources) -> Callable:
    def fallback_chatbot(state: StatState, runtime) -> dict:
        ctx = _ctx(runtime)
        fell = step(FallbackIn(question=state.question, golden_scope=state.golden_scope), res, chatbot=ctx.chatbot,
                    llm=ctx.llm)
        return {"records": [fell.call], "fallback": fell.answer}
    return fallback_chatbot


# ---------------------------------------------------------------- compute, caveats


def compute_node(step: Step, res: StatResources) -> Callable:
    def compute(state: StatState) -> dict:
        source = to_source(res.table.meta, org_id=res.org_id)
        built = state.built
        meta = {"type": state.plan.type, "pattern": built.pattern,
                "plan": built.plan.model_dump(mode="json", exclude_none=True)}  # the plan code computed (S9)
        done, started = timed(step, ComputeIn(plan=state.plan, built=built, default_applied=state.verdict.default_applied,
                                              computed=state.computed), res)
        out = {"source": source, "query": done.query, "built": done.built, "step_route": done.route}
        if done.route == "refused":  # a refusal is an answer, not a failure
            span = step_span("compute", started, outcome="refused", reason=done.refusal.reason)
            route = {"route": f"{(state.meta | meta)['route']}+calc_refused"}
            return out | {"refusal": done.refusal, "records": [span], "meta": meta | route}
        span = step_span("compute", started, facts=len(done.computed.result.facts))
        return out | {"computed": done.computed, "alternatives": done.alternatives, "records": [span], "meta": meta}
    return compute


def caveats_node(step: Step, res: StatResources) -> Callable:
    def caveats(state: StatState) -> dict:
        noted = step(CaveatsIn(computed=state.computed, built=state.built, alternatives=state.alternatives,
                               default_applied=state.verdict.default_applied), res)
        return {"caveats_out": noted.caveats, "shown": noted.shown, "context_ids": noted.context_ids}
    return caveats


# ---------------------------------------------------------------- compose <-> verify, template (the one loop, D3)


def compose_llm_node(step: Step, res: StatResources) -> Callable:
    def compose_llm(state: StatState, runtime) -> dict:
        written = step(ComposeLLMIn(question=state.question, computed=state.computed, cautions=state.shown,
                                    context_ids=state.context_ids, attempt=state.compose_attempt,
                                    problems=state.compose_problems, prompt=state.compose_prompt,
                                    called=called_before(state.records)), res, llm=_ctx(runtime).llm)
        return {"records": [written.call], "meta": written.meta, "compose_prompt": written.prompt, "parts": written.parts,
                "step_route": written.route}
    return compose_llm


def verify_node(step: Step, res: StatResources) -> Callable:
    def verify(state: StatState) -> dict:
        attempt = state.compose_attempt
        checked, started = timed(step, VerifyIn(parts=state.parts, computed=state.computed, attempt=attempt), res)
        problems = checked.problems
        span = step_span("verify", started, "error" if problems else "ok", attempt=attempt, problems=len(problems))
        meta: dict[str, Any] = {"verify_problems": [*state.meta.get("verify_problems", []), problems]}
        out = {"records": [span], "meta": meta}
        if checked.route == "ok":
            meta["compose"] = "model" if attempt == 0 else "model_retry"
            return out | {"verified": True, "step_route": "ok"}
        if checked.route == "retry" and attempt + 1 < COMPOSE_MAX_ATTEMPTS:  # the loop's cap, whatever verify says
            return out | {"compose_attempt": attempt + 1, "compose_problems": problems, "step_route": "retry"}
        return out | {"step_route": "template"}
    return verify


def template_node(step: Step, res: StatResources) -> Callable:
    def template(state: StatState) -> dict:
        meta = {"compose": "template"}
        # a template cites every main fact by construction; its problems are recorded, not assumed
        templated, started = timed(step, TemplateIn(type=state.meta["type"], computed=state.computed), res)
        if templated.problems:
            meta["template_problems"] = templated.problems
        span = step_span("template", started, "error" if templated.problems else "ok")
        return {"records": [span], "meta": meta, "parts": templated.parts, "verified": templated.verified}
    return template


def compose_done_node(res: StatResources) -> Callable:
    """The end of the compose path (verify ok or template): what the answer's parts are, before the join with chart."""
    def compose_done(state: StatState) -> dict:
        return {"meta": {"label_digits": label_digits(state.parts, state.computed), "numbers_verified": state.verified}}
    return compose_done


# ---------------------------------------------------------------- chart (beside compose), render, answer_final


def chart_node(step: Step, res: StatResources) -> Callable:
    """The chart branch: runs beside the compose path (it needs only the computed facts) and writes only `chart_out`."""
    def chart(state: StatState) -> dict:
        return {"chart_out": step(ChartIn(type=state.plan.type, computed=state.computed), res).chart}
    return chart


def render_node(step: Step, res: StatResources) -> Callable:
    def render(state: StatState) -> dict:
        chart = state.chart_out
        stat = step(RenderIn(question=state.question, query=state.query, computed=state.computed, parts=state.parts,
                             verified=state.verified, source=state.source, default_applied=state.verdict.default_applied,
                             caveats=state.caveats_out, shown=state.shown, context_ids=state.context_ids, chart=chart), res).stat
        return {"stat": stat, "meta": {"chart": chart.kind if chart else None, "caveats": [c.kind for c in state.caveats_out]}}
    return render


def answer_final_node(step: Step, res: StatResources) -> Callable:
    """The answers without facts: the judge's refusal or ask-back, the plan's ask-back, the calculation's refusal."""
    def answer_final(state: StatState) -> dict:
        if state.step_route == "refused":
            inp = AnswerFinalIn(kind="refused", question=state.question, refusal=state.refusal, query=state.query,
                                source=state.source)
            return {"stat": step(inp, res).stat}
        if state.step_route == "clarify":
            return {"stat": step(AnswerFinalIn(kind="clarify", question=state.question, clarify=state.clarify), res).stat}
        stat = step(AnswerFinalIn(kind="verdict", question=state.question, verdict=state.verdict), res).stat
        return {"stat": stat, "meta": {"route": "rules_final"}}
    return answer_final
