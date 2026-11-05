"""The pure stage steps of T1-28 S3a: step_<name>(In, StatResources) -> Out, on the made-up tables (tests/conftest.py).

Each step is called on its own (In -> Out, every route it can produce), and StatPipeline.ask, which now calls them,
keeps the decision_metadata keys and the span names it recorded before (plans/T1-28.md S3 Done).
"""

from __future__ import annotations

import inspect
import json
import typing

import pytest

from kosis_chat.analyze import blocks
from kosis_chat.analyze.node import step_plan_rules
from kosis_chat.analyze.rules import Unresolved
from kosis_chat.analyze.schemas import PlanRulesIn, PlanRulesOut
from kosis_chat.chart.node import step_chart
from kosis_chat.chart.schemas import ChartIn, ChartOut
from kosis_chat.compose.node import step_caveats, step_template, step_verify
from kosis_chat.compose.schemas import CaveatsIn, CaveatsOut, TemplateIn, TemplateOut, VerifyIn, VerifyOut
from kosis_chat.compose.template import template_parts
from kosis_chat.compute.engine import compute
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.resolve import NeedsClarify
from kosis_chat.compute.schemas import ComputeIn, ComputeOut
from kosis_chat.data.loader import load_table
from kosis_chat.judge.node import step_judge
from kosis_chat.judge.schemas import JudgeIn, JudgeOut, Verdict
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import StatPipeline
from kosis_chat.render.node import causal_note, step_answer_final, step_render
from kosis_chat.render.schemas import AnswerFinalIn, RenderIn, RenderOut
from kosis_chat.render.source import to_source
from kosis_chat.render.text import final_text, render_parts
from kosis_chat.schemas.resources import load_resources
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import FactPart, Refusal, TextPart  # isort: skip

STEPS = {
    step_judge: (JudgeIn, JudgeOut), step_plan_rules: (PlanRulesIn, PlanRulesOut), step_compute: (ComputeIn, ComputeOut),
    step_caveats: (CaveatsIn, CaveatsOut), step_verify: (VerifyIn, VerifyOut), step_template: (TemplateIn, TemplateOut),
    step_chart: (ChartIn, ChartOut), step_render: (RenderIn, RenderOut), step_answer_final: (AnswerFinalIn, RenderOut),
}
SEOUL_ONE = "2023년 서울특별시 1인 가구 수는?"


@pytest.fixture
def households(household_path):
    return load_resources(load_table(household_path), "101")


@pytest.fixture
def pop(pop_path):
    return load_resources(load_table(pop_path), "101")


def judged(res, question):
    return step_judge(JudgeIn(question=question), res)


def planned(res, question):
    verdict = judged(res, question).verdict
    return verdict, step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res)


def computed_of(res, question):
    verdict, rules = planned(res, question)
    done = step_compute(ComputeIn(plan=rules.rule, built=rules.built, default_applied=verdict.default_applied), res)
    return verdict, rules, done


# ---------------------------------------------------------------- the step contract


@pytest.mark.parametrize("step", list(STEPS), ids=lambda s: s.__name__)
def test_step_signature_and_docstring(step):
    params = list(inspect.signature(step).parameters.values())
    assert [p.name for p in params[:2]] == ["inp", "res"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[2:])  # only swappable helpers after (In, res)
    hints = typing.get_type_hints(step)
    assert (hints["inp"], hints["return"]) == STEPS[step]
    assert issubclass(hints["inp"], BoundaryModel) and issubclass(hints["return"], BoundaryModel)
    assert hints["res"].__name__ == "StatResources"
    lines = [line.strip() for line in inspect.getdoc(step).splitlines()]
    for head in ("In:", "Out:", "Side effects:", "Node:"):
        assert sum(line.startswith(head) for line in lines) == 1, f"{step.__name__}: {head}"


# ---------------------------------------------------------------- judge


def test_judge_routes(households):
    plan = judged(households, SEOUL_ONE)
    assert plan.route == "plan" and plan.verdict.outcome == "answered" and not plan.verdict.final
    refused = judged(households, "2030년 서울 1인 가구 수는?")
    assert refused.route == "final" and refused.verdict.outcome == "refused" and refused.verdict.refusal.reason == "no_period"
    asked = judged(households, "2023년 중구 1인 가구는?")
    assert asked.route == "final" and asked.verdict.outcome == "clarify" and len(asked.verdict.clarify_options) >= 2


# ---------------------------------------------------------------- analyze: plan_rules


def test_plan_rules_built(households):
    _, rules = planned(households, SEOUL_ONE)
    assert rules.route == "built" and rules.unresolved is None and rules.built.pattern == "value"
    assert rules.rule.question == SEOUL_ONE and rules.rule.type == "value"


def test_plan_rules_unresolved(pop):
    _, rules = planned(pop, "2023년 여자 인구는?")  # two items: the rules cannot pick one
    assert rules.route == "unresolved" and rules.built is None and rules.unresolved == ["항목"]


def test_plan_rules_build_can_be_swapped(households):
    def unplaced(plan, index, table):
        raise Unresolved(["셀 선택"])

    rules = step_plan_rules(PlanRulesIn(question=SEOUL_ONE), households, build=unplaced)
    assert rules.route == "unresolved" and rules.unresolved == ["셀 선택"] and rules.rule.type == "value"


# ---------------------------------------------------------------- compute


def test_compute_ok(households):
    _, rules, done = computed_of(households, SEOUL_ONE)
    assert done.route == "ok" and done.refusal is None and len(done.computed.result.facts) == 1
    assert done.query.question == SEOUL_ONE and done.built.plan is rules.built.plan and done.alternatives == []


def test_compute_refused(pop):
    _, rules = planned(pop, "2023년 여자 가상 인구는?")
    secret = blocks.value("FAKE_TBL_001|T1|1|2024")  # 'x' (비공개): no number to compute with
    done = step_compute(ComputeIn(plan=rules.rule, built=secret), pop)
    assert done.route == "refused" and done.computed is None
    assert isinstance(done.refusal, Refusal) and done.refusal.reason == "missing_value" and done.refusal.message


def test_compute_keeps_facts_already_computed(households):
    _, rules = planned(households, SEOUL_ONE)
    already = compute(households.table, rules.built)
    done = step_compute(ComputeIn(plan=rules.rule, built=rules.built, computed=already), households)
    assert done.route == "ok" and done.computed is already and done.alternatives == []


# ---------------------------------------------------------------- compose: caveats, verify, template


def test_caveats(households):
    verdict, rules, done = computed_of(households, "서울특별시 1인 가구 추이는?")  # the rules apply a period default
    assert verdict.default_applied
    noted = step_caveats(CaveatsIn(computed=done.computed, built=done.built, alternatives=done.alternatives,
                                   default_applied=verdict.default_applied), households)
    assert noted.shown == [c.text for c in noted.caveats if c.kind != "default_applied"]
    main = set(done.computed.main)
    assert all(i not in main for i in noted.context_ids)
    without = step_caveats(CaveatsIn(computed=done.computed, built=done.built), households)
    assert without.shown == [c.text for c in without.caveats]


def test_verify_routes(households):
    _, rules, done = computed_of(households, SEOUL_ONE)
    good = template_parts("value", done.computed)
    assert step_verify(VerifyIn(parts=good, computed=done.computed), households) == VerifyOut(problems=[], route="ok")
    typed = [TextPart(text="서울은 3102가구입니다.")]
    first = step_verify(VerifyIn(parts=typed, computed=done.computed, attempt=0), households)
    assert first.route == "retry" and first.problems
    last = step_verify(VerifyIn(parts=typed, computed=done.computed, attempt=1), households)
    assert last.route == "template" and last.problems == first.problems


def test_template(households):
    _, rules, done = computed_of(households, SEOUL_ONE)
    written = step_template(TemplateIn(type=rules.rule.type, computed=done.computed), households)
    assert written.verified and written.problems == [] and written.parts == template_parts("value", done.computed)
    assert any(isinstance(p, FactPart) and p.fact_id == done.computed.main[0] for p in written.parts)


# ---------------------------------------------------------------- chart


@pytest.mark.parametrize(("question", "kinds"), [
    (SEOUL_ONE, {"bar", "timeline"}),
    ("서울특별시 1인 가구 추이는?", {"line", "timeline", "multi_line", "grouped_bar", "bar"}),
    ("2023년 서울특별시와 부산광역시 1인 가구 비교", {"bar"}),
])
def test_chart(households, question, kinds):
    _, rules, done = computed_of(households, question)
    chart = step_chart(ChartIn(type=rules.rule.type, computed=done.computed), households).chart
    assert chart is not None and chart.kind in kinds
    assert set(chart.fact_ids) <= {f.id for f in done.computed.result.facts}


def test_chart_none_when_nothing_to_draw(households):
    _, rules, done = computed_of(households, SEOUL_ONE)
    empty = done.computed.model_copy(update={"main": []})  # no main fact: the builder finds nothing, no chart
    assert step_chart(ChartIn(type="value", computed=empty), households) == ChartOut(chart=None)


# ---------------------------------------------------------------- render


def test_render_answered(households):
    verdict, rules, done = computed_of(households, SEOUL_ONE)
    parts = template_parts("value", done.computed)
    source = to_source(households.table.meta, org_id="101")
    chart = step_chart(ChartIn(type="value", computed=done.computed), households).chart
    stat = step_render(RenderIn(question=SEOUL_ONE, query=done.query, computed=done.computed, parts=parts, verified=True,
                                source=source), households).stat
    assert stat.outcome == "answered" and stat.numbers_origin == "code" and stat.numbers_verified and chart is not None
    assert "chart" not in stat.model_dump()  # T1-28 S5: the chart is StatOutput.chart, not a StatAnswer field
    assert stat.text == final_text(render_parts(parts, done.computed.result), source, {}, [])
    causal = "2023년 서울특별시 1인 가구 수와 그렇게 된 원인은?"
    noted = step_render(RenderIn(question=causal, query=done.query, computed=done.computed, parts=parts, verified=False,
                                 source=source, shown=["메모"]), households).stat
    assert causal_note(causal) and noted.text.endswith(f"참고: 메모\n참고: {causal_note(causal)}\n\n출처: {source.citation()}")
    assert noted.numbers_verified is False


def test_answer_final_verdicts(households):
    refused = step_answer_final(AnswerFinalIn(kind="verdict", question="q", verdict=judged(households, "2030년 서울 1인 가구 수는?").verdict),
                                households).stat
    assert refused.outcome == "refused" and refused.source is not None and refused.numbers_origin == "none"
    assert refused.text.startswith(refused.refusal.message) and refused.text.endswith(refused.source.citation())
    out = Verdict(outcome="refused", rule="out_of_scope", refusal=Refusal(reason="out_of_scope", message="범위 밖"), final=True)
    outside = step_answer_final(AnswerFinalIn(kind="verdict", question="q", verdict=out), households).stat
    assert outside.source is None and outside.text == "범위 밖"
    asked = step_answer_final(AnswerFinalIn(kind="verdict", question="q", verdict=judged(households, "2023년 중구 1인 가구는?").verdict),
                              households).stat
    assert asked.outcome == "clarify" and len(asked.clarify_options) >= 2 and asked.clarify_choices == []


def test_answer_final_refused_and_clarify(households):
    source = to_source(households.table.meta, org_id="101")
    _, _, done = computed_of(households, SEOUL_ONE)
    refusal = Refusal(reason="missing_value", message="값이 없습니다.")
    refused = step_answer_final(AnswerFinalIn(kind="refused", question=SEOUL_ONE, refusal=refusal, query=done.query,
                                              source=source), households).stat
    assert refused.outcome == "refused" and refused.refusal is refusal and refused.query is done.query
    assert refused.text == final_text("값이 없습니다.", source)
    ask = NeedsClarify("행정구역별(읍면동)", [("11020", "서울특별시 > 중구"), ("21010", "부산광역시 > 중구")], "어느 중구인가요?")
    asked = step_answer_final(AnswerFinalIn(kind="clarify", question="중구는?", clarify=ask), households).stat
    assert asked.outcome == "clarify" and asked.clarify_options == ["서울특별시 > 중구", "부산광역시 > 중구"]
    assert [(c.code, c.axis) for c in asked.clarify_choices] == [("11020", "행정구역별(읍면동)"), ("21010", "행정구역별(읍면동)")]
    assert asked.text.startswith("어느 중구인가요?\n- 서울특별시 > 중구\n- 부산광역시 > 중구")


# ---------------------------------------------------------------- StatPipeline.ask: same metadata keys and spans


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


BASE = {"pipeline", "outcome_origin", "numbers_origin", "defaults_origin", "route", "spans"}
ANSWERED = BASE | {"type", "pattern", "plan", "compose", "label_digits", "numbers_verified", "chart", "caveats"}
# (table, question, model replies, decision_metadata keys, span names): recorded before S3a moved the steps
CASES = [
    ("household", SEOUL_ONE, [reply("서울은 {f1}입니다.")], ANSWERED | {"verify_problems"},
     ["judge", "analyze.rules", "compute", "compose", "verify"]),
    ("household", SEOUL_ONE, [reply("3102가구"), reply("3102가구")], ANSWERED | {"verify_problems"},
     ["judge", "analyze.rules", "compute", "compose", "verify", "compose", "verify", "template"]),
    ("household", "서울특별시 1인 가구 추이는?", ["not json"], ANSWERED | {"compose_status"},
     ["judge", "analyze.rules", "compute", "compose", "template"]),
    ("household", "2030년 서울 1인 가구 수는?", [], BASE, ["judge"]),
    ("household", "2023년 중구 1인 가구는?", [], BASE, ["judge"]),
    ("pop", "2023년 여자 가상 인구는?", ["not json"], ANSWERED | {"compose_status"},
     ["judge", "analyze.rules", "compute", "compose", "template"]),
    ("pop", "2023년 여자 인구는?", ["not json"],
     BASE | {"default_selection", "event_lookup", "fallback_prompt", "fill_status", "numbers_verified", "plan_status",
             "rule_default_plan", "rule_default_plan_validation", "semantic_validation", "unresolved"},
     ["judge", "analyze.rules", "analyze.llm_fill", "analyze.llm_plan", "fallback_chatbot"]),
]


@pytest.mark.parametrize(("which", "question", "replies", "keys", "spans"), CASES,
                         ids=[f"{c[0]}-{c[1]}-{len(c[2])}" for c in CASES])
def test_pipeline_metadata_keys_and_spans(household_path, pop_path, which, question, replies, keys, spans):
    path = household_path if which == "household" else pop_path
    answer = StatPipeline(load_table(path), client=MockClient(replies or "x"), org_id="101").ask(question)
    meta = answer.decision_metadata
    assert set(meta) == keys
    assert [s["name"] for s in meta["spans"]] == spans
