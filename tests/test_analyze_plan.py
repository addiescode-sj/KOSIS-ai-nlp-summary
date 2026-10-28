"""The model's own plan (analyze-plan-v1, T0-04 S5b) on the made-up tables: placed and computed by code, said back once
when wrong, asked back when it names several codes, and events missing from the calendar asked of the model."""

import json

import pytest
from pydantic import ValidationError

from kosis_chat import pipeline as pipeline_module
from kosis_chat.answer.chart import chart_spec
from kosis_chat.answer.template import template_parts
from kosis_chat.answer.verify import verify
from kosis_chat.compute.analyze import Plan, Unresolved
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.judge.question import build_index
from kosis_chat.llm import analyze_plan
from kosis_chat.llm.event_lookup import EventStart
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import StatPipeline

from kosis_ai.schemas import PlanV3  # isort: skip

REGION = "행정구역별(읍면동)"
BAD_FILL = json.dumps({"codes": {"항목": ["T9"]}})


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


def select(sid, item, periods, **dims):
    return {"id": sid, "op": "select", "item": {"codes": [item]}, "dims": {k: v for k, v in dims.items()},
            "periods": {"kind": "list", "periods": periods}}


def plan_json(steps, show, present):
    return json.dumps({"steps": steps, "show": show, "present": present}, ensure_ascii=False)


@pytest.fixture
def households(household_path):
    table = clean(load_table(household_path))
    return table, build_index(table)


def place(question, steps, show, present, households, lookup=None):
    table, index = households
    return analyze_plan.place(question, PlanV3(steps=steps, show=show, present=present), table, index, lookup)


# ---------------------------------------------------------------- place(): code checks and computes the plan


def test_plan_is_computed_by_code_with_labels_turned_into_codes(households):
    steps = [select("x", "T1", ["2023"], **{REGION: {"codes": ["서울특별시"]}}),
             select("y", "T1", ["2023"], **{REGION: {"codes": ["21"]}}),
             {"id": "d", "op": "combine", "a": "x", "b": "y", "fn": "diff", "orient": "larger_first"}]
    placed = place("2023년 서울과 부산 1인 가구 차이", steps, ["x", "y", "d"], "compare", households)
    assert placed.problems == [] and placed.built.pattern == "plan"
    assert [f.source_ref for f in placed.computed.result.facts if f.source_ref] == ["FAKE_TBL_002|T1|11|2023", "FAKE_TBL_002|T1|21|2023"]
    assert placed.plan.step("d").note == "차이"  # a combine without a note is named for the sentence
    assert placed.query_plan.type == "compare" and placed.query_plan.codes[REGION] == ["11", "21"]
    assert placed.query_plan.periods == ["2023"]


def test_plan_problems_are_words_the_model_can_act_on(households):
    steps = [{"id": "x", "op": "select", "item": {"codes": ["T1"]}, "dims": {"지역": {"codes": ["11"]}},
              "periods": {"kind": "list", "periods": ["2023"]}},
             {"id": "y", "op": "select", "cells": ["FAKE_TBL_002|T1|21|2023"]}]
    placed = place("2023년 서울 1인 가구", steps, ["x", "y"], "value", households)
    assert placed.computed is None
    assert placed.problems[0].startswith("select x: '지역'은(는) 분류 이름이 아닙니다.")
    assert placed.problems[1] == "select y: cells·table 대신 item·dims·periods로 칸을 적습니다."


def test_plan_must_read_the_periods_the_question_names(households):
    steps = [select("x", "T1", ["2024"], **{REGION: {"codes": ["11"]}})]
    placed = place("2015년과 2024년 서울 1인 가구", steps, ["x"], "trend", households)
    assert placed.problems == ["질문이 말한 시점 2015을(를) 읽는 select가 없습니다."]


def test_meaning_checks_still_refuse_a_wrong_plan(households):
    steps = [select("p", "T1", ["2023"], **{REGION: {"codes": ["11"]}}), select("t", "T0", ["2023"], **{REGION: {"codes": ["21"]}}),
             {"id": "s", "op": "combine", "a": "p", "b": "t", "fn": "share"}]
    placed = place("2023년 서울 1인 가구 비중", steps, ["s"], "share", households)
    assert placed.computed is None and placed.problems  # 서울 1인 ÷ 부산 전체: R5 refuses, said back to the model


def test_several_codes_for_one_label_ask_back(households):
    steps = [{"id": "x", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"path": ["중구"]}},
              "periods": {"kind": "list", "periods": ["2023"]}}]
    placed = place("2023년 중구 1인 가구", steps, ["x"], "value", households)
    assert placed.clarify is not None and [code for code, _ in placed.clarify.options] == ["11020", "21010"]


def test_event_missing_from_the_calendar_is_asked_and_recorded(households):
    asked = []
    steps = [{"id": "x", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"codes": ["11"]}},
              "periods": {"kind": "anchor", "event": "가상 사건", "offset": 0}}]
    placed = place("가상 사건 당시 서울 1인 가구", steps, ["x"], "value", households,
                   lookup=lambda name: asked.append(name) or "2020-03")
    assert asked == ["가상 사건"] and placed.computed.result.facts[0].period == "2020"
    assert "event_source: model (가상 사건 = 2020-03)" in placed.computed.cautions


@pytest.mark.parametrize("start", ["2020-13", "1800", "3000-01", "작년"])
def test_event_start_must_be_a_past_year_or_month(start):
    with pytest.raises(ValidationError):
        EventStart(start=start)
    assert EventStart(start="1997-11").start == "1997-11" and EventStart(start=None).start is None


def test_listed_template_and_chart_for_a_written_plan(households):
    steps = [select("x", "T1", ["2023"], **{REGION: {"codes": ["11"]}}), select("y", "T1", ["2023"], **{REGION: {"codes": ["21"]}}),
             {"id": "d", "op": "combine", "a": "x", "b": "y", "fn": "diff", "orient": "larger_first"}]
    computed = place("2023년 서울과 부산 1인 가구 차이", steps, ["x", "y", "d"], "compare", households).computed
    parts = template_parts("compare", computed)
    assert verify(parts, computed) == []
    chart = chart_spec("compare", computed)
    assert chart.kind == "bar" and [r["이름"] for r in chart.spec["data"]["values"]] == ["서울특별시", "부산광역시"]


def test_prompt_lists_every_category_and_the_periods(households):
    _, index = households
    rule = Plan(question="q", type=None, codes={REGION: ["11020", "21010"]}, periods=[])
    user = analyze_plan.user_prompt("2023년 중구 1인 가구", index, rule)
    assert "- 항목: T0=" in user and "11020=중구" in user and "21010=중구" in user and "11=서울특별시" in user
    assert "2015, 2020, 2023, 2024, 2025 (모두 5개, 단위 연(YYYY))" in user and "[예 1]" in user


# ---------------------------------------------------------------- the pipeline: after the fill, before the fallback


def test_pipeline_answers_with_the_models_plan_after_the_fill_fails(pop_path):
    good = plan_json([select("m", "T1", ["2023"], 성별={"codes": ["남자"]})], ["m"], "value")
    bot = StatPipeline(load_table(pop_path), client=MockClient([BAD_FILL, BAD_FILL, good, reply("2023년 남자는 {f1}입니다.")]),
                       org_id="101")
    answer = bot.ask("2023년 남자는?")
    stat, meta = answer.stat_answer, answer.decision_metadata
    assert meta["route"] == "llm_plan" and meta["pattern"] == "plan" and meta["compose"] == "model" and answer.attempts == 4
    assert stat.outcome == "answered" and stat.numbers_origin == "code" and stat.result.facts[0].value == 500
    assert stat.query.plan.steps[0].dims["성별"].codes == ["1"] and stat.query.item == "가상 인구"


def test_pipeline_says_the_plan_problems_back_once(pop_path):
    wrong = plan_json([select("m", "T1", ["2023"], 성={"codes": ["1"]})], ["m"], "value")
    good = plan_json([select("m", "T1", ["2023"], 성별={"codes": ["1"]})], ["m"], "value")
    bot = StatPipeline(load_table(pop_path), client=MockClient([BAD_FILL, BAD_FILL, wrong, good, reply("남자는 {f1}입니다.")]),
                       org_id="101")
    answer = bot.ask("2023년 남자는?")
    assert answer.decision_metadata["plan_problems"][0].startswith("select m: '성'은(는) 분류 이름이 아닙니다.")
    assert answer.stat_answer.outcome == "answered" and answer.attempts == 5


def test_pipeline_asks_back_when_the_plan_names_several_codes(household_path, monkeypatch):
    def unplaced(plan, index, table):
        raise Unresolved(["셀 선택"])

    monkeypatch.setattr(pipeline_module, "build_plan", unplaced)  # (the judge itself asks back on a bare '중구')
    ask = plan_json([{"id": "x", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"path": ["중구"]}},
                      "periods": {"kind": "list", "periods": ["2023"]}}], ["x"], "value")
    bot = StatPipeline(load_table(household_path), client=MockClient([BAD_FILL, BAD_FILL, ask, ask]), org_id="101")
    answer = bot.ask("2023년 도심 1인 가구는?")
    stat = answer.stat_answer
    assert answer.decision_metadata["route"] == "llm_plan+clarify" and stat.outcome == "clarify"
    assert stat.clarify_options == ["서울특별시 > 중구", "부산광역시 > 중구"] and stat.numbers_origin == "none"
