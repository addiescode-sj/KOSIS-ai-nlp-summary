"""The model steps of T1-28 S3b on the made-up tables (tests/conftest.py), with the MockClient.

Each LLM step is called on its own (In -> Out, its routes, its retry), the records they return become the same spans
and totals the pipeline reported before (graph/spans.py, graph/legacy.py), and the pipeline keeps its call ceilings:
fill 2 + plan 2 + event_lookup (one per event) + compose COMPOSE_MAX_ATTEMPTS + fallback 2 (plans/T1-28.md S3 Done).
"""

from __future__ import annotations

import inspect
import json
import time
import typing

import pytest

from kosis_chat.analyze.node import step_event_lookup, step_plan_fill_llm, step_plan_rules, step_plan_write_llm
from kosis_chat.analyze.event_lookup import EventStart
from kosis_chat.analyze.rules import Unresolved
from kosis_chat.analyze.schemas import (
    EventLookupIn,
    EventLookupOut,
    PlanFillIn,
    PlanFillOut,
    PlanRulesIn,
    PlanWriteIn,
    PlanWriteOut,
)
from kosis_chat.compose.node import step_caveats, step_compose_llm
from kosis_chat.compose.schemas import COMPOSE_MAX_ATTEMPTS, CaveatsIn, ComposeLLMIn, ComposeLLMOut
from kosis_chat.compute.node import step_compute
from kosis_chat.compute.resolve import NeedsClarify
from kosis_chat.compute.schemas import ComputeIn
from kosis_chat.data.loader import load_table
from kosis_chat.graph import legacy, spans
from kosis_chat.graph.legacy import DECISION_KEYS, totals_of
from kosis_chat.graph.spans import spans_of, step_span
from kosis_chat.judge.node import step_judge
from kosis_chat.judge.schemas import JudgeIn
from kosis_chat.legacy.fallback import FallbackIn, FallbackOut, step_fallback_chatbot
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.llm.model import STATUS_FORMAT_ERROR, STATUS_OK, LLMResult, MockClient
from kosis_chat.pipeline import StatPipeline
from kosis_chat.schemas.resources import LLMConfig, LLMKeyMissing, load_resources
from kosis_chat.schemas.state import LLMCall, StepSpan, called_before

STEPS = {
    step_plan_fill_llm: (PlanFillIn, PlanFillOut), step_plan_write_llm: (PlanWriteIn, PlanWriteOut),
    step_event_lookup: (EventLookupIn, EventLookupOut), step_compose_llm: (ComposeLLMIn, ComposeLLMOut),
    step_fallback_chatbot: (FallbackIn, FallbackOut),
}
REGION = "행정구역별(읍면동)"
BAD_FILL = json.dumps({"codes": {"항목": ["T9"]}})
GOOD_FILL = json.dumps({"codes": {"항목": ["T1"]}})


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


def plan_json(periods, show=("m",), present="value", dims=None):
    step = {"id": "m", "op": "select", "item": {"codes": ["T1"]}, "dims": dims or {"성별": {"codes": ["1"]}},
            "periods": periods}
    return json.dumps({"steps": [step], "show": list(show), "present": present}, ensure_ascii=False)


GOOD_PLAN = plan_json({"kind": "list", "periods": ["2023"]})


def access(client, retry_interval=0.0):
    return ModelAccess(client=lambda: client, model_name=lambda: client.name, retry_interval=retry_interval)


@pytest.fixture
def pop(pop_path):
    return load_resources(load_table(pop_path), "101")


@pytest.fixture
def households(household_path):
    return load_resources(load_table(household_path), "101")


def unresolved(res, question):
    verdict = step_judge(JudgeIn(question=question), res).verdict
    rules = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res)
    assert rules.route == "unresolved"
    return rules


# ---------------------------------------------------------------- the step contract


@pytest.mark.parametrize("step", list(STEPS), ids=lambda s: s.__name__)
def test_step_signature_and_docstring(step):
    params = list(inspect.signature(step).parameters.values())
    assert [p.name for p in params[:2]] == ["inp", "res"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params[2:])
    assert "llm" in [p.name for p in params[2:]]
    hints = typing.get_type_hints(step)
    assert (hints["inp"], hints["return"]) == STEPS[step]
    lines = [line.strip() for line in inspect.getdoc(step).splitlines()]
    for head in ("In:", "Out:", "Side effects:", "Node:"):
        assert any(line.startswith(head) for line in lines), (step.__name__, head)


# ---------------------------------------------------------------- plan_fill_llm


def test_fill_places_the_question(pop):
    rules = unresolved(pop, "2023년 여자 인구는?")
    client = MockClient([GOOD_FILL])
    out = step_plan_fill_llm(PlanFillIn(rule=rules.rule, unresolved=rules.unresolved), pop, llm=access(client))
    assert out.route == "built" and out.built is not None and out.plan.codes["항목"] == ["T1"]
    assert [c.name for c in out.calls] == ["analyze.llm_fill"] and len(client.calls) == 1
    assert list(out.meta) == ["route", "fill"] and out.meta["route"] == "llm_fill"


def test_fill_says_a_fault_back_once(pop):
    rules = unresolved(pop, "2023년 여자 인구는?")
    client = MockClient([BAD_FILL, GOOD_FILL])
    out = step_plan_fill_llm(PlanFillIn(rule=rules.rule, unresolved=rules.unresolved), pop, llm=access(client))
    assert out.route == "built" and [c.name for c in out.calls] == ["analyze.llm_fill", "analyze.llm_fill_retry"]
    assert "[이전 답변의 문제]" in client.calls[1][1] and "'T9'" in client.calls[1][1]
    assert out.meta["fill"]["ignored"] == ["코드 없음 항목=T9"]


def test_fill_stops_after_two_calls(pop):
    rules = unresolved(pop, "2023년 여자 인구는?")
    client = MockClient([BAD_FILL] * 5)
    out = step_plan_fill_llm(PlanFillIn(rule=rules.rule, unresolved=rules.unresolved), pop, llm=access(client))
    assert out.route == "unplaced" and out.built is None and len(client.calls) == 2


def test_fill_without_a_reply(pop):
    rules = unresolved(pop, "2023년 여자 인구는?")
    out = step_plan_fill_llm(PlanFillIn(rule=rules.rule, unresolved=rules.unresolved), pop, llm=access(MockClient("x")))
    assert out.route == "unplaced" and out.meta["fill_status"] == STATUS_FORMAT_ERROR
    assert out.plan.unresolved == rules.unresolved and out.calls[0].status == STATUS_FORMAT_ERROR


def test_fill_build_can_be_swapped(pop):
    rules = unresolved(pop, "2023년 여자 인구는?")

    def never(plan, index, table):
        raise Unresolved(["셀 선택"])

    out = step_plan_fill_llm(PlanFillIn(rule=rules.rule, unresolved=rules.unresolved), pop,
                             llm=access(MockClient([GOOD_FILL])), build=never)
    assert out.route == "unplaced" and out.meta["unresolved_after_fill"] == ["셀 선택"]


# ---------------------------------------------------------------- plan_write_llm and event_lookup


def test_write_places_the_models_plan(pop):
    rules = unresolved(pop, "2023년 남자는?")
    client = MockClient([GOOD_PLAN])
    out = step_plan_write_llm(PlanWriteIn(question="2023년 남자는?", rule=rules.rule), pop, llm=access(client))
    assert out.route == "placed" and out.placed.computed.result.facts[0].value == 500
    assert [c.name for c in out.calls] == ["analyze.llm_plan"] and list(out.meta) == ["event_lookup", "route", "plan"]


def test_write_says_the_problems_back_once(pop):
    rules = unresolved(pop, "2023년 남자는?")
    wrong = plan_json({"kind": "list", "periods": ["2023"]}, dims={"성": {"codes": ["1"]}})
    client = MockClient([wrong, GOOD_PLAN])
    out = step_plan_write_llm(PlanWriteIn(question="2023년 남자는?", rule=rules.rule), pop, llm=access(client))
    assert out.route == "placed" and [c.name for c in out.calls] == ["analyze.llm_plan", "analyze.llm_plan_retry"]
    assert "[이전 계획의 문제]" in client.calls[1][1] and out.meta["plan_problems"]


def test_write_fails_after_two_plans(pop):
    rules = unresolved(pop, "2023년 남자는?")
    wrong = plan_json({"kind": "list", "periods": ["2023"]}, dims={"성": {"codes": ["1"]}})
    client = MockClient([wrong] * 5)
    out = step_plan_write_llm(PlanWriteIn(question="2023년 남자는?", rule=rules.rule), pop, llm=access(client))
    assert out.route == "failed" and out.placed is None and len(client.calls) == 2
    assert list(out.meta) == ["event_lookup", "route", "plan", "plan_problems", "plan_retry", "plan_retry_problems"]


def test_write_asks_an_unknown_event_once_per_question(pop):
    rules = unresolved(pop, "2023년 남자는?")
    anchored = plan_json({"kind": "anchor", "event": "가상 사건", "offset": 0})
    client = MockClient([anchored, '{"start": null}', anchored])
    out = step_plan_write_llm(PlanWriteIn(question="2023년 남자는?", rule=rules.rule), pop, llm=access(client))
    names = [c.name for c in out.calls]
    assert names == ["analyze.llm_plan", "event_lookup", "analyze.llm_plan_retry"]  # the retry does not ask again
    assert out.meta["event_lookup"] == {"가상 사건": None} and out.calls[1].attributes == {"event": "가상 사건"}


def test_write_asks_back_when_the_plan_names_several_codes(households):
    question = "2023년 도심 1인 가구는?"
    verdict = step_judge(JudgeIn(question=question), households).verdict
    rule = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), households).rule
    ask = json.dumps({"steps": [{"id": "x", "op": "select", "item": {"codes": ["T1"]}, "dims": {REGION: {"path": ["중구"]}},
                                 "periods": {"kind": "list", "periods": ["2023"]}}], "show": ["x"], "present": "value"},
                     ensure_ascii=False)
    out = step_plan_write_llm(PlanWriteIn(question=question, rule=rule), households, llm=access(MockClient([ask, ask])))
    assert out.route == "clarify" and isinstance(out.clarify, NeedsClarify) and len(out.clarify.options) == 2


def test_event_lookup(pop):
    out = step_event_lookup(EventLookupIn(name="가상 사건"), pop, llm=access(MockClient(['{"start": "2020-03"}'])))
    assert out.start == "2020-03" and out.call.name == "event_lookup" and out.call.attributes == {"event": "가상 사건"}
    failed = step_event_lookup(EventLookupIn(name="가상 사건"), pop, llm=access(MockClient("모름")))
    assert failed.start is None and failed.call.status == STATUS_FORMAT_ERROR


# ---------------------------------------------------------------- compose_llm


def computed_of(res, question):
    verdict = step_judge(JudgeIn(question=question), res).verdict
    rules = step_plan_rules(PlanRulesIn(question=question, default_applied=verdict.default_applied), res)
    done = step_compute(ComputeIn(plan=rules.rule, built=rules.built, default_applied=verdict.default_applied), res)
    noted = step_caveats(CaveatsIn(computed=done.computed, built=done.built, alternatives=done.alternatives,
                                   default_applied=verdict.default_applied), res)
    return done.computed, noted


def test_compose_routes_and_retry_prompt(households):
    computed, noted = computed_of(households, "2023년 서울특별시 1인 가구 수는?")
    client = MockClient([reply("서울은 {f1}입니다."), reply("다시 {f1}입니다.")])
    first = step_compose_llm(ComposeLLMIn(question="q", computed=computed, cautions=noted.shown,
                                          context_ids=noted.context_ids), households, llm=access(client))
    assert first.route == "verify" and first.parts and first.call.attributes == {"attempt": 0} and first.meta == {}
    again = step_compose_llm(ComposeLLMIn(question="q", computed=computed, attempt=1, problems=["숫자 문제"],
                                          prompt=first.prompt), households, llm=access(client))
    assert again.prompt == first.prompt + "\n\n[이전 답변의 문제]\n- 숫자 문제\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요."
    assert client.calls[1][1] == again.prompt and again.call.attributes == {"attempt": 1}
    failed = step_compose_llm(ComposeLLMIn(question="q", computed=computed), households, llm=access(MockClient("x")))
    assert failed.route == "template" and failed.parts is None and failed.meta == {"compose_status": STATUS_FORMAT_ERROR}


# ---------------------------------------------------------------- fallback_chatbot


def test_fallback_step_records_one_unsummed_call(pop_path, mock_settings):
    bot = StatPipeline(load_table(pop_path), client=MockClient("x"), settings=mock_settings, org_id="101")
    out = step_fallback_chatbot(FallbackIn(question="2023년 여자 인구는?"), bot.resources, chatbot=lambda: bot.fallback,
                                llm=bot._model_access())
    assert out.call.name == "fallback_chatbot" and out.call.summed is False and out.call.attempts == out.answer.attempts
    assert out.answer.attempts <= 2 and totals_of([out.call]).attempts == 0


# ---------------------------------------------------------------- ModelAccess: the client and the wait


def test_model_access_waits_only_after_an_answer(pop, monkeypatch):
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    llm = access(MockClient("x"), retry_interval=2.5)
    _, first = llm.ask("event_lookup", "s", "u", EventStart, called=False)
    _, second = llm.ask("event_lookup", "s", "u", EventStart, called=True)
    assert waits.count(2.5) >= 1 and first.slept == 0.0 and second.prompt_chars == 2
    assert called_before([first]) and not called_before([step_span("judge", 0.0)])


def test_client_is_built_at_the_first_model_step(pop_path):
    """S2c: an LLMConfig without a key works for rule-only answers; LLMKeyMissing comes with the first model call."""
    bot = StatPipeline(load_table(pop_path), llm=LLMConfig(model="HCX-DASH-002"), org_id="101")
    assert bot.ask("2030년 여자 인구는?").decision_metadata["route"] == "rules_final"
    assert bot._client is None
    with pytest.raises(LLMKeyMissing):
        bot.ask("2023년 여자 인구는?")


# ---------------------------------------------------------------- spans and totals from the records


def test_spans_of_writes_the_old_span_dicts():
    t0 = 100.0
    call = LLMCall(name="compose", started=100.5, ended=101.5, model="HCX-DASH-002", status=STATUS_OK, attempts=2,
                   usage={"promptTokens": 10, "completionTokens": 3}, raw=["a", "b"], slept=0.25, retry_wait_seconds=0.25,
                   attributes={"attempt": 1})
    rule = StepSpan(name="analyze.rules", started=100.0, ended=100.25, status="error",
                    attributes={"type": None, "unresolved": ["항목"]})
    assert spans_of([rule, call], t0) == [
        {"name": "analyze.rules", "start_ms": 0.0, "duration_ms": 250.0, "status": "error",
         "attributes": {"unresolved": ["항목"]}},
        {"name": "compose", "start_ms": 500.0, "duration_ms": 500.0, "status": "ok",
         "attributes": {"gen_ai.request.model": "HCX-DASH-002", "gen_ai.usage.input_tokens": 10,
                        "gen_ai.usage.output_tokens": 3, "llm.attempts": 2, "llm.status": "ok", "attempt": 1}},
    ]
    assert list(spans_of([call], t0)[0]["attributes"])[:5] == ["gen_ai.request.model", "gen_ai.usage.input_tokens",
                                                                "gen_ai.usage.output_tokens", "llm.attempts", "llm.status"]


def test_totals_sum_the_calls_in_order():
    a = LLMCall(name="analyze.llm_fill", started=0.0, ended=1.0, model="m", status="ok", attempts=1, usage={"promptTokens": 5},
                raw=["r1"], errors=[], slept=0.0, retry_wait_seconds=0.0, prompt_chars=7)
    b = LLMCall(name="compose", started=1.0, ended=2.0, model="m", status="format_error", attempts=2,
                usage={"promptTokens": 6, "x": True}, raw=["r2", "r3"], errors=["e"], slept=1.0, retry_wait_seconds=0.5,
                prompt_chars=3)
    fallback = LLMCall(name="fallback_chatbot", started=2.0, ended=3.0, model="m", status="ok", attempts=9, summed=False)
    total = totals_of([a, step_span("verify", 1.0), b, fallback])
    assert (total.attempts, total.usage, total.raw, total.errors, total.waited, total.prompt_chars) == (
        3, {"promptTokens": 11}, ["r1", "r2", "r3"], ["e"], 1.5, 10)


def test_decision_keys_are_unchanged():
    assert DECISION_KEYS == {
        "pipeline", "outcome_origin", "numbers_origin", "defaults_origin", "route", "unresolved", "fill", "fill_status",
        "fill_retry", "fill_retry_status", "unresolved_after_fill", "event_lookup", "plan", "plan_status",
        "plan_problems", "plan_retry", "plan_retry_status", "plan_retry_problems", "type", "pattern", "compose",
        "compose_status", "verify_problems", "template_problems", "label_digits", "numbers_verified", "chart",
        "caveats", "fallback_prompt"}
    assert len(DECISION_KEYS) == 29
    assert set(spans.SPAN_NAMES) == {"judge", "analyze.rules", "analyze.llm_fill", "analyze.llm_fill_retry",
                                     "analyze.llm_plan", "analyze.llm_plan_retry", "event_lookup", "compute", "compose",
                                     "verify", "template", "fallback_chatbot"}
    assert legacy.add_usage is __import__("kosis_chat.pipeline").pipeline.add_usage


# ---------------------------------------------------------------- the pipeline: ceilings and the keys it writes

# The keys the compact-v2 chatbot's own record brings in (kosis_chat/chatbot.py), merged on the fallback route.
CHATBOT_KEYS = {"default_selection", "rule_default_plan", "rule_default_plan_validation", "semantic_validation",
                "absent_fields_normalized"}


def run(path, question, replies, monkeypatch=None, unplaced=False):
    if unplaced:
        from kosis_chat import pipeline as pipeline_module

        def never(plan, index, table):
            raise Unresolved(["셀 선택"])

        monkeypatch.setattr(pipeline_module, "build_plan", never)
    client = MockClient(replies)
    answer = StatPipeline(load_table(path), client=client, org_id="101").ask(question)
    return answer, client


def names(answer):
    return [s["name"] for s in answer.decision_metadata["spans"]]


def test_ceilings_when_every_reply_is_wrong(pop_path, monkeypatch):
    """Every reply has a fault the step can say back: fill 2 + plan 2 + one event lookup + fallback (at most 2)."""
    anchored = plan_json({"kind": "anchor", "event": "가상 사건", "offset": 0})

    def wrong(system, user):
        if system.startswith("당신은 통계 질문을 통계표의 칸으로"):
            return BAD_FILL
        if system.startswith("당신은 통계 질문을 계산 순서"):
            return anchored
        if system.startswith("당신은 사건이 언제"):
            return '{"start": null}'
        return "not json"

    answer, client = run(pop_path, "2023년 남자는?", wrong, monkeypatch, unplaced=True)
    assert names(answer) == ["judge", "analyze.rules", "analyze.llm_fill", "analyze.llm_fill_retry", "analyze.llm_plan",
                             "event_lookup", "analyze.llm_plan_retry", "fallback_chatbot"]
    assert len(client.calls) == 2 + 2 + 1 + 2  # the chatbot's answer and its one format retry
    assert answer.attempts == len(client.calls)
    assert set(answer.decision_metadata) <= DECISION_KEYS | {"spans"} | CHATBOT_KEYS


def test_compose_ceiling_is_the_constant(household_path):
    answer, client = run(household_path, "2023년 서울특별시 1인 가구 수는?", [reply("3102가구")] * 5)
    assert names(answer).count("compose") == COMPOSE_MAX_ATTEMPTS == len(client.calls)
    assert names(answer)[-1] == "template" and answer.decision_metadata["compose"] == "template"
    assert set(answer.decision_metadata) <= DECISION_KEYS | {"spans"}


@pytest.mark.parametrize(("replies", "route", "spans_"), [
    ([GOOD_FILL, reply("{f1}입니다.")], "llm_fill", ["analyze.llm_fill", "compute", "compose", "verify"]),
    ([BAD_FILL, GOOD_FILL, reply("{f1}입니다.")], "llm_fill",
     ["analyze.llm_fill", "analyze.llm_fill_retry", "compute", "compose", "verify"]),
    ([BAD_FILL, BAD_FILL, GOOD_PLAN, reply("{f1}입니다.")], "llm_plan",
     ["analyze.llm_fill", "analyze.llm_fill_retry", "analyze.llm_plan", "compute", "compose", "verify"]),
])
def test_pipeline_model_routes(pop_path, replies, route, spans_):
    answer, client = run(pop_path, "2023년 남자는?", replies)
    meta = answer.decision_metadata
    assert meta["route"] == route and names(answer) == ["judge", "analyze.rules", *spans_]
    assert answer.attempts == len(client.calls) == len(replies)
    assert answer.usage == {} and answer.raw_responses == replies
    assert set(meta) <= DECISION_KEYS | {"spans"}
