"""analyze-fill-v2: the menu names only the open categories, labels become codes, problems are listed for a retry."""

import pytest

from kosis_chat.compute.analyze import Plan
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.judge.question import build_index
from kosis_chat.llm import analyze_fill
from kosis_chat.llm.analyze_fill import AnalyzeFill, apply, asked_axes, menu, resolve_code, user_prompt

REGION = "행정구역별(읍면동)"


@pytest.fixture
def index(household_path):
    return build_index(clean(load_table(household_path)))


def plan(**kw):
    base = dict(question="q", type="value", codes={}, periods=["2023"], unresolved=["항목"])
    return Plan(**(base | kw))


def test_menu_lists_only_the_open_categories(index):
    p = plan(codes={REGION: ["11"]}, unresolved=["항목"])
    assert asked_axes(p, index) == ["항목"]
    text = user_prompt("2023년 서울 가구는?", index, p)
    assert "[채울 분류 목록]\n- 항목:" in text and f"- {REGION}:" not in text and "서울특별시" in text  # found, not listed
    narrowed = plan(codes={REGION: ["11", "21"]}, unresolved=[f"{REGION}: 2개 중 하나"])
    assert menu(index, narrowed, asked_axes(narrowed, index)).count("=") == 2  # just the two candidates
    assert menu(index, plan(unresolved=["질문 유형"])).count("\n") >= 1  # no category named: every category


def test_labels_and_codes_both_resolve(index):
    axis = index.axes[REGION]
    assert resolve_code(axis, "11") == "11" and resolve_code(axis, "서울특별시") == "11" and resolve_code(axis, "서울") == "11"
    assert resolve_code(axis, "달나라") is None and resolve_code(axis, "") is None
    assert resolve_code(axis, "11=서울특별시") == "11" and resolve_code(axis, "T1=1인 (가구)") is None  # a menu line copied
    assert resolve_code(index.axes["항목"], "T1=1인 (가구)") == "T1" and resolve_code(index.axes["항목"], "1인 (가구)") == "T1"


def test_fill_shapes_the_model_actually_wrote_are_read(index):
    """From the 2026-10-26 runs: the key as 'code=label', values as 'code=label', nested lists, numbers."""
    p = plan(codes={REGION: ["11"]}, unresolved=["항목"])
    keyed, ignored, problems = apply(p, AnalyzeFill(codes={"T1=1인 (가구)": ["2025"]}), index)
    assert keyed.codes["항목"] == ["T1"] and problems == [] and ignored == []
    both = plan(codes={}, unresolved=["항목", f"{REGION}: 2개 중 하나"])
    filled, _, problems = apply(both, AnalyzeFill(codes={"T2=2인 (가구)": ["11=서울특별시"]}), index)
    assert filled.codes["항목"] == ["T2"] and filled.codes[REGION] == ["11"] and problems == []
    nested = AnalyzeFill.model_validate({"codes": {REGION: ["전국", ["11", 21]], "항목": "T1"}})
    assert nested.codes == {REGION: ["전국", "11", "21"], "항목": ["T1"]}


def test_apply_maps_moves_and_reports(index):
    p = plan(codes={REGION: ["11"]}, unresolved=["항목"], pattern="avg", groups=[["2020", "2021"]])
    filled, ignored, problems = apply(p, AnalyzeFill(codes={"항목": ["1인"]}), index)
    assert filled.codes["항목"] == ["T1"] and ignored == [] and problems == [] and filled.unresolved == []
    assert filled.pattern == "avg" and filled.groups == [["2020", "2021"]] and filled.codes[REGION] == ["11"]  # rules kept
    moved, ignored, problems = apply(p, AnalyzeFill(codes={"가구원수": ["T1"]}), index)
    assert moved.codes["항목"] == ["T1"] and ignored == [] and problems == []
    bad, ignored, problems = apply(p, AnalyzeFill(codes={"시점": ["2023"], "항목": ["T99", "T1"]}), index)
    assert bad.codes["항목"] == ["T1"] and ignored == ["분류 없음 시점", "코드 없음 항목=T99"]
    assert problems == ["시점은 codes에 넣지 않습니다.", "'T99'은(는) 항목 목록에 없는 값입니다. 목록에서 '=' 앞의 코드를 씁니다."]
    none, ignored, problems = apply(p, AnalyzeFill(codes={}), index)
    assert "항목" not in none.codes and problems == ["항목: 질문이 가리키는 코드를 고르지 않았습니다."]
    narrowed = plan(codes={REGION: ["11", "21"]}, unresolved=[f"{REGION}: 2개 중 하나"])
    still, _, problems = apply(narrowed, AnalyzeFill(codes={}), index)
    assert still.codes[REGION] == ["11", "21"] and problems == [f"{REGION}: 질문이 가리키는 코드를 고르지 않았습니다."]
    assert "[이전 답변의 문제]\n- a\n- b" in analyze_fill.retry_prompt("u", ["a", "b"])
