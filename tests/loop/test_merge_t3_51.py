"""T1-33 merge with T3-51 (analyze/rules.py rule_plan): where T3-51's two rules meet the loop's rules.

- A place name that is in the table's name is left to the model by T3-51 (_names_the_table), but a flow question
  gives it a role ("경기도에서 서울로"): loop 5 (_flows) reads it as the place asked, and no note is left open.
- The same name without a role is still left to the model (T3-51 as it was).
- A word the item and another category share picks the item only (T3-51 _one_word_one_axis); the loop's rules
  that read the other category's names (a comparison of two ages) still read them.

The tables are made up (가상 데이터).
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.analyze.rules import rule_plan
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

PLACES = (("00", "전국"), ("11", "서울특별시"), ("31", "경기도"), ("23", "인천광역시"))


def flow_rows():
    cells = {}
    for i, (a, _) in enumerate(PLACES[1:], start=1):
        for j, (b, _) in enumerate(PLACES[1:], start=1):
            cells[(a, b)] = 90 * i + 37 * j * j + 500 * (a == b)
    totals: dict = defaultdict(int)
    for (a, b), v in cells.items():
        for aa in (a, "00"):
            for bb in (b, "00"):
                totals[(aa, bb)] += v
    return [{"period": "2020", "item_id": "T1", "item_name": "이동자수", "unit": "명", "dim1_name": "전출지별",
             "dim1_code": a, "dim1_label": dict(PLACES)[a], "dim2_name": "전입지별", "dim2_code": b,
             "dim2_label": dict(PLACES)[b], "value_raw": str(v), "value": v} for (a, b), v in totals.items()]


def age_rows():
    items = (("T0", "계"), ("T19", "미상"))
    ages = (("00", "계"), ("30", "30대"), ("40", "40대"), ("95", "미상"))
    return [{"period": "2024", "item_id": ic, "item_name": il, "unit": "건", "dim1_name": "연령별", "dim1_code": ac,
             "dim1_label": al, "value_raw": str(v), "value": float(v)}
            for n, (ic, il) in enumerate(items) for m, (ac, al) in enumerate(ages) for v in [100 * (n + 1) + 7 * m]]


def catalog(title, survey="가상 조사"):
    return {"survey_name": survey, "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE_M_FLOW": (flow_rows, catalog("가상 경기도 전출지별 전입지별 이동자")),
    "FAKE_M_DIV": (age_rows, catalog("가상 이혼사유별 이혼", "가상 이혼 동향")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("merge_t3_51") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def value(tbl, **want):
    return next(r["value"] for r in ROWS[tbl] if all(r[k] == v for k, v in want.items()))


def test_a_place_of_the_table_name_given_a_role_in_a_flow_is_the_place_asked(res):
    question = "2020년 경기도에서 서울로 전입한 사람은 몇 명이야?"
    plan = rule_plan(question, res["FAKE_M_FLOW"].index)
    assert not plan.unresolved
    got = calculate(question, [res["FAKE_M_FLOW"]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    answer = [f for f in got.facts if f.id in got.answer]
    assert [f.source_ref for f in answer] == ["FAKE_M_FLOW|T1|31|11|2020"]
    assert answer[0].value == value("FAKE_M_FLOW", dim1_code="31", dim2_code="11")


def test_a_place_of_the_table_name_without_a_role_is_still_left_to_the_model(res):
    plan = rule_plan("2020년 경기도 이동자수는 몇 명이야?", res["FAKE_M_FLOW"].index)
    assert plan.unresolved and all(u.split(":")[0] in ("전출지별", "전입지별") for u in plan.unresolved)
    assert calculate("2020년 경기도 이동자수는 몇 명이야?", [res["FAKE_M_FLOW"]]).status == "no_plan"


def test_a_word_of_the_item_and_an_age_keeps_the_item_and_the_ages_named_apart(res):
    plan = rule_plan("2024년 이혼사유 미상 이혼은 30대와 40대 중 어디가 많아?", res["FAKE_M_DIV"].index)
    assert plan.codes.get("항목") == ["T19"]
    assert sorted(plan.codes.get("연령별", [])) == ["30", "40"]
    assert not plan.unresolved
