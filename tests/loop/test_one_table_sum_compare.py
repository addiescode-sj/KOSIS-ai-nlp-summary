"""Loop 1, one table: sums of named targets or periods (analyze/rules.py), the answer of a comparison
(compute/answer_facts.py) and item names with a particle inside (judge/question.py label_forms).

The tables are made up (가상 데이터). Every test checks the answer's value and the cells it was computed from.
"""

from __future__ import annotations

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

CATALOG = {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": "가상 표", "period": "년",
           "url": "https://example.invalid/t"}
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"))
YEARS = ("2020", "2021", "2022", "2023")
TABLES = {
    # births and deaths: flows, two items of one table
    "FAKE_VITAL": [("T1", "출생아수", "명"), ("T2", "사망자수", "명")],
    # a stock: population by region
    "FAKE_POP": [("T1", "총인구", "명")],
    # an item name with a particle inside
    "FAKE_ELDER": [("T1", "고령자의 인구", "명"), ("T2", "부모와 동거하는 자녀", "명")],
}


def value_of(i_no, r_no, p_no):
    return 1000 * (5 - r_no) + 37 * p_no + 3 * i_no + r_no


def rows(items):
    out = []
    for p_no, period in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            for i_no, (item_id, item_name, unit) in enumerate(items):
                v = value_of(i_no, r_no, p_no)
                out.append({"period": period, "item_id": item_id, "item_name": item_name, "unit": unit,
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v})
    return out


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop1_one") / "data"
    return {t: load_resources(load_table(write_table(folder, t, rows(items), CATALOG))) for t, items in TABLES.items()}


def ask(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def derived(answer):
    return [f for f in answer if f.derivation]


def summed(facts, answer):
    (total,) = derived(answer)
    parts = total.derivation.split(" + ")
    assert len(parts) >= 2
    assert total.value == sum(facts[p].value for p in parts)
    return total, sorted(facts[p].source_ref for p in parts)


# ---------------------------------------------------------------- sums of targets one category names


@pytest.mark.parametrize("question, table, want", [
    ("2022년 서울과 부산 출생아 수를 합치면 몇 명이야?", "FAKE_VITAL", ["FAKE_VITAL|T1|11|2022", "FAKE_VITAL|T1|21|2022"]),
    ("2021년 대구, 인천, 부산의 인구를 모두 더하면?", "FAKE_POP",
     ["FAKE_POP|T1|21|2021", "FAKE_POP|T1|22|2021", "FAKE_POP|T1|23|2021"]),
    ("2023년 인천 출생아 수와 사망자 수를 합하면?", "FAKE_VITAL", ["FAKE_VITAL|T1|23|2023", "FAKE_VITAL|T2|23|2023"]),
])
def test_sum_of_named_targets(res, question, table, want):
    facts, answer = ask(res, question, table)
    _, parts = summed(facts, answer)
    assert parts == want


@pytest.mark.parametrize("question, want", [
    ("2020년과 2022년 부산 출생아 수를 합치면?", ["FAKE_VITAL|T1|21|2020", "FAKE_VITAL|T1|21|2022"]),
    ("서울의 2021년, 2023년 사망자 수를 더해 줘", ["FAKE_VITAL|T2|11|2021", "FAKE_VITAL|T2|11|2023"]),
])
def test_sum_of_periods_named_one_by_one(res, question, want):
    facts, answer = ask(res, question, "FAKE_VITAL")
    _, parts = summed(facts, answer)
    assert parts == want


# ---------------------------------------------------------------- a comparison answers what was asked


def test_two_values_asked_without_a_formula(res):
    _, answer = ask(res, "2022년 서울과 대구 인구 알려줘", "FAKE_POP")
    assert sorted(f.source_ref for f in answer) == ["FAKE_POP|T1|11|2022", "FAKE_POP|T1|22|2022"]
    assert not derived(answer)


@pytest.mark.parametrize("question, cells", [
    ("2023년 서울과 부산의 출생아 수 차이는?", ["FAKE_VITAL|T1|11|2023", "FAKE_VITAL|T1|21|2023"]),
    ("2020년 인천은 대구보다 사망자가 얼마나 적어?", ["FAKE_VITAL|T2|22|2020", "FAKE_VITAL|T2|23|2020"]),
])
def test_a_difference_without_the_ratio(res, question, cells):
    facts, answer = ask(res, question, "FAKE_VITAL")
    (d,) = derived(answer)
    assert " - " in d.derivation
    assert sorted(facts[i].source_ref for i in d.derivation.split(" - ")) == cells


@pytest.mark.parametrize("question, numerator, denominator", [
    ("2022년 서울 인구는 부산의 몇 배야?", "FAKE_POP|T1|11|2022", "FAKE_POP|T1|21|2022"),
    ("2021년 대구 인구는 인천 인구의 몇 배나 많아?", "FAKE_POP|T1|22|2021", "FAKE_POP|T1|23|2021"),
])
def test_a_ratio_without_the_difference(res, question, numerator, denominator):
    facts, answer = ask(res, question, "FAKE_POP")
    (r,) = derived(answer)
    a, b = r.derivation.split(" / ")
    assert (facts[a].source_ref, facts[b].source_ref) == (numerator, denominator)
    assert r.value == pytest.approx(facts[a].value / facts[b].value, abs=0.006)


# ---------------------------------------------------------------- an item name with a particle inside


@pytest.mark.parametrize("question, want", [
    ("2022년 서울 고령자 인구는?", "FAKE_ELDER|T1|11|2022"),
    ("2023년 부산의 부모 동거하는 자녀 수 알려줘", "FAKE_ELDER|T2|21|2023"),
])
def test_item_named_without_its_particle(res, question, want):
    _, answer = ask(res, question, "FAKE_ELDER")
    assert [f.source_ref for f in answer] == [want]
