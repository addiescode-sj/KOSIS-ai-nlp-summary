"""Loop 2, one table, two targets of one category (analyze/rules.py build_plan, analyze/blocks.py compare):

- values asked side by side are not refused when their units differ: no difference or ratio is planned that the
  question does not ask for (shared/patterns.py asked_comparison, read by the answer step too);
- one target per scale units of the other ("~ 1천 명당"): the per formula, the denominator being the target named
  before 당 (shared/patterns.py PER_UNIT, per_scale: the words the multi-table rules read).

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
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))
YEARS = ("2020", "2021", "2022", "2023")
TABLES = {
    # a count and a rate per 1,000 people in one table
    "FAKE_MARRIAGE": [("T1", "혼인건수", "건"), ("T2", "조혼인율", "천명당")],
    "FAKE_DEATHS": [("T1", "사망자수", "명"), ("T2", "조사망률", "천명당")],
    # two counts
    "FAKE_VITAL2": [("T1", "출생아수", "명"), ("T2", "사망자수", "명")],
    "FAKE_DOCTOR": [("T1", "인구", "명"), ("T2", "의사수", "명")],
}
BASES = {"T1": 3000, "T2": 7}


def value_of(item_id, r_no, p_no):
    if item_id == "T2":
        return round(4 + 0.3 * r_no + 0.1 * p_no, 1)
    return 3000 * (4 - r_no) + 41 * p_no + r_no


def doctor_value(item_id, r_no, p_no):
    return 500_000 * (4 - r_no) + 1000 * p_no if item_id == "T1" else 900 * (4 - r_no) + 13 * p_no + r_no


def rows(tbl, items):
    out = []
    for p_no, period in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            for i_no, (item_id, item_name, unit) in enumerate(items):
                if tbl == "FAKE_DOCTOR":
                    v = doctor_value(item_id, r_no, p_no)
                elif tbl == "FAKE_VITAL2":
                    v = 2000 * (4 - r_no) + 31 * p_no + 500 * i_no
                else:
                    v = value_of(item_id, r_no, p_no)
                out.append({"period": period, "item_id": item_id, "item_name": item_name, "unit": unit,
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v})
    return out


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop2_one") / "data"
    return {t: load_resources(load_table(write_table(folder, t, rows(t, items), CATALOG))) for t, items in TABLES.items()}


def ask(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


# ---------------------------------------------------------------- D-2: values of different units side by side


@pytest.mark.parametrize("question, table, cells", [
    ("2022년 서울의 혼인건수와 조혼인율 알려줘", "FAKE_MARRIAGE", ["FAKE_MARRIAGE|T1|11|2022", "FAKE_MARRIAGE|T2|11|2022"]),
    ("부산 2021년 사망자 수랑 조사망률은 각각 얼마야?", "FAKE_DEATHS", ["FAKE_DEATHS|T1|21|2021", "FAKE_DEATHS|T2|21|2021"]),
    ("대구의 2023년 조혼인율과 혼인 건수를 보여줘", "FAKE_MARRIAGE", ["FAKE_MARRIAGE|T1|22|2023", "FAKE_MARRIAGE|T2|22|2023"]),
])
def test_two_values_of_different_units_are_listed(res, question, table, cells):
    _, answer = ask(res, question, table)
    assert sorted(f.source_ref for f in answer) == cells
    assert not [f for f in answer if f.derivation]


def test_a_difference_of_different_units_is_still_refused(res):
    got = calculate("2022년 서울 혼인건수와 조혼인율의 차이는?", [res["FAKE_MARRIAGE"]])
    assert got.status == "refused" and got.reason == "unit_mismatch"


# ---------------------------------------------------------------- D-3: one target per scale units of the other


@pytest.mark.parametrize("question, table, numerator, denominator, scale", [
    ("2022년 부산의 사망자 1천 명당 출생아 수는?", "FAKE_VITAL2", "FAKE_VITAL2|T1|21|2022", "FAKE_VITAL2|T2|21|2022", 1000),
    ("2021년 대구 인구 1만 명당 의사 수는 몇 명이야?", "FAKE_DOCTOR", "FAKE_DOCTOR|T2|22|2021", "FAKE_DOCTOR|T1|22|2021", 10000),
    ("서울은 2023년에 인구 10만 명당 의사 수가 몇 명이야?", "FAKE_DOCTOR", "FAKE_DOCTOR|T2|11|2023", "FAKE_DOCTOR|T1|11|2023", 100000),
    ("2020년 서울은 사망자 1명당 출생아 수가 얼마야?", "FAKE_VITAL2", "FAKE_VITAL2|T1|11|2020", "FAKE_VITAL2|T2|11|2020", 1),
])
def test_a_value_per_units_of_another(res, question, table, numerator, denominator, scale):
    facts, answer = ask(res, question, table)
    (per,) = [f for f in answer if f.derivation]
    a_id, b_id = per.derivation.split(" * ")[0].split(" / ")
    assert (facts[a_id].source_ref, facts[b_id].source_ref) == (numerator, denominator)
    assert per.value == pytest.approx(facts[a_id].value / facts[b_id].value * scale, abs=0.01)
    assert sorted(f.source_ref for f in answer if f.source_ref) == sorted([numerator, denominator])


# ---------------------------------------------------------------- D-1: values of named targets at named periods


@pytest.mark.parametrize("question, table, cells", [
    ("2021년과 2023년 서울과 부산의 출생아 수 알려줘", "FAKE_VITAL2",
     ["FAKE_VITAL2|T1|11|2021", "FAKE_VITAL2|T1|11|2023", "FAKE_VITAL2|T1|21|2021", "FAKE_VITAL2|T1|21|2023"]),
    ("대구, 서울 인구를 2020년, 2022년 기준으로 각각 보여줘", "FAKE_DOCTOR",
     ["FAKE_DOCTOR|T1|11|2020", "FAKE_DOCTOR|T1|11|2022", "FAKE_DOCTOR|T1|22|2020", "FAKE_DOCTOR|T1|22|2022"]),
    ("부산 의사 수는 2020년과 2022년에 각각 몇 명이야?", "FAKE_DOCTOR", ["FAKE_DOCTOR|T2|21|2020", "FAKE_DOCTOR|T2|21|2022"]),
    ("2022년, 2023년 대구의 혼인건수와 조혼인율은?", "FAKE_MARRIAGE",
     ["FAKE_MARRIAGE|T1|22|2022", "FAKE_MARRIAGE|T1|22|2023", "FAKE_MARRIAGE|T2|22|2022", "FAKE_MARRIAGE|T2|22|2023"]),
])
def test_values_of_named_targets_at_named_periods(res, question, table, cells):
    _, answer = ask(res, question, table)
    assert sorted(f.source_ref for f in answer) == cells
    assert not [f for f in answer if f.derivation]


def test_two_named_periods_with_a_change_word_are_still_a_change(res):
    facts, answer = ask(res, "부산 의사 수는 2020년에 비해 2022년에 얼마나 늘었어?", "FAKE_DOCTOR")
    (d, *_) = [f for f in answer if f.derivation]
    assert sorted(facts[i].source_ref for i in d.derivation.split(" - ")) == ["FAKE_DOCTOR|T2|21|2020",
                                                                               "FAKE_DOCTOR|T2|21|2022"]


@pytest.mark.parametrize("question, table, cells", [
    ("2023년 서울과 부산의 혼인건수와 조혼인율 알려줘", "FAKE_MARRIAGE",
     ["FAKE_MARRIAGE|T1|11|2023", "FAKE_MARRIAGE|T1|21|2023", "FAKE_MARRIAGE|T2|11|2023", "FAKE_MARRIAGE|T2|21|2023"]),
    ("대구와 서울의 2020년 출생아 수, 사망자 수는 각각?", "FAKE_VITAL2",
     ["FAKE_VITAL2|T1|11|2020", "FAKE_VITAL2|T1|22|2020", "FAKE_VITAL2|T2|11|2020", "FAKE_VITAL2|T2|22|2020"]),
])
def test_values_of_two_targets_of_two_categories(res, question, table, cells):
    _, answer = ask(res, question, table)
    assert sorted(f.source_ref for f in answer) == cells
    assert not [f for f in answer if f.derivation]
