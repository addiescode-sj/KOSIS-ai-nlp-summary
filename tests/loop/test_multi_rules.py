"""Loop 1: the rules' plan over two or three tables (analyze/multi_rules.py), through kosis_chat.calc.calculate.

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
YEARS = ("2021", "2022", "2023")
MONTHS = tuple(f"2023{m:02d}" for m in range(1, 13))


def value_of(item_no: int, r_no: int, p_no: int, base: int) -> int:
    return base * (4 - r_no) + 10 * p_no + item_no


def rows(items, periods, base):
    out = []
    for p_no, period in enumerate(periods):
        for r_no, (code, label) in enumerate(REGIONS):
            for i_no, (item_id, item_name, unit) in enumerate(items):
                v = value_of(i_no, r_no, p_no, base)
                out.append({"period": period, "item_id": item_id, "item_name": item_name, "unit": unit,
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v, "prd_se": "M" if len(period) == 6 else "Y"})
    return out


SPECS = {
    "FAKE_BIRTH": ([("T1", "출생아수", "명"), ("T2", "조출생률", "천명당")], YEARS, 1000),
    "FAKE_DEATH": ([("T1", "사망자수", "명")], YEARS, 900),
    "FAKE_INFANT": ([("T1", "영아사망자수", "명")], YEARS, 30),
    "FAKE_MARRY": ([("T1", "혼인건수", "건"), ("T2", "조혼인율", "천명당")], YEARS, 500),
    "FAKE_POPK": ([("T1", "연앙인구", "천명")], YEARS, 3000),
    "FAKE_BIRTH_M": ([("T1", "출생아수", "명")], MONTHS, 100),
    "FAKE_DEATH_M": ([("T1", "사망자수", "명")], MONTHS, 90),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop1_multi") / "data"
    out = {}
    for tbl_id, (items, periods, base) in SPECS.items():
        path = write_table(folder, tbl_id, rows(items, periods, base), CATALOG)
        out[tbl_id] = load_resources(load_table(path))
    return out


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return got, facts, [facts[i] for i in got.answer]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def final(answer):
    derived = [f for f in answer if f.derivation and "환산" not in (f.note or "")]
    assert len(derived) == 1, [(f.id, f.derivation) for f in answer]
    return derived[0]


# ---------------------------------------------------------------- the values side by side


@pytest.mark.parametrize("question, tables, want", [
    ("2022년 부산 출생아 수와 사망자 수 알려줘", ("FAKE_BIRTH", "FAKE_DEATH"),
     ["FAKE_BIRTH|T1|21|2022", "FAKE_DEATH|T1|21|2022"]),
    ("대구의 2021년 혼인 건수, 사망자수, 출생아수는 각각?", ("FAKE_MARRY", "FAKE_DEATH", "FAKE_BIRTH"),
     ["FAKE_BIRTH|T1|22|2021", "FAKE_DEATH|T1|22|2021", "FAKE_MARRY|T1|22|2021"]),
])
def test_values_of_every_table_without_a_formula(res, question, tables, want):
    _, _, answer = ask(res, question, *tables)
    assert cells(answer) == want
    assert not [f for f in answer if f.derivation]


# ---------------------------------------------------------------- differences, in the question's direction


@pytest.mark.parametrize("question, tables, minuend, subtrahend", [
    ("2023년 서울은 사망자가 출생아보다 몇 명 많아?", ("FAKE_BIRTH", "FAKE_DEATH"),
     "FAKE_DEATH|T1|11|2023", "FAKE_BIRTH|T1|11|2023"),
    ("2022년 부산 출생아 수에서 사망자 수를 빼면?", ("FAKE_DEATH", "FAKE_BIRTH"),
     "FAKE_BIRTH|T1|21|2022", "FAKE_DEATH|T1|21|2022"),
    ("2021년 대구 출생아는 사망자보다 얼마나 적어?", ("FAKE_BIRTH", "FAKE_DEATH"),
     "FAKE_DEATH|T1|22|2021", "FAKE_BIRTH|T1|22|2021"),
])
def test_difference_follows_the_question(res, question, tables, minuend, subtrahend):
    _, facts, answer = ask(res, question, *tables)
    d = final(answer)
    a, b = d.derivation.split(" - ")
    assert (facts[a].source_ref, facts[b].source_ref) == (minuend, subtrahend)
    assert d.value == facts[a].value - facts[b].value
    assert cells(answer) == sorted([minuend, subtrahend])


def test_a_month_is_matched_on_both_monthly_tables(res):
    _, facts, answer = ask(res, "2023년 7월 부산 출생아 수와 사망자 수 차이는?", "FAKE_BIRTH_M", "FAKE_DEATH_M")
    assert cells(answer) == ["FAKE_BIRTH_M|T1|21|202307", "FAKE_DEATH_M|T1|21|202307"]
    assert final(answer).value == abs(value_of(0, 2, 6, 100) - value_of(0, 2, 6, 90))


# ---------------------------------------------------------------- sums


@pytest.mark.parametrize("question, tables, want", [
    ("2023년 서울 출생아 수와 사망자 수를 합치면 몇 명이야?", ("FAKE_BIRTH", "FAKE_DEATH"),
     ["FAKE_BIRTH|T1|11|2023", "FAKE_DEATH|T1|11|2023"]),
    ("2023년 3월 대구 출생아와 사망자를 더하면?", ("FAKE_DEATH_M", "FAKE_BIRTH_M"),
     ["FAKE_BIRTH_M|T1|22|202303", "FAKE_DEATH_M|T1|22|202303"]),
])
def test_sum_of_two_tables(res, question, tables, want):
    _, facts, answer = ask(res, question, *tables)
    s = final(answer)
    assert s.derivation.count("+") == 1
    assert sorted(facts[i].source_ref for i in s.derivation.split(" + ")) == want
    assert s.value == sum(facts[i].value for i in s.derivation.split(" + "))


# ---------------------------------------------------------------- ratios: one per unit of the other


@pytest.mark.parametrize("question, tables, numerator, denominator, scale", [
    ("2022년 서울의 혼인 1건당 출생아 수는?", ("FAKE_BIRTH", "FAKE_MARRY"),
     "FAKE_BIRTH|T1|11|2022", "FAKE_MARRY|T1|11|2022", 1),
    ("2023년 부산 사망자 1명당 출생아는 몇 명?", ("FAKE_DEATH", "FAKE_BIRTH"),
     "FAKE_BIRTH|T1|21|2023", "FAKE_DEATH|T1|21|2023", 1),
    ("2021년 대구 인구 1천 명당 사망자 수는?", ("FAKE_DEATH", "FAKE_POPK"),
     "FAKE_DEATH|T1|22|2021", "FAKE_POPK|T1|22|2021", 1000),
])
def test_value_per_unit_of_another_table(res, question, tables, numerator, denominator, scale):
    _, facts, answer = ask(res, question, *tables)
    r = final(answer)
    refs = {f.source_ref: f for f in facts.values() if f.source_ref}
    whole = refs[denominator].value * (1000 if denominator.startswith("FAKE_POPK") else 1)
    assert r.value == pytest.approx(refs[numerator].value / whole * scale, abs=0.01)
    assert cells(answer) == sorted([numerator, denominator])


# ---------------------------------------------------------------- an item name inside another table's item name


def test_an_item_inside_a_longer_item_name_belongs_to_the_longer_one(res):
    _, facts, answer = ask(res, "2023년 서울 사망자 수 대비 영아사망자 수 비율은?", "FAKE_INFANT", "FAKE_DEATH")
    r = final(answer)
    a, b = r.derivation.split(" / ")
    assert (facts[a].source_ref, facts[b].source_ref) == ("FAKE_INFANT|T1|11|2023", "FAKE_DEATH|T1|11|2023")


def test_no_period_named_uses_the_latest_period_of_every_table(res):
    _, _, answer = ask(res, "부산 출생아 수와 사망자 수의 차이", "FAKE_BIRTH", "FAKE_DEATH")
    assert cells(answer) == ["FAKE_BIRTH|T1|21|2023", "FAKE_DEATH|T1|21|2023"]


def test_a_named_month_a_yearly_table_lacks_is_not_replaced(res):
    got = calculate("2023년 7월 서울 출생아 수와 혼인 건수", [res["FAKE_BIRTH_M"], res["FAKE_MARRY"]])
    assert got.status == "no_plan"
