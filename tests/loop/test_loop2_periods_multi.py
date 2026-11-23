"""Loop 2: a year range on a table that holds yearly and monthly values (analyze/rules.py _range_periods), and the
multi-table plan beyond one cell per table (analyze/multi_rules.py): the same calculation per named target or per
named period, a change between two periods per table, and per-table sums over a range before the calculation.

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
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("24", "광주광역시"))
YEARS = ("2019", "2020", "2021", "2022", "2023")


def value_of(base: int, r_no: int, p_no: int, i_no: int = 0) -> int:
    return base * (5 - r_no) + 17 * p_no * (r_no + 1) + 7 * i_no


def year_rows(items, base):
    out = []
    for p_no, period in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            for i_no, (item_id, item_name, unit) in enumerate(items):
                v = value_of(base, r_no, p_no, i_no)
                out.append({"period": period, "item_id": item_id, "item_name": item_name, "unit": unit,
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v, "prd_se": "Y"})
    return out


def month_rows(item, base, years=("2021", "2022", "2023")):
    """A monthly table that also holds each year's value (the sum of its months)."""
    item_id, item_name, unit = item
    out = []
    for y_no, year in enumerate(years):
        for r_no, (code, label) in enumerate(REGIONS):
            months = [base * (5 - r_no) + 3 * m + 11 * y_no for m in range(1, 13)]
            values = {f"{year}{m:02d}": v for m, v in enumerate(months, 1)} | {year: sum(months)}
            out += [{"period": p, "item_id": item_id, "item_name": item_name, "unit": unit,
                     "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                     "value_raw": str(v), "value": v, "prd_se": "M" if len(p) == 6 else "Y"}
                    for p, v in values.items()]
    return out


SPECS = {
    "FAKE_BIRTH": lambda: year_rows([("T1", "출생아수", "명")], 900),
    "FAKE_DEATH": lambda: year_rows([("T1", "사망자수", "명")], 1000),
    "FAKE_MARRY": lambda: year_rows([("T1", "혼인건수", "건")], 500),
    "FAKE_DIVORCE": lambda: year_rows([("T1", "이혼건수", "건")], 200),
    "FAKE_BIRTH_YM": lambda: month_rows(("T1", "출생아수", "명"), 80),
    "FAKE_MARRY_YM": lambda: month_rows(("T1", "혼인건수", "건"), 40),
    "FAKE_MOVE_YM": lambda: month_rows(("T1", "전입자수", "명"), 300),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop2") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), CATALOG))) for t, make in SPECS.items()}


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def derived(answer):
    return [f for f in answer if f.derivation and "환산" not in (f.note or "")]


def refs_in(facts, fact):
    """The cells a derived fact was computed from, following derived inputs down to the cells."""
    import re

    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def ref(tbl, region, period):
    return f"{tbl}|T1|{region}|{period}"


# ---------------------------------------------------------------- B-1: a year range on a yearly + monthly table


@pytest.mark.parametrize("question, table, region, years, how", [
    ("2021~2023년 서울 출생아 수를 모두 더하면?", "FAKE_BIRTH_YM", "11", ("2021", "2022", "2023"), "sum"),
    ("2022년부터 2023년까지 부산의 혼인 건수 합계는?", "FAKE_MARRY_YM", "21", ("2022", "2023"), "sum"),
    ("2021년부터 2022년까지 대구 전입자 수 평균은 얼마야?", "FAKE_MOVE_YM", "22", ("2021", "2022"), "avg"),
    ("2021-2023년 광주광역시 출생아 수의 연평균은?", "FAKE_BIRTH_YM", "24", ("2021", "2022", "2023"), "avg"),
])
def test_a_year_range_uses_the_yearly_cells(res, question, table, region, years, how):
    facts, answer = ask(res, question, table)
    (total,) = derived(answer)
    want = sorted(ref(table, region, y) for y in years)
    assert refs_in(facts, total) == want
    values = [facts[i].value for i in total.derivation.replace("(", " ").replace(")", " ").split() if i in facts]
    expected = sum(values) / len(values) if how == "avg" else sum(values)
    assert total.value == pytest.approx(expected, abs=0.01)
    assert all(len(facts[i].period) == 4 for i in facts if facts[i].source_ref)


def test_months_are_still_used_when_the_question_asks_by_month(res):
    facts, answer = ask(res, "2022~2023년 서울 출생아 수 월별 추이 보여줘", "FAKE_BIRTH_YM")
    months = [f for f in facts.values() if f.source_ref and len(f.period) == 6]
    assert len(months) == 24


# ---------------------------------------------------------------- A-1: the same calculation per named target


@pytest.mark.parametrize("question, tables, regions, fn", [
    ("2022년 서울과 부산의 혼인 1건당 출생아 수는 각각 얼마야?", ("FAKE_BIRTH", "FAKE_MARRY"), ("11", "21"), "ratio"),
    ("2020년 대구와 광주 각각 사망자 수는 출생아 수보다 몇 명 많아?", ("FAKE_BIRTH", "FAKE_DEATH"), ("22", "24"), "diff"),
    ("2023년 부산, 대구, 서울의 이혼건수는 혼인건수의 몇 배야?", ("FAKE_MARRY", "FAKE_DIVORCE"), ("21", "22", "11"), "ratio"),
])
def test_one_calculation_per_named_target(res, question, tables, regions, fn):
    facts, answer = ask(res, question, *tables)
    results = derived(answer)
    assert len(results) == len(regions)
    year = next(y for y in YEARS if y in question)
    for got, region in zip(results, regions):
        cells = refs_in(facts, got)
        assert cells == sorted(ref(t, region, year) for t in tables)
        a, b = (facts[i].value for i in got.derivation.replace("/", " ").replace("-", " ").split())
        assert got.value == pytest.approx(a / b if fn == "ratio" else a - b, abs=0.01)
        if fn == "diff":
            assert got.value > 0  # the question's direction: deaths minus births
    assert sorted(f.source_ref for f in answer if f.source_ref) == sorted(
        ref(t, r, year) for t in tables for r in regions)


# ---------------------------------------------------------------- A-2: per named period, and a change per table


@pytest.mark.parametrize("question, tables, region, years", [
    ("2020년과 2023년 서울의 혼인 1건당 출생아 수를 각각 알려줘", ("FAKE_BIRTH", "FAKE_MARRY"), "11", ("2020", "2023")),
    ("부산은 2019년, 2021년에 출생아 수가 사망자 수보다 몇 명 적었어?", ("FAKE_BIRTH", "FAKE_DEATH"), "21", ("2019", "2021")),
])
def test_one_calculation_per_named_period(res, question, tables, region, years):
    facts, answer = ask(res, question, *tables)
    results = derived(answer)
    assert len(results) == len(years)
    for got, year in zip(results, years):
        assert refs_in(facts, got) == sorted(ref(t, region, year) for t in tables)


@pytest.mark.parametrize("question, tables, region, early, late, rate", [
    ("2019년 대비 2023년 부산 출생아 수와 사망자 수의 증감률은?", ("FAKE_BIRTH", "FAKE_DEATH"), "21", "2019", "2023", True),
    ("2020년부터 2022년까지 대구의 혼인 건수와 이혼 건수는 각각 몇 퍼센트 변했어?", ("FAKE_MARRY", "FAKE_DIVORCE"), "22",
     "2020", "2022", True),
    ("서울 출생아 수와 혼인 건수는 2021년에 비해 2023년에 얼마나 줄었어?", ("FAKE_BIRTH", "FAKE_MARRY"), "11", "2021", "2023",
     False),
])
def test_a_change_between_two_periods_per_table(res, question, tables, region, early, late, rate):
    facts, answer = ask(res, question, *tables)
    results = derived(answer)
    assert len(results) == len(tables)
    for got in results:
        cells = refs_in(facts, got)
        (tbl,) = {c.split("|")[0] for c in cells}
        assert cells == sorted([ref(tbl, region, early), ref(tbl, region, late)])
        first, last = facts[[i for i in facts if facts[i].source_ref == ref(tbl, region, early)][0]].value, \
            facts[[i for i in facts if facts[i].source_ref == ref(tbl, region, late)][0]].value
        want = (last / first - 1) * 100 if rate else last - first
        assert got.value == pytest.approx(want, abs=0.01)
    assert sorted(f.source_ref for f in answer if f.source_ref) == sorted(
        ref(t, region, y) for t in tables for y in (early, late))


# ---------------------------------------------------------------- A-3: sums over a range per table, then the calculation


@pytest.mark.parametrize("question, tables, region, years, fn", [
    ("2019~2021년 서울의 출생아 수 합계를 혼인 건수 합계로 나누면?", ("FAKE_BIRTH", "FAKE_MARRY"), "11",
     ("2019", "2020", "2021"), "ratio"),
    ("2021년부터 2023년까지 광주의 사망자 수 합은 출생아 수 합보다 얼마나 많아?", ("FAKE_BIRTH", "FAKE_DEATH"), "24",
     ("2021", "2022", "2023"), "diff"),
    ("2022~2023년 부산 혼인 1건당 출생아 수는?", ("FAKE_BIRTH_YM", "FAKE_MARRY_YM"), "21", ("2022", "2023"), "ratio"),
])
def test_per_table_sums_over_a_range(res, question, tables, region, years, fn):
    facts, answer = ask(res, question, *tables)
    finals = [f for f in derived(answer) if " + " not in f.derivation]
    assert len(finals) == 1
    (got,) = finals
    assert refs_in(facts, got) == sorted(ref(t, region, y) for t in tables for y in years)
    a_id, b_id = got.derivation.replace("/", " ").replace("-", " ").split()
    a, b = facts[a_id].value, facts[b_id].value
    assert got.value == pytest.approx(a / b if fn == "ratio" else a - b, abs=0.01)
    if fn == "diff":
        assert got.value > 0
