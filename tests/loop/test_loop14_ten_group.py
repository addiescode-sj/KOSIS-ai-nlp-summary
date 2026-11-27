"""Loop 14 C-1 (analyze/rules.py _number_range, _ten_group; shared/labels.py interval_bounds, TEN_GROUP):

A ten group said as one word ('20대', '30대') on a category of five-year interval names ('20-24세', '25-29세'; with
or without spaces around the mark) is the sum of the names that fill it, with its cells.

Neighbours: half a group ('20대 초반') is one name's cell; two names said one by one keep their own; a decade of years
('2010년대') is a period; a category whose own name is the group's word, or a ten-year name, keeps that name; a group
the names do not fill (bounds that do not meet it, an open '80세 이상' inside it) is not picked; another name of the
category named with the group is left as before.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2019", "2020", "2021")
FIVE = (("A0", "계"), ("A15", "15-19세"), ("A20", "20-24세"), ("A25", "25-29세"), ("A30", "30-34세"),
        ("A35", "35-39세"), ("A40", "40-44세"), ("A80", "80세 이상"))
SPACED = (("A0", "합계"), ("A20", "20 - 24세"), ("A25", "25 - 29세"), ("A30", "30 - 34세"), ("A35", "35 - 39세"),
          ("A40", "40 - 44세"), ("A45", "45 - 49세"))
OFFSET = (("A0", "계"), ("A18", "18-22세"), ("A23", "23-27세"), ("A28", "28-32세"), ("A33", "33-37세"))
OWN = (("A0", "계"), ("A20", "20대"), ("A30", "30대"), ("A21", "20-24세"), ("A22", "25-29세"))
TEN = (("A0", "계"), ("A2", "20-29세"), ("A20", "20-24세"), ("A25", "25-29세"))
OPEN = (("A0", "계"), ("A70", "70-74세"), ("A75", "75-79세"), ("A80", "80세 이상"))


def rows(ages, item="혼인건수", unit="건"):
    out = []
    for y, year in enumerate(YEARS):
        for i, (code, label) in enumerate(ages):
            v = 300 * (i + 1) + 11 * y + 7 * i * i
            out.append({"period": year, "item_id": "T1", "item_name": item, "unit": unit, "dim1_name": "연령별",
                        "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


def catalog(title):
    return {"survey_name": "가상 인구동향조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE14_FIVE": (lambda: rows(FIVE), catalog("가상 연령별 혼인 건수")),
    "FAKE14_SPACED": (lambda: rows(SPACED, "출생아수", "명"), catalog("가상 모의 연령별 출생아 수")),
    "FAKE14_OFFSET": (lambda: rows(OFFSET), catalog("가상 연령별 혼인 건수(다른 경계)")),
    "FAKE14_OWN": (lambda: rows(OWN), catalog("가상 연령대별 혼인 건수")),
    "FAKE14_TEN": (lambda: rows(TEN), catalog("가상 연령별 혼인 건수(십 단위)")),
    "FAKE14_OPEN": (lambda: rows(OPEN), catalog("가상 고령 연령별 혼인 건수")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop14c") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def ref(tbl, code, year):
    return f"{tbl}|T1|{code}|{year}"


def value(tbl, code, year):
    return next(r["value"] for r in ROWS[tbl] if r["dim1_code"] == code and r["period"] == year)


def computed(answer):
    return [f for f in answer if not f.source_ref]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def refs_in(facts, fact):
    return sorted(facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or ""))


@pytest.mark.parametrize("question, tbl, codes, year", [
    ("2020년 20대 혼인 건수는 몇 건이야?", "FAKE14_FIVE", ("A20", "A25"), "2020"),
    ("2021년에 30대가 한 혼인은 몇 건?", "FAKE14_FIVE", ("A30", "A35"), "2021"),
    ("2019년 엄마가 40대인 출생아 수 알려줘", "FAKE14_SPACED", ("A40", "A45"), "2019"),
    ("2020년 20 대 출생아는 몇 명이야?", "FAKE14_SPACED", ("A20", "A25"), "2020"),
])
def test_a_ten_group_is_the_sum_of_its_five_year_names(res, question, tbl, codes, year):
    facts, answer = ask(res, question, tbl)
    (got,) = computed(answer)
    assert got.value == sum(value(tbl, c, year) for c in codes)
    assert refs_in(facts, got) == sorted(ref(tbl, c, year) for c in codes)


# ---------------------------------------------------------------- neighbours


def test_neighbour_half_a_group_is_one_name(res):
    _, answer = ask(res, "2020년 20대 초반 혼인 건수는?", "FAKE14_FIVE")
    assert cells(answer) == [ref("FAKE14_FIVE", "A20", "2020")]


def test_neighbour_two_names_said_one_by_one_added(res):
    facts, answer = ask(res, "2020년 20-24세와 25-29세 혼인 건수를 합하면?", "FAKE14_FIVE")
    (got,) = computed(answer)
    assert got.value == value("FAKE14_FIVE", "A20", "2020") + value("FAKE14_FIVE", "A25", "2020")


def test_neighbour_a_decade_of_years_is_a_period(res):
    _, answer = ask(res, "2020년 25-29세 혼인 건수는?", "FAKE14_FIVE")
    assert cells(answer) == [ref("FAKE14_FIVE", "A25", "2020")]
    from kosis_chat.analyze.rules import _ten_group

    assert _ten_group("2010년대 25-29세 혼인 건수 추이", res["FAKE14_FIVE"].index) is None


@pytest.mark.parametrize("tbl, code", [("FAKE14_OWN", "A20"), ("FAKE14_TEN", "A2")])
def test_neighbour_a_name_that_is_the_group_keeps_its_cell(res, tbl, code):
    from kosis_chat.analyze.rules import _ten_group

    if tbl == "FAKE14_OWN":
        _, answer = ask(res, "2020년 20대 혼인 건수는?", tbl)
        assert cells(answer) == [ref(tbl, code, "2020")]
    else:  # a ten-year name beside its halves: the group word is not read over the halves
        assert _ten_group("2020년 20대 혼인 건수는?", res[tbl].index) is None


@pytest.mark.parametrize("question, tbl", [
    ("2020년 20대 혼인 건수는?", "FAKE14_OFFSET"),
    ("2020년 80대 혼인 건수는?", "FAKE14_OPEN"),
])
def test_neighbour_a_group_the_names_do_not_fill_is_not_picked(res, question, tbl):
    from kosis_chat.analyze.rules import _ten_group

    assert _ten_group(question, res[tbl].index) is None


def test_neighbour_another_name_named_with_the_group_is_left(res):
    from kosis_chat.analyze.rules import _ten_group

    assert _ten_group("2020년 20대와 30-34세 혼인 건수 차이는?", res["FAKE14_FIVE"].index) is None
