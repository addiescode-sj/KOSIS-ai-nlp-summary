"""Loop 14 B (analyze/rules.py _twin_axes, _pair_names, _role_words, _role_of, _flows; shared/patterns.py ORIGIN_AXIS,
ROLE_VERBS):

- A table whose title names two roles of places ('…별' twice: where people live now, where they were born) holds one
  role in a category of regions whose name says no role and the other in the items: the short names of the regions
  ('서울', '부산'), a role head and a mark on the first one only ('출생지-계'), and a region split into a 'name-계' and
  its parts ('경기-계', '경기-북부'). The items and the regions are a pair, compared by the forms a question says them
  by; the roles are read on the first item and the title.
- Being born or growing up somewhere (태어난, 출생한, 자란) is the place people came from.
- A region word given a role keeps its own category's code only: one flow is one cell; two flows swapped are two
  cells in the question's order.

Neighbours: a table whose categories say the roles by their names reads as before (loop 5); a region alone with no
role on this table stays as before (both categories); a table of the same items and regions whose title has no two
roles is no pair.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2015", "2020")
REGIONS = (("00", "전국"), ("11", "서울특별시"), ("11010", "종로구"), ("21", "부산광역시"), ("31", "경기도"),
           ("31010", "수원시"), ("23", "인천광역시"))
ITEMS = (("B0", "출생지-계"), ("B11", "서울"), ("B21", "부산"), ("B31", "경기-계"), ("B311", "경기-북부"),
         ("B312", "경기-남부"), ("B23", "인천"))
PLAIN_ITEMS = (("B0", "계"),) + ITEMS[1:]


def rows(items):
    out = []
    for y, year in enumerate(YEARS):
        for i, (rc, rl) in enumerate(REGIONS):
            for j, (ic, il) in enumerate(items):
                v = 1000 * (i + 1) + 37 * (j + 1) + 5 * y
                out.append({"period": year, "item_id": ic, "item_name": il, "unit": "명", "dim1_name": "행정구역별",
                            "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v})
    return out


def catalog(title):
    return {"survey_name": "가상 인구조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE14_BORN": (lambda: rows(ITEMS), catalog("가상 현거주지별 출생지별 인구")),
    "FAKE14_BORN_2": (lambda: rows(ITEMS), catalog("가상 거주지별 출생지별 내국인(5세 이상)")),
    "FAKE14_NOROLE": (lambda: rows(PLAIN_ITEMS), catalog("가상 행정구역별 지역별 인구")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop14b") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def ref(tbl, item, region, year):
    return f"{tbl}|{item}|{region}|{year}"


def value(tbl, item, region, year):
    return next(r["value"] for r in ROWS[tbl]
                if r["item_id"] == item and r["dim1_code"] == region and r["period"] == year)


def cells(answer):
    return [f.source_ref for f in answer if f.source_ref]


@pytest.mark.parametrize("tbl", ["FAKE14_BORN", "FAKE14_BORN_2"])
def test_items_and_regions_are_a_pair_of_roles(res, tbl):
    from kosis_chat.analyze.rules import _twin_axes

    assert _twin_axes(res[tbl].index) == ("항목", "행정구역별")  # the place born first (the start)


@pytest.mark.parametrize("question, tbl, item, region, year", [
    ("2020년 부산에서 태어나 서울에 사는 사람은 몇 명이야?", "FAKE14_BORN", "B21", "11", "2020"),
    ("2015년 서울에 거주하는 부산 출신 인구는?", "FAKE14_BORN", "B21", "11", "2015"),
    ("2020년 경기도에서 태어나서 인천에 살고 있는 사람은 몇 명?", "FAKE14_BORN_2", "B31", "23", "2020"),
    ("2015년 인천에 사는 사람 가운데 서울에서 자란 사람은 몇 명이야?", "FAKE14_BORN_2", "B11", "23", "2015"),
])
def test_one_flow_keeps_the_code_of_each_role(res, question, tbl, item, region, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, year)]
    assert answer[0].value == value(tbl, item, region, year)


def test_two_flows_swapped_in_question_order(res):
    _, answer = ask(res, "2020년 서울에서 태어나 부산에 사는 사람과 부산에서 태어나 서울에 사는 사람은 각각 몇 명이야?",
                    "FAKE14_BORN")
    assert cells(answer) == [ref("FAKE14_BORN", "B11", "21", "2020"), ref("FAKE14_BORN", "B21", "11", "2020")]


def test_ratio_of_two_flows(res):
    facts, answer = ask(res, "2020년 부산에서 태어나 서울에 사는 사람은 서울에서 태어나 부산에 사는 사람의 몇 배야?",
                        "FAKE14_BORN")
    (got,) = [f for f in answer if not f.source_ref]
    a, b = value("FAKE14_BORN", "B21", "11", "2020"), value("FAKE14_BORN", "B11", "21", "2020")
    assert got.value == pytest.approx(round(a / b, 2), abs=0.01)
    assert cells(answer) == [ref("FAKE14_BORN", "B21", "11", "2020"), ref("FAKE14_BORN", "B11", "21", "2020")]


def test_one_role_said_the_other_side_is_the_total(res):
    _, answer = ask(res, "2020년 서울에 사는 사람은 몇 명?", "FAKE14_BORN")
    assert cells(answer) == [ref("FAKE14_BORN", "B0", "11", "2020")]


# ---------------------------------------------------------------- neighbours


def test_neighbour_a_region_with_no_role_stays_as_before(res):
    _, answer = ask(res, "2020년 서울 인구는 몇 명이야?", "FAKE14_BORN")
    assert cells(answer) == [ref("FAKE14_BORN", "B11", "11", "2020")]


def test_neighbour_no_two_roles_in_the_title_is_no_pair(res):
    from kosis_chat.analyze.rules import _twin_axes

    assert _twin_axes(res["FAKE14_NOROLE"].index) is None


def test_neighbour_categories_named_by_role_read_as_before(tmp_path):
    from kosis_chat.analyze.rules import _twin_axes

    places = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("31", "경기도"))
    data = []
    for a, la in places:
        for b, lb in places:
            v = 100 + 10 * int(a) + int(b)
            data.append({"period": "2020", "item_id": "T1", "item_name": "이동자수", "unit": "명",
                         "dim1_name": "전출지별", "dim1_code": a, "dim1_label": la, "dim2_name": "전입지별",
                         "dim2_code": b, "dim2_label": lb, "value_raw": str(v), "value": v})
    r = load_resources(load_table(write_table(tmp_path / "data", "FAKE14_MOVE", data, catalog("가상 전출지별 전입지별 이동자"))))
    assert _twin_axes(r.index) == ("전출지별", "전입지별")
    got = calculate("2020년 부산에서 서울로 전입한 사람은 몇 명이야?", [r])
    facts = {f.id: f for f in got.facts}
    assert [facts[i].source_ref for i in got.answer] == ["FAKE14_MOVE|T1|21|11|2020"]
