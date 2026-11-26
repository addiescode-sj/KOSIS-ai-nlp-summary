"""Loop 12 E (analyze/rules.py _twin_axes, _role_of, _pair_reference; shared/patterns.py STAY_AXIS, VISIT_AXIS,
ROLE_VERBS):

- E-1: two categories of places are a pair when their names are the same but one or two that only one of them holds
  ('기타'), and the roles are read in the pair: a place stayed at (상주지) and a place gone to and from each day
  (통근통학지) put the place stayed at first.
- E-2: a name's role is the kind of the verb after its particle ("경기도에 사는": the place stayed at, "서울로
  통근하는": the place gone to), the particle alone when no such verb follows (loop 5).
- The flow named before "의 몇 배" is the reference of a ratio of two flows.

Neighbours: loop 5's pair of a start and an end (전출지 × 전입지) read as before (an end said by 에 and a verb of
arriving); categories whose names differ by more than two are no pair; one name given both roles is one cell.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.analyze.rules import _twin_axes
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ["2015", "2020"]
HOME = (("00", "전국"), ("11", "서울특별시"), ("31", "경기도"), ("23", "인천광역시"))
ELSE = HOME + (("99", "기타"),)
FAR = HOME + (("91", "강원도"), ("92", "충청북도"), ("93", "충청남도"))


def flow_rows(start, starts, end, ends, item, base):
    cells = {}
    for y, year in enumerate(YEARS):
        for i, (a, _) in enumerate(starts[1:], start=1):
            for j, (b, _) in enumerate(ends[1:], start=1):
                cells[(year, a, b)] = base * i + 37 * j * j + 11 * y + 500 * (a == b)
    totals: dict = defaultdict(int)
    for (year, a, b), v in cells.items():
        for aa in (a, "00"):
            for bb in (b, "00"):
                totals[(year, aa, bb)] += v
    return [{"period": year, "item_id": "T1", "item_name": item, "unit": "명", "dim1_name": start, "dim1_code": a,
             "dim1_label": dict(starts)[a], "dim2_name": end, "dim2_code": b, "dim2_label": dict(ends)[b],
             "value_raw": str(v), "value": v} for (year, a, b), v in totals.items()]


def catalog(title):
    return {"survey_name": "가상 조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE12_COMMUTE": (lambda: flow_rows("상주지별", HOME, "통근통학지별", ELSE, "통근통학인구", 100),
                       catalog("가상 상주지별 통근통학 인구")),
    "FAKE12_WORK": (lambda: flow_rows("거주지별", HOME, "종사지별", ELSE, "종사자수", 70),
                    catalog("가상 거주지별 종사지별 종사자")),
    "FAKE12_MIGRATE": (lambda: flow_rows("전출지별", ELSE, "전입지별", HOME, "이동자수", 90),
                       catalog("가상 전출지별 전입지별 이동자")),
    "FAKE12_APART": (lambda: flow_rows("상주지별", HOME, "통근통학지별", FAR, "통근통학인구", 60),
                     catalog("가상 통근통학 인구")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop12e") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, tbl):
    got = calculate(question, [res[tbl]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    return [facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or "")]


def value(tbl, a, b, year):
    return next(r["value"] for r in ROWS[tbl] if r["dim1_code"] == a and r["dim2_code"] == b and r["period"] == year)


def ref(tbl, a, b, year):
    return f"{tbl}|T1|{a}|{b}|{year}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


@pytest.mark.parametrize("tbl, pair", [
    ("FAKE12_COMMUTE", ("상주지별", "통근통학지별")),
    ("FAKE12_WORK", ("거주지별", "종사지별")),
    ("FAKE12_MIGRATE", ("전출지별", "전입지별")),   # one name more on one side
])
def test_pair_of_nearly_the_same_names(res, tbl, pair):
    assert _twin_axes(res[tbl].index) == pair


def test_neighbour_names_differing_by_more_than_two_are_no_pair(res):
    assert _twin_axes(res["FAKE12_APART"].index) is None


@pytest.mark.parametrize("question, tbl, a, b, year", [
    ("2020년 경기도에 사는 사람 가운데 서울로 통근하는 사람은 몇 명이야?", "FAKE12_COMMUTE", "31", "11", "2020"),
    ("2015년 인천에 거주하며 서울로 다니는 사람은 몇 명이야?", "FAKE12_COMMUTE", "23", "11", "2015"),
    ("2020년 서울에 살면서 경기도로 출근하는 종사자는?", "FAKE12_WORK", "11", "31", "2020"),
    ("2020년 경기도에서 서울로 통근하는 사람은?", "FAKE12_COMMUTE", "31", "11", "2020"),  # particles alone
])
def test_one_flow_by_the_verbs_of_its_names(res, question, tbl, a, b, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, a, b, year)]
    assert answer[0].value == value(tbl, a, b, year)


def test_share_of_a_place_gone_to_among_those_staying(res):
    facts, answer = ask(res, "2020년 경기도에 사는 사람 중 서울로 통근하는 사람의 비중은 몇 퍼센트야?", "FAKE12_COMMUTE")
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value("FAKE12_COMMUTE", "31", "11", "2020")
                                            / value("FAKE12_COMMUTE", "31", "00", "2020") * 100, 2), abs=0.01)
    assert refs_in(facts, got) == [ref("FAKE12_COMMUTE", "31", "11", "2020"), ref("FAKE12_COMMUTE", "31", "00", "2020")]


@pytest.mark.parametrize("question, tbl, year", [
    ("2020년 경기도에서 서울로 통근하는 인구는 서울에서 경기도로 통근하는 인구의 몇 배야?", "FAKE12_COMMUTE", "2020"),
    ("2015년 경기도에서 서울로 전입한 사람은 서울에서 경기도로 전입한 사람의 몇 배야?", "FAKE12_MIGRATE", "2015"),
])
def test_ratio_of_two_flows_against_the_one_before_of_how_many_times(res, question, tbl, year):
    facts, answer = ask(res, question, tbl)
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value(tbl, "31", "11", year) / value(tbl, "11", "31", year), 2), abs=0.01)
    assert refs_in(facts, got) == [ref(tbl, "31", "11", year), ref(tbl, "11", "31", year)]


def test_neighbour_end_said_by_its_particle_and_a_verb_of_arriving(res):
    _, answer = ask(res, "2020년 서울에 전입한 사람 중 인천에서 온 사람은 몇 명이야?", "FAKE12_MIGRATE")
    assert cells(answer) == [ref("FAKE12_MIGRATE", "23", "11", "2020")]


def test_neighbour_one_name_with_both_roles(res):
    _, answer = ask(res, "2020년 서울에 살면서 서울로 통근하는 사람은?", "FAKE12_COMMUTE")
    assert cells(answer) == [ref("FAKE12_COMMUTE", "11", "11", "2020")]
