"""Loop 5, A (analyze/rules.py _twin_axes, _flows, _pairs_plan; shared/patterns.py ORIGIN_*, DEST_*, REVERSE): a
table whose two categories hold the same names in two roles, where a flow starts and where it ends (전출지별 ×
전입지별, 출발지별 × 도착지별). A name the question gives a start role (에서, 부터, 떠난) goes to the start category, a
name given an end role (으로, 로, 에, 들어온) to the end category; a side no name was given is its total. Two flows
are compared (the flow before "보다" is the reference), a flow named as the first one turned around ("그 반대") is the
first one swapped, and a share is the flow over the same flow from (or to) everywhere.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
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
YEARS = ("2021", "2022", "2023")


def flow(o_no: int, d_no: int, y_no: int, base: int) -> int:
    return base * (o_no + 1) + 37 * d_no * d_no + 13 * y_no * (o_no + 2 * d_no) + 7 * (o_no == d_no)


def flow_value(o: str, d: str, y: str, base: int) -> int:
    codes = [c for c, _ in REGIONS[1:]]
    y_no = YEARS.index(y)
    origins = codes if o == "00" else [o]
    dests = codes if d == "00" else [d]
    return sum(flow(codes.index(a), codes.index(b), y_no, base) for a in origins for b in dests)


def flow_rows(start: str, end: str, item: str, base: int):
    out = []
    for y in YEARS:
        for o, o_label in REGIONS:
            for d, d_label in REGIONS:
                v = flow_value(o, d, y, base)
                out.append({"period": y, "item_id": "T1", "item_name": item, "unit": "명",
                            "dim1_name": start, "dim1_code": o, "dim1_label": o_label,
                            "dim2_name": end, "dim2_code": d, "dim2_label": d_label,
                            "value_raw": str(v), "value": v, "prd_se": "Y"})
    return out


SPECS = {
    "FAKE5_MIGRATE": lambda: flow_rows("전출지별", "전입지별", "이동자수", 300),
    "FAKE5_TRAVEL": lambda: flow_rows("출발지별", "도착지별", "여객수", 900),
}
BASE = {"FAKE5_MIGRATE": 300, "FAKE5_TRAVEL": 900}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop5_flows") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), CATALOG))) for t, make in SPECS.items()}


def ask(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [facts[i] for i in got.answer]


def ref(table, o, d, y):
    return f"{table}|T1|{o}|{d}|{y}"


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def derived(answer):
    return [f for f in answer if f.derivation]


@pytest.mark.parametrize("question, table, o, d, y", [
    ("2022년 서울에서 부산으로 이동한 사람은 몇 명이야?", "FAKE5_MIGRATE", "11", "21", "2022"),
    ("2021년 인천에서 대구로 옮긴 인구 알려줘", "FAKE5_MIGRATE", "23", "22", "2021"),
    ("2021년 대구에서 출발해 부산에 도착한 여객 수는?", "FAKE5_TRAVEL", "22", "21", "2021"),
    # one name and its role: the other side is its total
    ("2023년 대구로 전입한 사람 수는?", "FAKE5_MIGRATE", "00", "22", "2023"),
    ("2021년 부산에서 전출한 인구는 몇 명?", "FAKE5_MIGRATE", "21", "00", "2021"),
    ("2022년 서울을 떠난 여객 수 알려줘", "FAKE5_TRAVEL", "11", "00", "2022"),
])
def test_one_flow_is_one_cell(res, question, table, o, d, y):
    answer = ask(res, question, table)
    assert cells(answer) == [ref(table, o, d, y)]
    assert not derived(answer)


@pytest.mark.parametrize("question, table, first, second, y, reference", [
    ("2023년 서울에서 대구로 간 인구는 대구에서 서울로 간 인구보다 몇 명 많아?", "FAKE5_MIGRATE",
     ("11", "22"), ("22", "11"), "2023", 1),
    ("2021년 부산에서 서울로 이동한 사람보다 서울에서 부산으로 이동한 사람이 몇 명 더 많아?", "FAKE5_MIGRATE",
     ("21", "11"), ("11", "21"), "2021", 0),
    ("2022년 인천에서 부산으로 간 여객 수는 그 반대보다 몇 명 많아?", "FAKE5_TRAVEL",
     ("23", "21"), ("21", "23"), "2022", 1),
])
def test_two_flows_compared(res, question, table, first, second, y, reference):
    answer = ask(res, question, table)
    assert cells(answer) == sorted([ref(table, *first, y), ref(table, *second, y)])
    (got,) = derived(answer)
    a, b = flow_value(*first, y, BASE[table]), flow_value(*second, y, BASE[table])
    assert got.value == pytest.approx(a - b if reference == 1 else b - a)


@pytest.mark.parametrize("question, table, part, whole, y", [
    ("2023년 서울로 전입한 사람 중 부산에서 온 사람의 비중은?", "FAKE5_MIGRATE", ("21", "11"), ("00", "11"), "2023"),
    ("2022년 대구에서 출발한 여객 가운데 인천에 도착한 여객이 차지하는 비중은?", "FAKE5_TRAVEL",
     ("22", "23"), ("22", "00"), "2022"),
])
def test_a_flow_as_a_share_of_its_side(res, question, table, part, whole, y):
    answer = ask(res, question, table)
    assert cells(answer) == sorted([ref(table, *part, y), ref(table, *whole, y)])
    (got,) = derived(answer)
    want = flow_value(*part, y, BASE[table]) / flow_value(*whole, y, BASE[table]) * 100
    assert got.value == pytest.approx(want, abs=0.01)
