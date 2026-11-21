"""Loop 3, one table, rankings beyond the first place (analyze/rules.py _extreme_plan, analyze/blocks.py,
compute/plan_engine.py RankStep.report, shared/patterns.py rank_places, ASKS_PERIOD):

- a place other than the first ("세 번째로 많은", "2위", "두 번째로 적은"): the n-th candidate's cell only;
- the first n places ("상위 3개", "가장 적은 두 곳", "1~3위"): those n cells, in rank order;
- candidates ranked by their change between two periods (the amount, or the rate): the picked candidate's two cells
  and its change, a decline ranked most negative first;
- the largest and the smallest of a ranking together ("가장 많았던 해와 가장 적었던 해의 차이", "몇 배"): their
  difference or ratio and the two cells, whatever particle follows the period word.

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
YEARS = ("2018", "2019", "2020", "2021", "2022")
REGIONS = {"11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시", "24": "광주광역시",
           "25": "대전광역시"}
POP = {"11": [9700, 9760, 9650, 9500, 9430], "21": [3400, 3390, 3370, 3350, 3320],
       "22": [2450, 2440, 2420, 2400, 2380], "23": [2950, 2955, 2960, 2965, 2970],
       "24": [1460, 1455, 1450, 1445, 1440], "25": [1490, 1480, 1470, 1460, 1450]}
BIRTH = {"11": [58, 53, 47, 45, 42], "21": [19, 17, 15, 14, 13], "22": [14, 13, 11, 10, 10],
         "23": [20, 18, 16, 15, 14], "24": [9, 8, 9, 7, 7], "25": [10, 9, 7, 7, 6]}
IND_YEARS = ("2019", "2020", "2021", "2022")
INDUSTRIES = {"B": "농업", "C": "제조업", "D": "건설업", "E": "서비스업"}
FIRMS = {"B": [120, 118, 115, 110], "C": [400, 410, 405, 420], "D": [250, 260, 280, 275], "E": [900, 950, 980, 1000]}


def pop_rows():
    out = []
    for p_no, period in enumerate(YEARS):
        for item_id, name, values in (("T1", "인구", POP), ("T2", "출생아수", BIRTH)):
            regions = [("00", "전국", sum(v[p_no] for v in values.values()))]
            regions += [(code, REGIONS[code], values[code][p_no]) for code in REGIONS]
            for code, label, v in regions:
                out.append({"period": period, "item_id": item_id, "item_name": name, "unit": "명",
                            "dim1_name": "행정구역별", "dim1_code": code, "dim1_label": label,
                            "value_raw": str(v), "value": v})
    return out


def firm_rows():
    out = []
    for p_no, period in enumerate(IND_YEARS):
        cats = [("A", "계", sum(v[p_no] for v in FIRMS.values()))] + [(c, INDUSTRIES[c], FIRMS[c][p_no]) for c in INDUSTRIES]
        for code, label, v in cats:
            out.append({"period": period, "item_id": "T1", "item_name": "사업체수", "unit": "개",
                        "dim1_name": "산업별", "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop3_rank") / "data"
    return {"FAKE_RANK_POP": load_resources(load_table(write_table(folder, "FAKE_RANK_POP", pop_rows(), CATALOG))),
            "FAKE_RANK_IND": load_resources(load_table(write_table(folder, "FAKE_RANK_IND", firm_rows(), CATALOG)))}


def answer(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [facts[i] for i in got.answer]


def pop_ref(item, code, year):
    return f"FAKE_RANK_POP|{item}|{code}|{year}"


def firm_ref(code, year):
    return f"FAKE_RANK_IND|T1|{code}|{year}"


def cells_of(facts):
    return [f.source_ref for f in facts if f.source_ref]


def derived_of(facts):
    return [f.value for f in facts if not f.source_ref]


# ---------------------------------------------------------------- A-1: the n-th place alone


@pytest.mark.parametrize("question, table, ref", [
    ("2022년 인구가 세 번째로 많은 시도는?", "FAKE_RANK_POP", pop_ref("T1", "23", "2022")),  # 9430, 3320, 2970
    ("2020년 출생아 수 2위 지역은 어디야?", "FAKE_RANK_POP", pop_ref("T2", "23", "2020")),  # 47, 16
    ("2021년 사업체 수가 두 번째로 적은 산업은?", "FAKE_RANK_IND", firm_ref("D", "2021")),  # 115, 280
])
def test_nth_place_is_that_cell_only(res, question, table, ref):
    facts = answer(res, question, table)
    assert cells_of(facts) == [ref]
    assert derived_of(facts) == []


# ---------------------------------------------------------------- A-2: the first n places, in rank order


@pytest.mark.parametrize("question, table, refs", [
    ("2022년 인구 상위 3개 시도 알려줘", "FAKE_RANK_POP",
     [pop_ref("T1", "11", "2022"), pop_ref("T1", "21", "2022"), pop_ref("T1", "23", "2022")]),
    ("2019년 출생아 수가 가장 적은 두 곳은?", "FAKE_RANK_POP",
     [pop_ref("T2", "24", "2019"), pop_ref("T2", "25", "2019")]),  # 8, 9
    ("2020년 사업체 수 1~3위 산업은?", "FAKE_RANK_IND", [firm_ref("E", "2020"), firm_ref("C", "2020"), firm_ref("D", "2020")]),
])
def test_first_places_in_order(res, question, table, refs):
    facts = answer(res, question, table)
    assert cells_of(facts) == refs
    assert derived_of(facts) == []


# ---------------------------------------------------------------- A-3: ranked by the change between two periods


@pytest.mark.parametrize("question, table, refs, change", [
    # amounts: 인천 is the only region that grew (+20), 서울 fell the most (-270)
    ("2018년부터 2022년까지 인구가 가장 많이 늘어난 시도는?", "FAKE_RANK_POP",
     [pop_ref("T1", "23", "2018"), pop_ref("T1", "23", "2022")], 20),
    ("2018년과 2022년 사이 인구가 가장 많이 줄어든 시도는?", "FAKE_RANK_POP",
     [pop_ref("T1", "11", "2018"), pop_ref("T1", "11", "2022")], -270),
    # a rate: 2019 -> 2021 births, 광주 7/8 - 1 = -12.5% is the highest
    ("2019년 대비 2021년 출생아 수 증가율이 가장 높은 지역은?", "FAKE_RANK_POP",
     [pop_ref("T2", "24", "2019"), pop_ref("T2", "24", "2021")], -12.5),
    # another table and category: 서비스업 900 -> 1000
    ("2019년에서 2022년 사이 사업체 수가 가장 크게 증가한 산업은?", "FAKE_RANK_IND",
     [firm_ref("E", "2019"), firm_ref("E", "2022")], 100),
])
def test_ranked_by_change(res, question, table, refs, change):
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted(refs)
    assert derived_of(facts) == [pytest.approx(change, abs=0.005)]


# ---------------------------------------------------------------- A-4: the largest and the smallest together


@pytest.mark.parametrize("question, table, refs, value", [
    # 서울 인구: the most 2019 (9760), the fewest 2022 (9430)
    ("서울 인구가 가장 많았던 해와 가장 적었던 해의 차이는?", "FAKE_RANK_POP",
     [pop_ref("T1", "11", "2019"), pop_ref("T1", "11", "2022")], 330),
    # 부산 출생아: 2018 (19) over 2022 (13)
    ("부산 출생아 수가 가장 많았던 연도는 가장 적었던 연도의 몇 배야?", "FAKE_RANK_POP",
     [pop_ref("T2", "21", "2018"), pop_ref("T2", "21", "2022")], 19 / 13),
    # a category at one period: 서비스업 1000 - 농업 110
    ("2022년 사업체 수가 가장 많은 산업과 가장 적은 산업의 차이는?", "FAKE_RANK_IND",
     [firm_ref("E", "2022"), firm_ref("B", "2022")], 890),
])
def test_largest_and_smallest_combined(res, question, table, refs, value):
    facts = answer(res, question, table)
    assert sorted(cells_of(facts)) == sorted(refs)
    assert derived_of(facts) == [pytest.approx(value, abs=0.005)]


# ---------------------------------------------------------------- the first place stays as it was


@pytest.mark.parametrize("question, table, ref", [
    ("2022년 인구가 가장 많은 시도는?", "FAKE_RANK_POP", pop_ref("T1", "11", "2022")),
    ("2021년 사업체 수가 가장 적은 산업은?", "FAKE_RANK_IND", firm_ref("B", "2021")),
])
def test_first_place_unchanged(res, question, table, ref):
    facts = answer(res, question, table)
    assert cells_of(facts)[:1] == [ref]
