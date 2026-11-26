"""Loop 20:

- 0 (debt): over two or three tables, a whole's stem ('가구수' of '가구수-계') whose place starts inside another table's
  matched name and runs past its end ("맞벌이가구수") is the noun of what is counted (multi_rules._measure_noun); a
  describing word read by its class, a syllable closed by ㄴ (judge/question.py _ADNOMINAL_END, _MEASURE_BETWEEN,
  _DESCRIBED: "키운", "산", "가진", "자녀가 독립한").
- G: a count of places after the noun a ranking word's describing word modifies, before a possessive ("가장 많은 시도
  세 곳의 인구"): the first n places (shared/patterns.py rank_places, _count_after_candidates).
- E: a lower end over numbered names of any counting word, said by a digit or a native number word ("2명 이상",
  "둘 이상", "두 대 이상", "1대 초과"): their sum (analyze/rules.py _counted_range).

Neighbours for every condition. The tables are made up (가상 데이터); every test checks the values and the cells.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2021", "2022", "2023")
SIDO = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"), ("23", "인천광역시"))


def catalog(title):
    return {"survey_name": "가상 통계조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


def item_rows(items, base, unit="가구", whole=True, regions=SIDO, years=YEARS):
    """One category of items by region; with whole, the first item is the sum of the others."""
    rows = []
    for y, year in enumerate(years):
        for r, (rc, rl) in enumerate(regions):
            parts = [base + 37 * r + 11 * y + 19 * i + (r * i + y) % 5 for i in range(1, len(items) + (0 if whole else 1))]
            for i, (ic, il) in enumerate(items):
                if whole:
                    v = sum(parts) if i == 0 else parts[i - 1]
                else:
                    v = parts[i]
                if rc == "00":
                    v *= 9
                rows.append({"period": year, "item_id": ic, "item_name": il, "unit": unit, "dim1_name": "시도별",
                             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v})
    return rows


def counted_items(thing, unit):
    return (("T0", "가구수-계"), ("T1", f"{thing}-없음"), ("T2", f"1{unit}"), ("T3", f"2{unit}"),
            ("T4", f"3{unit} 이상"), ("T9", "미상"))


CHILD, CAR = counted_items("자녀", "명"), counted_items("자동차", "대")
WORK = (("W1", "맞벌이가구"), ("W2", "외벌이가구"))
ONE = (("S1", "1인가구"), ("S2", "다인가구"))

SPECS = {
    "F20_CHILD": (lambda: item_rows(CHILD, 300), catalog("가상 시도별 자녀수별 가구")),
    "F20_CAR": (lambda: item_rows(CAR, 200), catalog("가상 시도별 자동차보유대수별 가구")),
    "F20_WORK": (lambda: item_rows(WORK, 500, whole=False), catalog("가상 시도별 맞벌이 가구")),
    "F20_ONE": (lambda: item_rows(ONE, 700, whole=False), catalog("가상 시도별 1인 가구")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop20") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, *tbls):
    got = calculate(question, [res[t] for t in tbls])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def status(res, question, *tbls):
    return calculate(question, [res[t] for t in tbls]).status


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


def refs_in(facts, fact):
    """The cells a computed fact was made of, following computed facts down to the cells."""
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += [f.source_ref] if f.source_ref else refs_in(facts, f)
    return sorted(out)


def ref(tbl, *codes, year="2023"):
    return "|".join([tbl, *codes, year])


def value(tbl, *codes, year="2023"):
    for r in ROWS[tbl]:
        if r["period"] == year and [r["item_id"], r["dim1_code"]] == list(codes):
            return r["value"]
    raise KeyError((tbl, codes, year))


CH, CR, WK, ON = "F20_CHILD", "F20_CAR", "F20_WORK", "F20_ONE"


# ---------------------------------------------------------------- 0 (debt)


@pytest.mark.parametrize("question, tbls, year, region, items", [
    ("2023년 서울의 맞벌이가구수, 자녀가 없는 가구수, 자동차가 1대인 가구수를 알려줘", (WK, CH, CR), "2023", "11",
     ("W1", "T1", "T2")),
    ("2022년 부산 자녀 1명 가구수와 1인가구수와 자동차 없는 가구수는?", (CH, ON, CR), "2022", "21",
     ("T2", "S1", "T1")),
    ("2021년 대구의 외벌이가구수와 자녀가 2명인 가구수를 알려줘", (CH, WK), "2021", "22", ("T3", "W2")),
])
def test_0_stem_written_on_after_another_tables_name(res, question, tbls, year, region, items):
    _, answer = ask(res, question, *tbls)
    assert cells(answer) == sorted(ref(t, i, region, year=year) for t, i in zip(tbls, items))


@pytest.mark.parametrize("question, year, region, items", [
    ("2023년 인천에서 자녀가 1명 있는 가구의 가구수와 자동차를 2대 가진 가구의 가구수는?", "2023", "23", ("T2", "T3")),
    ("2022년 부산의 자녀가 없는 가구수와 자동차를 1대 산 가구수는?", "2022", "21", ("T1", "T2")),
])
def test_0_describing_word_closed_by_n(res, question, year, region, items):
    _, answer = ask(res, question, CH, CR)
    assert cells(answer) == sorted([ref(CH, items[0], region, year=year), ref(CR, items[1], region, year=year)])


@pytest.mark.parametrize("question, region, year, item", [
    ("2023년 서울의 자녀가 1명 있는 가구의 가구수는?", "11", "2023", "T2"),
    ("2022년 대구의 자동차를 2대 가진 가구수는?", "22", "2022", "T3"),
])
def test_0_one_table_describing_word_closed_by_n(res, question, region, year, item):
    tbl = CH if "자녀" in question else CR
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, year=year)]


# neighbours: what must stay as it was


def test_0_neighbour_whole_and_part_listed(res):
    _, answer = ask(res, "2023년 서울의 가구수와 자녀가 없는 가구수를 알려줘", CH)
    assert cells(answer) == sorted([ref(CH, "T0", "11"), ref(CH, "T1", "11")])


def test_0_neighbour_whole_after_a_region_closed_by_n(res):
    # a region name closed by ㄴ ('인천') before the whole: not a describing word, the whole stays the target
    _, answer = ask(res, "2023년 인천 가구수는?", CH)
    assert cells(answer) == [ref(CH, "T0", "23")]


def test_0_neighbour_whole_said_on_its_own_stays_open(res):
    assert status(res, "2023년 서울의 가구수와 자녀가 없는 가구수, 자동차가 없는 가구수를 알려줘", CH, CR) != "ok"


def test_0_neighbour_names_without_the_stem(res):
    _, answer = ask(res, "2023년 서울의 맞벌이가구와 1인가구를 알려줘", WK, ON)
    assert cells(answer) == sorted([ref(WK, "W1", "11"), ref(ON, "S1", "11")])


def test_0_neighbour_parts_difference(res):
    facts, answer = ask(res, "2023년 서울의 자녀가 없는 가구수는 자동차가 없는 가구수보다 몇 가구 많아?", CH, CR)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(CH, "T1", "11"), ref(CR, "T1", "11")])
    assert got.value == value(CH, "T1", "11") - value(CR, "T1", "11")


# ---------------------------------------------------------------- G: a count of places after the candidates' noun

SPECS.update({
    "F20_POP": (lambda: item_rows((("P1", "인구"),), 1000, unit="명", whole=False), catalog("가상 시도별 인구")),
    "F20_BIRTH": (lambda: [dict(r, value=v, value_raw=str(v))
                           for r in item_rows((("B1", "출생아수"),), 400, unit="명", whole=False)
                           for v in [r["value"] * (1 if r["dim1_code"] != "22" else 3)]],
                  catalog("가상 시도별 출생")),
})
ROWS.update({t: SPECS[t][0]() for t in ("F20_POP", "F20_BIRTH")})


def ranked(tbl, item, year, top, largest=True):
    vals = sorted(((value(tbl, item, rc, year=year), rc) for rc, _ in SIDO if rc != "00"), reverse=largest)
    return sorted(ref(tbl, item, rc, year=year) for _, rc in vals[:top])


@pytest.mark.parametrize("question, tbl, item, year, top, largest", [
    ("2023년 인구가 가장 많은 시도 세 곳의 인구를 각각 알려줘", "F20_POP", "P1", "2023", 3, True),
    ("2022년 출생아수가 제일 적은 지역 2곳의 출생아수는 각각 얼마야?", "F20_BIRTH", "B1", "2022", 2, False),
    ("2021년 인구가 가장 적은 시도 두 곳의 인구는 각각?", "F20_POP", "P1", "2021", 2, False),
])
def test_g_count_after_the_candidates_noun_before_possessive(res, question, tbl, item, year, top, largest):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == ranked(tbl, item, year, top, largest)


def test_g_neighbour_pool_said_before_the_ranking_word(res):
    # a count before the ranking word names the pool: the first place alone, as before
    _, answer = ask(res, "2023년 4개 시도 중 인구가 가장 많은 곳은?", "F20_POP")
    assert cells(answer) == ranked("F20_POP", "P1", "2023", 1)


def test_g_neighbour_count_with_topic_particle(res):
    _, answer = ask(res, "2023년 인구가 가장 많은 시도 세 곳은?", "F20_POP")
    assert cells(answer) == ranked("F20_POP", "P1", "2023", 3)


# ---------------------------------------------------------------- E: a lower end over numbered names of a counting word


@pytest.mark.parametrize("question, tbl, region, year, items", [
    ("2023년 서울의 자녀가 2명 이상인 가구수는?", CH, "11", "2023", ("T3", "T4")),
    ("2022년 부산에서 자녀가 둘 이상인 가구는 몇 가구야?", CH, "21", "2022", ("T3", "T4")),
    ("2023년 대구의 자동차를 1대 이상 가진 가구수는?", CR, "22", "2023", ("T2", "T3", "T4")),
    ("2021년 인천에서 자동차가 두 대 이상인 가구수 알려줘", CR, "23", "2021", ("T3", "T4")),
    ("2023년 서울의 자동차가 1대 초과인 가구수는?", CR, "11", "2023", ("T3", "T4")),
])
def test_e_lower_end_of_counted_names(res, question, tbl, region, year, items):
    facts, answer = ask(res, question, tbl)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted(ref(tbl, i, region, year=year) for i in items)
    assert got.value == sum(value(tbl, i, region, year=year) for i in items)


def test_e_neighbour_one_name_said_with_a_number_word(res):
    _, answer = ask(res, "2023년 서울의 자녀가 두 명인 가구수는?", CH)
    assert cells(answer) == [ref(CH, "T3", "11")]


def test_e_neighbour_the_open_name_itself(res):
    _, answer = ask(res, "2023년 서울의 자녀가 3명 이상인 가구수는?", CH)
    assert cells(answer) == [ref(CH, "T4", "11")]


def test_e_neighbour_another_name_named_on_its_own(res):
    _, answer = ask(res, "2023년 서울의 자녀가 1명인 가구수와 3명 이상인 가구수를 알려줘", CH)
    assert cells(answer) == sorted([ref(CH, "T2", "11"), ref(CH, "T4", "11")])
