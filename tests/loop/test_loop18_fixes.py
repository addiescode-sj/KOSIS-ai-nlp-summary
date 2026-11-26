"""Loop 18:

- 0 (debt): over two or three tables, a whole matched by its stem only ('가구수' of '가구수-계') said at every place as
  the noun of what is counted after a matched name, beside one other item of its table, leaves that item the
  table's (multi_rules._assign_items, _measure_noun); a name matched by parts said apart whose place runs over another
  table's place to its end is the other table's (_closes_other); one table: a noun and the possessive particle may
  stand between a part and the whole's stem (judge/question.py _MEASURE_BETWEEN).
- A: in a table counting a couple's events (marriages, divorces), a sex said by the spouse's word ('남편', '아내':
  judge/question.py spouse_forms, shared SPOUSE_ROLES, COUPLE_EVENTS).
- B: a calculation between two tables at two periods and its change (multi_rules._operation, _gap_change).
- C: a percent asked with a word that puts parts together (analyze/rules.py _percent_of_other, _asks_share).
- D: a ranking repeated for each named target over another category (analyze/rules.py).
- E: a copula and a bound noun ('인 경우', '인 쪽') between a name and the word naming the reference
  (analyze/roles.py _skip_names, shared BOUND_NOUNS).

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


# ---------------------------------------------------------------- tables

# 0: items a whole ('가구수-계'), '<thing>-없음', numbers with a counting word, an open name, an unknown.
def counted_items(thing, unit):
    return (("T0", "가구수-계"), ("T1", f"{thing}-없음"), ("T2", f"1{unit}"), ("T3", f"2{unit}"),
            ("T4", f"3{unit} 이상"), ("T9", "미상"))


def counted_rows(items, base):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO):
            parts = [base + 37 * r + 11 * y + 19 * i + (r + i) % 4 for i in range(1, len(items))]
            for i, (ic, il) in enumerate(items):
                v = sum(parts) if i == 0 else parts[i - 1]
                rows.append({"period": year, "item_id": ic, "item_name": il, "unit": "가구", "dim1_name": "시도별",
                             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v})
    return rows


# two tables of one item each, the one's name ending with the other's ('이혼건수', '협의이혼건수')
def one_item_rows(item, name, unit, base):
    return [{"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별",
             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v}
            for y, year in enumerate(YEARS) for r, (rc, rl) in enumerate(SIDO)
            for v in [base + 53 * r + 29 * y + (r * y) % 7]]


# A: a couple's events counted by each spouse: the two sexes (no total) by a category of the party's kind, one item.
SEXES = (("1", "남자"), ("2", "여자"))


def couple_rows(item, name, other_name, others, base, sexes=SEXES):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO[:3]):
            for s, (sc, sl) in enumerate(sexes):
                parts = [base + 41 * r + 13 * y + 7 * s + 31 * k + (r + s + k) % 3 for k in range(1, len(others))]
                for k, (oc, ol) in enumerate(others):
                    v = sum(parts) if k == 0 else parts[k - 1]
                    rows.append({"period": year, "item_id": item, "item_name": name, "unit": "건",
                                 "dim1_name": "시도별", "dim1_code": rc, "dim1_label": rl,
                                 "dim2_name": "성별", "dim2_code": sc, "dim2_label": sl,
                                 "dim3_name": other_name, "dim3_code": oc, "dim3_label": ol,
                                 "value_raw": str(v), "value": v})
    return rows


KINDS = (("0", "계"), ("1", "초혼"), ("2", "재혼"))
AGES = (("0", "계"), ("1", "20-29세"), ("2", "30-39세"), ("3", "40세이상"))

# C, D, E: people by region, sex (with a total) and age (with a total), one item; a second table of workers by
# region and industry for D's ranking.
POP_SEX = (("0", "계"), ("1", "남자"), ("2", "여자"))
POP_AGE = (("0", "계"), ("1", "0-14세"), ("2", "15-64세"), ("3", "65세이상"))


def people_rows(item, name, unit, base):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO):
            grid = {}
            for s in (1, 2):
                for a in (1, 2, 3):
                    grid[s, a] = base * (3 if a == 2 else 1) + 97 * r + 13 * y + 29 * s + 61 * a + (r * a + s) % 5
            for sc, sl in POP_SEX:
                for ac, al in POP_AGE:
                    ss = (1, 2) if sc == "0" else (int(sc),)
                    aa = (1, 2, 3) if ac == "0" else (int(ac),)
                    v = sum(grid[s, a] for s in ss for a in aa)
                    if rc == "00":
                        v = v * 9
                    rows.append({"period": year, "item_id": item, "item_name": name, "unit": unit,
                                 "dim1_name": "시도별", "dim1_code": rc, "dim1_label": rl,
                                 "dim2_name": "성별", "dim2_code": sc, "dim2_label": sl,
                                 "dim3_name": "연령별", "dim3_code": ac, "dim3_label": al,
                                 "value_raw": str(v), "value": v})
    return rows


CHILD, CAR, PET = counted_items("자녀", "명"), counted_items("자동차", "대"), counted_items("반려동물", "마리")
SPECS = {
    "F18_CHILD": (lambda: counted_rows(CHILD, 300), catalog("가상 시도별 자녀수별 가구")),
    "F18_CAR": (lambda: counted_rows(CAR, 200), catalog("가상 시도별 자동차보유대수별 가구")),
    "F18_PET": (lambda: counted_rows(PET, 100), catalog("가상 시도별 반려동물수별 가구")),
    "F18_DIV": (lambda: one_item_rows("D1", "이혼건수", "건", 800), catalog("가상 시도별 이혼")),
    "F18_DIV2": (lambda: one_item_rows("D2", "협의이혼건수", "건", 500), catalog("가상 시도별 협의이혼")),
    # B: three tables of one metric each by region
    "F18_BIRTH": (lambda: one_item_rows("B1", "출생아수", "명", 2000), catalog("가상 시도별 출생")),
    "F18_DEAD": (lambda: [dict(r, value=v, value_raw=str(v)) for r in one_item_rows("X1", "사망자수", "명", 1900)
                          for v in [r["value"] + (300 if r["period"] == "2023" else 0)]],
                 catalog("가상 시도별 사망")),
    "F18_WED": (lambda: one_item_rows("W1", "혼인건수", "건", 1500), catalog("가상 시도별 혼인")),
    "F18_MAR": (lambda: couple_rows("M1", "혼인건수", "혼인종류별", KINDS, 400),
                catalog("가상 시도별 성별 혼인종류별 혼인")),
    "F18_DIVAGE": (lambda: couple_rows("V1", "이혼건수", "연령별", AGES, 150),
                   catalog("가상 시도별 성별 연령별 이혼")),
    # not a couple's event: a death counted by sex and age
    "F18_DEATH": (lambda: couple_rows("X1", "사망자수", "연령별", AGES, 900), catalog("가상 시도별 성별 연령별 사망")),
    "F18_POP": (lambda: people_rows("P1", "인구", "명", 1000), catalog("가상 시도별 성별 연령별 인구")),
    # a marital status splits people: no couple's event counted
    "F18_STATUS": (lambda: couple_rows("S1", "인구", "혼인상태별", (("0", "계"), ("1", "미혼"), ("2", "유배우")), 700),
                   catalog("가상 시도별 성별 혼인상태별 인구")),
    # the spouse's word already a category's name
    "F18_HUSB": (lambda: [dict(r, dim3_name="남편 연령별") for r in couple_rows("H1", "혼인건수", "x", AGES, 600)],
                 catalog("가상 시도별 성별 남편 연령별 혼인")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop18") / "data"
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
        keys = [r["item_id"], r["dim1_code"], *([r["dim2_code"]] if "dim2_code" in r else []),
                *([r["dim3_code"]] if "dim3_code" in r else [])]
        if r["period"] == year and keys == list(codes):
            return r["value"]
    raise KeyError((tbl, codes, year))


CH, CR, PT = "F18_CHILD", "F18_CAR", "F18_PET"


# ---------------------------------------------------------------- 0 (debt)


@pytest.mark.parametrize("question, year, region, items", [
    ("2023년 서울의 자녀가 없는 가구수, 자동차가 1대인 가구수, 반려동물이 2마리인 가구수를 알려줘", "2023", "11",
     ("T1", "T2", "T3")),
    ("2022년 부산 자녀 1명 가구수와 자동차 없는 가구수와 반려동물 1마리 가구수는?", "2022", "21", ("T2", "T1", "T2")),
    ("2021년 대구에서 자녀가 2명인 가구의 가구수, 자동차가 없는 가구의 가구수, 반려동물이 없는 가구의 가구수는?",
     "2021", "22", ("T3", "T1", "T1")),
])
def test_0_three_tables_part_per_table(res, question, year, region, items):
    _, answer = ask(res, question, CH, CR, PT)
    assert cells(answer) == sorted(ref(t, i, region, year=year) for t, i in zip((CH, CR, PT), items))


def test_0_two_tables_part_per_table(res):
    _, answer = ask(res, "2023년 인천에서 자녀가 없는 가구의 가구수와 자동차가 1대인 가구의 가구수를 알려줘", CH, CR)
    assert cells(answer) == sorted([ref(CH, "T1", "23"), ref(CR, "T2", "23")])


def test_0_apart_name_does_not_run_over_another_table(res):
    _, answer = ask(res, "2023년 서울의 자동차가 2대인 가구수와 반려동물이 없는 가구수를 알려줘", CR, PT)
    assert cells(answer) == sorted([ref(CR, "T3", "11"), ref(PT, "T1", "11")])


@pytest.mark.parametrize("question, region, year, item", [
    ("2023년 서울의 자녀가 없는 가구의 가구수는?", "11", "2023", "T1"),
    ("2022년 부산의 자녀가 두 명인 집의 가구수는?", "21", "2022", "T3"),
])
def test_0_one_table_noun_and_possessive_between(res, question, region, year, item):
    _, answer = ask(res, question, CH)
    assert cells(answer) == [ref(CH, item, region, year=year)]


def test_0_neighbour_whole_and_part_listed(res):
    _, answer = ask(res, "2023년 서울의 가구수와 자녀가 없는 가구수를 알려줘", CH)
    assert cells(answer) == sorted([ref(CH, "T0", "11"), ref(CH, "T1", "11")])


def test_0_neighbour_share_of_the_whole(res):
    facts, answer = ask(res, "2023년 서울 가구수 중 자녀가 없는 가구의 비중은?", CH)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(CH, "T0", "11"), ref(CH, "T1", "11")])
    assert got.value == pytest.approx(round(100 * value(CH, "T1", "11") / value(CH, "T0", "11"), 2), abs=0.01)


def test_0_neighbour_whole_only_over_tables(res):
    _, answer = ask(res, "2023년 서울의 가구수를 표마다 알려줘", CH, CR)
    assert cells(answer) == sorted([ref(CH, "T0", "11"), ref(CR, "T0", "11")])


def test_0_neighbour_whole_said_on_its_own_stays_open(res):
    # the whole said first on its own: not a noun of what is counted, the table's item stays open as before
    assert status(res, "2023년 서울의 가구수와 자녀가 없는 가구수, 자동차가 없는 가구수를 알려줘", CH, CR) != "ok"


def test_0_neighbour_parts_difference(res):
    facts, answer = ask(res, "2023년 서울의 자녀가 없는 가구수는 자동차가 없는 가구수보다 몇 가구 많아?", CH, CR)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(CH, "T1", "11"), ref(CR, "T1", "11")])
    assert got.value == value(CH, "T1", "11") - value(CR, "T1", "11")


@pytest.mark.parametrize("question", [
    "2023년 서울의 협의이혼건수와 이혼건수를 알려줘",
    "2023년 서울의 이혼건수와 협의이혼건수는?",
])
def test_0_neighbour_long_name_ending_with_a_short_one_listed(res, question):
    _, answer = ask(res, question, "F18_DIV", "F18_DIV2")
    assert cells(answer) == sorted([ref("F18_DIV", "D1", "11"), ref("F18_DIV2", "D2", "11")])


def test_0_neighbour_long_name_ending_with_a_short_one_difference(res):
    facts, answer = ask(res, "2023년 서울의 협의이혼건수는 이혼건수보다 몇 건 적어?", "F18_DIV", "F18_DIV2")
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref("F18_DIV", "D1", "11"), ref("F18_DIV2", "D2", "11")])
    assert got.value == value("F18_DIV", "D1", "11") - value("F18_DIV2", "D2", "11")


def test_0_neighbour_long_name_ending_with_a_short_one_changes(res):
    facts, answer = ask(res, "2021년부터 2023년까지 부산의 협의이혼건수와 이혼건수는 각각 얼마나 변했어?",
                        "F18_DIV", "F18_DIV2")
    used = sorted({r for f in computed(answer) for r in refs_in(facts, f)})
    assert used == sorted([ref(t, i, "21", year=y) for t, i in (("F18_DIV", "D1"), ("F18_DIV2", "D2"))
                           for y in ("2021", "2023")])


# ---------------------------------------------------------------- A


MAR, DVA = "F18_MAR", "F18_DIVAGE"


@pytest.mark.parametrize("question, tbl, region, sex, other, year", [
    ("2023년 서울에서 남편이 재혼인 혼인 건수는?", MAR, "11", "1", "2", "2023"),
    ("2022년 부산의 아내가 초혼인 혼인은 몇 건이야?", MAR, "21", "2", "1", "2022"),
    ("2023년 서울 신부가 재혼인 혼인 건수 알려줘", MAR, "11", "2", "2", "2023"),
    ("2021년 부산에서 남편이 30-39세인 이혼 건수는?", DVA, "21", "1", "2", "2021"),
])
def test_a_sex_said_by_the_spouse(res, question, tbl, region, sex, other, year):
    item = "M1" if tbl == MAR else "V1"
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region, sex, other, year=year)]


def test_a_two_spouses_difference(res):
    facts, answer = ask(res, "2023년 서울에서 남편이 재혼인 혼인과 아내가 재혼인 혼인의 차이는?", MAR)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(MAR, "M1", "11", "1", "2"), ref(MAR, "M1", "11", "2", "2")])
    assert abs(got.value) == abs(value(MAR, "M1", "11", "1", "2") - value(MAR, "M1", "11", "2", "2"))


def test_a_neighbour_sex_name_as_it_is(res):
    _, answer = ask(res, "2023년 서울에서 남자가 재혼인 혼인 건수는?", MAR)
    assert cells(answer) == [ref(MAR, "M1", "11", "1", "2")]


def test_a_neighbour_not_a_couples_event(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res["F18_DEATH"].table)
    assert [m for m in match_labels(index, "2023년 서울에서 남편이 40세이상인 사망자수") if m.axis == "성별"] == []


def test_a_neighbour_marital_status_splits_people(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res["F18_STATUS"].table)
    assert [m for m in match_labels(index, "2023년 서울에서 아내가 유배우인 인구") if m.axis == "성별"] == []


def test_a_neighbour_spouse_word_in_a_category_name(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res["F18_HUSB"].table)
    assert [m for m in match_labels(index, "2023년 서울에서 남편이 30-39세인 혼인 건수") if m.axis == "성별"] == []


def test_a_neighbour_no_sex_category(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res["F18_DIV"].table)
    assert {m.code for m in match_labels(index, "2023년 서울의 남편 이혼건수")} == {"11", "D1"}


# ---------------------------------------------------------------- B


BR, DE, WE = "F18_BIRTH", "F18_DEAD", "F18_WED"


def gap_change(facts, answer):
    """The last computed answer fact (the change) and the cells it was made of."""
    got = computed(answer)[-1]
    return got, refs_in(facts, got)


@pytest.mark.parametrize("question, region, early, late", [
    ("서울의 출생아 수와 사망자 수의 차이는 2021년에 비해 2023년에 얼마나 변했어?", "11", "2021", "2023"),
    ("부산 출생아 수와 사망자 수 격차가 2022년부터 2023년 사이 얼마나 벌어졌어?", "21", "2022", "2023"),
])
def test_b_gap_between_tables_changed(res, question, region, early, late):
    facts, answer = ask(res, question, BR, DE)
    got, used = gap_change(facts, answer)
    assert used == sorted(ref(t, i, region, year=y) for t, i in ((BR, "B1"), (DE, "X1")) for y in (early, late))
    # one direction at both periods: the table larger at the later period first
    b = {y: value(BR, "B1", region, year=y) for y in (early, late)}
    d = {y: value(DE, "X1", region, year=y) for y in (early, late)}
    sign = 1 if b[late] >= d[late] else -1
    assert got.value == sign * ((b[late] - d[late]) - (b[early] - d[early]))
    assert len(cells(answer)) == 4


@pytest.mark.parametrize("question, region, fewer", [
    ("부산에서 출생아가 사망자보다 많은 수는 2021년보다 2023년에 얼마나 줄었어?", "21", False),
    ("대구는 사망자 수가 출생아 수보다 적은 정도가 2021년과 2023년 사이에 얼마나 달라졌어?", "22", True),
])
def test_b_reference_table_difference_changed(res, question, region, fewer):
    facts, answer = ask(res, question, BR, DE)
    got, used = gap_change(facts, answer)
    assert used == sorted(ref(t, i, region, year=y) for t, i in ((BR, "B1"), (DE, "X1")) for y in ("2021", "2023"))
    gap = {y: value(BR, "B1", region, year=y) - value(DE, "X1", region, year=y) for y in ("2021", "2023")}
    assert got.value == gap["2023"] - gap["2021"]


@pytest.mark.parametrize("question, region", [
    ("서울의 혼인 1건당 출생아 수는 2021년과 2023년 사이 얼마나 변했어?", "11"),
    ("인천의 혼인 건당 출생아 수가 2022년에 비해 2023년에 얼마나 달라졌어?", "23"),
])
def test_b_per_unit_between_tables_changed(res, question, region):
    facts, answer = ask(res, question, BR, WE)
    got, used = gap_change(facts, answer)
    early = "2021" if "2021" in question else "2022"
    assert used == sorted(ref(t, i, region, year=y) for t, i in ((BR, "B1"), (WE, "W1")) for y in (early, "2023"))
    per = {y: value(BR, "B1", region, year=y) / value(WE, "W1", region, year=y) for y in (early, "2023")}
    assert got.value == pytest.approx(per["2023"] - per[early], abs=0.011)


def test_b_neighbour_each_tables_change(res):
    facts, answer = ask(res, "서울의 출생아 수와 사망자 수는 2021년보다 2023년에 얼마나 늘었어?", BR, DE)
    got = computed(answer)
    assert [refs_in(facts, f) for f in got] == [
        sorted([ref(BR, "B1", "11", year="2021"), ref(BR, "B1", "11")]),
        sorted([ref(DE, "X1", "11", year="2021"), ref(DE, "X1", "11")])]


def test_b_neighbour_two_periods_without_change(res):
    facts, answer = ask(res, "2021년과 2023년 서울의 혼인 1건당 출생아 수는?", BR, WE)
    assert len(computed(answer)) == 2
    assert all(f.value != 0 and "/" in f.derivation for f in computed(answer))


def test_b_neighbour_gap_at_two_periods_without_change(res):
    _, answer = ask(res, "2021년과 2023년 서울의 출생아 수와 사망자 수 차이는?", BR, DE)
    assert len(computed(answer)) == 2


def test_b_neighbour_summed_range(res):
    facts, answer = ask(res, "2021년부터 2023년까지 서울 출생아 수 합계는 사망자 수 합계보다 얼마나 많아?", BR, DE)
    got = computed(answer)[-1]
    total = sum(value(BR, "B1", "11", year=y) - value(DE, "X1", "11", year=y) for y in YEARS)
    assert got.value == total


def test_b_neighbour_gap_at_one_period(res):
    facts, answer = ask(res, "2023년 서울의 출생아 수와 사망자 수 차이는?", BR, DE)
    (got,) = computed(answer)
    assert abs(got.value) == abs(value(BR, "B1", "11") - value(DE, "X1", "11"))


# ---------------------------------------------------------------- C


PO = "F18_POP"


def pop(region, sex, age, year="2023"):
    return value(PO, "P1", region, sex, age, year=year)


def pref(region, sex, age, year="2023"):
    return ref(PO, "P1", region, sex, age, year=year)


@pytest.mark.parametrize("question, region, year", [
    ("2023년 서울 인구 중 0-14세와 65세이상을 합치면 몇 퍼센트야?", "11", "2023"),
    ("2022년 부산 인구 가운데 0-14세와 65세이상을 더하면 몇 %야?", "21", "2022"),
])
def test_c1_parts_put_together_in_a_group_percent(res, question, region, year):
    facts, answer = ask(res, question, PO)
    got = computed(answer)[-1]
    assert refs_in(facts, got) == sorted([pref(region, "0", "1", year), pref(region, "0", "3", year),
                                          pref(region, "0", "0", year)])
    whole = pop(region, "0", "0", year)
    assert got.value == pytest.approx(100 * (pop(region, "0", "1", year) + pop(region, "0", "3", year)) / whole,
                                      abs=0.01)


@pytest.mark.parametrize("question, region, sex", [
    ("2023년 서울 0-14세와 65세이상을 합하면 계 여자 인구의 몇 퍼센트야?", "11", "2"),
    ("2023년 대구 0-14세와 65세이상을 합치면 계 남자 인구의 몇 %야?", "22", "1"),
])
def test_c2_parts_put_together_of_the_whole_past_names(res, question, region, sex):
    facts, answer = ask(res, question, PO)
    got = computed(answer)[-1]
    assert refs_in(facts, got) == sorted([pref(region, sex, "1"), pref(region, sex, "3"), pref(region, sex, "0")])
    assert got.value == pytest.approx(100 * (pop(region, sex, "1") + pop(region, sex, "3")) / pop(region, sex, "0"),
                                      abs=0.01)


@pytest.mark.parametrize("question", [
    "2023년 서울 0-14세는 65세이상의 몇 퍼센트야?",
    "2023년 서울 0-14세는 65세이상 인구의 몇 퍼센트야?",
])
def test_c_neighbour_one_target_a_percent_of_another(res, question):
    facts, answer = ask(res, question, PO)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([pref("11", "0", "1"), pref("11", "0", "3")])
    assert got.value == pytest.approx(100 * pop("11", "0", "1") / pop("11", "0", "3"), abs=0.01)


def test_c_neighbour_sum_without_percent(res):
    facts, answer = ask(res, "2023년 서울 0-14세와 65세이상을 합하면 몇 명이야?", PO)
    (got,) = computed(answer)
    assert got.value == pop("11", "0", "1") + pop("11", "0", "3")


def test_c_neighbour_one_part_in_a_group(res):
    facts, answer = ask(res, "2023년 서울 인구 중 0-14세는 몇 퍼센트야?", PO)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([pref("11", "0", "1"), pref("11", "0", "0")])


def test_c_neighbour_percent_change(res):
    facts, answer = ask(res, "2021년 대비 2023년 서울 0-14세 인구는 몇 퍼센트 늘었어?", PO)
    got = computed(answer)[-1]
    assert got.value == pytest.approx(100 * (pop("11", "0", "1") / pop("11", "0", "1", "2021") - 1), abs=0.01)


# ---------------------------------------------------------------- D


@pytest.mark.parametrize("question, year, sex, regions", [
    ("2023년 서울과 부산에서 각각 인구가 가장 많은 연령대는?", "2023", "0", ("11", "21")),
    ("2022년 대구와 인천 각각 남자 인구가 제일 많은 연령대는 어디야?", "2022", "1", ("22", "23")),
])
def test_d_ranking_repeated_per_named_region(res, question, year, sex, regions):
    _, answer = ask(res, question, PO)
    best = [max(("1", "2", "3"), key=lambda a: pop(r, sex, a, year)) for r in regions]
    assert cells(answer) == sorted(pref(r, sex, a, year) for r, a in zip(regions, best))


def test_d_ranking_repeated_per_named_sex(res):
    _, answer = ask(res, "2022년 남자와 여자 각각 두 번째로 인구가 많은 시도는?", PO)
    picked = []
    for s in ("1", "2"):
        order = sorted(("11", "21", "22", "23"), key=lambda r: pop(r, s, "0", "2022"), reverse=True)
        picked.append(pref(order[1], s, "0", "2022"))
    assert cells(answer) == sorted(picked)


def test_d_neighbour_one_named_target(res):
    _, answer = ask(res, "2023년 서울에서 인구가 가장 많은 연령대는?", PO)
    assert cells(answer) == [pref("11", "0", max(("1", "2", "3"), key=lambda a: pop("11", "0", a)))]


@pytest.mark.parametrize("question", [
    "2023년 서울과 부산 중 인구가 가장 많은 시도는?",
    "2023년 서울과 부산 각각 인구가 가장 많은 시도는?",
])
def test_d_neighbour_named_candidates(res, question):
    _, answer = ask(res, question, PO)
    assert cells(answer) == [pref(max(("11", "21"), key=lambda r: pop(r, "0", "0")), "0", "0")]


# ---------------------------------------------------------------- E


@pytest.mark.parametrize("question, region, year, subject, reference", [
    ("2023년 서울에서 남자인 경우가 여자인 경우보다 인구가 몇 퍼센트 많아?", "11", "2023", ("1", "0"), ("2", "0")),
    ("2022년 부산은 65세이상인 쪽이 0-14세인 쪽보다 몇 퍼센트 많아?", "21", "2022", ("0", "3"), ("0", "1")),
])
def test_e_reference_said_with_copula_and_bound_noun(res, question, region, year, subject, reference):
    facts, answer = ask(res, question, PO)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([pref(region, *subject, year), pref(region, *reference, year)])
    a, b = pop(region, *subject, year), pop(region, *reference, year)
    assert got.value == pytest.approx(100 * (a / b - 1), abs=0.01)


def test_e_neighbour_no_reference_named(res):
    facts, answer = ask(res, "2023년 서울에서 남자인 경우와 여자인 경우의 인구 차이는?", PO)
    (got,) = computed(answer)
    assert got.value == abs(pop("11", "1", "0") - pop("11", "2", "0"))


def test_e_neighbour_reference_right_after_the_name(res):
    facts, answer = ask(res, "2023년 서울에서 여자가 남자보다 인구가 몇 명 많아?", PO)
    (got,) = computed(answer)
    assert got.value == pop("11", "2", "0") - pop("11", "1", "0")
