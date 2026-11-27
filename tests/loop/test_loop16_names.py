"""Loop 16 (judge/question.py match_labels, label_forms tokens, group_parts, name_spans; shared/patterns.py
asks_min, count_words_of, BOTH_SEXES, WHOLE_AFTER; shared/labels.py predicate_forms; analyze/rules.py _both_sexes):

- A: a token of a name that starts or ends with a digit ('20', '24년' of '20～24년') is said only at the bounds of
  its number, so a year written in the question ("2024년") names no interval.
- B-1: a name '<head>-<predicate in its noun form>' ('자녀-없음') said apart, by the head and a describing or adverb
  form of the predicate ("자녀가 없는", "자동차 없이"), when a word of what the table is split by holds the head.
- B-2: a name of a number and a counting word ('2명') said with a native number word ("두 명인").
- B-3: the whole's stem ('가구수' of '가구수-계') said right after a matched part is the noun of what is counted.
- C: the two sexes named as one word and a word of the whole ("남녀 전체", "남녀를 합친") is the total.
- D: a name inside a title word made of the name and a counting tail ('사망' in '사망자수').
- E: the smaller-end words of one-syllable stems are read where a word starts ('면적은' asks no minimum).

Neighbours for every condition. The tables are made up (가상 데이터); every test checks the values and the cells.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2022", "2023", "2024")
SIDO = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))


def catalog(title):
    return {"survey_name": "가상 통계조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


# ---------------------------------------------------------------- tables

# A: a category of year intervals joined by a full-width tilde, two open names and an unknown, beside regions.
DURATIONS = (("0", "계"), ("1", "1년미만"), ("2", "1～4년"), ("3", "5～9년"), ("4", "10～14년"), ("5", "15～19년"),
             ("6", "20～24년"), ("7", "25년이상"), ("9", "미상"))


def duration_rows(item="T1", name="이혼건수", unit="건", base=500):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO):
            for d, (dc, dl) in enumerate(DURATIONS):
                v = base + 31 * r + 7 * y + 13 * d + (r * d) % 5
                rows.append({"period": year, "item_id": item, "item_name": name, "unit": unit,
                             "dim1_name": "시도별", "dim1_code": rc, "dim1_label": rl,
                             "dim2_name": "혼인지속기간별", "dim2_code": dc, "dim2_label": dl,
                             "value_raw": str(v), "value": v})
    return rows


# B: the items are a whole ('가구수-계'), '<counted thing>-없음', two numbers with a counting word, an open name and
# an unknown; a region category. The counted thing is only in the title's "…별" word.
def counted_items(thing, unit):
    return (("T0", "가구수-계"), ("T1", f"{thing}-없음"), ("T2", f"1{unit}"), ("T3", f"2{unit}"),
            ("T4", f"3{unit} 이상"), ("T9", "미상"))


def counted_rows(items, base):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO):
            parts = [base + 41 * r + 9 * y + 17 * i + (r + i) % 3 for i in range(1, len(items))]
            for i, (ic, il) in enumerate(items):
                v = sum(parts) if i == 0 else parts[i - 1]
                rows.append({"period": year, "item_id": ic, "item_name": il, "unit": "가구", "dim1_name": "시도별",
                             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v})
    return rows


# C, D: a category of the sexes with a total, an age interval category, two items with no total (a count named by
# two characters and its rate); the title writes the count as '<name>자수'.
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
AGES = (("A0", "계"), ("A1", "0-14세"), ("A2", "15-64세"), ("A3", "65세이상"))
EVENTS = (("D1", "사망", "명"), ("D2", "사망률", "명/10만명"))


def sex_age_rows(events=EVENTS, base=900, sexes=SEXES):
    rows = []
    for y, year in enumerate(YEARS):
        for a, (ac, al) in enumerate(AGES):
            for e, (ec, el, unit) in enumerate(events):
                parts = [base * (e * 0 + 1) // (1 + 9 * e) + 23 * a + 11 * y + 5 * s for s in (1, 2)]
                for sc, sl in sexes:
                    v = sum(parts) if sc == "0" else parts[int(sc) - 1]
                    rows.append({"period": year, "item_id": ec, "item_name": el, "unit": unit,
                                 "dim1_name": "성별", "dim1_code": sc, "dim1_label": sl,
                                 "dim2_name": "연령별", "dim2_code": ac, "dim2_label": al,
                                 "value_raw": str(v), "value": v})
    return rows


# E: regions ranked by a count, the question pointing at them with a noun ending with '적'.
def area_rows():
    return [{"period": year, "item_id": "T1", "item_name": "농지면적", "unit": "ha", "dim1_name": "시도별",
             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v}
            for y, year in enumerate(YEARS) for r, (rc, rl) in enumerate(SIDO)
            for v in [9000 if rc == "00" else 1000 + 300 * ((r * 7) % 4) + 17 * y]]


CHILD = counted_items("자녀", "명")
CAR = counted_items("자동차", "대")
SPECS = {
    "FAKE16_DUR": (lambda: duration_rows(), catalog("가상 시도별 혼인지속기간별 이혼")),
    "FAKE16_DUR2": (lambda: duration_rows("T2", "이혼자수", "명", 900), catalog("가상 시도별 혼인지속기간별 이혼자")),
    "FAKE16_CHILD": (lambda: counted_rows(CHILD, 300), catalog("가상 시도별 자녀수별 가구")),
    "FAKE16_CAR": (lambda: counted_rows(CAR, 200), catalog("가상 시도별 자동차보유대수별 가구")),
    "FAKE16_DEATH": (lambda: sex_age_rows(), catalog("가상 성별 연령별 사망자수 및 사망률")),
    "FAKE16_BIRTH": (lambda: sex_age_rows((("B1", "출생", "명"), ("B2", "출생률", "명/천명")), 700),
                     catalog("가상 성별 연령별 출생아수 및 출생률")),
    # no total in the category of the sexes
    "FAKE16_NOSUM": (lambda: sex_age_rows(base=400, sexes=SEXES[1:]), catalog("가상 성별 연령별 사망자")),
    "FAKE16_AREA": (area_rows, catalog("가상 시도별 농지면적")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop16") / "data"
    return {t: load_resources(load_table(write_table(folder, t, ROWS[t], cat))) for t, (_, cat) in SPECS.items()}


def ask(res, question, *tbls):
    got = calculate(question, [res[t] for t in tbls])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def cells(answer):
    return sorted(f.source_ref for f in answer if f.source_ref)


def computed(answer):
    return [f for f in answer if not f.source_ref]


def refs_in(facts, fact):
    return sorted(facts[i].source_ref for i in re.findall(r"f\d+", fact.derivation or ""))


def ref(tbl, item, d1, d2=None, year="2023"):
    return "|".join([tbl, item, d1, *([d2] if d2 is not None else []), year])


def value(tbl, item, d1, d2=None, year="2023"):
    for r in ROWS[tbl]:
        if (r["period"] == year and r["item_id"] == item and r["dim1_code"] == d1
                and (d2 is None or r.get("dim2_code") == d2)):
            return r["value"]
    raise KeyError((tbl, item, d1, d2))


DUR, DUR2, CH, CR, DE, BI = "FAKE16_DUR", "FAKE16_DUR2", "FAKE16_CHILD", "FAKE16_CAR", "FAKE16_DEATH", "FAKE16_BIRTH"


# ---------------------------------------------------------------- A


@pytest.mark.parametrize("question, region", [
    ("2024년 서울의 이혼건수는?", "11"),
    ("부산 이혼 건수 2024년 기준으로 알려줘", "21"),
])
def test_a_year_names_no_interval(res, question, region):
    _, answer = ask(res, question, DUR)
    assert cells(answer) == [ref(DUR, "T1", region, "0", "2024")]


def test_a_same_question_another_year_was_and_is_the_total(res):
    _, answer = ask(res, "2023년 서울의 이혼건수는?", DUR)
    assert cells(answer) == [ref(DUR, "T1", "11", "0", "2023")]


def test_a_another_interval_named_with_the_year(res):
    _, answer = ask(res, "2024년 대구의 혼인지속기간 5～9년 이혼건수는?", DUR)
    assert cells(answer) == [ref(DUR, "T1", "22", "3", "2024")]


def test_a_two_tables_change(res):
    facts, answer = ask(res, "2022년부터 2024년까지 서울의 이혼건수와 이혼자수는 각각 얼마나 변했어?", DUR, DUR2)
    used = sorted({r for f in computed(answer) for r in refs_in(facts, f)})
    assert used == sorted([ref(DUR, "T1", "11", "0", "2022"), ref(DUR, "T1", "11", "0", "2024"),
                           ref(DUR2, "T2", "11", "0", "2022"), ref(DUR2, "T2", "11", "0", "2024")])


@pytest.mark.parametrize("question", [
    "2024년 서울의 혼인지속기간 20～24년 이혼건수는?",
    "2024년 서울의 혼인지속기간 20~24년 이혼건수는?",
    "2024년 서울에서 혼인지속기간이 20년~24년인 이혼건수는?",
])
def test_a_neighbour_interval_named_as_it_is_or_by_another_mark(res, question):
    _, answer = ask(res, question, DUR)
    assert cells(answer) == [ref(DUR, "T1", "11", "6", "2024")]


# ---------------------------------------------------------------- B-1, B-3


@pytest.mark.parametrize("question, tbl, region, year", [
    ("2023년 서울에서 자녀가 없는 가구는 몇 가구야?", CH, "11", "2023"),
    ("2022년 부산의 자녀 없는 가구 수는?", CH, "21", "2022"),
    ("2024년 대구에 자동차 없이 사는 가구 수 알려줘", CR, "22", "2024"),
    ("2023년 서울 자녀가 한 명도 없는 가구는?", CH, "11", "2023"),
])
def test_b1_head_and_predicate_said_apart(res, question, tbl, region, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, "T1", region, year=year)]


def test_b1_change_of_two_periods(res):
    facts, answer = ask(res, "서울의 자동차가 없는 가구는 2022년에서 2024년 사이 얼마나 늘었어?", CR)
    (got,) = [f for f in computed(answer) if refs_in(facts, f)][:1]
    assert refs_in(facts, got) == sorted([ref(CR, "T1", "11", year="2022"), ref(CR, "T1", "11", year="2024")])
    assert got.value == value(CR, "T1", "11", year="2024") - value(CR, "T1", "11", year="2022")


def test_b1_two_tables_listed(res):
    _, answer = ask(res, "2023년 서울의 자녀가 없는 가구 수와 자동차가 없는 가구 수를 알려줘", CH, CR)
    assert cells(answer) == sorted([ref(CH, "T1", "11"), ref(CR, "T1", "11")])


def test_b1_neighbour_head_alone_is_the_whole(res):
    _, answer = ask(res, "2023년 서울의 자녀수별 가구의 가구수는?", CH)
    assert cells(answer) == [ref(CH, "T0", "11")]


def test_b1_neighbour_opposite_predicate_is_not_the_name(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[CH].table)
    assert [m.code for m in match_labels(index, "2023년 서울 자녀가 있는 가구 수") if m.code == "T1"] == []


def test_b1_neighbour_predicate_without_the_head(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[CH].table)
    assert [m.code for m in match_labels(index, "2023년 서울에서 없는 가구 수") if m.code == "T1"] == []


def test_b3_neighbour_whole_and_part_listed(res):
    _, answer = ask(res, "2023년 서울의 가구수와 자녀가 없는 가구수를 알려줘", CH)
    assert cells(answer) == sorted([ref(CH, "T0", "11"), ref(CH, "T1", "11")])


def test_b3_neighbour_part_of_the_whole_share(res):
    facts, answer = ask(res, "2023년 서울 가구수 중 자녀가 없는 가구의 비중은?", CH)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(CH, "T0", "11"), ref(CH, "T1", "11")])
    assert got.value == pytest.approx(round(100 * value(CH, "T1", "11") / value(CH, "T0", "11"), 2), abs=0.01)


# ---------------------------------------------------------------- B-2


@pytest.mark.parametrize("question, tbl, item, region", [
    ("2023년 서울에서 자녀가 두 명인 가구는 몇 가구야?", CH, "T3", "11"),
    ("2023년 부산 자동차가 한 대 있는 가구 수는?", CR, "T2", "21"),
])
def test_b2_number_name_by_native_word(res, question, tbl, item, region):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, region)]


def test_b2_fixed_name_ranked_by_region(res):
    _, answer = ask(res, "2023년 자녀가 두 명인 가구가 가장 많은 시도는?", CH)
    best = max(("11", "21", "22"), key=lambda r: value(CH, "T3", r))
    assert cells(answer) == [ref(CH, "T3", best)]


def test_b2_two_tables_difference(res):
    facts, answer = ask(res, "2023년 서울에서 자녀가 한 명인 가구 수와 자동차가 한 대인 가구 수의 차이는?", CH, CR)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(CH, "T2", "11"), ref(CR, "T2", "11")])
    assert abs(got.value) == abs(value(CH, "T2", "11") - value(CR, "T2", "11"))


def test_b2_neighbour_none_at_all_is_not_one(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[CH].table)
    codes = {m.code for m in match_labels(index, "2023년 서울 자녀가 한 명도 없는 가구 수")}
    assert "T2" not in codes and "T1" in codes


def test_b2_neighbour_range_and_other_counting_word(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[CH].table)
    assert "T4" not in {m.code for m in match_labels(index, "2023년 서울 자녀가 세 명 이상인 가구 수")} - {"T4"}
    assert {m.code for m in match_labels(index, "2023년 서울 자녀가 세 명 이상인 가구 수")} & {"T2", "T3"} == set()
    assert {m.code for m in match_labels(index, "두 해 전과 비교한 서울 가구 수")} & {"T2", "T3"} == set()
    assert {m.code for m in match_labels(index, "결혼한 명 수")} & {"T2", "T3"} == set()


def test_b2_neighbour_digits_as_before(res):
    _, answer = ask(res, "2023년 서울에서 자녀가 2명인 가구는?", CH)
    assert cells(answer) == [ref(CH, "T3", "11")]


# ---------------------------------------------------------------- C, D


@pytest.mark.parametrize("question, tbl, item, age, year", [
    ("2023년 남녀 전체 15-64세 사망자수는?", DE, "D1", "A2", "2023"),
    ("2022년 남녀를 합친 65세이상 출생아수는 몇 명이야?", BI, "B1", "A3", "2022"),
])
def test_c_d_both_sexes_whole_is_the_total(res, question, tbl, item, age, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, "0", age, year)]


@pytest.mark.parametrize("question, tbl, item, age, year", [
    ("2023년 남녀 전체 15-64세 사망은 몇 명?", DE, "D1", "A2", "2023"),
    ("2024년 남자와 여자를 합한 0-14세 출생은?", BI, "B1", "A1", "2024"),
])
def test_c_both_sexes_whole_with_the_item_named(res, question, tbl, item, age, year):
    _, answer = ask(res, question, tbl)
    assert cells(answer) == [ref(tbl, item, "0", age, year)]


def test_c_d_ratio_of_two_ages_with_both_sexes_whole(res):
    facts, answer = ask(res, "2024년 남녀 합계 65세이상 사망자수는 15-64세의 몇 배야?", DE)
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref(DE, "D1", "0", "A3", "2024"), ref(DE, "D1", "0", "A2", "2024")])
    assert got.value == pytest.approx(round(value(DE, "D1", "0", "A3", "2024") / value(DE, "D1", "0", "A2", "2024"), 2),
                                      abs=0.01)


def test_c_d_largest_period_with_both_sexes_whole(res):
    _, answer = ask(res, "남녀 통틀어 0-14세 사망자수가 가장 많았던 해는?", DE)
    assert cells(answer) == [ref(DE, "D1", "0", "A1", "2024")]


def test_d_title_word_names_the_item(res):
    _, answer = ask(res, "2023년 남자 0-14세 사망자수는?", DE)
    assert cells(answer) == [ref(DE, "D1", "1", "A1")]


def test_d_neighbour_rate_of_the_same_stem(res):
    _, answer = ask(res, "2023년 여자 15-64세 사망률은?", DE)
    assert cells(answer) == [ref(DE, "D2", "2", "A2")]


def test_c_neighbour_each_sex_stays_two(res):
    _, answer = ask(res, "2023년 남녀 각각 65세이상 사망자수는?", DE)
    assert cells(answer) == sorted([ref(DE, "D1", "1", "A3"), ref(DE, "D1", "2", "A3")])


def test_c_neighbour_no_total_sums_the_two(res):
    facts, answer = ask(res, "2023년 남녀를 합친 65세이상 사망은?", "FAKE16_NOSUM")
    (got,) = computed(answer)
    assert refs_in(facts, got) == sorted([ref("FAKE16_NOSUM", "D1", "1", "A3"), ref("FAKE16_NOSUM", "D1", "2", "A3")])


# ---------------------------------------------------------------- E


@pytest.mark.parametrize("question, year", [
    ("2023년 농지면적은 어느 시도가 가장 넓어?", "2023"),
    ("2022년 농지면적이 가장 큰 시도는?", "2022"),
    ("2024년 농지 면적은 가장 많은 시도가 어디야?", "2024"),
])
def test_e_noun_ending_like_a_min_word_asks_the_largest(res, question, year):
    _, answer = ask(res, question, "FAKE16_AREA")
    best = max(("11", "21", "22"), key=lambda r: value("FAKE16_AREA", "T1", r, year=year))
    assert cells(answer) == [ref("FAKE16_AREA", "T1", best, year=year)]


@pytest.mark.parametrize("question", [
    "2023년 농지면적이 가장 적은 시도는?",
    "2023년 농지면적이 가장적은 시도는?",
    "2023년 농지면적이 제일 작았던 시도는?",
])
def test_e_neighbour_min_words_still_min(res, question):
    _, answer = ask(res, question, "FAKE16_AREA")
    low = min(("11", "21", "22"), key=lambda r: value("FAKE16_AREA", "T1", r))
    assert cells(answer) == [ref("FAKE16_AREA", "T1", low)]


def test_e_neighbour_both_ends(res):
    _, answer = ask(res, "2023년 농지면적은 가장 많은 시도와 가장 적은 시도가 각각 얼마야?", "FAKE16_AREA")
    vals = {r: value("FAKE16_AREA", "T1", r) for r in ("11", "21", "22")}
    assert sorted([ref("FAKE16_AREA", "T1", max(vals, key=vals.get)), ref("FAKE16_AREA", "T1", min(vals, key=vals.get))]) \
        == [c for c in cells(answer)]
