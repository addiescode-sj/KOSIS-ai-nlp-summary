"""Loop 15 (judge/question.py label_forms, match_labels; shared/labels.py LABEL_ALIASES, FAMILY_ROLES):

- A-1: a family role written as one Hanja character ('부', '모', '처') is said by its everyday words ('아버지',
  '엄마', '아내'), on one table (a list, a share change, a ratio of two names) and on two tables.
- A-2: a name for a person ending with a person suffix ('취업자') said by its stem and a describing ending ('취업한').
- E-1: an ordinal name with a person suffix ('첫째아') said by the ordinal alone ('첫째', '둘째인'), on one table
  and on two tables.
- G: a pair of nations said by the first characters of the fixed nation and of the other one ('한중 부부').

Neighbours: a longer name holding the everyday word wins; a word for both roles ('부모') is neither; an ordinal
followed by 로 + a describing word is still a rank; a name said with its suffix is read as before; a stem that is
another name of the table stays that name; a first character two nations share makes no shortening.
The tables are made up (가상 데이터); every test checks the values and the cells they come from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

YEARS = ("2021", "2022", "2023")


def catalog(title):
    return {"survey_name": "가상 가족조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


# ---------------------------------------------------------------- tables

# A: a category of two one-character roles with no total, and a category of people (a total and three names ending
# with a person suffix).
PARENTS = (("P1", "부"), ("P2", "모"))
STATES = (("0", "계"), ("1", "취업자"), ("2", "실업자"), ("3", "비경활자"))


def parents_rows(item="T1", name="자녀수", base=100):
    rows = []
    for y, year in enumerate(YEARS):
        for p, (pc, pl) in enumerate(PARENTS):
            parts = [base * (p + 1) + 13 * y + 7, base // 2 + 11 * y + 3 * p, base // 3 + 5 * y + p]
            for sc, sl in STATES:
                v = sum(parts) if sc == "0" else parts[int(sc) - 1]
                rows.append({"period": year, "item_id": item, "item_name": name, "unit": "명",
                             "dim1_name": "부모별", "dim1_code": pc, "dim1_label": pl,
                             "dim2_name": "경제활동상태별", "dim2_code": sc, "dim2_label": sl,
                             "value_raw": str(v), "value": v})
    return rows


# A, another family: spouses ('부' for a husband is not used: '처' and '남편').
SPOUSES = (("H", "남편"), ("W", "처"))


def spouses_rows():
    rows = []
    for y, year in enumerate(YEARS):
        for s, (sc, sl) in enumerate(SPOUSES):
            for ac, al in (("0", "계"), ("1", "취업자"), ("2", "실업자")):
                v = 300 + 40 * s + 9 * y + 17 * int(ac)
                rows.append({"period": year, "item_id": "T1", "item_name": "인원", "unit": "명",
                             "dim1_name": "배우자별", "dim1_code": sc, "dim1_label": sl,
                             "dim2_name": "취업여부별", "dim2_code": ac, "dim2_label": al,
                             "value_raw": str(v), "value": v})
    return rows


# A neighbour: a table whose names hold the everyday word ('아버지와 동거' beside '부').
NEAR_PARENTS = (("P0", "계"), ("P1", "부"), ("P3", "아버지와 동거"))


def near_rows():
    return [{"period": year, "item_id": "T1", "item_name": "가구수", "unit": "가구", "dim1_name": "가구유형별",
             "dim1_code": c, "dim1_label": label, "value_raw": str(500 + 30 * i + y), "value": 500 + 30 * i + y}
            for y, year in enumerate(YEARS) for i, (c, label) in enumerate(NEAR_PARENTS)]


# E-1: the items are ordinals with a person suffix; a region category.
ORDINALS = (("B0", "계"), ("B1", "첫째아"), ("B2", "둘째아"), ("B3", "셋째아"), ("B4", "넷째아 이상"), ("B9", "미상"))
SIDO = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))


def births_rows(base=1000):
    rows = []
    for y, year in enumerate(YEARS):
        for r, (rc, rl) in enumerate(SIDO):
            for i, (ic, il) in enumerate(ORDINALS):
                v = base - 90 * i + 37 * r + 11 * y + (r * i) % 7
                rows.append({"period": year, "item_id": ic, "item_name": il, "unit": "명", "dim1_name": "시도별",
                             "dim1_code": rc, "dim1_label": rl, "value_raw": str(v), "value": v})
    return rows


def deaths_rows():
    return [{"period": year, "item_id": "D1", "item_name": "영아사망자수", "unit": "명", "dim1_name": "시도별",
             "dim1_code": rc, "dim1_label": rl, "value_raw": str(20 + 3 * r + y), "value": 20 + 3 * r + y}
            for y, year in enumerate(YEARS) for r, (rc, rl) in enumerate(SIDO)]


# G: two tables of the same shape, a category of nations, the fixed nation in the category's name.
NATIONS = (("N0", "계"), ("N1", "중국"), ("N2", "베트남"), ("N3", "필리핀"), ("N4", "일본"), ("N5", "미국"),
           ("N6", "몽골"), ("N7", "캄보디아"), ("N8", "태국"), ("N9", "러시아(연방)"), ("N10", "미얀마"),
           ("N11", "기타국가"), ("N12", "중국"))   # '미' starts two names; '중국' twice under two codes


def nations_rows(axis, base, nations=NATIONS):
    rows = []
    for y, year in enumerate(YEARS):
        for n, (nc, nl) in enumerate(nations):
            for gc, gl in (("0", "계"), ("1", "초혼"), ("2", "재혼")):
                v = base + 41 * n + 7 * y + 3 * int(gc)
                rows.append({"period": year, "item_id": "M1", "item_name": "혼인건수", "unit": "건",
                             "dim1_name": axis, "dim1_code": nc, "dim1_label": nl,
                             "dim2_name": "혼인종류별", "dim2_code": gc, "dim2_label": gl,
                             "value_raw": str(v), "value": v})
    return rows


SPECS = {
    "FAKE15_PARENT": (lambda: parents_rows(), catalog("가상 부모의 경제활동상태별 자녀수")),
    "FAKE15_PARENT_INCOME": (lambda: parents_rows("T2", "소득자수", 60), catalog("가상 부모의 경제활동상태별 소득자수")),
    "FAKE15_SPOUSE": (spouses_rows, catalog("가상 배우자별 취업여부별 인원")),
    "FAKE15_NEAR": (near_rows, catalog("가상 가구유형별 가구")),
    "FAKE15_BIRTH": (lambda: births_rows(), catalog("가상 시도별 출산순위별 출생")),
    "FAKE15_DEATH": (deaths_rows, catalog("가상 시도별 영아사망")),
    "FAKE15_WIFE": (lambda: nations_rows("한국인 남편의 아내 국적별", 200), catalog("가상 아내 국적별 혼인")),
    "FAKE15_HUSB": (lambda: nations_rows("한국인 아내의 남편 국적별", 90), catalog("가상 남편 국적별 혼인")),
    # no nation fixed: the pair words are not made
    "FAKE15_FREE": (lambda: nations_rows("외국인 아내 국적별", 50, NATIONS[:-1]), catalog("가상 외국인 아내 국적별 혼인")),
}
ROWS = {t: make() for t, (make, _) in SPECS.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop15n") / "data"
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
    return "|".join([tbl, item, d1, *( [d2] if d2 is not None else []), year])


def value(tbl, item, d1, d2=None, year="2023"):
    for r in ROWS[tbl]:
        if (r["period"] == year and r["item_id"] == item and r["dim1_code"] == d1
                and (d2 is None or r.get("dim2_code") == d2)):
            return r["value"]
    raise KeyError((tbl, item, d1, d2))


P = "FAKE15_PARENT"


# ---------------------------------------------------------------- A-1


@pytest.mark.parametrize("question, year, state", [
    ("2023년 아버지와 어머니가 취업자인 자녀는 각각 몇 명이야?", "2023", "1"),
    ("2022년 아빠가 실업자인 자녀와 엄마가 실업자인 자녀 수 알려줘", "2022", "2"),
    ("2021년 부친과 모친이 비경활자인 자녀 수는?", "2021", "3"),
])
def test_a1_two_roles_by_everyday_words_listed(res, question, year, state):
    _, answer = ask(res, question, P)
    assert cells(answer) == sorted([ref(P, "T1", "P1", state, year), ref(P, "T1", "P2", state, year)])
    assert sorted(f.value for f in answer) == sorted([value(P, "T1", "P1", state, year), value(P, "T1", "P2", state, year)])


def test_a1_one_role_value(res):
    _, answer = ask(res, "2023년 엄마가 취업자인 자녀는 몇 명?", P)
    assert cells(answer) == [ref(P, "T1", "P2", "1")]


def test_a1_ratio_of_two_states_of_one_role(res):
    facts, answer = ask(res, "2023년 아버지가 취업자인 자녀 수는 아버지가 실업자인 자녀 수의 몇 배야?", P)
    (got,) = computed(answer)
    assert got.value == pytest.approx(round(value(P, "T1", "P1", "1") / value(P, "T1", "P1", "2"), 2), abs=0.01)
    assert refs_in(facts, got) == sorted([ref(P, "T1", "P1", "1"), ref(P, "T1", "P1", "2")])


def test_a1_spouse_role_on_another_table(res):
    _, answer = ask(res, "2022년 아내가 취업자인 경우는 몇 명이야?", "FAKE15_SPOUSE")
    assert cells(answer) == [ref("FAKE15_SPOUSE", "T1", "W", "1", "2022")]


def test_a1_two_tables(res):
    _, answer = ask(res, "2023년 어머니가 취업자인 자녀수와 소득자수를 알려줘", P, "FAKE15_PARENT_INCOME")
    assert cells(answer) == sorted([ref(P, "T1", "P2", "1"), ref("FAKE15_PARENT_INCOME", "T2", "P2", "1")])


def test_a1_neighbour_longer_name_holding_the_word_wins(res):
    _, answer = ask(res, "2023년 아버지와 동거하는 가구 수는?", "FAKE15_NEAR")
    assert cells(answer) == [ref("FAKE15_NEAR", "T1", "P3")]


def test_a1_neighbour_word_for_both_roles_is_neither(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[P].table)
    assert [m.code for m in match_labels(index, "2023년 부모가 취업자인 자녀 수") if m.axis == "부모별"] == []


# ---------------------------------------------------------------- A-2


@pytest.mark.parametrize("question, year, role", [
    ("2023년 아버지가 취업한 자녀는 몇 명이야?", "2023", "P1"),
    ("2021년 어머니가 실업한 자녀 수는?", "2021", "P2"),
])
def test_a2_person_name_said_by_stem_and_describing_ending(res, question, year, role):
    state = "1" if "취업" in question else "2"
    _, answer = ask(res, question, P)
    assert cells(answer) == [ref(P, "T1", role, state, year)]


def test_a2_neighbour_stem_followed_by_other_characters(res):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[P].table)
    assert [m.code for m in match_labels(index, "2023년 아버지 취업률") if m.axis == "경제활동상태별"] == []


# ---------------------------------------------------------------- E-1

B = "FAKE15_BIRTH"


@pytest.mark.parametrize("question, item, region, year", [
    ("2023년 서울에서 태어난 아이 중 둘째인 아이는 몇 명이야?", "B2", "11", "2023"),
    ("2022년 부산 셋째 출생아 수 알려줘", "B3", "21", "2022"),
    ("2021년 전국에서 첫째는 몇 명 태어났어?", "B1", "00", "2021"),
])
def test_e1_ordinal_alone(res, question, item, region, year):
    _, answer = ask(res, question, B)
    assert cells(answer) == [ref(B, item, region, year=year)]


def test_e1_ordinal_alone_two_tables(res):
    _, answer = ask(res, "2023년 대구의 첫째 출생아 수와 영아사망자 수를 알려줘", B, "FAKE15_DEATH")
    assert cells(answer) == sorted([ref(B, "B1", "22"), ref("FAKE15_DEATH", "D1", "22")])


def test_e1_neighbour_rank_with_ordinal(res):
    _, answer = ask(res, "2023년 첫째아가 두 번째로 많은 시도는?", B)
    expected = sorted(((value(B, "B1", rc), rc) for rc, _ in SIDO[1:]), reverse=True)[1][1]
    assert ref(B, "B1", expected) in cells(answer)


def test_e1_neighbour_ordinal_then_ro_is_a_rank(res):
    _, answer = ask(res, "2023년 첫째아 출생아가 둘째로 많은 시도는?", B)
    expected = sorted(((value(B, "B1", rc), rc) for rc, _ in SIDO[1:]), reverse=True)[1][1]
    assert ref(B, "B1", expected) in cells(answer)
    assert all("|B2|" not in c for c in cells(answer))


def test_e1_neighbour_name_with_suffix(res):
    _, answer = ask(res, "2023년 서울 둘째아 수는?", B)
    assert cells(answer) == [ref(B, "B2", "11")]


# ---------------------------------------------------------------- G

W, H = "FAKE15_WIFE", "FAKE15_HUSB"


@pytest.mark.parametrize("question, nation, year", [
    ("2022년 한베 부부의 혼인 건수를 표마다 알려줘", "N2", "2022"),
    ("2023년 한필 커플 혼인건수는?", "N3", "2023"),
])
def test_g_pair_of_nations_shortened_two_tables(res, question, nation, year):
    _, answer = ask(res, question, W, H)
    assert cells(answer) == sorted([ref(W, "M1", nation, "0", year), ref(H, "M1", nation, "0", year)])
    assert sorted(f.value for f in answer) == sorted([value(W, "M1", nation, "0", year), value(H, "M1", nation, "0", year)])


@pytest.mark.parametrize("question, nation, year", [
    ("2023년 한일 부부 혼인건수는?", "N4", "2023"),
    ("2021년 한몽 부부의 재혼 건수는?", "N6", "2021"),
])
def test_g_one_table(res, question, nation, year):
    _, answer = ask(res, question, W)
    kind = "2" if "재혼" in question else "0"
    assert cells(answer) == [ref(W, "M1", nation, kind, year)]


def test_g_same_name_twice_as_when_named(res):
    _, short = ask(res, "2023년 한중 부부 혼인건수는?", W)
    _, named = ask(res, "2023년 중국 혼인건수는?", W)
    assert cells(short) == cells(named)


def test_g_neighbour_nation_named_as_it_is(res):
    _, answer = ask(res, "2023년 몽골 혼인건수는?", W)
    assert cells(answer) == [ref(W, "M1", "N6", "0")]


@pytest.mark.parametrize("question", ["2023년 한미 부부 혼인건수", "2023년 한일전 관객 혼인건수"])
def test_g_neighbour_shared_initial_or_longer_word_makes_none(res, question):
    from kosis_chat.judge.question import index_of, match_labels
    index = index_of(res[W].table)
    # '미' starts 미국 and 미얀마; '한일' is the start of a longer word ('한일전')
    assert [m.code for m in match_labels(index, question) if m.axis.startswith("한국인")] == []


def test_g_neighbour_no_fixed_nation_makes_none(res):
    from kosis_chat.judge.question import index_of, nation_pairs
    assert nation_pairs(index_of(res["FAKE15_FREE"].table)) == {}
    assert nation_pairs(index_of(res[W].table))[("한국인 남편의 아내 국적별", "N4")] == ("한일",)
