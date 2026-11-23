"""Loop 7 (judge/question.py, judge/schemas.py, analyze/rules.py, analyze/blocks.py, shared/patterns.py):

- I: two periods of one series and a question for how many times the one is the other: one ratio, the period named
  as the reference (before a comparing word, or before "의 몇 배") as the denominator; no change, no rate of change.
- C: two targets of one category, two periods and a change asked with a verb (늘었, 줄었, 올랐, 몇 퍼센트 늘었) or a
  word of "each" (각각, 저마다): each target's asked change and its two cells, no difference between the changes.
  A question that asks which changed more keeps the difference.
- B: an item named "group + attribute" ('귀농가구원-여자', '귀어인여자') beside the group's total, asked with the
  group and the attribute apart ("귀농가구원 중 여자", "귀촌인 남자"): that item, not the group's total.

The tables are made up (가상 데이터). Every test checks the answer's values and the cells they were computed from.
"""

from __future__ import annotations

import re

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"), ("22", "대구광역시"))
YEARS = [str(y) for y in range(2013, 2024)]


def birth_rows():
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            v = 40000 - 1700 * y_no - 9000 * r_no + 37 * r_no * y_no
            out.append({"period": year, "item_id": "B1", "item_name": "출생아수", "unit": "명", "dim1_name": "시도별",
                        "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


def rate_rows():
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            v = round(55.0 + 0.4 * y_no + 1.3 * r_no - 0.05 * r_no * y_no, 1)
            out.append({"period": year, "item_id": "E1", "item_name": "고용률", "unit": "%", "dim1_name": "시도별",
                        "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


def house_rows():
    out = []
    for y_no, year in enumerate(YEARS):
        for r_no, (code, label) in enumerate(REGIONS):
            v = 900 + 35 * y_no + 211 * r_no
            out.append({"period": year, "item_id": "H1", "item_name": "1인가구수", "unit": "가구", "dim1_name": "시도별",
                        "dim1_code": code, "dim1_label": label, "value_raw": str(v), "value": v})
    return out


# group + attribute items beside the group's total, the first item not a total of everything; codes do not nest
RETURN = (("T01", "귀농가구원-계"), ("T02", "귀농가구원-남자"), ("T03", "귀농가구원-여자"),
          ("T04", "귀촌인-계"), ("T05", "귀촌인-남자"), ("T06", "귀촌인-여자"))
# the same shape without a mark: the group's name alone is its total, the details are the name and the attribute
FISH = (("F01", "귀어인"), ("F02", "귀어인남자"), ("F03", "귀어인여자"))


def item_rows(items, unit):
    out = []
    for y_no, year in enumerate(("2022", "2023")):
        for r_no, (code, label) in enumerate(REGIONS):
            base = 300 * (4 - r_no) + 11 * y_no
            vals, total = {}, None
            for n, (item, name) in enumerate(items):
                if n % 3 == 0:
                    total = item
                    continue
                vals[item] = base + 17 * n
                vals[total] = vals.get(total, 0) + vals[item]
            for item, name in items:
                out.append({"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별",
                            "dim1_code": code, "dim1_label": label, "value_raw": str(vals[item]), "value": vals[item]})
    return out


# a total marked on a stem, then details that do not share the stem; nothing links them (one marked name only)
MEMBERS = (("G0", "가구원-계"), ("G1", "청년"), ("G2", "중장년"), ("G3", "노년"))


def member_rows():
    out = []
    for y_no, year in enumerate(("2022", "2023")):
        for r_no, (code, label) in enumerate(REGIONS):
            v = {"G1": 500 - 60 * r_no + 9 * y_no, "G2": 800 - 90 * r_no + 4 * y_no, "G3": 300 - 30 * r_no + 21 * y_no}
            v["G0"] = sum(v.values())
            for item, name in MEMBERS:
                out.append({"period": year, "item_id": item, "item_name": name, "unit": "명", "dim1_name": "시도별",
                            "dim1_code": code, "dim1_label": label, "value_raw": str(v[item]), "value": v[item]})
    return out


def worker_sex_rows():
    """'취업자' and '취업자여자' items beside a sex category that holds '여자' too: the category is the attribute's."""
    out = []
    for r_no, (code, label) in enumerate(REGIONS):
        for item, name, k in (("W1", "취업자", 1), ("W2", "취업자여자", 2)):
            for sex, s_name, s in (("0", "계", 3), ("1", "남자", 1), ("2", "여자", 2)):
                v = 1000 * k + 100 * s + r_no
                out.append({"period": "2023", "item_id": item, "item_name": name, "unit": "명", "dim1_name": "시도별",
                            "dim1_code": code, "dim1_label": label, "dim2_name": "성별", "dim2_code": sex,
                            "dim2_label": s_name, "value_raw": str(v), "value": v})
    return out


def catalog(survey, title):
    return {"survey_name": survey, "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE7_BIRTH": (birth_rows, catalog("가상 인구동향조사", "가상 시도별 출생")),
    "FAKE7_RATE": (rate_rows, catalog("가상 노동력조사", "가상 시도별 고용 지표")),
    "FAKE7_HOUSE": (house_rows, catalog("가상 가구조사", "가상 시도별 1인가구")),
    "FAKE7_RETURN": (lambda: item_rows(RETURN, "명"), catalog("가상 귀농귀촌조사", "가상 시도별 귀농귀촌 인구")),
    "FAKE7_FISH": (lambda: item_rows(FISH, "명"), catalog("가상 귀어조사", "가상 시도별 귀어 인구")),
    "FAKE7_WSEX": (worker_sex_rows, catalog("가상 고용조사", "가상 시도별 성별 취업자")),
    "FAKE7_MEMBER": (member_rows, catalog("가상 가족실태조사", "가상 시도별 연령대별 구성")),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop7") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), cat))) for t, (make, cat) in SPECS.items()}


def ask(res, question, *tbl_ids):
    got = calculate(question, [res[t] for t in tbl_ids])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return facts, [facts[i] for i in got.answer]


def refs_in(facts, fact):
    """The cells a derived fact was computed from, following derived inputs down to the cells."""
    out = []
    for i in re.findall(r"f\d+", fact.derivation or ""):
        f = facts[i]
        out += refs_in(facts, f) if f.derivation else [f.source_ref]
    return sorted(out)


def value_of(tbl, item, region, year):
    rows = SPECS[tbl][0]()
    return next(r["value"] for r in rows if r["item_id"] == item and r["dim1_code"] == region and r["period"] == year)


def cells(answer):
    return [f.source_ref for f in answer if f.source_ref]


def derived(answer):
    return [f for f in answer if f.derivation]


# ---------------------------------------------------------------- I: how many times one period is another


@pytest.mark.parametrize("question, table, item, region, subject, reference", [
    # the reference before "의 몇 배"
    ("2023년 서울 출생아 수는 2013년의 몇 배야?", "FAKE7_BIRTH", "B1", "11", "2023", "2013"),
    # the reference before 대비 / 에 비해, named first or second
    ("2015년 대비 2022년 부산 출생아 수는 몇 배인가요?", "FAKE7_BIRTH", "B1", "21", "2022", "2015"),
    ("대구의 2020년 1인가구수는 2014년에 비해 몇 배 늘었어?", "FAKE7_HOUSE", "H1", "22", "2020", "2014"),
    # the earlier period is the subject: not later over earlier by default
    ("2013년 전국 출생아 수는 2023년보다 몇 배 많았어?", "FAKE7_BIRTH", "B1", "00", "2013", "2023"),
])
def test_ratio_of_two_periods(res, question, table, item, region, subject, reference):
    facts, answer = ask(res, question, table)
    ratios = derived(answer)
    assert len(ratios) == 1
    sub, ref = f"{table}|{item}|{region}|{subject}", f"{table}|{item}|{region}|{reference}"
    assert refs_in(facts, ratios[0]) == sorted([sub, ref])
    assert ratios[0].derivation.count("/") == 1 and "-" not in ratios[0].derivation
    want = value_of(table, item, region, subject) / value_of(table, item, region, reference)
    assert ratios[0].value == pytest.approx(round(want, 2), abs=0.01)
    assert sorted(cells(answer)) == sorted([sub, ref])


# ---------------------------------------------------------------- C: each target's change


def _each(facts, answer, table, item, regions, years, fn):
    changes = derived(answer)
    assert len(changes) == len(regions)  # one per target, no difference of the changes
    early, late = years
    for change, region in zip(sorted(changes, key=lambda f: refs_in(facts, f)), sorted(regions)):
        a, b = f"{table}|{item}|{region}|{early}", f"{table}|{item}|{region}|{late}"
        assert refs_in(facts, change) == sorted([a, b])
        assert change.value == pytest.approx(round(fn(value_of(table, item, region, late),
                                                      value_of(table, item, region, early)), 2), abs=0.01)
    assert sorted(cells(answer)) == sorted(f"{table}|{item}|{r}|{y}" for r in regions for y in years)


@pytest.mark.parametrize("question, table, item, regions, years", [
    ("2015년부터 2020년까지 서울과 부산의 출생아 수는 각각 얼마나 줄었어?", "FAKE7_BIRTH", "B1", ("11", "21"), ("2015", "2020")),
    ("2014년과 2021년 사이에 부산과 대구의 1인가구는 몇 가구 늘었어?", "FAKE7_HOUSE", "H1", ("21", "22"), ("2014", "2021")),
    # a rate of percent cells asked in points
    ("2016년 대비 2023년 서울과 대구의 고용률은 저마다 몇 %p 올랐어?", "FAKE7_RATE", "E1", ("11", "22"), ("2016", "2023")),
])
def test_each_targets_amount(res, question, table, item, regions, years):
    facts, answer = ask(res, question, table)
    _each(facts, answer, table, item, regions, years, lambda late, early: late - early)


@pytest.mark.parametrize("question, table, item, regions, years", [
    ("2018년 대비 2023년 서울과 대구 출생아 수는 각각 몇 퍼센트 줄었어?", "FAKE7_BIRTH", "B1", ("11", "22"), ("2018", "2023")),
    # the noun of a rate with a word of each: the two rates only
    ("전국과 부산의 2013년 대비 2019년 1인가구수 증가율을 각각 알려줘", "FAKE7_HOUSE", "H1", ("00", "21"), ("2013", "2019")),
])
def test_each_targets_rate(res, question, table, item, regions, years):
    facts, answer = ask(res, question, table)
    _each(facts, answer, table, item, regions, years, lambda late, early: (late / early - 1) * 100)


def test_which_changed_more_keeps_the_difference(res):
    """Neighbour: a question for which of the two changed more still has the difference of the two rates."""
    facts, answer = ask(res, "2013년 대비 2019년 서울과 부산 중 어느 쪽 출생아 수 감소율이 더 커?", "FAKE7_BIRTH")
    assert len(derived(answer)) == 3


def test_one_target_change_unchanged(res):
    """Neighbour: one target, the same words: its change, as before."""
    facts, answer = ask(res, "2015년부터 2020년까지 서울의 출생아 수는 얼마나 줄었어?", "FAKE7_BIRTH")
    diff = next(f for f in derived(answer) if "-" in f.derivation and "/" not in f.derivation)
    assert refs_in(facts, diff) == ["FAKE7_BIRTH|B1|11|2015", "FAKE7_BIRTH|B1|11|2020"]


# ---------------------------------------------------------------- B: a group and an attribute named apart


@pytest.mark.parametrize("question, table, want", [
    ("2023년 서울 귀농가구원 중 여자는 몇 명이야?", "FAKE7_RETURN", "FAKE7_RETURN|T03|11|2023"),
    ("2022년 부산에 사는 귀촌인 남자 수 알려줘", "FAKE7_RETURN", "FAKE7_RETURN|T05|21|2022"),
    ("2023년 대구 귀농가구원 가운데 남성은 몇 명?", "FAKE7_RETURN", "FAKE7_RETURN|T02|22|2023"),
    # no mark: the group's name is its total, the details join the name and the attribute
    ("2023년 서울 귀어인 가운데 여자는?", "FAKE7_FISH", "FAKE7_FISH|F03|11|2023"),
    ("2022년 부산에 사는 귀어인 중 남자는 몇 명이야?", "FAKE7_FISH", "FAKE7_FISH|F02|21|2022"),
])
def test_group_and_attribute_apart(res, question, table, want):
    _, answer = ask(res, question, table)
    assert cells(answer) == [want]


def test_group_and_attribute_two_regions(res):
    facts, answer = ask(res, "2023년 서울과 부산의 귀촌인 여자 수 차이는?", "FAKE7_RETURN")
    diff = next(f for f in derived(answer))
    assert refs_in(facts, diff) == ["FAKE7_RETURN|T06|11|2023", "FAKE7_RETURN|T06|21|2023"]


@pytest.mark.parametrize("question, table, want", [
    # neighbours: the group alone is its total, the detail named whole is the detail
    ("2023년 서울 귀농가구원은 몇 명?", "FAKE7_RETURN", "FAKE7_RETURN|T01|11|2023"),
    ("2022년 부산 귀촌인-여자 수", "FAKE7_RETURN", "FAKE7_RETURN|T06|21|2022"),
    ("2023년 대구 귀어인 수 알려줘", "FAKE7_FISH", "FAKE7_FISH|F01|22|2023"),
])
def test_group_alone_unchanged(res, question, table, want):
    _, answer = ask(res, question, table)
    assert cells(answer) == [want]


# ---------------------------------------------------------------- A-1: a group named only as where to look


@pytest.mark.parametrize("question, table, want", [
    ("2023년 서울 가구원 중 청년은 몇 명?", "FAKE7_MEMBER", ["FAKE7_MEMBER|G1|11|2023"]),
    ("부산의 2022년 가구원 가운데 노년 인구 알려줘", "FAKE7_MEMBER", ["FAKE7_MEMBER|G3|21|2022"]),
    # the region's total named before 중: the region named after it
    ("2023년 전국 중에서 대구의 출생아 수는 몇 명이야?", "FAKE7_BIRTH", ["FAKE7_BIRTH|B1|22|2023"]),
])
def test_group_only_where_to_look(res, question, table, want):
    _, answer = ask(res, question, table)
    assert cells(answer) == want


def test_group_only_where_to_look_change(res):
    facts, answer = ask(res, "2022년부터 2023년까지 대구 가구원 중 노년은 얼마나 늘었어?", "FAKE7_MEMBER")
    detail = ["FAKE7_MEMBER|G3|22|2022", "FAKE7_MEMBER|G3|22|2023"]
    assert all(refs_in(facts, f) == detail for f in derived(answer))  # nothing of the group's total
    assert sorted(cells(answer)) == detail
    diff = next(f for f in derived(answer) if "/" not in f.derivation)
    assert diff.value == value_of("FAKE7_MEMBER", "G3", "22", "2023") - value_of("FAKE7_MEMBER", "G3", "22", "2022")


def test_group_share_unchanged(res):
    """Neighbour: the same words with a share word: the group is the whole, as before."""
    facts, answer = ask(res, "2023년 서울 가구원 중 청년 비중은?", "FAKE7_MEMBER")
    assert refs_in(facts, answer[-1]) == ["FAKE7_MEMBER|G0|11|2023", "FAKE7_MEMBER|G1|11|2023"]


def test_attribute_of_another_category_unchanged(res):
    """Neighbour: the attribute is a code of another category (성별): that category takes it, the item stays."""
    _, answer = ask(res, "2023년 서울 취업자 중 여자는 몇 명?", "FAKE7_WSEX")
    assert cells(answer) == ["FAKE7_WSEX|W1|11|2|2023"]


def test_group_compared_with_its_detail(res):
    """Neighbour: the group said twice, once on its own: the group's total and the detail are both named."""
    facts, answer = ask(res, "2023년 서울 귀농가구원 전체와 귀농가구원 여자의 차이는?", "FAKE7_RETURN")
    diff = next(f for f in derived(answer))
    assert refs_in(facts, diff) == ["FAKE7_RETURN|T01|11|2023", "FAKE7_RETURN|T03|11|2023"]
