"""Loop 6 (prepare/index.py, judge/question.py, shared/labels.py, shared/patterns.py, analyze/rules.py,
analyze/multi_rules.py):

- A-1: a total marked on a stem by a hyphen, an underscore or a space ('취업자-계', '사업체_합계') is the category's
  total, and is asked by its stem ('취업자').
- A-2: a whole, subtotals with a total mark and their details, in table order, are linked as parent and children
  even though their codes do not nest; a share asked inside a named group ("임금근로자 중", "취업자 가운데") takes that
  group as the whole.
- A-3: a ranking inside a named group ranks the group's members only.
- B-1 / B-2: an interval name ('20-24세') is also asked with another joining mark, with the counting word after each
  bound, "A부터 B까지", and as a half of its ten-year group ('20대 초반', '40대 후반').
- C: a label inside a longer word that comes from the survey's name or the table's title is not named; a two-character
  region name inside a word is not named; a short label said on its own beside a longer one that holds it is named.
- G: one metric kept in two tables, one per survey year: the periods are split between the tables (values, or the
  change between the two years).

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

# an item column: a whole, two subtotals, their details; the codes do not nest by prefix
WORKERS = (("T10", "취업자-계"), ("T20", "임금근로자-계"), ("T21", "상용근로자"), ("T22", "임시근로자"),
           ("T30", "비임금근로자-계"), ("T31", "자영업자"), ("T32", "무급가족종사자"))


def worker_rows():
    out = []
    for y_no, year in enumerate(("2022", "2023")):
        for r_no, (code, label) in enumerate(REGIONS):
            base = 100 * (4 - r_no) + 7 * y_no
            v = {"T21": base * 3, "T22": base * 2 - 11, "T31": base + 13, "T32": base // 3}
            v["T20"], v["T30"] = v["T21"] + v["T22"], v["T31"] + v["T32"]
            v["T10"] = v["T20"] + v["T30"]
            for item, name in WORKERS:
                out.append({"period": year, "item_id": item, "item_name": name, "unit": "천명", "dim1_name": "행정구역별",
                            "dim1_code": code, "dim1_label": label, "value_raw": str(v[item]), "value": v[item]})
    return out


# a category (not the item): the same shape with an underscore and a space for the mark
SIZES = (("A", "사업체_합계"), ("B", "소규모_합계"), ("B1", "1~4인"), ("B2", "5~9인"), ("C", "중대규모 합계"),
         ("C1", "10~99인"), ("C2", "100인 이상"))


def size_rows():
    out = []
    for y_no, year in enumerate(("2021", "2022")):
        for r_no, (code, label) in enumerate(REGIONS[:3]):
            base = 50 * (3 - r_no) + 3 * y_no
            v = {"B1": base * 4, "B2": base * 2 + 5, "C1": base + 9, "C2": base // 2}
            v["B"], v["C"] = v["B1"] + v["B2"], v["C1"] + v["C2"]
            v["A"] = v["B"] + v["C"]
            for size, name in SIZES:
                out.append({"period": year, "item_id": "E1", "item_name": "사업체수", "unit": "개", "dim1_name": "시도별",
                            "dim1_code": code, "dim1_label": label, "dim2_name": "종사자규모별", "dim2_code": size,
                            "dim2_label": name, "value_raw": str(v[size]), "value": v[size]})
    return out


AGES = [("000", "계")] + [(f"{a:03d}", f"{a}-{a + 4}세") for a in range(15, 65, 5)] + [("065", "65세 이상")]


def age_rows():
    out = []
    for y_no, year in enumerate(("2022", "2023")):
        for r_no, (code, label) in enumerate(REGIONS):
            for a_no, (age, name) in enumerate(AGES):
                v = 5000 if age == "000" else 50 + a_no * 3 + r_no * 7 + y_no
                out.append({"period": year, "item_id": "T1", "item_name": "취업자수", "unit": "명", "dim1_name": "행정구역별",
                            "dim1_code": code, "dim1_label": label, "dim2_name": "연령별", "dim2_code": age,
                            "dim2_label": name, "value_raw": str(v), "value": v})
    return out


GROUPS = (("0", "전체"), ("1", "청년"), ("2", "남자"), ("3", "여자"), ("4", "수도권"), ("5", "비수도권"))
ANSWERS = (("0", "계"), ("1", "매우 만족"), ("2", "약간 만족"), ("3", "보통"), ("4", "불만족"))


def survey_rows():
    out = []
    for year in ("2022", "2023"):
        for g_no, (group, name) in enumerate(GROUPS):
            for (answer, label), v in zip(ANSWERS, (100.0, 10 + g_no, 30 + g_no, 40 - 2 * g_no, 20 + g_no)):
                out.append({"period": year, "item_id": "S1", "item_name": "생활만족도", "unit": "%",
                            "dim1_name": "응답자특성별", "dim1_code": group, "dim1_label": name, "dim2_name": "응답별",
                            "dim2_code": answer, "dim2_label": label, "value_raw": str(v), "value": v})
    return out


def year_rows(year, item, base):
    return [{"period": year, "item_id": "H1", "item_name": item, "unit": "%", "dim1_name": "시도별", "dim1_code": c,
             "dim1_label": l, "value_raw": str(base + n * 1.5), "value": base + n * 1.5} for n, (c, l) in enumerate(REGIONS)]


def catalog(survey, title):
    return {"survey_name": survey, "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE6_WORK": (worker_rows, catalog("가상 지역별고용조사", "가상 시도별 종사상지위별 취업자")),
    "FAKE6_SIZE": (size_rows, catalog("가상 사업체조사", "가상 시도별 종사자규모별 사업체수")),
    "FAKE6_AGE": (age_rows, catalog("가상 경제활동조사", "가상 시도별 연령별 취업자")),
    "FAKE6_LIFE": (survey_rows, catalog("가상 청년삶 실태조사", "가상 응답자특성별 생활만족도")),
    "FAKE6_LIFE_PLAIN": (survey_rows, catalog("가상 생활조사", "가상 응답자특성별 생활만족도")),
    "FAKE6_SMOKE20": (lambda: year_rows("2020", "흡연율", 20.5), catalog("가상 건강조사", "가상 시도별 흡연율 (2020)")),
    "FAKE6_SMOKE23": (lambda: year_rows("2023", "흡연율", 18.25), catalog("가상 건강조사", "가상 시도별 흡연율 (2023)")),
    "FAKE6_DRINK23": (lambda: year_rows("2023", "고위험음주율", 11.0), catalog("가상 건강조사", "가상 시도별 고위험음주율")),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop6") / "data"
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


def cell_value(facts, ref):
    return next(f.value for f in facts.values() if f.source_ref == ref)


def cells(answer):
    return [f.source_ref for f in answer if f.source_ref]


# ---------------------------------------------------------------- A-1: a total marked on a stem


@pytest.mark.parametrize("question, table, want", [
    # no item named: the whole of the item column, not a subtotal
    ("2023년 서울 취업자 수는?", "FAKE6_WORK", "FAKE6_WORK|T10|11|2023"),
    ("부산의 2022년 취업자는 몇 명이야?", "FAKE6_WORK", "FAKE6_WORK|T10|21|2022"),
    ("2023년 대구광역시 전체 규모는 얼마야?", "FAKE6_WORK", "FAKE6_WORK|T10|22|2023"),
    # a subtotal asked by its stem
    ("2023년 부산 임금근로자는 몇 명이야?", "FAKE6_WORK", "FAKE6_WORK|T20|21|2023"),
    ("2022년 서울 비임금근로자 수 알려줘", "FAKE6_WORK", "FAKE6_WORK|T30|11|2022"),
    # a category of another table: the whole by an underscore mark, a subtotal by a space mark
    ("2022년 서울 사업체수는?", "FAKE6_SIZE", "FAKE6_SIZE|E1|11|A|2022"),
    ("2021년 부산 중대규모 사업체는 몇 개야?", "FAKE6_SIZE", "FAKE6_SIZE|E1|21|C|2021"),
])
def test_marked_total_and_stem(res, question, table, want):
    _, answer = ask(res, question, table)
    assert cells(answer) == [want]


# ---------------------------------------------------------------- A-2: a share inside a named group


@pytest.mark.parametrize("question, table, part, whole", [
    # the subtotal named before 중 / 가운데 is the whole
    ("2023년 서울 임금근로자 중 상용근로자 비중은?", "FAKE6_WORK", "FAKE6_WORK|T21|11|2023", "FAKE6_WORK|T20|11|2023"),
    ("부산의 2022년 비임금근로자 가운데 무급가족종사자가 차지하는 비중은?", "FAKE6_WORK", "FAKE6_WORK|T32|21|2022",
     "FAKE6_WORK|T30|21|2022"),
    # the whole named: a detail's share of the whole, not of its subtotal, and the region stays on both
    ("2023년 서울 취업자 중 자영업자 비중은?", "FAKE6_WORK", "FAKE6_WORK|T31|11|2023", "FAKE6_WORK|T10|11|2023"),
    ("2022년 대구 전체 취업자 가운데 임시근로자 구성비는?", "FAKE6_WORK", "FAKE6_WORK|T22|22|2022", "FAKE6_WORK|T10|22|2022"),
    # a category of another table
    ("2022년 서울 소규모 사업체 중 1~4인 사업체 비중은?", "FAKE6_SIZE", "FAKE6_SIZE|E1|11|B1|2022", "FAKE6_SIZE|E1|11|B|2022"),
])
def test_share_inside_named_group(res, question, table, part, whole):
    facts, answer = ask(res, question, table)
    share = answer[-1]
    assert refs_in(facts, share) == sorted([part, whole])
    assert share.value == pytest.approx(round(cell_value(facts, part) / cell_value(facts, whole) * 100, 2))


# ---------------------------------------------------------------- A-3: a ranking inside a named group


@pytest.mark.parametrize("question, table, pool, pick", [
    ("2023년 서울 비임금근로자 가운데 가장 많은 종사상지위는?", "FAKE6_WORK", ("T31", "T32"), max),
    ("2022년 부산 임금근로자 중에서 가장 적은 것은 뭐야?", "FAKE6_WORK", ("T21", "T22"), min),
    ("2022년 서울 중대규모 사업체 중 가장 많은 규모는?", "FAKE6_SIZE", ("C1", "C2"), max),
])
def test_ranking_inside_named_group(res, question, table, pool, pick):
    facts, answer = ask(res, question, table)
    rows = {r["item_id"] if table == "FAKE6_WORK" else r["dim2_code"]: r["value"] for r in SPECS[table][0]()
            if r["period"] == re.search(r"20\d\d", question).group(0)
            and r["dim1_code"] == ("11" if "서울" in question else "21")}
    best = pick(pool, key=lambda c: rows[c])
    assert len(cells(answer)) == 1 and f"|{best}|" in cells(answer)[0]
    assert answer[0].value == rows[best]


def test_value_named_in_full_still_works(res):
    """Neighbour: the name written in full with its mark is still the same cell."""
    _, answer = ask(res, "2023년 서울 취업자-계는?", "FAKE6_WORK")
    assert cells(answer) == ["FAKE6_WORK|T10|11|2023"]


# ---------------------------------------------------------------- B-1 / B-2: an interval joined another way


@pytest.mark.parametrize("question, want", [
    ("2023년 서울 20~24세 취업자 수", "FAKE6_AGE|T1|11|020|2023"),
    ("2022년 부산의 30세부터 34세까지 취업자는?", "FAKE6_AGE|T1|21|030|2022"),
    ("2023년 대구광역시 45세~49세 취업자는 몇 명이야?", "FAKE6_AGE|T1|22|045|2023"),
    ("2023년 서울 20대 초반 취업자 수", "FAKE6_AGE|T1|11|020|2023"),
    ("2022년 부산 40대 후반 취업자는 몇 명?", "FAKE6_AGE|T1|21|045|2022"),
    ("2023년 서울 50대 전반 취업자 수 알려줘", "FAKE6_AGE|T1|11|050|2023"),
])
def test_interval_joined_another_way(res, question, want):
    _, answer = ask(res, question, "FAKE6_AGE")
    assert cells(answer) == [want]


def test_interval_halves_compared(res):
    facts, answer = ask(res, "2023년 서울 30대 초반과 30대 후반 취업자 수 차이는?", "FAKE6_AGE")
    diff = next(f for f in answer if f.derivation and "-" in f.derivation)
    assert refs_in(facts, diff) == ["FAKE6_AGE|T1|11|030|2023", "FAKE6_AGE|T1|11|035|2023"]


def test_region_inside_a_word_is_not_named(res):
    """'연령대 구간' holds 대구 only when the spaces are taken out: no second region."""
    _, answer = ask(res, "2023년 서울의 연령대 구간별 취업자 가운데 25-29세는 몇 명이야?", "FAKE6_AGE")
    assert cells(answer) == ["FAKE6_AGE|T1|11|025|2023"]


# ---------------------------------------------------------------- C: words of the survey's name


@pytest.mark.parametrize("question, table, groups", [
    # 청년 inside the survey's name (청년삶) is not the 청년 group
    ("2023년 청년삶 실태조사에서 남자와 여자의 보통 응답 비율 차이는?", "FAKE6_LIFE", ("2", "3")),
    ("청년삶조사 결과로 2022년 남자와 여자의 불만족 비율 차이는 얼마야?", "FAKE6_LIFE", ("2", "3")),
    # a short label said on its own beside a longer one that holds it (수도권, 비수도권)
    ("2023년 수도권과 비수도권의 보통 응답 비율 차이는?", "FAKE6_LIFE_PLAIN", ("4", "5")),
])
def test_survey_name_words(res, question, table, groups):
    facts, answer = ask(res, question, table)
    diff = next(f for f in answer if f.derivation and "-" in f.derivation)
    year = re.search(r"20\d\d", question).group(0)
    answer_code = "3" if "보통" in question else "4"
    assert refs_in(facts, diff) == sorted(f"{table}|S1|{g}|{answer_code}|{year}" for g in groups)


def test_label_said_as_its_own_word_is_kept(res):
    """Neighbour: 청년 said as a word of its own is the group, even in a table whose survey name holds it."""
    _, answer = ask(res, "2023년 청년삶 실태조사의 청년 응답자 중 보통 응답 비율은?", "FAKE6_LIFE")
    assert cells(answer) == ["FAKE6_LIFE|S1|1|3|2023"]


# ---------------------------------------------------------------- G: one metric in a table per survey year


@pytest.mark.parametrize("question, tables, region", [
    ("2020년과 2023년 서울 흡연율을 각각 알려줘", ("FAKE6_SMOKE20", "FAKE6_SMOKE23"), "11"),
    ("부산 흡연율 2020년 값과 2023년 값을 알려줘", ("FAKE6_SMOKE23", "FAKE6_SMOKE20"), "21"),
])
def test_split_years_listed(res, question, tables, region):
    _, answer = ask(res, question, *tables)
    assert cells(answer) == [f"FAKE6_SMOKE20|H1|{region}|2020", f"FAKE6_SMOKE23|H1|{region}|2023"]


@pytest.mark.parametrize("question, region, rate", [
    ("대구 흡연율은 2020년에서 2023년으로 몇 퍼센트 줄었어?", "22", True),
    ("2020년 대비 2023년 서울 흡연율 증감률은?", "11", True),
    ("2020년과 2023년 사이 부산 흡연율은 얼마나 변했어?", "21", False),
])
def test_split_years_change(res, question, region, rate):
    facts, answer = ask(res, question, "FAKE6_SMOKE23", "FAKE6_SMOKE20")
    early, late = f"FAKE6_SMOKE20|H1|{region}|2020", f"FAKE6_SMOKE23|H1|{region}|2023"
    derived = [f for f in answer if f.derivation]
    assert len(derived) == 1 and refs_in(facts, derived[0]) == sorted([early, late])
    a, b = cell_value(facts, early), cell_value(facts, late)
    want = round((b / a - 1) * 100, 2) if rate else round(b - a, 2)
    assert derived[0].value == pytest.approx(want)


def test_split_years_not_for_two_metrics(res):
    """Neighbour: two tables of different metrics are not one metric split by year."""
    got = calculate("2020년과 2023년 서울 흡연율과 고위험음주율", [res["FAKE6_SMOKE20"], res["FAKE6_DRINK23"]])
    assert got.status != "ok" or not any(f.source_ref.endswith("|2020") and "DRINK" in f.source_ref
                                         for f in got.facts if f.source_ref)
