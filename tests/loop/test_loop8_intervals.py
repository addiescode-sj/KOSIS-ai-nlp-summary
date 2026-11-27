"""Loop 8 (analyze/rules.py _number_range), A-1: a category with closed intervals and several open intervals of
different bounds ('65세이상', '70세이상', '75세이상'). A question for "N 이상" where an open interval starts at N is
that interval's cell, not the next bound's. A category of one-unit names and one open tail ('1인'..'4인', '5인이상')
still adds the names up to the tail.

The tables are made up (가상 데이터).
"""

from __future__ import annotations

import pytest

from conftest import write_table
from kosis_chat.adapters.parquet import load_table
from kosis_chat.calc import calculate
from kosis_chat.schemas.resources import load_resources

REGIONS = (("00", "전국"), ("11", "서울특별시"), ("21", "부산광역시"))
AGES = (("A0", "계"), ("A1", "0-14세"), ("A2", "15-64세"), ("A3", "65세이상"), ("A4", "70세이상"), ("A5", "75세이상"),
        ("A6", "85세이상"))
# another table: the bounds spaced by ten, with sex
OLD = (("B0", "합계"), ("B1", "20-59세"), ("B2", "60세 이상"), ("B3", "70세 이상"), ("B4", "80세 이상"))
SIZES = (("S0", "계"), ("S1", "1인"), ("S2", "2인"), ("S3", "3인"), ("S4", "4인"), ("S5", "5인이상"))


def rows_of(tbl, item, name, unit, dim_name, labels, sexes=None):
    out = []
    for y_no, year in enumerate(("2021", "2022", "2023")):
        for r_no, (rc, rl) in enumerate(REGIONS):
            for s_no, sex in enumerate(sexes or [None]):
                for n, (code, label) in enumerate(labels):
                    v = 1000 * (len(labels) - n) + 37 * y_no + 11 * r_no + 5 * s_no
                    row = {"period": year, "item_id": item, "item_name": name, "unit": unit, "dim1_name": "시도별",
                           "dim1_code": rc, "dim1_label": rl, "dim2_name": dim_name, "dim2_code": code,
                           "dim2_label": label, "value_raw": str(v), "value": v}
                    if sex:
                        row.update({"dim3_name": "성별", "dim3_code": sex[0], "dim3_label": sex[1]})
                    out.append(row)
    return out


def catalog(title):
    return {"survey_name": "가상 인구조사", "organization": "가상 기관", "table_name": title, "period": "년",
            "url": "https://example.invalid/t"}


SPECS = {
    "FAKE8_AGE": (lambda: rows_of("FAKE8_AGE", "P1", "인구", "명", "연령별", AGES), catalog("가상 시도별 연령별 인구")),
    "FAKE8_OLD": (lambda: rows_of("FAKE8_OLD", "W1", "취업자", "명", "연령별", OLD,
                                  [("0", "계"), ("1", "남자"), ("2", "여자")]), catalog("가상 시도별 성별 연령별 취업자")),
    "FAKE8_SIZE": (lambda: rows_of("FAKE8_SIZE", "H1", "가구", "가구", "가구원수별", SIZES), catalog("가상 가구원수별 가구")),
}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    folder = tmp_path_factory.mktemp("loop8i") / "data"
    return {t: load_resources(load_table(write_table(folder, t, make(), cat))) for t, (make, cat) in SPECS.items()}


def answer_refs(res, question, table):
    got = calculate(question, [res[table]])
    assert got.status == "ok", (got.status, got.reason, got.detail)
    facts = {f.id: f for f in got.facts}
    return [facts[i] for i in got.answer]


@pytest.mark.parametrize("question, table, want", [
    ("2023년 65세 이상 인구는?", "FAKE8_AGE", "FAKE8_AGE|P1|00|A3|2023"),
    ("2022년 서울의 75세 이상 인구 알려줘", "FAKE8_AGE", "FAKE8_AGE|P1|11|A5|2022"),
    ("부산 70세 이상 인구는 2021년에 몇 명이었어?", "FAKE8_AGE", "FAKE8_AGE|P1|21|A4|2021"),
    ("2023년 서울 60세 이상 여자 취업자 수", "FAKE8_OLD", "FAKE8_OLD|W1|11|B2|2|2023"),
    ("2022년 부산의 70세 이상 남성 취업자는?", "FAKE8_OLD", "FAKE8_OLD|W1|21|B3|1|2022"),
    ("2021년 전국 80세 이상 취업자 수", "FAKE8_OLD", "FAKE8_OLD|W1|00|B4|0|2021"),
])
def test_open_interval_at_the_bound(res, question, table, want):
    answer = answer_refs(res, question, table)
    assert [f.source_ref for f in answer] == [want]
    assert answer[0].value == next(r["value"] for r in SPECS[table][0]()
                                   if "|".join([table, r["item_id"], r["dim1_code"], r["dim2_code"],
                                                *([r["dim3_code"]] if "dim3_code" in r else []), r["period"]]) == want)


def test_one_unit_names_and_a_tail_unchanged(res):
    """Neighbour: '4인 이상' on 1인..4인, 5인이상 still takes 4인 and 5인이상 (the open tail above the bound)."""
    answer = answer_refs(res, "2023년 서울 4인 이상 가구 수는?", "FAKE8_SIZE")
    assert sorted(f.source_ref for f in answer if f.source_ref) == ["FAKE8_SIZE|H1|11|S4|2023", "FAKE8_SIZE|H1|11|S5|2023"]
