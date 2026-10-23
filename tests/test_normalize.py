"""Tests for src/kosis_ai/normalize.py: fetch_table on the provided data, unit and measure_kind rules."""

import json
from pathlib import Path

import pandas as pd
import pytest

from kosis_ai.normalize import LABEL_COLUMNS, LookupFailed, fetch_table, measure_kind, normalize, resolve_unit
from kosis_ai.schemas import Cell

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden" / "stat_golden.jsonl"
VALUES = ROOT / "data" / "values"
GOLDEN_TABLES = ["101_DT_1B8000F", "101_DT_1B8000H", "101_DT_1JC1502", "101_DT_1B08024", "101_DT_101094A001"]


@pytest.fixture(autouse=True)
def _needs_data(request):
    if "offline" not in request.keywords and not VALUES.exists():
        pytest.skip("provided data is not in the repo (gitignored); place it at data/")


offline = pytest.mark.offline


# --- fetch_table on the provided data ------------------------------------------------------------------------


def test_golden_cell_facts_keep_their_unit_through_fetch_table():
    if not GOLDEN.exists():
        pytest.skip("golden set is not in the repo (gitignored)")
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    facts = [f for r in rows for f in r.get("required_facts", []) + r.get("aux_facts", []) if f.get("source_ref")]
    cells = {}
    for tid in {"101_" + f["source_ref"].split("|")[0] for f in facts}:
        for _, row in fetch_table(tid).iterrows():
            cells[Cell.from_row({k: (None if pd.isna(v) else v) for k, v in row.items()}).ref()] = row
    assert len(facts) == 67
    assert [f["source_ref"] for f in facts if cells[f["source_ref"]]["unit_resolved"] != f["unit"]] == []


@pytest.mark.parametrize("tid", GOLDEN_TABLES)
def test_column_rules_match_the_single_cell_rules(tid):
    df = fetch_table(tid)
    labels = [c for c in LABEL_COLUMNS if c in df]
    for r in df.drop_duplicates(["unit", "item_name", *labels]).itertuples():
        dims = [getattr(r, c) for c in labels if isinstance(getattr(r, c), str)]
        unit, source = resolve_unit(r.unit, r.item_name, dims, r.unit_resolved if r.unit_source == "meta" else None)
        assert (unit, source) == (r.unit_resolved, r.unit_source)
        assert measure_kind(r.tbl_id, unit, r.item_name, dims) == r.measure_kind


def test_periods_are_filtered_and_missing_ones_refused():
    df = fetch_table("101_DT_1B08024", {"periods": ["2020"], "행정구역별(시도)": "충청북도"})
    assert list(df["period"]) == ["2020"]
    with pytest.raises(LookupFailed) as err:
        fetch_table("101_DT_1B08024", {"periods": ["2003"]})
    assert err.value.reason == "no_period"


def test_missing_table_is_an_error_not_an_empty_frame():
    with pytest.raises(FileNotFoundError):
        fetch_table("101_DT_NOT_A_TABLE")


@pytest.mark.parametrize("tid", ["101_DT_1PD1006", "101_DT_1JC2025"])
def test_tables_with_a_hidden_category_are_refused(tid):
    with pytest.raises(LookupFailed) as err:
        fetch_table(tid)
    assert err.value.reason == "class_mismatch"


def test_symbols_are_labelled_and_never_zero():
    df = fetch_table("101_DT_1B8000F", {"periods": ["2025"], "기본항목별": "기대수명(출생시 기대여명)"})
    row = df.iloc[0]
    assert row["value_raw"].strip() == "-" and pd.isna(row["value"]) and row["value_label"] == "해당 없음"


def test_measure_kinds_on_golden_tables():
    f = fetch_table("101_DT_1B8000F", {"periods": ["2024"]}).set_index("dim1_label")["measure_kind"]
    assert f["출생아수(명)"] == "count_flow"  # dim label decides, not the title-like item '... 동태율 추이'
    assert f["조출생률(천명당)"] == "ratio"
    assert f["합계출산율(명)"] == "ratio"
    assert f["기대수명(출생시 기대여명)"] == "average"
    h = fetch_table("101_DT_1JC1502", {"periods": ["2020"], "행정구역별(읍면동)": "전국"}).set_index("item_name")["measure_kind"]
    assert h["1인 (가구)"] == "count_stock"
    assert fetch_table("101_DT_1B08024", {"periods": ["2020"]})["measure_kind"].unique().tolist() == ["ratio"]


# --- measure_kind for the R8 cases (the refusal itself is in compute, T1-07) -------------------------------------


def test_golden_rate_sum_case_is_marked_ratio():
    """S-REF-06: 2024년 17개 시도 조출생률을 다 더하면? — every rate is a ratio, so R8 can refuse the sum."""
    df = fetch_table("101_DT_1B8000H", {"periods": ["2024"], "item": "조출생률 (천명당)"})
    assert set(df["measure_kind"]) == {"ratio"}


def test_r8_checklist_cases_are_marked():
    percent = fetch_table("101_DT_101094A001", {"periods": ["2023"], "년생별": "1992"})
    assert set(percent["measure_kind"]) == {"ratio"}  # '% 항목 두 개의 합산' must be refused
    homes = fetch_table("101_DT_1JC1502", {"item": "1인", "행정구역별(읍면동)": "전국", "periods": ["2023", "2024"]})
    assert set(homes["measure_kind"]) == {"count_stock"}  # '연말 인구·가구를 여러 해 합산' must be refused
    births = fetch_table("101_DT_1B8000H", {"item": "출생건수 (명)", "행정구역별": "전국", "periods": ["2023", "2024"]})
    assert set(births["measure_kind"]) == {"count_flow"}  # flows over complete periods may add up


def test_unknown_item_and_dim_are_refused():
    with pytest.raises(LookupFailed) as err:
        fetch_table("101_DT_1B8000H", {"periods": ["2024"], "item": "없는 항목"})
    assert err.value.reason == "no_item"
    with pytest.raises(LookupFailed) as err:
        fetch_table("101_DT_1B8000H", {"periods": ["2024"], "행정구역별": "없는 지역"})
    assert err.value.reason == "class_mismatch"


def test_same_label_at_two_levels_resolves_to_top_level():
    df = fetch_table("101_DT_1JC1502", {"periods": ["2025"], "item": "일반가구_계", "행정구역별(읍면동)": "동부"})
    assert df["dim1_code"].tolist() == ["03"]  # not 부산 동부 21003 and the other 13


# --- unit rules on hand-made input -----------------------------------------------------------------------------


@offline
@pytest.mark.parametrize(
    "unit, item, dims, meta, expected",
    [
        pytest.param("가구", "1인", [], None, ("가구", "unit_column"), id="unit column"),
        pytest.param("월세없음", "보증금 (만원)", [], None, ("만원", "parentheses"), id="fake unit column value is ignored"),
        pytest.param(None, "출생건수 (명)", ["서울"], None, ("명", "parentheses"), id="item parentheses"),
        pytest.param(None, "총조사인구", ["15세 이상 (미취학 포함)"], None, (None, "unknown"), id="non-unit parentheses"),
        pytest.param(None, "합계출산율", ["서울"], None, ("명", "override"), id="override"),
        pytest.param(None, "총인구", ["전국"], "명", ("명", "meta"), id="safe metadata unit"),
        pytest.param(None, "총부양비", ["전국"], "명", (None, "unknown"), id="metadata unit never on a ratio item"),
    ],
)
def test_resolve_unit(unit, item, dims, meta, expected):
    assert resolve_unit(unit, item, dims, meta) == expected


@offline
def test_metadata_unit_is_dropped_when_another_unit_appears(tmp_path):
    (tmp_path / "table_metadata").mkdir()
    (tmp_path / "table_metadata" / "T.json").write_text(json.dumps({"unitNmKor": "가구"}), encoding="utf-8")
    df = pd.DataFrame(
        {"tbl_id": "T", "unit": [None, "십만원"], "item_name": ["가구수", "보증금"], "dim1_label": ["전국", "전국"],
         "value_kind": "ok"}
    )  # fmt: skip
    out = normalize(df, "T", tmp_path)
    assert out["unit_resolved"].tolist() == [None, "십만원"]  # '가구' from the metadata would be a guess here
