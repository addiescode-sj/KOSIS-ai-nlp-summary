"""Input contract, the Parquet loader and baseline cleaning."""

from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from kosis_chat.data import preprocessing, serializer
from kosis_chat.data.loader import UNKNOWN_UNIT, infer_unit, load_table, sha256_of
from kosis_chat.data.schema import SchemaError, validate
from tests.conftest import pop_rows, write_table


def test_parquet_loads_with_catalog_meta_and_raw_values(pop_path):
    table = load_table(pop_path)
    assert table.meta.input_format == "parquet"
    assert table.meta.file_sha256 == sha256_of(pop_path)
    assert (table.meta.tbl_id, table.meta.tbl_nm, table.meta.survey_nm) == ("FAKE_TBL_001", "가상 성별 인구", "가상 인구조사")
    assert table.meta.source_url == "https://example.invalid/fake-table-001"
    assert "년 (2022 ~ 2024)" in table.meta.note
    assert table.cells[0].dims == {"성별": "계"} and table.cells[0].dim_codes == {"성별": "0"}
    assert table.cells[0].value_raw == "1,000"
    assert table.period_range == ("2022", "2024")
    assert validate(table) == []


def test_source_ref_and_item_code_come_from_kosis_codes(pop_path, household_path):
    assert load_table(pop_path).cells[2].source_ref == "FAKE_TBL_001|T1|2|2022"
    cell = load_table(household_path).cells[1]
    assert (cell.item_code, cell.source_ref) == ("T1", "FAKE_TBL_002|T1|00|2015")


def test_unit_is_inferred_from_item_name(household_path):
    table = load_table(household_path)
    assert {(c.unit, c.unit_source) for c in table.cells} == {("가구", "item_name")}


@pytest.mark.parametrize(
    ("unit", "item", "dims", "expected"),
    [
        ("명", "출생건수 (명)", [], ("명", "given")),
        (None, "출생건수 (명)", ["경기도"], ("명", "item_name")),
        ("", "인구동태건수", ["영아사망률(출생아 천명당)"], ("출생아 천명당", "dim_label")),
        (None, "인구동태건수", ["기대수명(출생시 기대여명) - 남"], ("년", "override")),  # parentheses are not a unit
        (None, "합계출산율", ["서울특별시"], ("명", "override")),
        (None, "가상 항목", ["가상(분류)"], (UNKNOWN_UNIT, "unknown")),
    ],
)
def test_infer_unit(unit, item, dims, expected):
    assert infer_unit(unit, item, dims) == expected


def test_other_formats_are_rejected(tmp_path):
    data = tmp_path / "pop.csv"
    data.write_text("item,period,value,unit\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="unsupported input format"):
        load_table(data)


def test_table_without_catalog_is_rejected(pop_path, data_dir):
    (data_dir / "tables.csv").unlink()
    with pytest.raises(SchemaError, match="tables.csv"):
        load_table(pop_path)


def test_table_missing_from_catalog_is_rejected(pop_path, data_dir):
    path = pop_path.with_name("101_OTHER.parquet")
    path.write_bytes(pop_path.read_bytes())
    with pytest.raises(SchemaError, match="101_OTHER"):
        load_table(path)


def test_missing_meta_field_is_rejected_with_field_name(pop_path, data_dir):
    catalog = data_dir / "tables.csv"
    with catalog.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["organization"] = ""
    with catalog.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SchemaError) as exc:
        load_table(pop_path)
    assert any("org_nm" in str(p) for p in exc.value.problems)


@pytest.mark.parametrize(
    ("raw", "value", "flag"),
    [
        ("1,000", 1000.0, "ok"),
        ("90.9", 90.9, "ok"),
        ("-", None, "missing"),
        ("", None, "missing"),
        ("x", None, "suppressed"),
        ("560p", 560.0, "provisional"),
        ("abc", None, "unparsed"),
    ],
)
def test_parse_value(raw, value, flag):
    assert preprocessing.parse_value(raw) == (value, flag)


def test_clean_keeps_original_fields(pop_path):
    table = load_table(pop_path)
    cleaned = preprocessing.clean(table)
    for before, after in zip(table.cells, cleaned.cells, strict=True):
        assert after.value_raw == before.value_raw
        assert after.unit == before.unit
        assert after.source_ref == before.source_ref
        assert after.value_flag is not None
        assert replace(after, value=None, value_flag=None) == before


def test_markdown_shows_meta_header_and_unit_per_row(pop_path):
    text = serializer.to_markdown(preprocessing.clean(load_table(pop_path)))
    assert "조사명: 가상 인구조사" in text
    assert "| 항목 | 성별 | 시점 | 값 | 단위 |" in text
    assert "| 가상 성비 | 계 | 2023 | 90.9 | % |" in text
    assert "| 가상 인구 | 여자 | 2024 | 560p | 명 |" in text  # raw value, not our parse


def test_parse_value_flags_caution_marks():
    assert preprocessing.parse_value("*") == (None, "flagged")


def test_clean_uses_parquet_value_kind(pop_path, household_path):
    flags = {c.value_raw: (c.value, c.value_flag) for c in preprocessing.clean(load_table(pop_path)).cells}
    assert flags["1,000"] == (1000.0, "ok")
    assert flags["-"] == (None, "missing")
    assert flags["x"] == (None, "suppressed")
    assert flags["560p"] == (560.0, "provisional")  # other_text falls back to parse_value
    blank = [c for c in preprocessing.clean(load_table(household_path)).cells if c.period_raw == "2025"]
    assert blank and {(c.value, c.value_flag) for c in blank} == {(None, "missing")}


@pytest.mark.parametrize(
    ("kind", "flag"),
    [("ok", "ok"), ("blank", "missing"), ("dash", "missing"), ("ellipsis", "missing"),
     ("x_secret", "suppressed"), ("asterisk", "flagged")],
)
def test_value_kind_to_flag(pop_path, kind, flag):
    cell = replace(load_table(pop_path).cells[0], value_kind=kind)
    assert preprocessing.clean_cell(cell).value_flag == flag


@pytest.mark.parametrize(
    ("raw", "shown"),
    [("65975.0000", "65975"), ("12.5000", "12.5"), ("0.0000", "0"), ("1,000", "1,000"), ("90.9", "90.9"),
     ("560p", "560p"), ("X", "X"), ("-", "-"), ("*", "*"), ("", ""), ("100", "100")],
)
def test_display_value_drops_trailing_decimal_zeros(raw, shown):
    assert serializer.display_value(raw) == shown


def test_markdown_drops_zip_decimal_zeros(household_path):
    long_text = serializer.to_markdown(load_table(household_path))
    wide_text = serializer.to_wide_markdown(load_table(household_path))
    assert ".0000" not in long_text and ".0000" not in wide_text


def test_duplicate_source_ref_is_rejected(data_dir):
    rows = pop_rows()
    path = write_table(data_dir, "FAKE_TBL_003", rows + [rows[0]], {
        "survey_name": "가상 인구조사", "organization": "가상 기관", "table_name": "가상 중복 표",
        "period": "년 (2022 ~ 2024)", "url": "https://example.invalid/fake-table-003"})
    with pytest.raises(SchemaError) as err:
        load_table(path)
    (problem,) = err.value.problems
    assert problem.where == f"cells[{len(rows)}].source_ref" and problem.message.startswith("duplicate of cells[0] (FAKE_TBL_003|T1|")
