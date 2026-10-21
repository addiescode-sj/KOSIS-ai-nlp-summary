"""Input contract, loader adapters and baseline cleaning."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kosis_chat.data import preprocessing, serializer
from kosis_chat.data.loader import load_table, sha256_of
from kosis_chat.data.schema import SchemaError, validate


def test_format_a_loads_with_row_units_and_raw_values(fixtures):
    table = load_table(fixtures / "pop_a.csv")
    assert table.meta.input_format == "A"
    assert table.meta.file_sha256 == sha256_of(fixtures / "pop_a.csv")
    assert table.cells[0].dims == {"성별": "계"}
    assert table.cells[0].value_raw == "1,000"
    assert table.units_summary == ["명", "%"]
    assert table.period_range == ("2022", "2024")
    assert validate(table) == []


def test_format_a_without_codes_uses_file_hash_and_row_number(fixtures):
    table = load_table(fixtures / "pop_a.csv")
    digest = sha256_of(fixtures / "pop_a.csv")[:12]
    assert table.cells[0].source_ref == f"sha256:{digest}#row2"
    assert table.cells[-1].source_ref == f"sha256:{digest}#row11"


def test_format_a_with_codes_keeps_codes_in_source_ref(fixtures):
    table = load_table(fixtures / "codes_a.csv")
    assert table.cells[0].dim_codes == {"성별": "1"}
    assert table.cells[0].source_ref == "FAKE_TBL_001|가상 인구|성별=1|2023"


def test_format_b_kosis_json_maps_categories_and_codes(fixtures):
    table = load_table(fixtures / "pop_b.json")
    assert table.meta.input_format == "B"
    assert table.meta.tbl_id == "FAKE_TBL_003"  # taken from the response, meta.json had none
    assert table.cells[1].dims == {"연령": "0~14세"}
    assert table.cells[1].dim_codes == {"연령": "1"}
    assert table.cells[1].source_ref == "FAKE_TBL_003|T1|1|2023"


def test_missing_meta_field_is_rejected_with_field_name(fixtures):
    with pytest.raises(SchemaError) as exc:
        load_table(fixtures / "missing_meta.csv")
    assert any("org_nm" in str(p) for p in exc.value.problems)


def test_missing_meta_file_is_rejected(fixtures, tmp_path):
    data = tmp_path / "t.csv"
    data.write_text((fixtures / "pop_a.csv").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SchemaError, match="meta file not found"):
        load_table(data)


def test_wide_format_is_rejected(fixtures):
    with pytest.raises(SchemaError, match="wide format"):
        load_table(fixtures / "wide.csv")


def test_cp949_file_is_rejected(fixtures, tmp_path):
    data = tmp_path / "cp.csv"
    data.write_bytes((fixtures / "pop_a.csv").read_text(encoding="utf-8").encode("cp949"))
    (tmp_path / "cp.meta.json").write_text((fixtures / "pop_a.meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SchemaError, match="not UTF-8"):
        load_table(data)


def test_missing_required_column_is_rejected(fixtures, tmp_path):
    data = tmp_path / "nounit.csv"
    data.write_text("item,period,value\n가상 인구,2023,1\n", encoding="utf-8")
    (tmp_path / "nounit.meta.json").write_text((fixtures / "pop_a.meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SchemaError, match="unit"):
        load_table(data)


def test_meta_conflicting_with_response_is_rejected(fixtures, tmp_path):
    (tmp_path / "b.json").write_text((fixtures / "pop_b.json").read_text(encoding="utf-8"), encoding="utf-8")
    meta = json.loads((fixtures / "pop_b.meta.json").read_text(encoding="utf-8"))
    meta["tbl_id"] = "OTHER"
    (tmp_path / "b.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SchemaError, match="tbl_id"):
        load_table(tmp_path / "b.json")


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


def test_clean_keeps_original_fields(fixtures):
    table = load_table(fixtures / "pop_a.csv")
    cleaned = preprocessing.clean(table)
    for before, after in zip(table.cells, cleaned.cells, strict=True):
        assert after.value_raw == before.value_raw
        assert after.unit == before.unit
        assert after.source_ref == before.source_ref
        assert after.value_flag is not None
        assert replace(after, value=None, value_flag=None) == before


def test_markdown_shows_meta_header_and_unit_per_row(fixtures):
    text = serializer.to_markdown(preprocessing.clean(load_table(fixtures / "pop_a.csv")))
    assert "조사명: 가상 인구조사" in text
    assert "| 항목 | 성별 | 시점 | 값 | 단위 |" in text
    assert "| 가상 성비 | 계 | 2023 | 90.9 | % |" in text
    assert "| 가상 인구 | 여자 | 2024 | 560p | 명 |" in text  # raw value, not our parse
