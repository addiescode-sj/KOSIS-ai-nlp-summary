"""scripts/scan_tbl_ids.py (T1-28 S5): the read-only scan run before the tables.csv tbl_id check is switched on."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scan_tbl_ids", ROOT / "scripts" / "scan_tbl_ids.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


def test_matching_tables_pass(data_dir, capsys):
    assert scan.main(["--data", str(data_dir)]) == 0
    assert "0 tables break it" in capsys.readouterr().out


def test_a_catalog_row_naming_another_tbl_id_is_reported(data_dir, tmp_path, capsys):
    catalog = data_dir / "tables.csv"
    before = (data_dir / "values" / "101_FAKE_TBL_002.parquet").read_bytes()
    catalog.write_text(catalog.read_text(encoding="utf-8-sig").replace("101_FAKE_TBL_002,101,FAKE_TBL_002,",
                                                                       "101_FAKE_TBL_002,101,FAKE_TBL_009,"), encoding="utf-8-sig")
    out = tmp_path / "scan.csv"
    assert scan.main(["--data", str(data_dir), "--out", str(out)]) == 1
    text = capsys.readouterr().out
    assert "tbl_id_mismatch: 1" in text and "101_FAKE_TBL_002" in text
    assert "101_FAKE_TBL_002" in out.read_text(encoding="utf-8-sig")
    assert (data_dir / "values" / "101_FAKE_TBL_002.parquet").read_bytes() == before  # read only


def test_blank_and_missing_tbl_ids_are_counted_without_python_rows():
    import pyarrow as pa
    ids, blank = scan.tbl_id_counts(pa.chunked_array([["A", None, " ", "A"], ["B", ""]]))
    assert ids == ["A", "B"] and blank == 3
    assert scan.tbl_id_counts(pa.chunked_array([[float("nan"), None]], type=pa.float64())) == ([], 2)
