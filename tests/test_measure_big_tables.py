"""scripts/measure_big_tables.py (T1-31): the measurement of big tables, run end to end on two made-up tables, child
processes included. --big-rows 0 (a test-only option) sends these small tables down the big-table path: one is
file-backed, the other breaks an eligibility condition and is loaded whole."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kosis_chat.prepare.schemas import Problem, SchemaError
from tests.conftest import CATALOG_COLUMNS, PARQUET_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("measure_big_tables", ROOT / "scripts" / "measure_big_tables.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)

LABEL = "가상라벨"  # every label of the made-up tables starts with it; none may reach the output


def _write(data: Path, tbl_id: str, rows: list[dict]) -> None:
    values = data / "values"
    values.mkdir(parents=True, exist_ok=True)
    full = [{"org_id": "101", "tbl_id": tbl_id, "source": "api", "prd_se": "Y", "unit": "명", "value_kind": "ok"} | r
            for r in rows]
    schema = pa.schema([(c, pa.float64() if c == "value" else pa.string()) for c in PARQUET_COLUMNS])
    pq.write_table(pa.table({c: [r.get(c) for r in full] for c in PARQUET_COLUMNS}, schema=schema),
                   values / f"101_{tbl_id}.parquet", row_group_size=7)
    catalog = data / "tables.csv"
    new = not catalog.exists()
    with catalog.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({"key": f"101_{tbl_id}", "org_id": "101", "tbl_id": tbl_id, "source": "api",
                         "parquet": f"values/101_{tbl_id}.parquet", "survey_name": LABEL, "organization": LABEL,
                         "table_name": LABEL, "period": "년 (2015 ~ 2024)", "url": "https://example.invalid/t"})


def _row(year: int, item: str, region: str, sex: str, value: float, *, second: str = "성별") -> dict:
    return {"period": str(year), "item_id": item, "item_name": f"{LABEL} {item}", "value_raw": f"{value:g}",
            "value": value, "dim1_name": "지역별", "dim1_code": region, "dim1_label": f"{LABEL} {region}",
            "dim2_name": second, "dim2_code": sex, "dim2_label": f"{LABEL} {sex}"}


def _tables(data: Path) -> None:
    good = [_row(y, item, r, s, 1000.5 + y + i)
            for i, (y, item, r, s) in enumerate((y, item, r, s) for y in range(2015, 2025) for item in ("T1", "T2")
                                                for r in ("00", "11", "21", "22", "23", "31") for s in ("0", "1", "2"))]
    _write(data, "FAKE_GOOD", good)
    bad = [_row(y, "T1", r, "0", 7.25 + y) for y in range(2020, 2023) for r in ("00", "11")]
    bad.append(_row(2023, "T1", "0", "00", 9.5, second="지역별") | {"dim1_name": "성별"})  # a name on two levels
    _write(data, "FAKE_BAD", bad)


def test_two_small_tables_through_the_big_table_path(tmp_path, capsys):
    data, index, out = tmp_path / "작업 폴더" / "data", tmp_path / "index", tmp_path / "out.json"
    _tables(data)
    args = ["--data", str(data), "--out", str(out), "--index-dir", str(index), "--min-rows", "0", "--big-rows", "0"]
    assert measure.main(args + ["--fresh"]) == 0
    printed = capsys.readouterr().out
    text = out.read_text(encoding="utf-8")
    assert LABEL not in printed and LABEL not in text and str(tmp_path) not in text
    report = json.loads(text)
    good, bad = report["tables"]["101_FAKE_GOOD"], report["tables"]["101_FAKE_BAD"]

    assert good["route"] == "file" and good["first"]["scan"] == "fresh" and good["reuse"]["scan"] == "saved"
    assert good["first"]["exit"] == 0 and good["first"]["saved_scan_bytes"] > 0 and good["first"]["peak_mb"] > 0
    probes = {p["name"]: p for p in good["first"]["probes"]}
    assert probes["one cell"]["cells"] == 1
    assert probes["one series"]["cells"] == 10 and probes["series_at 5 keys"]["cells"] == 50
    assert probes["20 cells of one period"]["cells"] == 20 and probes["by_refs 20"]["cells"] == 20
    assert probes["summary of one item, all periods"]["cells"] == 180
    assert probes["summary of one item, one period"]["cells"] == 18
    assert all(p["row_groups_read"] > 0 for p in probes.values())  # each on its own store: none from a memo

    assert bad["route"] == "memory" and bad["condition"] == "level" and bad["reuse"] is None
    assert report["summary"]["routes"] == {"file": 1, "memory": 1}
    assert "101_FAKE_GOOD" in printed and "routes: file 1, memory 1" in printed

    # a rerun of one key keeps the scan, merges into the file, and a target missed gives exit 1
    assert measure.main(args + ["--tables", "101_FAKE_GOOD", "--max-gb", "0.001"]) == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report["tables"]) == {"101_FAKE_GOOD", "101_FAKE_BAD"}
    assert report["tables"]["101_FAKE_GOOD"]["first"]["scan"] == "saved"
    assert report["summary"]["over_targets"]["101_FAKE_GOOD"][0]["what"] == "peak (first load)"


def test_bad_arguments_and_a_missing_folder_give_2(tmp_path):
    assert measure.main(["--data", str(tmp_path / "none"), "--out", str(tmp_path / "o.json")]) == 2
    _tables(tmp_path / "data")
    assert measure.main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "o.json"), "--tables", "NOPE"]) == 2
    try:
        measure.main(["--out", str(tmp_path / "o.json")])
    except SystemExit as exc:
        assert exc.code == 2


def test_the_condition_comes_from_the_scan_or_names_the_outcome(tmp_path):
    _tables(tmp_path / "data")
    values = tmp_path / "data" / "values"
    assert measure.why_not_file_backed(values / "101_FAKE_BAD.parquet") == "small"  # 7 rows: never scanned
    error = SchemaError([Problem("x.parquet", "over the 3-row limit of a whole-table load")])
    assert measure.why_not_file_backed(values / "101_FAKE_BAD.parquet", error) == "level"
    assert measure.why_not_file_backed(values / "101_FAKE_GOOD.parquet", error) == "row_limit"
    assert measure.why_not_file_backed(values / "101_NONE.parquet", error) == "error"


def test_probe_descriptors_are_evenly_spaced_whether_the_scan_was_saved_or_not(tmp_path, monkeypatch):
    from kosis_chat.adapters import parquet, parquet_store

    _tables(tmp_path / "data")
    path = tmp_path / "data" / "values" / "101_FAKE_GOOD.parquet"
    for folder in (tmp_path / "index", None):  # saved: the descriptors are read back from the file; not saved: held
        monkeypatch.setattr(parquet_store, "INDEX_DIR", folder)
        monkeypatch.setattr(parquet, "index_folder", (lambda p: folder) if folder is None else parquet.index_folder)
        store = parquet.load_table(path, big_rows=0).store
        assert (store.facts.descriptors is None) == (folder is not None)
        every = list(store.index_cells())
        picked = measure.descriptor_cells(store, measure.PROBE_DESCRIPTORS)
        assert picked == [every[i * len(every) // measure.PROBE_DESCRIPTORS] for i in range(measure.PROBE_DESCRIPTORS)]
