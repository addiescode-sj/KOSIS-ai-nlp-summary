import csv
import importlib.util
import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

spec = importlib.util.spec_from_file_location(
    "map_zip_item_ids", Path(__file__).resolve().parents[1] / "scripts" / "map_zip_item_ids.py")
mz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mz)


def make_data(tmp_path):
    (tmp_path / "values").mkdir()
    (tmp_path / "table_metadata").mkdir()
    with (tmp_path / "tables.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "source", "parquet"])
        w.writerow(["101_Z", "zip", "values/101_Z.parquet"])
        w.writerow(["101_A", "api", "values/101_A.parquet"])
    items = [{"itmId": "T0", "scrKor": "일반가구_계"}, {"itmId": "T1", "scrKor": "1인"},
             {"itmId": "T6", "scrKor": "6인 이상"}, {"itmId": "M1", "scrKor": "남자"}, {"itmId": "M2", "scrKor": "남자"}]
    (tmp_path / "table_metadata" / "101_Z.json").write_text(
        json.dumps({"itemInfo": {"itmList": items}}, ensure_ascii=False), encoding="utf-8")
    names = ["일반가구_계 (가구)", "1인 (가구)", "6인이상 (가구)", "남자 (명)", "미상"]
    table = pa.table({"tbl_id": ["Z"] * 5, "item_id": names, "item_name": names, "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    pq.write_table(table, tmp_path / "values" / "101_Z.parquet")
    pq.write_table(table, tmp_path / "values" / "101_A.parquet")
    return tmp_path


def read_ids(path):
    return pq.read_table(path).column("item_id").to_pylist()


def test_zip_item_names_become_meta_codes(tmp_path):
    data = make_data(tmp_path)
    assert mz.main(["--data", str(data)]) == 0
    assert read_ids(data / "values" / "101_Z.parquet") == ["T0", "T1", "T6", "남자 (명)", "미상"]
    assert pq.read_table(data / "values" / "101_Z.parquet").column("item_name").to_pylist()[1] == "1인 (가구)"
    assert read_ids(data / "values" / "101_A.parquet")[1] == "1인 (가구)"  # api 표는 건드리지 않음
    with (data / "zip_item_map.csv").open(encoding="utf-8-sig") as f:
        status = {r["item_name"]: r["status"] for r in csv.DictReader(f)}
    assert status == {"일반가구_계 (가구)": "mapped", "1인 (가구)": "mapped", "6인이상 (가구)": "mapped",
                      "남자 (명)": "dup", "미상": "nomatch"}


def test_dry_run_keeps_files_and_rerun_is_noop(tmp_path):
    data = make_data(tmp_path)
    mz.main(["--data", str(data), "--dry-run"])
    assert read_ids(data / "values" / "101_Z.parquet")[1] == "1인 (가구)"
    mz.main(["--data", str(data)])
    mz.main(["--data", str(data)])
    assert read_ids(data / "values" / "101_Z.parquet")[:3] == ["T0", "T1", "T6"]
