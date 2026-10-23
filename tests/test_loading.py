"""Unit fallback, the Parquet loader (format C) and the wide serializer."""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from eval import run_eval
from kosis_chat.chatbot import Chatbot
from kosis_chat.data import preprocessing, serializer
from kosis_chat.data.loader import UNKNOWN_UNIT, infer_unit, load_table
from kosis_chat.data.schema import SchemaError
from kosis_chat.llm.model import MockClient


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


def _write_kosis_json(folder, rows):
    (folder / "t.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    meta = {"tbl_id": "FAKE_TBL_009", "tbl_nm": "가상 동태", "survey_nm": "가상 조사", "org_nm": "가상 기관",
            "period_type": "Y", "source_url": "https://example.invalid/9", "fetched_at": "2026-10-22"}
    (folder / "t.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return folder / "t.json"


def test_kosis_json_without_unit_nm_loads_with_inferred_units(tmp_path):
    rows = [
        {"TBL_ID": "FAKE_TBL_009", "ITM_ID": "T1", "ITM_NM": "가상 출생 (명)", "C1": "00", "C1_NM": "전국",
         "C1_OBJ_NM": "지역", "PRD_DE": "2023", "DT": "100"},
        {"TBL_ID": "FAKE_TBL_009", "ITM_ID": "T2", "ITM_NM": "가상 항목", "C1": "00", "C1_NM": "전국",
         "C1_OBJ_NM": "지역", "PRD_DE": "2023", "DT": "5"},
    ]
    table = load_table(_write_kosis_json(tmp_path, rows))
    assert [(c.unit, c.unit_source) for c in table.cells] == [("명", "item_name"), (UNKNOWN_UNIT, "unknown")]


def test_wide_markdown_puts_periods_in_columns(fixtures):
    table = preprocessing.clean(load_table(fixtures / "pop_a.csv"))
    text = serializer.serialize(table, "wide")
    assert "단위: 행마다 다름(단위 열)" in text  # pop_a has 명 and %
    assert "| 항목 | 성별 | 단위 | 2022 | 2023 | 2024 |" in text
    assert "| 가상 인구 | 여자 | 명 |" in text and "560p" in text  # raw values kept
    assert len(text) < len(serializer.serialize(table, "long"))


def test_wide_markdown_writes_single_unit_once_and_splits_many_periods(fixtures):
    table = preprocessing.clean(load_table(fixtures / "pop_a.csv"))
    people = replace(table, cells=tuple(c for c in table.cells if c.unit == "명"))
    text = serializer.to_wide_markdown(people, max_period_columns=2)
    assert "단위: 명" in text and "| 단위 |" not in text
    assert "시점 2022 ~ 2023" in text and "시점 2024 ~ 2024" in text


def test_unknown_serialization_is_rejected(fixtures):
    with pytest.raises(ValueError, match="unknown serialization"):
        serializer.serialize(load_table(fixtures / "pop_a.csv"), "json")


def test_chatbot_uses_configured_serialization(fixtures, mock_settings):
    table = load_table(fixtures / "pop_a.csv")
    client = MockClient("x")
    Chatbot(table, client=client, settings=replace(mock_settings, serialization="wide")).ask("q")
    assert "| 항목 | 성별 | 단위 | 2022 | 2023 | 2024 |" in client.calls[0][1]


def test_run_eval_records_serialization(fixtures, tmp_path):
    table = fixtures / "pop_a.csv"
    from kosis_chat.data.loader import sha256_of

    golden = tmp_path / "golden" / "v1"
    golden.mkdir(parents=True)
    item = {"id": "q1", "kind": "answerable", "question": "q", "table": str(table), "table_sha256": sha256_of(table)}
    (golden / "g.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    out = tmp_path / "results"
    argv = ["--stage", "B0", "--golden", str(golden), "--backend", "mock", "--serialization", "wide", "--out", str(out),
            "--log-dir", str(tmp_path / "logs")]
    assert run_eval.main(argv) == 0
    manifest = json.loads((next(out.iterdir()) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["serialization"] == "wide"


# Parquet (format C): same long layout as data/values/*.parquet, meta from tables.csv.
def _write_parquet(folder, rows):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    columns = ["org_id", "tbl_id", "source", "prd_se", "period", "item_id", "item_name", "unit"]
    columns += [f"dim{i}_{p}" for i in range(1, 9) for p in ("name", "code", "label")]
    columns += ["value_raw", "value", "value_kind"]
    data = {c: [r.get(c) for r in rows] for c in columns}
    data["value"] = [float(r["value"]) if r.get("value") is not None else None for r in rows]
    schema = pa.schema([(c, pa.float64() if c == "value" else pa.string()) for c in columns])
    values = folder / "values"
    values.mkdir()
    pq.write_table(pa.table(data, schema=schema), values / "101_FAKE_TBL_010.parquet")
    with (folder / "tables.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "org_id", "tbl_id", "survey_name", "organization", "table_name", "period", "url"])
        writer.writeheader()
        writer.writerow({"key": "101_FAKE_TBL_010", "org_id": "101", "tbl_id": "FAKE_TBL_010", "survey_name": "가상 조사",
                         "organization": "가상 기관", "table_name": "가상 표", "period": "년 (2023 ~ 2024)",
                         "url": "https://example.invalid/10"})
    return values / "101_FAKE_TBL_010.parquet"


def _row(period, value_raw, unit=None, item="가상 출생 (명)", kind="ok"):
    return {"org_id": "101", "tbl_id": "FAKE_TBL_010", "source": "api", "prd_se": "Y", "period": period,
            "item_id": "T1", "item_name": item, "unit": unit, "dim1_name": "지역", "dim1_code": "00",
            "dim1_label": "전국", "value_raw": value_raw, "value": None if value_raw in ("", "-") else value_raw,
            "value_kind": kind}


def test_parquet_loads_with_catalog_meta_and_inferred_unit(tmp_path):
    path = _write_parquet(tmp_path, [_row("2023", "100"), _row("2024", "-", kind="dash")])
    table = preprocessing.clean(load_table(path))
    assert table.meta.input_format == "C"
    assert (table.meta.tbl_id, table.meta.tbl_nm, table.meta.survey_nm) == ("FAKE_TBL_010", "가상 표", "가상 조사")
    assert table.meta.source_url == "https://example.invalid/10"
    assert table.cells[0].dims == {"지역": "전국"} and table.cells[0].dim_codes == {"지역": "00"}
    assert table.cells[0].source_ref == "FAKE_TBL_010|T1|00|2023"
    assert [(c.unit, c.unit_source) for c in table.cells] == [("명", "item_name")] * 2
    assert [c.value_flag for c in table.cells] == ["ok", "missing"]


def test_parquet_without_catalog_is_rejected(tmp_path):
    path = _write_parquet(tmp_path, [_row("2023", "100")])
    (tmp_path / "tables.csv").unlink()
    with pytest.raises(SchemaError, match="tables.csv"):
        load_table(path)
