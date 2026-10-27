"""Shared pytest fixtures: made-up Parquet tables, offline settings and a local fake HTTP server.

The tables are **made up** (가상 데이터) in the same layout as data/values/*.parquet + data/tables.csv;
they contain no real KOSIS values.
"""

from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kosis_chat.config import load_settings

PARQUET_COLUMNS = ["org_id", "tbl_id", "source", "prd_se", "period", "item_id", "item_name", "unit"]
PARQUET_COLUMNS += [f"dim{i}_{part}" for i in range(1, 9) for part in ("name", "code", "label")]
PARQUET_COLUMNS += ["value_raw", "value", "value_kind"]
CATALOG_COLUMNS = ["key", "org_id", "tbl_id", "survey_name", "organization", "table_name", "period", "source", "parquet", "url"]


def write_table(data_dir: Path, tbl_id: str, rows: list[dict], catalog: dict) -> Path:
    """Write values/101_<tbl_id>.parquet and add its tables.csv row; returns the Parquet path."""
    values = data_dir / "values"
    values.mkdir(parents=True, exist_ok=True)
    full = [{"org_id": "101", "tbl_id": tbl_id, "source": "api", "prd_se": "Y", "value_kind": "ok", **r} for r in rows]
    columns = {c: [r.get(c) for r in full] for c in PARQUET_COLUMNS}
    columns["value"] = [None if r.get("value") is None else float(r["value"]) for r in full]
    schema = pa.schema([(c, pa.float64() if c == "value" else pa.string()) for c in PARQUET_COLUMNS])
    path = values / f"101_{tbl_id}.parquet"
    pq.write_table(pa.table(columns, schema=schema), path)
    catalog_path = data_dir / "tables.csv"
    new = not catalog_path.exists()
    with catalog_path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({"key": f"101_{tbl_id}", "org_id": "101", "tbl_id": tbl_id, "source": "api",
                         "parquet": f"values/101_{tbl_id}.parquet", **catalog})
    return path


def _num(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


# FAKE_TBL_001: 가상 성별 인구 (two items, 성별 0/1/2, 2022~2024, with -, x and 560p values).
SEXES = (("0", "계"), ("1", "남자"), ("2", "여자"))
POP_VALUES = {"2022": ("1,000", "480", "520"), "2023": ("1,050", "500", "550"), "2024": ("-", "x", "560p")}
# value_kind as the Parquet build writes it.
POP_KINDS = {"-": "dash", "x": "x_secret", "560p": "other_text"}


def pop_rows() -> list[dict]:
    rows = []
    for period, values in POP_VALUES.items():
        for (code, label), raw in zip(SEXES, values, strict=True):
            rows.append({"period": period, "item_id": "T1", "item_name": "가상 인구", "unit": "명", "dim1_name": "성별",
                         "dim1_code": code, "dim1_label": label, "value_raw": raw,
                         "value": _num(raw) if raw not in POP_KINDS else None, "value_kind": POP_KINDS.get(raw, "ok")})
    rows.append({"period": "2023", "item_id": "T2", "item_name": "가상 성비", "unit": "%", "dim1_name": "성별",
                 "dim1_code": "0", "dim1_label": "계", "value_raw": "90.9", "value": 90.9})
    return rows


# FAKE_TBL_002: 가상 가구원수별 가구, regions nested by code length (시도 > 시군구 > 읍면동), no unit column,
# ZIP-style values ("1000.0000") and a blank 2025.
REGIONS = (("00", "전국"), ("03", "동부"), ("11", "서울특별시"), ("21", "부산광역시"), ("39", "제주특별자치도"),
           ("11010", "종로구"), ("11020", "중구"), ("21010", "중구"), ("11010530", "사직동"), ("11010540", "삼청동"))
HOUSEHOLD_ITEMS = (("T0", "일반가구_계 (가구)"), ("T1", "1인 (가구)"), ("T2", "2인 (가구)"))
HOUSEHOLD_PERIODS = ("2015", "2020", "2023", "2024", "2025")


def household_rows() -> list[dict]:
    rows = []
    for p_no, period in enumerate(HOUSEHOLD_PERIODS):
        for r_no, (code, label) in enumerate(REGIONS):
            for i_no, (item_id, item_name) in enumerate(HOUSEHOLD_ITEMS):
                # 일반가구_계 (T0) is ten times a region's base, so a part's share of it stays below 100% (S3 checks that).
                base = 10000 * (r_no + 1) if item_id == "T0" else 1000 * (r_no + 1) + 100 * i_no
                raw = "" if period == "2025" else f"{base + p_no}.0000"  # ZIP style
                rows.append({"source": "zip", "period": period, "item_id": item_id, "item_name": item_name,
                             "dim1_name": "행정구역별(읍면동)", "dim1_code": code, "dim1_label": label,
                             "value_raw": raw, "value": _num(raw), "value_kind": "ok" if raw else "blank"})
    return rows


@pytest.fixture
def data_dir(tmp_path) -> Path:
    folder = tmp_path / "data"
    write_table(folder, "FAKE_TBL_001", pop_rows(), {
        "survey_name": "가상 인구조사", "organization": "가상 기관", "table_name": "가상 성별 인구",
        "period": "년 (2022 ~ 2024)", "url": "https://example.invalid/fake-table-001"})
    write_table(folder, "FAKE_TBL_002", household_rows(), {
        "survey_name": "가상 총조사", "organization": "가상 기관", "table_name": "가상 가구원수별 가구",
        "period": "년 (2015 ~ 2025)", "url": "https://example.invalid/fake-table-002"})
    return folder


@pytest.fixture
def pop_path(data_dir) -> Path:
    return data_dir / "values" / "101_FAKE_TBL_001.parquet"


@pytest.fixture
def household_path(data_dir) -> Path:
    return data_dir / "values" / "101_FAKE_TBL_002.parquet"


@pytest.fixture
def mock_settings():
    return load_settings(backend="mock")


class FakeServer:
    """Answers every POST with `status` / `body` / `headers` (changeable between requests) and records requests."""

    def __init__(self, status=200, body=None, headers=None, delay=0.0):
        self.status, self.body, self.headers, self.delay = status, body, headers or {}, delay
        self.requests: list[dict] = []
        self.request_headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                outer.request_headers.append(dict(self.headers))
                if outer.delay:
                    threading.Event().wait(outer.delay)
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                for name, value in outer.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server_factory():
    servers = []

    def make(**kwargs):
        servers.append(FakeServer(**kwargs))
        return servers[-1]

    yield make
    for s in servers:
        s.close()
