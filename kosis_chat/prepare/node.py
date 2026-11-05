"""prepare: a loaded table (ParquetTable) -> TableInput, without opening a file (T1-28 S2).

Rows -> cells and the unit inference are prepare/convert.py, value cleaning prepare/clean.py, the index
prepare/index.py. The table meta comes from ParquetTable.info, else from the caller's catalog rows (tables.csv read by
kosis_chat/adapters/parquet.py; D23 c). A file read by adapters.parquet.load_table gives the same cells as the same rows
passed here; only file_sha256 (here: the content hash) and fetched_at (here: ParquetTable.fetched_at or "") differ.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from kosis_chat.prepare.clean import clean
from kosis_chat.prepare.convert import check_meta, convert, frame_content, meta_from_row, no_rows, period_codes, rows_of
from kosis_chat.prepare.schemas import Problem, SchemaError, TableInput, validate
from kosis_chat.schemas.io import ParquetTable

HASH_VERSION = "prepare-content-v1"
PARTIAL_NOTE = "; coverage partial: 일부 행만 받은 표"


def content_hash(table: ParquetTable) -> str:
    """sha256 over everything that changes a prepared table: the column names and dtypes (sorted), every row,
    the info (all fields), the coverage and fetched_at. Never a subset of columns: two tables would share a key. Computed once
    per ParquetTable object."""
    if table._content_hash is None:
        typed, _, rows = frame_content(table.frame)
        digest = hashlib.sha256()
        head = {"v": HASH_VERSION, "columns": typed, "coverage": table.coverage, "fetched_at": table.fetched_at,
                "info": table.info.model_dump(mode="json") if table.info is not None else None}
        digest.update(json.dumps(head, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        for row in rows:
            digest.update(b"\n")
            digest.update(json.dumps(row, ensure_ascii=False, default=repr).encode("utf-8"))
        table._content_hash = digest.hexdigest()
    return table._content_hash


def _catalog_row(tbl_id: str, catalog: Mapping[str, Mapping[str, str]] | None) -> Mapping[str, str]:
    found = [row for row in (catalog or {}).values() if row.get("tbl_id") == tbl_id]
    if len(found) != 1:
        what = "no row" if not found else f"{len(found)} rows"
        raise SchemaError([Problem("catalog", f"{what} with tbl_id {tbl_id!r}; pass ParquetTable.info")])
    return found[0]


def _meta_source(table: ParquetTable, tbl_id: str, catalog) -> tuple[Mapping[str, str], str]:
    """(row in tables.csv names, fetched_at) from info, else from the catalog."""
    if table.info is not None:
        info = table.info
        row = {"tbl_id": info.tbl_id, "table_name": info.table_name, "survey_name": info.survey_name,
               "organization": info.organization, "url": info.url, "period": info.period}
        return row, table.fetched_at or ""
    row = _catalog_row(tbl_id, catalog)
    return row, table.fetched_at or str(row.get("fetched_at") or "")


def table_input(table: ParquetTable, catalog: Mapping[str, Mapping[str, str]] | None = None) -> TableInput:
    """The converted (not yet cleaned) TableInput; SchemaError with every problem found.

    catalog: tables.csv rows by key, used only when table.info is None."""
    rows = rows_of(table.frame)
    if not rows:
        raise no_rows("frame")
    row, fetched_at = _meta_source(table, str(rows[0]["tbl_id"]), catalog)
    meta = meta_from_row(row, period_codes(rows), fetched_at)
    if table.coverage == "partial":
        meta["note"] += PARTIAL_NOTE
    allow_empty = () if fetched_at else ("fetched_at",)
    check_meta(meta, where="frame", allow_empty=allow_empty)
    out = convert(rows, meta, content_hash(table))
    problems = validate(out, allow_empty=allow_empty)
    if problems:
        raise SchemaError(problems)
    return out


def prepare(table: ParquetTable, catalog: Mapping[str, Mapping[str, str]] | None = None) -> TableInput:
    """table_input, cleaned (value / value_flag filled), as StatPipeline holds it."""
    return clean(table_input(table, catalog))
