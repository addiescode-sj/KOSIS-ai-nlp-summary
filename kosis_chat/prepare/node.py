"""prepare: a loaded table (ParquetTable) -> TableInput, without opening a file (T1-28 S2).

Rows -> cells and the unit inference are prepare/convert.py, value cleaning prepare/clean.py, the index
prepare/index.py. The table meta comes from ParquetTable.info, else from the caller's catalog rows (tables.csv read by
kosis_chat/adapters/parquet.py; D23 c). A file read by adapters.parquet.load_table gives the same cells as the same rows
passed here; only file_sha256 (here: the content hash) and fetched_at (here: ParquetTable.fetched_at or "") differ.
"""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from collections.abc import Mapping

from kosis_chat.prepare.clean import clean
from kosis_chat.prepare.convert import (check_meta, convert, frame_digest, meta_from_row, no_rows, period_codes, rows_of,
                                        tidy_note, tidy_rows)
from kosis_chat.prepare.schemas import Problem, SchemaError, TableInput, validate
from kosis_chat.schemas.io import ParquetTable

HASH_VERSION = "prepare-content-v3"  # v3 (T1-28 S5 review): the frame part is the Arrow IPC stream, not JSON rows
PARTIAL_NOTE = "; coverage partial: 일부 행만 받은 표"


# Frame hashes of pyarrow Tables (immutable) by the frame object, not by the ParquetTable wrapper: a new StatInput
# around the same Table reuses it. id -> (weak reference, hash); the entry goes when the Table is freed, and the weak
# reference is checked so a reused id never answers. A DataFrame can change in place: hashed on every call.
_FRAME_HASHES: dict[int, tuple[weakref.ref, str]] = {}
_FRAME_LOCK = threading.Lock()


def _forget(key: int):
    def drop(_ref) -> None:
        with _FRAME_LOCK:
            _FRAME_HASHES.pop(key, None)
    return drop


def _frame_hash(table: ParquetTable) -> str:
    """sha256 over the frame's schema (sorted columns) and its Arrow IPC stream (prepare.convert.frame_digest). Never a
    subset of columns: two tables would share a key."""
    frame = table.frame
    arrow = hasattr(frame, "to_pylist")  # pyarrow Table, not a pandas DataFrame
    if arrow:
        with _FRAME_LOCK:
            hit = _FRAME_HASHES.get(id(frame))
        if hit is not None and hit[0]() is frame:
            table._frame_hash = hit[1]
            return hit[1]
    digest = hashlib.sha256()
    frame_digest(frame, digest)
    value = digest.hexdigest()
    if arrow:
        table._frame_hash = value
        with _FRAME_LOCK:
            _FRAME_HASHES[id(frame)] = (weakref.ref(frame, _forget(id(frame))), value)
    return value


def _hash(table: ParquetTable, meta: tuple[Mapping[str, str], str] | None) -> str:
    head = {"v": HASH_VERSION, "frame": _frame_hash(table), "coverage": table.coverage, "fetched_at": table.fetched_at,
            "info": table.info.model_dump(mode="json") if table.info is not None else None,
            # the meta actually applied: from info, else the catalog row prepare picked (and the fetched_at used)
            "meta": None if meta is None else {"row": dict(meta[0]), "fetched_at": meta[1]}}
    return hashlib.sha256(json.dumps(head, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def content_hash(table: ParquetTable, catalog: Mapping[str, Mapping[str, str]] | None = None) -> str:
    """sha256 over everything that changes a prepared table: the frame (columns, dtypes, every row), the info, the
    coverage and the meta prepare applies (info, else the catalog row for the frame's tbl_id, and fetched_at).
    Without info and catalog the meta part is empty: pass the catalog prepare will use to get its key."""
    meta = None
    if table.info is not None or catalog is not None:
        meta = _meta_source(table, table.tbl_id(), catalog)
    return _hash(table, meta)


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
    tbl_id = str(rows[0]["tbl_id"])
    if {str(r["tbl_id"]) for r in rows} != {tbl_id}:  # ParquetTable checks this; a frame changed since must not pass
        raise SchemaError([Problem("frame", "rows of more than one table, or rows without tbl_id")])
    row, fetched_at = _meta_source(table, tbl_id, catalog)
    if str(row.get("tbl_id")) != tbl_id:
        raise SchemaError([Problem("meta", f"meta tbl_id {row.get('tbl_id')!r} is not the frame's tbl_id {tbl_id!r}")])
    meta = meta_from_row(row, period_codes(rows), fetched_at)
    rows, counts = tidy_rows(rows)
    if not rows:
        raise no_rows("frame")
    meta["note"] += tidy_note(counts)
    if table.coverage == "partial":
        meta["note"] += PARTIAL_NOTE
    allow_empty = () if fetched_at else ("fetched_at",)
    check_meta(meta, where="frame", allow_empty=allow_empty)
    out = convert(rows, meta, _hash(table, (row, fetched_at)))
    problems = validate(out, allow_empty=allow_empty)
    if problems:
        raise SchemaError(problems)
    return out


def prepare(table: ParquetTable, catalog: Mapping[str, Mapping[str, str]] | None = None) -> TableInput:
    """table_input, cleaned (value / value_flag filled), as StatPipeline holds it."""
    return clean(table_input(table, catalog))
