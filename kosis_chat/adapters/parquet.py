"""Outside the module: open one table's files, data/values/<org>_<tbl>.parquet and its data/tables.csv row (T1-28 S2).

The file side of kosis_chat/data/loader.py: reading the Parquet, finding the catalog row, the file date (fetched_at)
and the file hash. Turning the rows into a TableInput is kosis_chat/prepare/convert.py. Entry points, eval and the
CLI use this module; the graph and stage packages never import it. data/loader.py re-exports every name.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

from kosis_chat.prepare.convert import (PARQUET_COLUMNS, ROW_LIMIT, check_meta, convert, meta_from_row, no_rows, period_codes,
                                        rows_of, tidy_note, tidy_rows)
from kosis_chat.prepare.schemas import Problem, SchemaError, TableInput, validate


# The rows' tbl_id must be the tables.csv row's tbl_id (plans/T1-28.md 검토 반영 2: the file path picks the catalog row
# by file name and never read the rows' tbl_id). Off until the user's scan of every provided table
# (scripts/scan_tbl_ids.py) finds no table that breaks it; then set True (one line; tests/test_adapters_tbl_id.py).
CHECK_CATALOG_TBL_ID = False


def read_catalog(path: Path) -> dict[str, dict]:
    """tables.csv rows by key (what prepare and StatModule take as `catalog`); LookupError when there is no file."""
    if not path.exists():
        raise LookupError(f"no tables.csv at {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["key"]: row for row in csv.DictReader(f) if row.get("key")}


def file_stamp(path: Path) -> tuple[int, int] | None:
    """(mtime in ns, size) of a file, None when it is missing: what api's caches compare to notice a replaced file."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def parquet_stamp(path: Path) -> tuple | None:
    """file_stamp plus the Parquet footer's row count and serialized size (read_metadata reads only the footer):
    a file rewritten inside one mtime tick with the same byte size is still seen when its rows or footer differ."""
    stamp = file_stamp(path)
    if stamp is None:
        return None
    try:
        import pyarrow.parquet as pq

        footer = pq.read_metadata(path)
        return (*stamp, footer.num_rows, footer.serialized_size)
    except Exception:  # not readable as Parquet: the load reports it
        return (*stamp, None, None)


def catalog_tbl_id_problems(rows: list[dict], catalog: dict, where: str) -> list[Problem]:
    """The rows' tbl_id against the catalog row's: one value on every row, equal to it."""
    ids = {str(r.get("tbl_id")) for r in rows}
    if ids != {str(catalog.get("tbl_id"))}:
        return [Problem(where, f"rows' tbl_id {sorted(ids)[:3]} is not the tables.csv tbl_id {catalog.get('tbl_id')!r}")]
    return []


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_catalog(path: Path) -> Path | None:
    """tables.csv next to the Parquet file or one level up (data/values/x.parquet -> data/tables.csv)."""
    for folder in (path.parent, path.parent.parent):
        if (folder / "tables.csv").exists():
            return folder / "tables.csv"
    return None


def catalog_row(path: Path) -> dict:
    """The tables.csv row whose key is the file stem; SchemaError when there is no catalog or no such row."""
    catalog = find_catalog(path)
    if catalog is None:
        raise SchemaError([Problem(path.name, "no tables.csv next to it or one level up")])
    with catalog.open(encoding="utf-8-sig", newline="") as f:
        row = next((r for r in csv.DictReader(f) if r.get("key") == path.stem), None)
    if row is None:
        raise SchemaError([Problem(catalog.name, f"no row with key {path.stem!r}")])
    return row


def fetched_at_of(path: Path) -> str:
    """tables.csv has no fetch date; the Parquet file date stands in for it."""
    return dt.date.fromtimestamp(path.stat().st_mtime).isoformat()


def parquet_meta(path: Path, prd_se: set[str]) -> dict:
    """Meta from the tables.csv row whose key is the file stem, fetched_at = the file date."""
    row = catalog_row(path)
    return meta_from_row(row, prd_se, fetched_at_of(path))


def footer_rows(path: Path) -> int | None:
    """The row count in the Parquet footer (no data read); None when it cannot be read (read_parquet reports why)."""
    try:
        import pyarrow.parquet as pq

        return pq.read_metadata(path).num_rows
    except Exception:
        return None


def read_parquet(path: Path) -> Any:
    """The table's columns (prepare.convert.PARQUET_COLUMNS) as a pyarrow Table; SchemaError when it cannot be read."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SchemaError([Problem(path.name, "reading .parquet needs pyarrow (pip install pyarrow)")]) from exc
    try:
        return pq.read_table(path, columns=PARQUET_COLUMNS)
    except Exception as exc:  # wrong schema or broken file
        raise SchemaError([Problem(path.name, f"cannot read Parquet: {exc}")]) from exc


def load_parquet(path: Path) -> TableInput:
    """One long Parquet file per table (data/values), meta from tables.csv; not yet validated as a whole."""
    row_count = footer_rows(path)
    if row_count is not None and row_count > ROW_LIMIT:
        raise SchemaError([Problem(path.name, f"table has {row_count:,} rows, over the {ROW_LIMIT:,}-row limit of a whole-table load")])
    frame = read_parquet(path)
    try:
        rows = rows_of(frame)
    except Exception as exc:  # as read_parquet: a column of an unexpected type
        raise SchemaError([Problem(path.name, f"cannot read Parquet: {exc}")]) from exc
    if not rows:
        raise no_rows(path.name)
    if CHECK_CATALOG_TBL_ID:
        problems = catalog_tbl_id_problems(rows, catalog_row(path), path.name)
        if problems:
            raise SchemaError(problems)
    meta = parquet_meta(path, period_codes(rows))
    rows, counts = tidy_rows(rows)
    if not rows:
        raise no_rows(path.name)
    meta["note"] += tidy_note(counts)
    check_meta(meta, where=path.name)
    return convert(rows, meta, sha256_of(path))


def load_table(path: str | Path) -> TableInput:
    """Load one Parquet table; raises SchemaError with every problem found."""
    data_path = Path(path)
    if not data_path.exists():
        raise SchemaError([Problem(str(data_path), "file not found")])
    if data_path.suffix.lower() != ".parquet":
        raise SchemaError(
            [Problem(data_path.name, f"unsupported input format {data_path.suffix!r} (use data/values/*.parquet)")]
        )
    table = load_parquet(data_path)
    problems = validate(table)
    if problems:
        raise SchemaError(problems)
    return table
