"""Loaded Parquet rows (long format: one row per value) -> TableInput cells, with the unit inferred from the names.

The body of kosis_chat/data/loader.py moved here in T1-28 S2: this module never opens a file. The file side
(reading the Parquet, tables.csv, the file date and hash) is kosis_chat/adapters/parquet.py; data/loader.py
re-exports every name. The unit rule (infer_unit and its constants) moved to kosis_ai/units.py in T1-28 S2b; the
names below are the same objects, so data/loader.py and the callers here are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from kosis_chat.prepare.schemas import Cell, Problem, SchemaError, TableInput, TableMeta, validate_meta_dict

try:
    from kosis_ai import units
except ImportError:  # same fallback as prepare/index.py: src/ is not on the path for `python -m ...` (S6 removes it)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kosis_ai import units

# Parquet has up to eight category levels: dim1_* .. dim8_*.
DIM_LEVELS = range(1, 9)

# Unit fallback when KOSIS gives no unit (e.g. DT_1B8000F/H put it in names: "출생건수 (명)", "조사망률(천명당)").
UNKNOWN_UNIT = units.UNKNOWN_UNIT
KNOWN_UNITS = units.INFER_KNOWN_UNITS  # 16 units, the loader's set (D-u (b)); kosis_ai.normalize.KNOWN_UNITS has 26
_PAREN = units.PAREN
# Names whose parentheses are not a unit, or that have none; checked when no unit is found in parentheses.
UNIT_OVERRIDES = units.INFER_UNIT_OVERRIDES
_unit_in_parens = units.unit_in_parens
infer_unit = units.infer_unit

# KOSIS period codes in the Parquet prd_se column -> TableMeta.period_type.
PRD_SE = {"Y": "Y", "A": "Y", "M": "M", "Q": "Q", "H": "H"}
# The columns the conversion reads (the input contract of a loaded table; other columns are ignored).
PARQUET_COLUMNS = ["tbl_id", "prd_se", "period", "item_id", "item_name", "unit", "value_raw", "value", "value_kind"] + [
    f"dim{i}_{part}" for i in range(1, 9) for part in ("name", "code", "label")
]


def rows_of(frame: Any) -> list[dict]:
    """The rows of a loaded table (pyarrow Table, or pandas DataFrame), PARQUET_COLUMNS only, as Python values.

    A DataFrame goes through pyarrow so that missing values are None (not NaN), as when the file is read."""
    if not hasattr(frame, "to_pylist"):  # pandas
        import pyarrow as pa

        frame = pa.Table.from_pandas(frame, preserve_index=False)
    return frame.select(PARQUET_COLUMNS).to_pylist()


class _HashSink:
    """A write-only file that feeds what it is given to a hash (Arrow IPC writes here; nothing is kept)."""

    closed = False

    def __init__(self, digest) -> None:
        self.digest = digest

    def write(self, data) -> int:
        self.digest.update(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def as_arrow(frame: Any):
    """A pandas DataFrame as a pyarrow Table (once; missing values None as in rows_of); a pyarrow Table as is."""
    if hasattr(frame, "to_pylist"):
        return frame
    import pyarrow as pa

    return pa.Table.from_pandas(frame, preserve_index=False)


def frame_digest(frame: Any, digest) -> None:
    """Feed a loaded table's content to `digest` (hashlib object): the schema (names and types, columns sorted by
    name, no metadata) and the Arrow IPC stream of every column chunk. IPC writes each chunk's visible slice and its
    dictionary, so two different contents never give the same bytes; equal contents chunked differently (or with
    different bytes under null slots) only cost a cache miss. No Python value per cell (T1-28 S5 review: a 35M-cell
    table hashed row by row took ~100 s)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = as_arrow(frame)
    table = table.select(sorted(table.column_names)).replace_schema_metadata(None)
    schema = pa.schema([pa.field(f.name, f.type, f.nullable) for f in table.schema])
    table = pa.Table.from_arrays(table.columns, schema=schema)
    digest.update(str(schema).encode("utf-8"))
    sink = pa.PythonFile(_HashSink(digest), mode="w")
    with ipc.new_stream(sink, schema) as writer:
        writer.write_table(table)


def tbl_id_counts(frame: Any) -> tuple[set, int]:
    """(the distinct tbl_id values, the rows without one: None, NaN or blank) without a Python value per row."""
    import pyarrow as pa
    import pyarrow.compute as pc

    column = as_arrow(frame).column("tbl_id")
    if pa.types.is_floating(column.type):  # an all-NaN pandas column
        column = pc.if_else(pc.is_nan(column), pa.scalar(None, column.type), column)
    if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
        column = pc.cast(column, pa.string())
    blank = pc.equal(pc.utf8_trim_whitespace(column), "")
    missing = column.null_count + pc.sum(pc.fill_null(blank, False)).as_py() if len(column) else 0
    present = pc.filter(column, pc.fill_null(pc.invert(blank), False))
    return set(pc.unique(present).to_pylist()), int(missing or 0)


def frame_content(frame: Any) -> tuple[list[tuple[str, str]], list[str], list[list]]:
    """Every column of a loaded table, for the content hash: ((name, dtype) sorted by name, the names in that order,
    each row's values in that order). Missing values are None, through pyarrow as in rows_of."""
    if not hasattr(frame, "to_pylist"):  # pandas
        import pyarrow as pa

        frame = pa.Table.from_pandas(frame, preserve_index=False)
    typed = sorted((str(f.name), str(f.type)) for f in frame.schema)
    names = [name for name, _ in typed]
    return typed, names, [[r[n] for n in names] for r in frame.to_pylist()]


def period_codes(rows: Iterable[Mapping]) -> set[str]:
    return {str(r["prd_se"]) for r in rows if r["prd_se"]}


# A table over this many rows is not loaded whole: every row becomes a Python Cell (T1-29: the provided files over it
# have 9 to 111 million rows and ran out of memory). adapters/parquet.py checks the Parquet footer before reading.
ROW_LIMIT = 5_000_000


def _blank(value: Any) -> bool:
    return value is None or not str(value).strip()


def _cell_key(r: Mapping) -> tuple:
    """What cells_of makes the source_ref from: the item, the codes of the named category levels and the period."""
    return (r["item_id"], tuple((level, r[f"dim{level}_code"]) for level in DIM_LEVELS if r[f"dim{level}_name"]), r["period"])


def tidy_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Rows a whole table used to be rejected for, settled row by row (T1-29); no number is chosen or made up.

    1. A row with neither a period nor a value is dropped (empty shells in some provided zip tables).
    2. Two period codes writing one period string (month and quarter both use 197001): the rows of every code other
       than the table's own (the first code in order, as meta_from_row takes period_type) are dropped.
    3. Rows with one key: equal values are kept once; different values become one blank cell (value_kind "blank" ->
       value_flag "missing"), so a question about it is refused as a missing value.
    Returns (rows in the first-seen order, counts by rule: blank_rows, cycle_rows, same_rows, conflict_cells)."""
    counts = {"blank_rows": 0, "cycle_rows": 0, "same_rows": 0, "conflict_cells": 0}
    kept = [r for r in rows if not (_blank(r["period"]) and r["value"] is None and _blank(r["value_raw"]))]
    counts["blank_rows"] = len(rows) - len(kept)
    codes = sorted(period_codes(kept))
    if len(codes) > 1:
        seen: dict[tuple, set] = {}
        for r in kept:
            seen.setdefault(_cell_key(r), set()).add(r["prd_se"])
        losing = set().union(*(cycles for cycles in seen.values() if len(cycles) > 1)) - {codes[0]}
        if losing:
            before = len(kept)
            kept = [r for r in kept if r["prd_se"] not in losing]
            counts["cycle_rows"] = before - len(kept)
    groups: dict[tuple, list[dict]] = {}
    for r in kept:
        groups.setdefault(_cell_key(r), []).append(r)
    if len(groups) == len(kept):
        return kept, counts
    out = []
    for same in groups.values():
        first = same[0]
        if len(same) > 1:
            if len({(r["value_raw"], r["value"], r["value_kind"]) for r in same}) == 1:
                counts["same_rows"] += len(same) - 1
            else:
                counts["conflict_cells"] += 1
                first = {**first, "value_raw": "", "value": None, "value_kind": "blank"}
        out.append(first)
    return out, counts


def tidy_note(counts: Mapping[str, int]) -> str:
    """What tidy_rows did, for TableMeta.note ("" when nothing)."""
    names = (("blank_rows", "시점·값이 빈 행 {} 제외"), ("cycle_rows", "시점 표기가 겹친 다른 주기 행 {} 제외"),
             ("same_rows", "같은 행 {} 제외"), ("conflict_cells", "값이 서로 다른 겹친 셀 {}개는 값 없음으로 처리"))
    parts = [text.format(f"{counts[key]:,}") for key, text in names if counts.get(key)]
    return f"; 행 정리: {', '.join(parts)}" if parts else ""


def meta_from_row(row: Mapping, prd_se: set[str], fetched_at: str) -> dict:
    """Meta of a table from its tables.csv row (or a TableInfo in the same names), its prd_se codes and fetch date."""
    codes = sorted(prd_se)
    period_type = PRD_SE.get(codes[0], "Y") if codes else "Y"
    note = f"tables.csv 수록기간: {row.get('period', '')}"
    if len(codes) > 1 or (codes and codes[0] not in PRD_SE):
        note += f"; prd_se {codes} -> {period_type}"
    return {
        "tbl_id": row.get("tbl_id"),
        "tbl_nm": row.get("table_name"),
        "survey_nm": row.get("survey_name"),
        "org_nm": row.get("organization"),
        "period_type": period_type,
        "source_url": row.get("url"),
        "fetched_at": fetched_at,
        "note": note,
    }


def check_meta(meta: dict, where: str, *, allow_empty: tuple[str, ...] = ()) -> None:
    """Raise SchemaError with every problem of the meta dict (before a TableMeta is built)."""
    problems = validate_meta_dict(meta, where=where, allow_empty=allow_empty)
    if problems:
        raise SchemaError(problems)


def build_meta(meta: dict, digest: str) -> TableMeta:
    return TableMeta(
        tbl_id=str(meta["tbl_id"]),
        tbl_nm=str(meta["tbl_nm"]),
        survey_nm=str(meta["survey_nm"]),
        org_nm=str(meta["org_nm"]),
        period_type=str(meta["period_type"]),
        source_url=str(meta["source_url"]),
        fetched_at=str(meta["fetched_at"]),
        input_format="parquet",
        file_sha256=digest,
        note=str(meta.get("note") or ""),
    )


def cells_of(rows: Iterable[Mapping], table_meta: TableMeta) -> tuple[Cell, ...]:
    """One Cell per row, in row order; source_ref = tbl_id|item_id|dim codes|period."""
    cells = []
    for r in rows:
        dims: dict[str, str] = {}
        codes: dict[str, str] = {}
        for level in DIM_LEVELS:
            name = r[f"dim{level}_name"]
            if not name:
                continue
            dims[str(name)] = str(r[f"dim{level}_label"] or "").strip()
            if r[f"dim{level}_code"] is not None:
                codes[str(name)] = str(r[f"dim{level}_code"])
        item = str(r["item_name"] or "").strip()
        unit, unit_source = infer_unit(r["unit"], item, list(dims.values()))
        code_part = "|".join(codes.get(name, "") for name in dims)
        cells.append(
            Cell(
                item=item,
                dims=dims,
                dim_codes=codes,
                item_code=str(r["item_id"] or ""),
                period_raw=str(r["period"]).strip(),
                value_raw=str(r["value_raw"] if r["value_raw"] is not None else ""),
                value_kind=r["value_kind"],
                source_value=r["value"],
                unit=unit,
                unit_source=unit_source,
                source_ref=f"{table_meta.tbl_id}|{r['item_id']}|{code_part}|{r['period']}",
            )
        )
    return tuple(cells)


def convert(rows: list[dict], meta: dict, digest: str) -> TableInput:
    """Rows and a checked meta dict (check_meta) -> TableInput, not yet validated or cleaned."""
    table_meta = build_meta(meta, digest)
    return TableInput(meta=table_meta, cells=cells_of(rows, table_meta))


def no_rows(where: str) -> SchemaError:
    return SchemaError([Problem(where, "table has no rows")])
