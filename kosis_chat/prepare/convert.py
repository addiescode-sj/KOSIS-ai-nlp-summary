"""Loaded Parquet rows (long format: one row per value) -> TableInput cells, with the unit inferred from the names.

The body of kosis_chat/data/loader.py moved here in T1-28 S2: this module never opens a file. The file side
(reading the Parquet, tables.csv, the file date and hash) is kosis_chat/adapters/parquet.py; data/loader.py
re-exports every name. infer_unit stays with the conversion until T1-28 S2b moves the unit rules to kosis_ai/units.py.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from kosis_chat.prepare.schemas import Cell, Problem, SchemaError, TableInput, TableMeta, validate_meta_dict

# Parquet has up to eight category levels: dim1_* .. dim8_*.
DIM_LEVELS = range(1, 9)

# Unit fallback when KOSIS gives no unit (e.g. DT_1B8000F/H put it in names: "출생건수 (명)", "조사망률(천명당)").
UNKNOWN_UNIT = "단위 미상"
KNOWN_UNITS = {"명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "백만원", "명/㎢", "㎢", "배", "쌍"}
_PAREN = re.compile(r"\(([^()]*)\)")
# Names whose parentheses are not a unit, or that have none; checked when no unit is found in parentheses.
UNIT_OVERRIDES = (("합계출산율", "명"), ("기대수명", "년"))

# KOSIS period codes in the Parquet prd_se column -> TableMeta.period_type.
PRD_SE = {"Y": "Y", "A": "Y", "M": "M", "Q": "Q", "H": "H"}
# The columns the conversion reads (the input contract of a loaded table; other columns are ignored).
PARQUET_COLUMNS = ["tbl_id", "prd_se", "period", "item_id", "item_name", "unit", "value_raw", "value", "value_kind"] + [
    f"dim{i}_{part}" for i in range(1, 9) for part in ("name", "code", "label")
]


def _unit_in_parens(name: str) -> str | None:
    for text in reversed(_PAREN.findall(name or "")):
        text = text.strip()
        if text in KNOWN_UNITS or text.endswith("당"):  # "천명당", "출생아 천명당"
            return text
    return None


def infer_unit(unit: str | None, item: str, dim_labels: list[str]) -> tuple[str, str]:
    """Return (unit, unit_source): the given unit, else a unit in the item / category names, else UNKNOWN_UNIT."""
    if unit and str(unit).strip():
        return str(unit).strip(), "given"
    found = _unit_in_parens(item)
    if found:
        return found, "item_name"
    for label in dim_labels:
        found = _unit_in_parens(label)
        if found:
            return found, "dim_label"
    for name in (item, *dim_labels):
        for keyword, override in UNIT_OVERRIDES:
            if keyword in (name or ""):
                return override, "override"
    return UNKNOWN_UNIT, "unknown"


def rows_of(frame: Any) -> list[dict]:
    """The rows of a loaded table (pyarrow Table, or pandas DataFrame), PARQUET_COLUMNS only, as Python values.

    A DataFrame goes through pyarrow so that missing values are None (not NaN), as when the file is read."""
    if not hasattr(frame, "to_pylist"):  # pandas
        import pyarrow as pa

        frame = pa.Table.from_pandas(frame, preserve_index=False)
    return frame.select(PARQUET_COLUMNS).to_pylist()


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
