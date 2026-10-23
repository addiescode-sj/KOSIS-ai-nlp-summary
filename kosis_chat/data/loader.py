"""Read one table from data/values/<org>_<tbl>.parquet (long format, meta from data/tables.csv) into a TableInput."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from pathlib import Path

from kosis_chat.data.schema import Cell, Problem, SchemaError, TableInput, TableMeta, validate, validate_meta_dict

# Parquet has up to eight category levels: dim1_* .. dim8_*.
_DIM_LEVELS = range(1, 9)

# Unit fallback when KOSIS gives no unit (e.g. DT_1B8000F/H put it in names: "출생건수 (명)", "조사망률(천명당)").
UNKNOWN_UNIT = "단위 미상"
KNOWN_UNITS = {"명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "백만원", "명/㎢", "㎢", "배", "쌍"}
_PAREN = re.compile(r"\(([^()]*)\)")
# Names whose parentheses are not a unit, or that have none; checked when no unit is found in parentheses.
UNIT_OVERRIDES = (("합계출산율", "명"), ("기대수명", "년"))


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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_meta(meta: dict, digest: str) -> TableMeta:
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


# KOSIS period codes in the Parquet prd_se column -> TableMeta.period_type.
_PRD_SE = {"Y": "Y", "A": "Y", "M": "M", "Q": "Q", "H": "H"}
_PARQUET_COLUMNS = ["tbl_id", "prd_se", "period", "item_id", "item_name", "unit", "value_raw"] + [
    f"dim{i}_{part}" for i in range(1, 9) for part in ("name", "code", "label")
]


def _find_catalog(path: Path) -> Path | None:
    """tables.csv next to the Parquet file or one level up (data/values/x.parquet -> data/tables.csv)."""
    for folder in (path.parent, path.parent.parent):
        if (folder / "tables.csv").exists():
            return folder / "tables.csv"
    return None


def _parquet_meta(path: Path, prd_se: set[str]) -> dict:
    """Meta from the tables.csv row whose key is the file stem."""
    catalog = _find_catalog(path)
    if catalog is None:
        raise SchemaError([Problem(path.name, "no tables.csv next to it or one level up")])
    with catalog.open(encoding="utf-8-sig", newline="") as f:
        row = next((r for r in csv.DictReader(f) if r.get("key") == path.stem), None)
    if row is None:
        raise SchemaError([Problem(catalog.name, f"no row with key {path.stem!r}")])
    codes = sorted(prd_se)
    period_type = _PRD_SE.get(codes[0], "Y") if codes else "Y"
    note = f"tables.csv 수록기간: {row.get('period', '')}"
    if len(codes) > 1 or (codes and codes[0] not in _PRD_SE):
        note += f"; prd_se {codes} -> {period_type}"
    return {
        "tbl_id": row.get("tbl_id"),
        "tbl_nm": row.get("table_name"),
        "survey_nm": row.get("survey_name"),
        "org_nm": row.get("organization"),
        "period_type": period_type,
        "source_url": row.get("url"),
        # tables.csv has no fetch date; the Parquet file date stands in for it.
        "fetched_at": dt.date.fromtimestamp(path.stat().st_mtime).isoformat(),
        "note": note,
    }


def _load_parquet(path: Path) -> TableInput:
    """One long Parquet file per table (data/values), meta from tables.csv."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SchemaError([Problem(path.name, "reading .parquet needs pyarrow (pip install pyarrow)")]) from exc
    try:
        rows = pq.read_table(path, columns=_PARQUET_COLUMNS).to_pylist()
    except Exception as exc:  # wrong schema or broken file
        raise SchemaError([Problem(path.name, f"cannot read Parquet: {exc}")]) from exc
    if not rows:
        raise SchemaError([Problem(path.name, "table has no rows")])
    meta = _parquet_meta(path, {str(r["prd_se"]) for r in rows if r["prd_se"]})
    problems = validate_meta_dict(meta, where=path.name)
    if problems:
        raise SchemaError(problems)

    table_meta = _build_meta(meta, sha256_of(path))
    cells = []
    for r in rows:
        dims: dict[str, str] = {}
        codes: dict[str, str] = {}
        for level in _DIM_LEVELS:
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
                unit=unit,
                unit_source=unit_source,
                source_ref=f"{table_meta.tbl_id}|{r['item_id']}|{code_part}|{r['period']}",
            )
        )
    return TableInput(meta=table_meta, cells=tuple(cells))


def load_table(path: str | Path) -> TableInput:
    """Load one Parquet table; raises SchemaError with every problem found."""
    data_path = Path(path)
    if not data_path.exists():
        raise SchemaError([Problem(str(data_path), "file not found")])
    if data_path.suffix.lower() != ".parquet":
        raise SchemaError(
            [Problem(data_path.name, f"unsupported input format {data_path.suffix!r} (use data/values/*.parquet)")]
        )
    table = _load_parquet(data_path)
    problems = validate(table)
    if problems:
        raise SchemaError(problems)
    return table
