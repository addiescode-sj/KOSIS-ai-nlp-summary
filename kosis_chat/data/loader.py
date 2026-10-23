"""Read supported input formats (A: CSV + meta.json, B: KOSIS JSON + meta.json, C: Parquet + tables.csv) into a TableInput."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path

from kosis_chat.data.schema import Cell, Problem, SchemaError, TableInput, TableMeta, validate, validate_meta_dict

CSV_REQUIRED_COLUMNS = ("item", "period", "value", "unit")
DIM_PREFIX = "dim:"
CODE_PREFIX = "code:"
# A header like "2023", "202401" or "2024 Q1" means periods are columns (wide format C).
_PERIOD_HEADER = re.compile(r"^\d{4}(\s*\.?\s*(\d{2}|Q\d|H\d))?\s*(년|월)?$", re.IGNORECASE)
# KOSIS OpenAPI keys for up to eight category levels.
_KOSIS_LEVELS = range(1, 9)

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


def meta_path_for(data_path: Path) -> Path:
    """`pop.csv` / `pop.json` -> `pop.meta.json` next to it."""
    return data_path.with_name(data_path.stem + ".meta.json")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SchemaError(
            [Problem(path.name, f"not UTF-8 (KOSIS web downloads are often CP949; convert first): {exc.reason}")]
        ) from exc


def _read_meta(data_path: Path) -> dict:
    meta_path = meta_path_for(data_path)
    if not meta_path.exists():
        raise SchemaError([Problem(meta_path.name, "meta file not found; every table needs <name>.meta.json")])
    try:
        meta = json.loads(_read_text(meta_path))
    except json.JSONDecodeError as exc:
        raise SchemaError([Problem(meta_path.name, f"invalid JSON: {exc.msg}")]) from exc
    if not isinstance(meta, dict):
        raise SchemaError([Problem(meta_path.name, "must be a JSON object")])
    return meta


def _build_meta(meta: dict, input_format: str, digest: str) -> TableMeta:
    return TableMeta(
        tbl_id=str(meta["tbl_id"]),
        tbl_nm=str(meta["tbl_nm"]),
        survey_nm=str(meta["survey_nm"]),
        org_nm=str(meta["org_nm"]),
        period_type=str(meta["period_type"]),
        source_url=str(meta["source_url"]),
        fetched_at=str(meta["fetched_at"]),
        input_format=input_format,
        file_sha256=digest,
        note=str(meta.get("note") or ""),
    )


def _load_csv(path: Path) -> TableInput:
    """Format A: long CSV, one row per cell."""
    meta = _read_meta(path)
    text = _read_text(path)
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    problems: list[Problem] = []
    if not header:
        raise SchemaError([Problem(path.name, "empty file or missing header row")])
    if any(_PERIOD_HEADER.match(h) for h in header):
        raise SchemaError(
            [Problem(path.name, "wide format (periods as columns) is not supported in B0; provide long format")]
        )
    missing = [c for c in CSV_REQUIRED_COLUMNS if c not in header]
    if missing:
        problems.append(Problem(path.name, f"missing required columns: {', '.join(missing)}"))
    problems += validate_meta_dict(meta, where=meta_path_for(path).name)
    if problems:
        raise SchemaError(problems)

    digest = sha256_of(path)
    table_meta = _build_meta(meta, "A", digest)
    dim_cols = [h for h in header if h.startswith(DIM_PREFIX)]
    code_cols = [h for h in header if h.startswith(CODE_PREFIX)]
    cells = []
    for row_no, raw in enumerate(reader, start=2):  # header is line 1
        row = {(k or "").strip(): (v if v is not None else "") for k, v in raw.items()}
        dims = {c[len(DIM_PREFIX):]: row[c].strip() for c in dim_cols}
        codes = {c[len(CODE_PREFIX):]: row[c].strip() for c in code_cols if row[c].strip()}
        if codes:
            code_part = "|".join(f"{k}={v}" for k, v in sorted(codes.items()))
            source_ref = f"{table_meta.tbl_id}|{row['item'].strip()}|{code_part}|{row['period'].strip()}"
        else:
            source_ref = f"sha256:{digest[:12]}#row{row_no}"
        cells.append(
            Cell(
                item=row["item"].strip(),
                dims=dims,
                dim_codes=codes,
                period_raw=row["period"].strip(),
                value_raw=row["value"],
                unit=row["unit"].strip(),
                source_ref=source_ref,
            )
        )
    return TableInput(meta=table_meta, cells=tuple(cells))


def _load_kosis_json(path: Path) -> TableInput:
    """Format B: KOSIS OpenAPI JSON list; meta.json supplies what the response does not."""
    meta = dict(_read_meta(path))
    try:
        rows = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise SchemaError([Problem(path.name, f"invalid JSON: {exc.msg}")]) from exc
    if isinstance(rows, dict) and "err" in rows:
        raise SchemaError([Problem(path.name, f"KOSIS error response: {rows.get('errMsg') or rows['err']}")])
    if not isinstance(rows, list) or not rows:
        raise SchemaError([Problem(path.name, "expected a non-empty JSON list of KOSIS rows")])

    problems: list[Problem] = []
    # tbl_id / tbl_nm may come from the response; if both sides have them they must agree.
    for meta_key, row_key in (("tbl_id", "TBL_ID"), ("tbl_nm", "TBL_NM")):
        found = {str(r.get(row_key)) for r in rows if r.get(row_key)}
        if len(found) > 1:
            problems.append(Problem(path.name, f"{row_key} differs between rows: {sorted(found)}"))
        elif found:
            value = found.pop()
            if meta.get(meta_key) and str(meta[meta_key]) != value:
                problems.append(Problem(meta_key, f"meta.json says {meta[meta_key]!r}, response says {value!r}"))
            meta.setdefault(meta_key, value)
    problems += validate_meta_dict(meta, where=meta_path_for(path).name)
    for i, r in enumerate(rows):
        for key in ("ITM_NM", "PRD_DE", "DT"):  # UNIT_NM may be absent; infer_unit fills it
            if key not in r:
                problems.append(Problem(f"{path.name}[{i}]", f"missing key {key}"))
    if problems:
        raise SchemaError(problems)

    table_meta = _build_meta(meta, "B", sha256_of(path))
    cells = []
    for r in rows:
        dims: dict[str, str] = {}
        codes: dict[str, str] = {}
        for level in _KOSIS_LEVELS:
            label = r.get(f"C{level}_NM")
            if label is None:
                continue
            dim_name = str(r.get(f"C{level}_OBJ_NM") or f"C{level}")
            dims[dim_name] = str(label).strip()
            if r.get(f"C{level}") is not None:
                codes[dim_name] = str(r[f"C{level}"])
        item_code = str(r.get("ITM_ID") or r["ITM_NM"])
        code_part = "|".join(codes.get(name, "") for name in dims)
        item = str(r["ITM_NM"]).strip()
        unit, unit_source = infer_unit(r.get("UNIT_NM"), item, list(dims.values()))
        cells.append(
            Cell(
                item=item,
                dims=dims,
                dim_codes=codes,
                period_raw=str(r["PRD_DE"]).strip(),
                value_raw=str(r["DT"]),
                unit=unit,
                unit_source=unit_source,
                source_ref=f"{table_meta.tbl_id}|{item_code}|{code_part}|{r['PRD_DE']}",
            )
        )
    return TableInput(meta=table_meta, cells=tuple(cells))


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
    """Meta from <name>.meta.json when present, else from the tables.csv row whose key is the file stem."""
    if meta_path_for(path).exists():
        return _read_meta(path)
    catalog = _find_catalog(path)
    if catalog is None:
        raise SchemaError([Problem(path.name, "no <name>.meta.json and no tables.csv next to it or one level up")])
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
    """Format C: one long Parquet file per table (data/values), meta from tables.csv."""
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

    table_meta = _build_meta(meta, "C", sha256_of(path))
    cells = []
    for r in rows:
        dims: dict[str, str] = {}
        codes: dict[str, str] = {}
        for level in _KOSIS_LEVELS:
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
                period_raw=str(r["period"]).strip(),
                value_raw=str(r["value_raw"] if r["value_raw"] is not None else ""),
                unit=unit,
                unit_source=unit_source,
                source_ref=f"{table_meta.tbl_id}|{r['item_id']}|{code_part}|{r['period']}",
            )
        )
    return TableInput(meta=table_meta, cells=tuple(cells))


def load_table(path: str | Path) -> TableInput:
    """Load one table, choosing the adapter by extension; raises SchemaError with every problem found."""
    data_path = Path(path)
    if not data_path.exists():
        raise SchemaError([Problem(str(data_path), "file not found")])
    if data_path.name.endswith(".meta.json"):
        raise SchemaError([Problem(data_path.name, "pass the data file, not the meta file")])
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        table = _load_csv(data_path)
    elif suffix == ".json":
        table = _load_kosis_json(data_path)
    elif suffix == ".parquet":
        table = _load_parquet(data_path)
    else:
        raise SchemaError(
            [Problem(data_path.name, f"unsupported input format {suffix!r} (use .csv, KOSIS .json or .parquet)")]
        )
    problems = validate(table)
    if problems:
        raise SchemaError(problems)
    return table
