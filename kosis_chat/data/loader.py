"""Read supported input formats (A: CSV + meta.json, B: KOSIS JSON + meta.json) into a TableInput."""

from __future__ import annotations

import csv
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
        for key in ("ITM_NM", "PRD_DE", "DT", "UNIT_NM"):
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
        cells.append(
            Cell(
                item=str(r["ITM_NM"]).strip(),
                dims=dims,
                dim_codes=codes,
                period_raw=str(r["PRD_DE"]).strip(),
                value_raw=str(r["DT"]),
                unit=str(r["UNIT_NM"]).strip(),
                source_ref=f"{table_meta.tbl_id}|{item_code}|{code_part}|{r['PRD_DE']}",
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
    else:
        raise SchemaError([Problem(data_path.name, f"unsupported input format {suffix!r} (use .csv or KOSIS .json)")])
    problems = validate(table)
    if problems:
        raise SchemaError(problems)
    return table
