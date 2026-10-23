"""Read and normalize one provided table: fetch_table(table_id, filters) -> DataFrame (plans/T1-05.md).

Source is data/values/<org>_<tbl>.parquet (long format, one row per cell). Original columns are kept as they are;
derived columns are added: unit_resolved, unit_source, measure_kind, value_label.
Conditions the table cannot meet raise LookupFailed with a RefusalReason, so T1-08 can answer with the reason.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from kosis_ai.schemas import MAX_DIMS, MeasureKind, RefusalReason

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KEY_COLUMNS = ["period", "item_id", *(f"dim{i}_code" for i in range(1, MAX_DIMS + 1))]
LABEL_COLUMNS = [f"dim{i}_label" for i in range(1, MAX_DIMS + 1)]

# Units seen in the provided data (docs/eda-full.md 2-4). Anything else in a unit column or in parentheses is not a
# unit: '월세없음', '15세 이상', '4년제이상', '미취학 포함' ...
KNOWN_UNITS = {
    "명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "만원", "십만원", "백만원", "명/㎢", "㎢", "㎡",
    "배", "쌍", "점", "개사", "세대", "평", "척", "대", "여자백명당남자수",
}  # fmt: skip
# Names with no unit anywhere in the data; checked after the unit column and parentheses.
UNIT_OVERRIDES = {"합계출산율": "명", "기대수명": "년"}
_PAREN = re.compile(r"\(([^()]*)\)")
# Items that are rates, averages or indexes must not inherit a count unit from the table metadata.
_NOT_COUNT = re.compile(r"율|비율|성비|부양비|평균|중위|지수")

VALUE_LABELS = {
    "ok": None,
    "dash": "해당 없음",
    "x_secret": "비공개",
    "asterisk": "비공개",
    "blank": "그 시점에 집계하지 않음",
    "ellipsis": "자료 없음",
    "other_text": "해석할 수 없는 값",
}

# measure_kind (PREPROCESSING.md 1장). The first label with a keyword decides; dim labels before the item name,
# because some items are table titles ('인구동태건수 및 동태율 추이') while the dim names the measure ('출생아수(명)').
_KIND_KEYWORDS: list[tuple[re.Pattern, MeasureKind]] = [
    (re.compile(r"지수"), "index"),
    (re.compile(r"평균|중위|기대수명|기대여명"), "average"),
    (re.compile(r"율|비율|성비|부양비|밀도"), "ratio"),
    (re.compile(r"건수|출생아수|사망자수|이동자|전입|전출|혼인|이혼|출생|사망"), "count_flow"),
    (re.compile(r"인구|가구|세대|주택|인원"), "count_stock"),
]
_RATIO_UNITS = {"%", "명/㎢", "배", "여자백명당남자수"}
_STOCK_UNITS = {"명", "가구", "호", "세대", "개", "개사", "천명"}
_FLOW_UNITS = {"건", "쌍"}
# Per table or item fixes for the keyword rules: {(tbl_id, item_name or dim label): kind}.
MEASURE_OVERRIDES: dict[tuple[str, str], MeasureKind] = {}


_UNIT_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")


class LookupFailed(Exception):
    """The table has no cell for the requested condition; T1-08 turns it into a Refusal."""

    def __init__(self, reason: RefusalReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def _bare(label: str) -> str:
    """'1인 (가구)' -> '1인': item and dim labels without a trailing unit suffix."""
    return _UNIT_SUFFIX.sub("", label)


def _matches(labels: pd.Series, label: str) -> pd.Series:
    return (labels == label) | (labels.map(_bare) == label)


def _dim_hit(df: pd.DataFrame, name: str, label: str) -> pd.Series:
    """Rows whose `name` dim is `label` (or that code). The same label at several levels — '동부' is both
    code 03 under 전국 and 21003 under 부산 — resolves to the shortest code, i.e. the top level."""
    col = next((f"dim{i}" for i in range(1, MAX_DIMS + 1) if f"dim{i}_name" in df and (df[f"dim{i}_name"] == name).any()), None)
    if col is None:
        raise LookupFailed("class_mismatch", f"표에 '{name}' 분류가 없습니다")
    labels, codes = df[f"{col}_label"], df[f"{col}_code"]
    hit = _matches(labels, label) | (codes == label)
    found = codes[hit].unique()
    if len(found) > 1:
        top = [c for c in found if len(c) == min(map(len, found))]
        if len(top) > 1:
            raise ValueError(f"'{label}' is {len(top)} different '{name}' cells {top[:5]}; pass a code instead")
        hit &= codes == top[0]
    return hit


def select(df: pd.DataFrame, *, item: str | None = None, dims: dict[str, str] | None = None) -> pd.DataFrame:
    """Rows of df whose item and dims match, or LookupFailed naming the condition the table cannot meet."""
    out = df
    if item is not None:
        out = out[_matches(out["item_name"], item)]
        if out.empty:
            raise LookupFailed("no_item", f"표에 '{item}' 항목이 없습니다")
    for name, label in (dims or {}).items():
        hit = _dim_hit(out, name, label)
        if not hit.any():
            raise LookupFailed("class_mismatch", f"'{name}' 분류에 '{label}'이(가) 없습니다")
        out = out[hit]
    return out


def _known(unit: str | None) -> str | None:
    if not isinstance(unit, str):  # None or NaN from pandas
        return None
    text = unit.strip()
    return text if text in KNOWN_UNITS or (text.endswith("당") and len(text) <= 12) else None


def _paren_unit(name: str | None) -> str | None:
    for text in reversed(_PAREN.findall(name or "")):
        if found := _known(text):
            return found
    return None


def _override_unit(name: str | None) -> str | None:
    return next((unit for keyword, unit in UNIT_OVERRIDES.items() if keyword in (name or "")), None)


def _keyword_kind(name: str | None) -> MeasureKind | None:
    return next((kind for pattern, kind in _KIND_KEYWORDS if pattern.search(name or "")), None)


def resolve_unit(
    unit: str | None, item_name: str, dim_labels: list[str], meta_unit: str | None = None
) -> tuple[str | None, str]:
    """(unit, source). Order: unit column (known units only) -> '(unit)' in item / dim names -> overrides ->
    meta_unit -> None. Pass meta_unit only when it is safe for the table (see fetch_table)."""
    if found := _known(unit):
        return found, "unit_column"
    for name in (item_name, *dim_labels):
        if found := _paren_unit(name):
            return found, "parentheses"
    for name in (item_name, *dim_labels):
        if found := _override_unit(name):
            return found, "override"
    if meta_unit and not any(_NOT_COUNT.search(n or "") for n in (item_name, *dim_labels)):
        return meta_unit, "meta"
    return None, "unknown"


def measure_kind(tbl_id: str, unit: str | None, item_name: str, dim_labels: list[str]) -> MeasureKind | None:
    """count_stock / count_flow / ratio / average / index, or None when the rules cannot tell (then nothing sums)."""
    for name in (*dim_labels, item_name):
        if (tbl_id, name) in MEASURE_OVERRIDES:
            return MEASURE_OVERRIDES[(tbl_id, name)]
    if unit in _RATIO_UNITS or (unit or "").endswith("당"):
        return "ratio"
    for name in (*dim_labels, item_name):
        if kind := _keyword_kind(name):
            return kind
    if unit in _FLOW_UNITS:
        return "count_flow"
    if unit in _STOCK_UNITS:
        return "count_stock"
    return None


@lru_cache(maxsize=256)
def _meta_unit(table_id: str, data_dir: Path) -> str | None:
    path = data_dir / "table_metadata" / f"{table_id}.json"
    if not path.exists():
        return None
    return _known(json.loads(path.read_text(encoding="utf-8")).get("unitNmKor"))


def _table_path(table_id: str, data_dir: Path) -> Path:
    path = data_dir / "values" / f"{table_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{table_id} is not in the provided data ({path})")
    return path


def normalize(df: pd.DataFrame, table_id: str = "", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Add unit_resolved, unit_source, measure_kind and value_label to a long-format cell frame."""
    # Same rules as resolve_unit / measure_kind, applied to each column's distinct values and filled in rule order
    # (a 450k-row slice of a 100M-row table normalizes in well under a second; tests check both give the same).
    labels = [c for c in LABEL_COLUMNS if c in df]
    texts = ["item_name", *labels]

    def per_value(col: str, fn) -> pd.Series:
        values = df[col].dropna().unique()
        return df[col].map(dict(zip(values, map(fn, values))))

    unit = per_value("unit", _known)
    source = pd.Series("unit_column", index=df.index, dtype=object).where(unit.notna())
    steps = [(col, _paren_unit, "parentheses") for col in texts] + [(col, _override_unit, "override") for col in texts]
    for col, fn, name in steps:
        found = per_value(col, fn)
        take = unit.isna() & found.notna()
        unit, source = unit.where(~take, found), source.where(~take, name)
    # The metadata unit describes the whole table: use it only when no row names a different unit.
    meta = _meta_unit(table_id, data_dir) if table_id else None
    if meta is not None and set(unit.dropna().unique()) - {meta}:
        meta = None
    if meta is not None:
        counts = ~pd.concat([per_value(c, lambda s: bool(_NOT_COUNT.search(s))).fillna(False) for c in texts], axis=1).any(axis=1)
        take = unit.isna() & counts
        unit, source = unit.where(~take, meta), source.where(~take, "meta")
    source = source.fillna("unknown")

    kind = pd.Series(None, index=df.index, dtype=object)
    for col in [*labels, "item_name"]:
        found = df[["tbl_id", col]].apply(tuple, axis=1).map(MEASURE_OVERRIDES) if MEASURE_OVERRIDES else None
        if found is not None:
            kind = kind.where(kind.notna(), found)
    kind = kind.where(kind.notna(), unit.map(lambda u: "ratio" if u in _RATIO_UNITS or str(u).endswith("당") else None))
    for col in [*labels, "item_name"]:
        kind = kind.where(kind.notna(), per_value(col, _keyword_kind))
    kind = kind.where(kind.notna(), unit.map(lambda u: "count_flow" if u in _FLOW_UNITS else "count_stock" if u in _STOCK_UNITS else None))

    out = df.copy()
    out["unit_resolved"] = unit.astype(object).where(unit.notna(), None)
    out["unit_source"] = source
    out["measure_kind"] = kind.where(kind.notna(), None)
    out["value_label"] = out["value_kind"].map(VALUE_LABELS)
    return out


def fetch_table(table_id: str, filters: dict | None = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Normalized cells of one table ('101_DT_1B8000H') that match `filters`.

    filters: {"item": label, "periods": [...], "start": "1970", "end": ..., "<dim name>": label or code}.
    Periods are filtered while reading, so a 100M-row table stays fast. Missing conditions raise LookupFailed
    (no_item / class_mismatch / no_period). A table whose cells cannot be told apart (same key twice, ZIP tables
    with a hidden category) raises LookupFailed("class_mismatch") instead of returning mixed values.
    """
    filters = dict(filters or {})
    item = filters.pop("item", None)
    periods = filters.pop("periods", None)
    start, end = filters.pop("start", None), filters.pop("end", None)
    conds = [
        ds.field("period").isin(list(periods)) if periods is not None else None,
        ds.field("period") >= start if start else None,
        ds.field("period") <= end if end else None,
    ]
    expr = None
    for cond in conds:
        if cond is not None:
            expr = cond if expr is None else expr & cond
    df = ds.dataset(_table_path(table_id, data_dir)).to_table(filter=expr).to_pandas()
    if periods is not None:  # read with only these periods, so any absent here is absent from the table
        missing = sorted(set(periods) - set(df["period"]))
        if missing:
            raise LookupFailed("no_period", f"표에 {', '.join(missing)} 시점이 없습니다")
    if df.empty:
        raise LookupFailed("no_period", "요청한 기간에 해당하는 시점이 표에 없습니다")
    df = select(df, item=item, dims=filters or None)
    if df.duplicated([c for c in KEY_COLUMNS if c in df]).any():
        raise LookupFailed("class_mismatch", "표에 이름 없는 분류가 숨어 있어 칸을 구분할 수 없습니다")
    return normalize(df, table_id, data_dir)
