"""Unit and measure-kind inference from KOSIS item / category names (T1-28 S2b, decisions D-u (b) and N6).

Three entry points, each keeping the rules and the source values it had before the move (only the place changed):

- infer_unit(unit, item, dim_labels) -> (unit, source): the loader's rule (was kosis_chat/data/loader.py, then
  kosis_chat/prepare/convert.py). A non-blank unit column is taken as it is ("given"); else a unit in parentheses
  from INFER_KNOWN_UNITS (16) or ending in 당, loose (any length), item name first ("item_name", "dim_label"); else an
  override; else UNKNOWN_UNIT ("unknown"). Every TableInput cell's unit comes from here.
- resolve_unit(unit, item_name, dim_labels, meta_unit=None) -> (unit | None, source): the T1-05 normalize rule (was
  src/kosis_ai/normalize.py). Known units only (RESOLVE_KNOWN_UNITS, 26, or ending in 당 up to
  12 characters), also
  for the unit column ("unit_column"); then parentheses ("parentheses"), overrides ("override"), the table's meta
  unit when no name is a rate ("meta"), else (None, "unknown"). measure_kind reads its result.
- unit_or_suffix(unit, texts): the contract's rule (was the body of kosis_ai.schemas.Cell.unit_or_suffix): a truthy
  unit as it is, else a short trailing "(unit)" (SUFFIX_UNIT: no space, digit or ~, up to 5 characters), else None.

The three are not merged: D-u (b) keeps today's results and leaves unifying them to a later analysis. Shared here
are only the internals that were textually the same: the parentheses pattern PAREN (loader and normalize) and the
trailing "(...)" pattern UNIT_PAREN (normalize, and the label code in kosis_chat that kosis_chat/shared/patterns.py
re-exports). The old modules re-export every name they had.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only: kosis_ai.schemas imports this module
    from kosis_ai.schemas import MeasureKind

# ---------------------------------------------------------------- shared internals

# Every "(...)" in a name, innermost text captured (findall gives the texts, the last one is read first).
PAREN = re.compile(r"\(([^()]*)\)")
# A trailing "(...)" with the spaces around it, e.g. "출생건수 (명)" -> "출생건수" with .sub(""); group 1 is the text.
UNIT_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")

# ---------------------------------------------------------------- loader rule: infer_unit

# Unit fallback when KOSIS gives no unit (e.g. DT_1B8000F/H put it in names: "출생건수 (명)", "조사망률(천명당)").
UNKNOWN_UNIT = "단위 미상"
INFER_KNOWN_UNITS = {"명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "백만원", "명/㎢", "㎢", "배", "쌍"}
# Names whose parentheses are not a unit, or that have none; checked when no unit is found in parentheses.
INFER_UNIT_OVERRIDES = (("합계출산율", "명"), ("기대수명", "년"))


def unit_in_parens(name: str) -> str | None:
    for text in reversed(PAREN.findall(name or "")):
        text = text.strip()
        if text in INFER_KNOWN_UNITS or text.endswith("당"):  # "천명당", "출생아 천명당"
            return text
    return None


def infer_unit(unit: str | None, item: str, dim_labels: list[str]) -> tuple[str, str]:
    """Return (unit, unit_source): the given unit, else a unit in the item / category names, else UNKNOWN_UNIT."""
    if unit and str(unit).strip():
        return str(unit).strip(), "given"
    found = unit_in_parens(item)
    if found:
        return found, "item_name"
    for label in dim_labels:
        found = unit_in_parens(label)
        if found:
            return found, "dim_label"
    for name in (item, *dim_labels):
        for keyword, override in INFER_UNIT_OVERRIDES:
            if keyword in (name or ""):
                return override, "override"
    return UNKNOWN_UNIT, "unknown"


# ---------------------------------------------------------------- normalize rule: resolve_unit, measure_kind

# Units seen in the provided data (docs/eda-full.md 2-4). Anything else in a unit column or in parentheses is not a
# unit: '월세없음', '15세 이상', '4년제이상', '미취학 포함' ...
RESOLVE_KNOWN_UNITS = {
    "명", "건", "가구", "호", "개", "세", "년", "%", "천명", "원", "천원", "만원", "십만원", "백만원", "명/㎢", "㎢", "㎡",
    "배", "쌍", "점", "개사", "세대", "평", "척", "대", "여자백명당남자수",
}  # fmt: skip
# Names with no unit anywhere in the data; checked after the unit column and parentheses.
RESOLVE_UNIT_OVERRIDES = {"합계출산율": "명", "기대수명": "년"}
# Items that are rates, averages or indexes must not inherit a count unit from the table metadata.
NOT_COUNT = re.compile(r"율|비율|성비|부양비|평균|중위|지수")

# measure_kind (PREPROCESSING.md 1장). The first label with a keyword decides; dim labels before the item name,
# because some items are table titles ('인구동태건수 및 동태율 추이') while the dim names the measure ('출생아수(명)').
KIND_KEYWORDS: list[tuple[re.Pattern, MeasureKind]] = [
    (re.compile(r"지수"), "index"),
    (re.compile(r"평균|중위|기대수명|기대여명"), "average"),
    (re.compile(r"율|비율|성비|부양비|밀도"), "ratio"),
    (re.compile(r"건수|출생아수|사망자수|이동자|전입|전출|혼인|이혼|출생|사망"), "count_flow"),
    (re.compile(r"인구|가구|세대|주택|인원"), "count_stock"),
]
RATIO_UNITS = {"%", "명/㎢", "배", "여자백명당남자수"}
STOCK_UNITS = {"명", "가구", "호", "세대", "개", "개사", "천명"}
FLOW_UNITS = {"건", "쌍"}
# Per table or item fixes for the keyword rules: {(tbl_id, item_name or dim label): kind}.
MEASURE_OVERRIDES: dict[tuple[str, str], MeasureKind] = {}


def known_unit(unit: str | None) -> str | None:
    if not isinstance(unit, str):  # None or NaN from pandas
        return None
    text = unit.strip()
    return text if text in RESOLVE_KNOWN_UNITS or (text.endswith("당") and len(text) <= 12) else None


def paren_unit(name: str | None) -> str | None:
    for text in reversed(PAREN.findall(name or "")):
        if found := known_unit(text):
            return found
    return None


def override_unit(name: str | None) -> str | None:
    return next((unit for keyword, unit in RESOLVE_UNIT_OVERRIDES.items() if keyword in (name or "")), None)


def keyword_kind(name: str | None) -> MeasureKind | None:
    return next((kind for pattern, kind in KIND_KEYWORDS if pattern.search(name or "")), None)


def resolve_unit(
    unit: str | None, item_name: str, dim_labels: list[str], meta_unit: str | None = None
) -> tuple[str | None, str]:
    """(unit, source). Order: unit column (known units only) -> '(unit)' in item / dim names -> overrides ->
    meta_unit -> None. Pass meta_unit only when it is safe for the table (see normalize.fetch_table)."""
    if found := known_unit(unit):
        return found, "unit_column"
    for name in (item_name, *dim_labels):
        if found := paren_unit(name):
            return found, "parentheses"
    for name in (item_name, *dim_labels):
        if found := override_unit(name):
            return found, "override"
    if meta_unit and not any(NOT_COUNT.search(n or "") for n in (item_name, *dim_labels)):
        return meta_unit, "meta"
    return None, "unknown"


def measure_kind(tbl_id: str, unit: str | None, item_name: str, dim_labels: list[str]) -> MeasureKind | None:
    """count_stock / count_flow / ratio / average / index, or None when the rules cannot tell (then nothing sums)."""
    for name in (*dim_labels, item_name):
        if (tbl_id, name) in MEASURE_OVERRIDES:
            return MEASURE_OVERRIDES[(tbl_id, name)]
    if unit in RATIO_UNITS or (unit or "").endswith("당"):
        return "ratio"
    for name in (*dim_labels, item_name):
        if kind := keyword_kind(name):
            return kind
    if unit in FLOW_UNITS:
        return "count_flow"
    if unit in STOCK_UNITS:
        return "count_stock"
    return None


# ---------------------------------------------------------------- contract rule: unit_or_suffix

# Trailing "(unit)" in an item name or dim label, e.g. "출생건수 (명)", "출생아수(명)".
# ponytail: short no-space/no-digit token only, so "(출생시 기대여명)" or "(2012~)" are not units; a unit table if this misses too many.
SUFFIX_UNIT = re.compile(r"\(([^()\s\d~]{1,5})\)\s*$")


def unit_or_suffix(unit: str | None, texts: list[str]) -> str | None:
    """Unit column, else a trailing '(unit)' in the first of texts that has one (item name, then dim labels).
    None when unknown. kosis_ai.schemas.Cell.unit_or_suffix passes its own fields."""
    if unit:
        return unit
    for text in texts:
        if m := SUFFIX_UNIT.search(text):
            return m.group(1)
    return None
