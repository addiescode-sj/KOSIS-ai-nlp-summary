"""Numbers as code writes them (T1-28 S2b): a fact as '123,456가구', a raw cell value as shown, what a derived fact is.

Moved with their bodies unchanged:
- SYMBOL_MEANING, format_number, format_fact, is_change, direction, derived_kind: were answer/render.py (re-exported
  there); compose (template, verify, caveats), chart and render all use them.
- display_value: was data/serializer.py (re-exported there); compute/engine.py read the decimals of a cell from it and
  so imported the B0 serializer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kosis_ai.schemas import Fact

SYMBOL_MEANING = {"-": "자료 없음", "x": "비공개", "X": "비공개", "…": "미상", "...": "미상", "*": "통계 미작성"}


def _number(value: float, digits: int | None) -> str:
    if digits is None:  # a cell value: as many decimals as it has (stored values never carry float noise)
        digits = 0 if float(value).is_integer() else min(len(repr(float(value)).split(".")[1]), 4)
    return f"{value:,.{digits}f}"


def format_number(value: float, unit: str | None, digits: int | None = None) -> str:
    """'123,456가구' for a plain number (charts use it for values that are not facts of the answer)."""
    return f"{_number(value, digits)}{unit or ''}"


def format_fact(fact: Fact) -> str:
    """'123,456가구'; a change without its sign (the sentence says 늘었다/줄었다); a symbol as "'-'(자료 없음)"."""
    if fact.value is None:
        raw = fact.raw or "-"
        return f"'{raw}'({SYMBOL_MEANING.get(raw, '수치 아님')})"
    value = abs(fact.value) if is_change(fact) else fact.value
    return format_number(value, fact.unit, fact.round)


_CHANGE = re.compile(r"\s*(?:f\d+\s*-\s*f\d+|\(f\d+\s*/\s*f\d+\s*-\s*1\)\s*\*\s*100)\s*")


def is_change(fact: Fact) -> bool:
    """A difference ('f4 - f1') or a growth rate ('(f4 / f1 - 1) * 100'): shown without its sign."""
    return bool(fact.derivation) and _CHANGE.fullmatch(fact.derivation) is not None


def direction(fact: Fact) -> str:
    """'증가' / '감소' / '같음' for a change fact (what the sentence must say, since the number has no sign)."""
    if fact.value is None or fact.value == 0:
        return "같음"
    return "증가" if fact.value > 0 else "감소"


_KIND_WORDS = {"growth": "증감률", "difference": "차이", "share": "비중", "ratio": "배수", "avg": "평균", "sum": "합계"}
_SUM = re.compile(r"^f\d+(?:\s*\+\s*f\d+)+$")
_AVG = re.compile(r"^\(f\d+(?:\s*\+\s*f\d+)+\)\s*/\s*\d+$")


def derived_kind(fact: Fact) -> str:
    """What a computed fact is, read from its derivation: 차이, 증감률, 비중, 배수, 평균 or 합계."""
    expr = fact.derivation or ""
    if _SUM.match(expr):
        return _KIND_WORDS["sum"]
    if _AVG.match(expr):
        return _KIND_WORDS["avg"]
    if "* 100" in expr and "- 1)" in expr:
        return _KIND_WORDS["growth"]
    if "* 100" in expr:
        return _KIND_WORDS["share"]
    if "/" in expr:
        return _KIND_WORDS["ratio"]
    return _KIND_WORDS["difference"]


# "65975.0000" -> "65975", "12.5000" -> "12.5": ZIP-built tables store numbers with four decimals.
_TRAILING_ZEROS = re.compile(r"^(-?\d[\d,]*)\.(\d*?)0+$")


def display_value(value_raw: str) -> str:
    """The raw value as shown to the model: trailing decimal zeros dropped, symbols (-, X, *) and p/e marks kept."""
    text = value_raw.strip()
    match = _TRAILING_ZEROS.match(text)
    if not match:
        return text
    whole, decimals = match.groups()
    return f"{whole}.{decimals}" if decimals else whole
