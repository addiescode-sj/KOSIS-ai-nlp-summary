"""Cut a table down to the part a question needs, so a big Parquet table fits in the model input.

Modes (eval --scope, cli --scope):
  oracle  the golden item's own `scope` (upper bound: retrieval is perfect)
  auto    derived from the question text only (years, category labels, item names)
  full    the whole table
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from kosis_chat.data import serializer
from kosis_chat.data.schema import Cell, TableInput

MODES = ("oracle", "auto", "full")
# auto mode drops the oldest periods until the table text fits; DASH-002 answered at 31k characters and failed at 65k.
AUTO_MAX_CHARS = 30_000
# "최근 추이" with no number of years.
RECENT_PERIODS = 5

_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_YEAR_RANGE = re.compile(r"((?:19|20)\d{2})\s*년?\s*(?:~|-|부터)\s*((?:19|20)\d{2})")
_YEAR_SINCE = re.compile(r"((?:19|20)\d{2})\s*년?\s*이후")
_RECENT_N = re.compile(r"최근\s*(\d+)\s*년")
_UNIT_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_REGION_SUFFIX = re.compile(r"(특별자치시|특별자치도|특별시|광역시|도)$")
# Words people use for a label; matched without spaces.
LABEL_ALIASES = {
    "여자": ("여성",),
    "남자": ("남성",),
    "충청북도": ("충북",),
    "충청남도": ("충남",),
    "전라북도": ("전북",),
    "전북특별자치도": ("전북", "전라북도"),
    "전라남도": ("전남",),
    "경상북도": ("경북",),
    "경상남도": ("경남",),
    "세종특별자치시": ("세종시",),
}
# Items kept next to a matched item when the question asks for a share (the denominator).
_SHARE_WORDS = ("비중", "비율", "차지")


def _is_total(name: str) -> bool:
    core = _UNIT_PAREN.sub("", name).strip()
    return core.endswith("계") or core.startswith("총") or "전체" in core


@dataclass(frozen=True)
class Scope:
    """Which cells to keep. None / missing means "all" for that part."""

    items: tuple[str, ...] | None = None  # item codes (Cell.item_code)
    dims: dict[str, tuple[str, ...]] = field(default_factory=dict)  # category name -> codes
    periods: tuple[str, str] | None = None  # inclusive [start, end]; compared on the leading digits

    @classmethod
    def from_dict(cls, data: dict | None) -> Scope:
        data = data or {}
        periods = data.get("periods")
        return cls(
            items=tuple(data["items"]) if data.get("items") else None,
            dims={name: tuple(codes) for name, codes in (data.get("dims") or {}).items()},
            periods=(str(periods[0]), str(periods[1])) if periods else None,
        )

    def to_dict(self) -> dict:
        out: dict = {}
        if self.items:
            out["items"] = list(self.items)
        if self.dims:
            out["dims"] = {name: list(codes) for name, codes in self.dims.items()}
        if self.periods:
            out["periods"] = list(self.periods)
        return out


def _in_range(period: str, bounds: tuple[str, str]) -> bool:
    start, end = bounds
    return start <= period[: len(start)] and period[: len(end)] <= end


def keep(cell: Cell, scope: Scope) -> bool:
    if scope.items is not None and cell.item_code not in scope.items:
        return False
    for name, codes in scope.dims.items():
        if cell.dim_codes.get(name) not in codes:
            return False
    return scope.periods is None or _in_range(cell.period_raw, scope.periods)


def apply(table: TableInput, scope: Scope) -> TableInput:
    """Cells inside the scope; the meta (and its file hash) stays that of the whole table."""
    return replace(table, cells=tuple(c for c in table.cells if keep(c, scope)))


# ---------------------------------------------------------------- category levels


def dim_codes(table: TableInput) -> dict[str, dict[str, str]]:
    """Category name -> {code: label} in first-seen order."""
    out: dict[str, dict[str, str]] = {}
    for c in table.cells:
        for name, code in c.dim_codes.items():
            out.setdefault(name, {}).setdefault(code, c.dims.get(name, ""))
    return out


def same_level(codes: list[str], code: str) -> list[str]:
    """Codes at the level of `code`: KOSIS numeric codes nest by length (11 > 11010 > 11010530),
    so siblings share the parent prefix; other codes (S1, A10) are one flat level."""
    if not code.isdigit():
        return list(codes)
    lengths = sorted({len(c) for c in codes if c.isdigit()})
    shorter = [n for n in lengths if n < len(code)]
    parent = code[: shorter[-1]] if shorter else ""
    return [c for c in codes if c.isdigit() and len(c) == len(code) and c.startswith(parent)]


def top_level(codes: list[str]) -> list[str]:
    """The shortest numeric codes (e.g. 전국 + 시도), or every code when they are not numeric."""
    numeric = [c for c in codes if c.isdigit()]
    if not numeric:
        return list(codes)
    shortest = min(len(c) for c in numeric)
    return [c for c in numeric if len(c) == shortest]


# ---------------------------------------------------------------- auto


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _label_forms(label: str, min_len: int = 2) -> set[str]:
    core = _UNIT_PAREN.sub("", label).strip()
    forms = {_squash(label), _squash(core), *(_squash(a) for a in LABEL_ALIASES.get(core, ()))}
    short = _REGION_SUFFIX.sub("", core)
    if short != core:
        forms.add(_squash(short))
    return {f for f in forms if len(f) >= min_len}


def _matches(label: str, question: str, min_len: int = 2) -> bool:
    return any(form in question for form in _label_forms(label, min_len))


def question_periods(question: str, periods: list[str]) -> tuple[str, str] | None:
    """Year range the question asks about, or None when it names none the table has.
    "1985년생" is a cohort label, not a period."""
    if not periods:
        return None
    first, last = periods[0][:4], periods[-1][:4]
    text = re.sub(r"(?:19|20)\d{2}\s*년생", "", question)
    recent = _RECENT_N.search(text)
    if recent:
        years = sorted({p[:4] for p in periods})
        return years[-int(recent.group(1)) :][0], last
    if "최근" in text and not _YEAR.search(text):
        years = sorted({p[:4] for p in periods})
        return years[-RECENT_PERIODS:][0], last
    found = [(a, b) for a, b in _YEAR_RANGE.findall(text)]
    found += [(y, last) for y in _YEAR_SINCE.findall(text)]
    years = _YEAR.findall(text)
    if not found and not years:
        return None
    lows = [a for a, _ in found] + years
    highs = [b for _, b in found] + years
    bounds = (min(lows), max(highs))
    if not any(_in_range(p, bounds) for p in periods):
        return None  # asked for years the table lacks: show every period so the model can say so
    return max(bounds[0], first), min(bounds[1], last)


def auto_scope(table: TableInput, question: str) -> Scope:
    q = _squash(question)
    items: dict[str, str] = {}
    for c in table.cells:
        items.setdefault(c.item_code, c.item)
    matched_items = [code for code, name in items.items() if _matches(name, q)]
    if matched_items and any(w in q for w in _SHARE_WORDS):
        matched_items += [code for code, name in items.items() if code not in matched_items and _is_total(name)]
    dims = {}
    for name, codes in dim_codes(table).items():
        top = top_level(list(codes))
        # Lower-level names need 3+ characters: 2-character 읍면동 names hide in ordinary words ("더하면" has 하면).
        hit = [code for code, label in codes.items() if _matches(label, q, 2 if code in top else 3)]
        dims[name] = tuple(hit or top)
    periods = sorted({c.period_raw for c in table.cells})
    return Scope(
        items=tuple(matched_items) if matched_items and len(matched_items) < len(items) else None,
        dims=dims,
        periods=question_periods(question, periods),
    )


def fit(table: TableInput, scope: Scope, style: str, max_chars: int = AUTO_MAX_CHARS) -> tuple[TableInput, Scope, int]:
    """Apply the scope, then drop the oldest periods while the table text is longer than max_chars.
    Returns (table, final scope, number of periods dropped)."""
    cut = apply(table, scope)
    if not cut.cells and scope.periods:  # e.g. 세종 before 2012: show every period so the model can say so
        scope = replace(scope, periods=None)
        cut = apply(table, scope)
    periods = sorted({c.period_raw for c in cut.cells})
    dropped = 0
    while len(periods) > 1 and len(serializer.serialize(cut, style)) > max_chars:
        dropped += 1
        periods = periods[1:]
        scope = replace(scope, periods=(periods[0], scope.periods[1] if scope.periods else periods[-1]))
        cut = apply(table, scope)
    return cut, scope, dropped


def select(table: TableInput, mode: str, question: str, golden_scope: dict | None, style: str) -> tuple[TableInput, dict]:
    """The table part for one question and a record of what was chosen (saved with eval results)."""
    if mode == "full":
        return table, {"mode": mode, "scope": {}, "cells": len(table.cells)}
    if mode == "oracle":
        if golden_scope is None:
            raise ValueError("oracle scope needs a `scope` in the golden item")
        chosen = Scope.from_dict(golden_scope)
        cut, dropped = apply(table, chosen), 0
    elif mode == "auto":
        cut, chosen, dropped = fit(table, auto_scope(table, question), style)
    else:
        raise ValueError(f"unknown scope mode {mode!r}; expected one of {MODES}")
    return cut, {"mode": mode, "scope": chosen.to_dict(), "cells": len(cut.cells), "dropped_periods": dropped}
