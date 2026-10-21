"""Baseline cleaning: derive value / value_flag from value_raw and keep every original field unchanged."""

from __future__ import annotations

import re
from dataclasses import replace

from kosis_chat.data.schema import Cell, TableInput

MISSING_MARKS = {"", "-", "…", "...", "..", "－"}
SUPPRESSED_MARKS = {"x", "X", "Ｘ"}
# Trailing provisional / estimate markers such as "123 p" or "123(p)".
_PROVISIONAL = re.compile(r"^(?P<num>.+?)\s*\(?\s*(p|e)\s*\)?$", re.IGNORECASE)


def parse_value(value_raw: str) -> tuple[float | None, str]:
    """Return (value, value_flag) for one raw value string."""
    text = (value_raw or "").strip()
    if text in MISSING_MARKS:
        return None, "missing"
    if text in SUPPRESSED_MARKS:
        return None, "suppressed"
    flag = "ok"
    match = _PROVISIONAL.match(text)
    if match and not text[-1].isdigit():
        text, flag = match.group("num"), "provisional"
    try:
        return float(text.replace(",", "")), flag
    except ValueError:
        return None, "unparsed"


def clean_cell(cell: Cell) -> Cell:
    value, flag = parse_value(cell.value_raw)
    return replace(cell, value=value, value_flag=flag)


def clean(table: TableInput) -> TableInput:
    """Fill derived fields for every cell. B1 adds the PREPROCESSING.md rules here."""
    return replace(table, cells=tuple(clean_cell(c) for c in table.cells))
