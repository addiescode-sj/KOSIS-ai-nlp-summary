"""Baseline cleaning: derive value / value_flag and keep every original field unchanged.

Cells from Parquet carry the build's value_kind and number, which are used as they are (KIND_FLAGS);
other cells, and Parquet "other_text" values such as "560p", are parsed from value_raw (parse_value)."""

from __future__ import annotations

import re
from dataclasses import replace

from kosis_chat.data.schema import Cell, TableInput

MISSING_MARKS = {"", "-", "…", "...", "..", "－"}
SUPPRESSED_MARKS = {"x", "X", "Ｘ"}
# KOSIS caution marks; their meaning differs by table, so they are kept apart from missing / suppressed.
FLAGGED_MARKS = {"*", "**", "***"}
# Parquet value_kind -> value_flag. "other_text" and unknown kinds go through parse_value.
KIND_FLAGS = {
    "ok": "ok",
    "blank": "missing",
    "dash": "missing",
    "ellipsis": "missing",
    "x_secret": "suppressed",
    "asterisk": "flagged",
}
# Trailing provisional / estimate markers such as "123 p" or "123(p)".
_PROVISIONAL = re.compile(r"^(?P<num>.+?)\s*\(?\s*(p|e)\s*\)?$", re.IGNORECASE)


def parse_value(value_raw: str) -> tuple[float | None, str]:
    """Return (value, value_flag) for one raw value string."""
    text = (value_raw or "").strip()
    if text in MISSING_MARKS:
        return None, "missing"
    if text in SUPPRESSED_MARKS:
        return None, "suppressed"
    if text in FLAGGED_MARKS:
        return None, "flagged"
    flag = "ok"
    match = _PROVISIONAL.match(text)
    if match and not text[-1].isdigit():
        text, flag = match.group("num"), "provisional"
    try:
        return float(text.replace(",", "")), flag
    except ValueError:
        return None, "unparsed"


def clean_cell(cell: Cell) -> Cell:
    flag = KIND_FLAGS.get(cell.value_kind or "")
    if flag is None:
        value, flag = parse_value(cell.value_raw)
    else:
        value = cell.source_value if flag == "ok" else None
    return replace(cell, value=value, value_flag=flag)


def clean(table: TableInput) -> TableInput:
    """Fill derived fields for every cell. B1 adds the PREPROCESSING.md rules here."""
    return replace(table, cells=tuple(clean_cell(c) for c in table.cells))
