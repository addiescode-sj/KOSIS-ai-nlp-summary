"""Turn a TableInput into prompt text: a meta header plus a markdown table (long: one value per row, wide: periods as columns)."""

from __future__ import annotations

import re

from kosis_chat.data.schema import TableInput

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


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def meta_header(table: TableInput) -> str:
    m = table.meta
    period = table.period_range
    lines = [
        f"조사명: {m.survey_nm}",
        f"통계표: {m.tbl_nm} ({m.tbl_id})",
        f"작성기관: {m.org_nm}",
        f"수록 시점: {period[0]} ~ {period[1]}" if period else "수록 시점: 없음",
        f"출처: KOSIS {m.source_url} (받은 날짜 {m.fetched_at})",
    ]
    if m.note:
        lines.append(f"주석: {m.note}")
    return "\n".join(lines)


STYLES = ("long", "wide")
# Wide tables split into blocks of at most this many period columns, so the model reads columns reliably.
MAX_PERIOD_COLUMNS = 20


def serialize(table: TableInput, style: str = "long") -> str:
    if style == "long":
        return to_markdown(table)
    if style == "wide":
        return to_wide_markdown(table)
    raise ValueError(f"unknown serialization {style!r}; expected one of {STYLES}")


def to_markdown(table: TableInput) -> str:
    """Values are shown as received (value_raw, minus trailing decimal zeros) so the model quotes the source, not our parsing."""
    dims = table.dim_names
    header = ["항목", *dims, "시점", "값", "단위"]
    lines = [
        meta_header(table),
        "",
        "| " + " | ".join(_escape(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for c in table.cells:
        row = [c.item, *(c.dims.get(d, "") for d in dims), c.period_raw, display_value(c.value_raw), c.unit]
        lines.append("| " + " | ".join(_escape(v) for v in row) + " |")
    return "\n".join(lines)


def to_wide_markdown(table: TableInput, max_period_columns: int = MAX_PERIOD_COLUMNS) -> str:
    """One row per (item, categories) with periods as columns; a single table unit is written once above the table."""
    dims = table.dim_names
    periods = sorted({c.period_raw for c in table.cells})
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    units: dict[tuple[str, ...], list[str]] = {}
    for c in table.cells:
        key = (c.item, *(c.dims.get(d, "") for d in dims))
        rows.setdefault(key, {})[c.period_raw] = display_value(c.value_raw)
        if c.unit not in units.setdefault(key, []):
            units[key].append(c.unit)
    all_units = {u for us in units.values() for u in us}
    single_unit = len(all_units) == 1
    lines = [meta_header(table), "", f"단위: {next(iter(all_units))}" if single_unit else "단위: 행마다 다름(단위 열)"]
    for start in range(0, len(periods), max_period_columns):
        block = periods[start : start + max_period_columns]
        if len(periods) > max_period_columns:
            lines += ["", f"시점 {block[0]} ~ {block[-1]}"]
        header = ["항목", *dims, *([] if single_unit else ["단위"]), *block]
        lines.append("| " + " | ".join(_escape(h) for h in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for key, values in rows.items():
            row = [*key, *([] if single_unit else ["/".join(units[key])]), *(values.get(p, "") for p in block)]
            lines.append("| " + " | ".join(_escape(v) for v in row) + " |")
    return "\n".join(lines)
