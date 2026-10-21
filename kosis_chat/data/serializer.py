"""Turn a TableInput into prompt text: a meta header plus a markdown table with a unit on every row."""

from __future__ import annotations

from kosis_chat.data.schema import TableInput


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


def to_markdown(table: TableInput) -> str:
    """Values are shown exactly as received (value_raw) so the model quotes the source, not our parsing."""
    dims = table.dim_names
    header = ["항목", *dims, "시점", "값", "단위"]
    lines = [
        meta_header(table),
        "",
        "| " + " | ".join(_escape(h) for h in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for c in table.cells:
        row = [c.item, *(c.dims.get(d, "") for d in dims), c.period_raw, c.value_raw.strip(), c.unit]
        lines.append("| " + " | ".join(_escape(v) for v in row) + " |")
    return "\n".join(lines)
