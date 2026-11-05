"""Period strings as people read them (T1-28 S2b). Periods follow the Parquet files: 'YYYY' a year, 'YYYYMM' a month.

Moved from kosis_chat/answer/render.py, which re-exports them: compute/build.py and compute/plan_engine.py wrote the
notes of their computed facts with them and so imported the answer layer (a layer turned the wrong way).
judge/answerability._period_text was the same body as period_text and is now an alias of it.
"""

from __future__ import annotations


def period_text(period: str) -> str:
    """'2024' -> '2024년'; '202406' -> '2024년 6월'."""
    return f"{period[:4]}년 {int(period[4:])}월" if len(period) == 6 and period.isdigit() else f"{period}년"


def span_text(first: str, last: str) -> str:
    """'2019년~2023년'; a half or a quarter of a year by name ('2024년 상반기', '2025년 1분기'), else '2024년 1월~6월'."""
    if first == last:
        return period_text(first)
    if len(first) == 6 and len(last) == 6 and first[:4] == last[:4]:
        a, b = int(first[4:]), int(last[4:])
        if (a, b) == (1, 6):
            return f"{first[:4]}년 상반기"
        if (a, b) == (7, 12):
            return f"{first[:4]}년 하반기"
        if b - a == 2 and a % 3 == 1:
            return f"{first[:4]}년 {a // 3 + 1}분기"
        return f"{first[:4]}년 {a}월~{b}월"
    return f"{period_text(first)}~{period_text(last)}"
