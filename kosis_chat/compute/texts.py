"""Refusal and ask-back sentences the calculation writes itself (T1-28 S2b, D-c (a)).

These are the ten sentences that compute/resolve.py and compute/semantics.py wrote as f-strings in place; the other
calculation refusals already use judge/messages.refusal_message and stay so. Each function returns the same
characters the f-string at its call site wrote (the arguments are formatted exactly as they were there).
"""

from __future__ import annotations

from collections.abc import Sequence

# ---------------------------------------------------------------- resolve.py (NeedsClarify)


def event_unknown(name: str) -> str:
    """An event missing from the calendar that the model could not date either (resolve.event_start)."""
    return f"'{name}'이(가) 언제인지 알 수 없습니다. 연도를 알려 주세요."


def path_ambiguous(path: Sequence[str]) -> str:
    """A label path that names several codes (resolve._by_path)."""
    return f"'{' > '.join(path)}'이(가) 여러 곳입니다. 어느 것인지 골라 주세요."


def pick_category(axis: str) -> str:
    """A category the plan left out that has no total (resolve._default)."""
    return f"{axis}을(를) 골라 주세요."


# ---------------------------------------------------------------- resolve.py (CalcRefused)


def span_not_supported(year: object, part: object) -> str:
    """A part of a year asked of a quarterly or half-yearly table (resolve.resolve_periods)."""
    return f"{year}년 {part}: 분기·반기 표의 시점 해석은 아직 지원하지 않습니다."


def span_on_yearly(year: object, part: object) -> str:
    """A part of a year asked of a yearly table (resolve.resolve_periods)."""
    return f"이 표는 연 단위라 {year}년 {part} 값이 없습니다."


def other_table(table: object, tbl_id: object) -> str:
    """A select that names another table (resolve.resolve_select)."""
    return f"{table}: 이 질문의 표({tbl_id})가 아닙니다(여러 표 결합은 아직 엔진만 지원)."


def star_outside_map(axis: str) -> str:
    """A '*' code outside a map block (resolve.resolve_select)."""
    return f"{axis}의 '*'는 map 안에서만 쓸 수 있습니다."


# ---------------------------------------------------------------- semantics.py (CalcRefused)


def pp_diff_units(what: str, a: str | None, b: str | None) -> str:
    """A %p difference of values that are not both % (semantics.check_same_unit)."""
    return f"{what}: %p 차이는 두 값이 모두 %일 때만 구할 수 있습니다(단위 {a} / {b})."


def share_outside_total(wrong: Sequence[str]) -> str:
    """A share whose denominator does not hold the numerator (semantics.check_share_scope)."""
    return f"비중의 분모가 분자를 포함하지 않습니다({', '.join(wrong)})."


def share_out_of_bounds(what: str, value: float) -> str:
    """A share outside 0..100% (semantics.check_share_bound)."""
    return f"{what} 비중이 {value:.1f}%로 0~100% 밖이라 분모가 분자의 전체가 아닙니다."
