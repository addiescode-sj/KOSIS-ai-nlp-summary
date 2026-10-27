"""Refusal and ask-back sentences, written by code so the reason never depends on the model's wording."""

from __future__ import annotations

import re

REFUSAL_MESSAGES = {
    "no_period": "이 통계표에는 {asked} 자료가 없어 답할 수 없습니다.",
    "no_item": "이 통계표에는 질문하신 항목이 없어 답할 수 없습니다.",
    "class_mismatch": "이 통계표에는 {what} 값이 없습니다(그 시점에 해당 분류가 없거나 집계하지 않았습니다).",
    "unit_mismatch": "{what}은(는) 단위가 서로 달라({units}) 더하거나 나눌 수 없습니다.",
    "not_aggregatable": "{what}은(는) 비율·평균이거나 합계가 없는 구분이라 더한 값은 의미가 없습니다.",
    "causal": "통계표는 수치만 담고 있어 원인이나 효과는 판단할 수 없습니다.",
    "missing_value": "해당 칸의 값이 '{raw}'({meaning})로 표시되어 있어 수치로 답할 수 없습니다.",
    "out_of_scope": "제공된 통계(인구·가구 분야) 범위 밖의 질문이라 답할 수 없습니다.",
}

CLARIFY_LABEL = "'{label}'은(는) 여러 지역에 있습니다. 어느 지역을 말씀하시나요?"
CLARIFY_ITEM = "이 표에는 여러 항목이 있습니다. 어떤 항목을 말씀하시나요?"


class _Fields(dict):
    def __missing__(self, key: str) -> str:
        return "질문하신 조건"


def refusal_message(reason: str, **fields: str) -> str:
    return REFUSAL_MESSAGES[reason].format_map(_Fields(fields))


_DEFAULT_KEYS = {"period": "시점"}
_LATEST = re.compile(r"latest \((.+)\)")
_RECENT = re.compile(r"recent (\d+) points \((.+)\)")
_PREVIOUS = re.compile(r"latest \((.+)\) vs previous \((.+)\)")


def _default_text(value: str) -> str:
    if match := _PREVIOUS.fullmatch(value):
        return f"가장 최근 시점 {match.group(1)}과 그 직전 시점 {match.group(2)}"
    if match := _LATEST.fullmatch(value):
        return f"가장 최근 시점 {match.group(1)}"
    if match := _RECENT.fullmatch(value):
        return f"최근 {match.group(1)}개 시점 {match.group(2)}"
    return value


def default_note(applied: dict[str, str]) -> str:
    """Line added under the question so the model answers with, and states, the conditions the rules assumed."""
    if not applied:
        return ""
    conditions = ", ".join(f"{_DEFAULT_KEYS.get(key, key)}={_default_text(value)}" for key, value in applied.items())
    return f"\n(질문에 없는 조건은 다음으로 정했다: {conditions}. 이 조건으로 답하고 답변 첫 문장에 이 조건을 밝힌다.)"
