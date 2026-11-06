"""Refusal and ask-back sentences, written by code so the reason never depends on the model's wording.

T1-28 S3a: moved here from judge/messages.py (a re-export shim now)."""

from __future__ import annotations

from kosis_chat.shared import defaults

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

# not_aggregatable for a stock summed over years: the rate wording of REFUSAL_MESSAGES does not fit (T1-14)
STOCK_OVER_TIME = ("{what}은(는) 해마다 그 시점에 있는 수를 센 값이라 여러 해를 더하면 같은 대상이 겹쳐 세어집니다. "
                   "연도별 값은 따로 확인할 수 있습니다.")

CLARIFY_LABEL = "'{label}'은(는) 여러 지역에 있습니다. 어느 지역을 말씀하시나요?"
CLARIFY_ITEM = "이 표에는 여러 항목이 있습니다. 어떤 항목을 말씀하시나요?"
CLARIFY_COMPARE = "'{word} 편'인지는 무엇과 비교하느냐에 따라 달라집니다. 어떤 기준과 비교할까요?"
COMPARE_OPTIONS = ("같은 표의 다른 지역·분류와 비교", "성별 등 다른 집단과 비교", "과거 시점과 비교")


# The particle helpers moved to kosis_chat/shared/texts.py in T1-28 S2b (re-exported here by name).
from kosis_chat.shared.texts import _FORM, _PAIRS, _PLACEHOLDER_JOSA, _READ_LAST, josa, with_josa  # noqa: E402, F401


class _Fields(dict):
    def __missing__(self, key: str) -> str:
        return "질문하신 조건"


def refusal_message(reason: str, **fields: str) -> str:
    return with_josa(REFUSAL_MESSAGES[reason].format_map(_Fields(fields)))


# The default_applied strings are written and read in kosis_chat/shared/defaults.py (T1-28 S2b); old names kept.
_DEFAULT_KEYS = defaults.DEFAULT_KEYS
_LATEST = defaults.LATEST
_RECENT = defaults.RECENT
_PREVIOUS = defaults.PREVIOUS


def _default_text(value: str) -> str:
    parsed = defaults.parse_period(value)
    if parsed is None:
        return value
    kind, groups = parsed
    if kind == "previous":
        return f"가장 최근 시점 {groups[0]}과 그 직전 시점 {groups[1]}"
    if kind == "latest":
        return f"가장 최근 시점 {groups[0]}"
    return f"최근 {groups[0]}개 시점 {groups[1]}"


def default_note(applied: dict[str, str]) -> str:
    """Line added under the question so the model answers with, and states, the conditions the rules assumed."""
    if not applied:
        return ""
    conditions = ", ".join(f"{_DEFAULT_KEYS.get(key, key)}={_default_text(value)}" for key, value in applied.items())
    return f"\n(질문에 없는 조건은 다음으로 정했다: {conditions}. 이 조건으로 답하고 답변 첫 문장에 이 조건을 밝힌다.)"
