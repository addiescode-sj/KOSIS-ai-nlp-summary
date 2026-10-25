"""compact-v2 model output: one strict type per outcome (docs/plans/20261024-output-contract-v2/PLAN.md).

answered = {outcome, message}, refused = {outcome, message, code}, clarify = {outcome, message, options}.
When the outcome is valid, a field that belongs to another outcome and holds only an empty value (null, [], {}, "")
is dropped before validation and its name kept in `absent_normalized`; a filled one is still an error.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, RootModel, ValidationError, field_validator, model_validator

from kosis_chat.llm.answer_schema import LLMAnswer, RefusalReason
from kosis_chat.llm.structured import extract_json

VERSION = "compact-v2"
OUTCOMES = ("answered", "refused", "clarify")
BRANCH_FIELDS = {"answered": ("outcome", "message"), "refused": ("outcome", "message", "code"),
                 "clarify": ("outcome", "message", "options")}
# Names the model may leave empty in the wrong branch; anything else is an unknown field and always rejected.
KNOWN_FIELDS = ("outcome", "message", "code", "options", "reason")
REFUSAL_CODES = ("no_period", "no_item", "class_mismatch", "unit_mismatch", "not_aggregatable", "causal",
                 "missing_value", "out_of_scope")
SKELETONS = {
    "answered": '{"outcome":"answered","message":"<답변 문장>"}',
    "refused": '{"outcome":"refused","message":"<거절 이유 문장>","code":"<거절 코드 하나>"}',
    "clarify": '{"outcome":"clarify","message":"<되묻는 질문>","options":["<선택지 1>","<선택지 2>"]}',
}
# Field-placement examples shown in the system prompt; tests check each one validates.
EXAMPLES = (
    '{"outcome":"answered","message":"표에서 확인한 값은 1,234명입니다."}',
    '{"outcome":"refused","message":"해당 칸은 기호로 표시되어 값이 없습니다.","code":"missing_value"}',
    '{"outcome":"clarify","message":"어떤 항목을 확인할까요?","options":["항목 A","항목 B"]}',
)
_STRICT = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
Message = Annotated[str, Field(min_length=1, description="답변·거절 이유·되묻는 질문. 출처 없이 일반 문장으로.")]


class Answered(BaseModel):
    model_config = _STRICT
    outcome: Literal["answered"]
    message: Message


class Refused(BaseModel):
    model_config = _STRICT
    outcome: Literal["refused"]
    message: Message
    code: RefusalReason = Field(description="거절 코드 하나. 설명 문장은 message에.")


class Clarify(BaseModel):
    model_config = _STRICT
    outcome: Literal["clarify"]
    message: Message
    options: list[str] = Field(min_length=2, description="서로 다른 선택지 2개 이상.",
                               json_schema_extra={"uniqueItems": True, "items": {"type": "string", "minLength": 1}})

    @field_validator("options")
    @classmethod
    def distinct_nonblank(cls, options: list[str]) -> list[str]:
        cleaned = [o.strip() for o in options]
        if any(not o for o in cleaned):
            raise ValueError("options cannot contain blank choices")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("options must be distinct")
        return cleaned


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, dict, str)) and not (value.strip() if isinstance(value, str) else value))


def strip_absent(data: Any) -> tuple[Any, tuple[str, ...]]:
    """Drop empty fields that belong to another outcome; returns (data, dropped names). Unchanged when outcome is invalid."""
    if not isinstance(data, dict) or data.get("outcome") not in OUTCOMES:
        return data, ()
    allowed = BRANCH_FIELDS[data["outcome"]]
    dropped = tuple(k for k, v in data.items() if k in KNOWN_FIELDS and k not in allowed and _is_empty(v))
    return {k: v for k, v in data.items() if k not in dropped}, dropped


class CompactAnswerV2(RootModel[Annotated[Union[Answered, Refused, Clarify], Field(discriminator="outcome")]]):
    _absent_normalized: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="wrap")
    @classmethod
    def _normalize_absent(cls, data: Any, handler):
        if isinstance(data, cls):
            return handler(data)
        cleaned, dropped = strip_absent(data)
        value = handler(cleaned)
        value._absent_normalized = dropped
        return value

    @property
    def outcome(self) -> str:
        return self.root.outcome

    @property
    def absent_normalized(self) -> tuple[str, ...]:
        return self._absent_normalized

    def to_legacy(self) -> LLMAnswer:
        a = self.root
        if isinstance(a, Answered):
            return LLMAnswer(outcome="answered", answer=a.message)
        if isinstance(a, Refused):
            return LLMAnswer(outcome="refused", refusal_reason=a.code, refusal_message=a.message)
        return LLMAnswer(outcome="clarify", clarify_question=a.message, clarify_options=list(a.options))

    # Hooks read by structured.ask_structured (other answer models do not define them).
    @staticmethod
    def outcome_of(text: str) -> str | None:
        """The first attempt's outcome, kept fixed during the format retry; None when it cannot be read."""
        try:
            outcome = extract_json(text).get("outcome")
        except (ValueError, RecursionError):
            return None
        return outcome if outcome in OUTCOMES else None

    @staticmethod
    def repair_hint(text: str, exc: Exception) -> str:
        return repair_hint(text, exc)


def _safe_name(part: Any) -> str:
    return str(part) if re.fullmatch(r"[A-Za-z_]\w*", str(part)) else "<key>"


def repair_hint(text: str, exc: Exception) -> str:
    """Korean retry guidance: the skeleton of the outcome the model chose, plus one line per error.
    Never echoes the model's text; only the outcome code and field names (sanitised) appear."""
    outcome = CompactAnswerV2.outcome_of(text)
    codes = ", ".join(REFUSAL_CODES)
    if outcome is None:
        lines = ["JSON 객체 하나만 출력합니다. outcome은 answered, refused, clarify 중 하나입니다.",
                 "out_of_scope 같은 거절 코드는 outcome이 아니라 refused의 code에 씁니다.",
                 "가능한 형태는 아래 세 가지뿐입니다.", *SKELETONS.values()]
        return "\n".join(lines)
    lines = [f"이전 응답의 outcome은 {outcome}입니다. outcome을 바꾸지 말고 아래 형태로 다시 쓰세요.", SKELETONS[outcome]]
    errors = exc.errors(include_url=False, include_input=False, include_context=False) if isinstance(exc, ValidationError) else []
    for error in errors[:8]:
        loc = [p for p in error.get("loc", ()) if p not in OUTCOMES]  # union errors start with the branch tag
        field = _safe_name(loc[0]) if loc else ""
        kind = error.get("type", "")
        if kind == "extra_forbidden":
            lines.append(f"{outcome}에서는 {field} 필드를 쓰지 않습니다. 이 필드를 지우세요.")
        elif field == "message":
            lines.append("message는 비어 있지 않은 문장입니다(답변, 거절 이유 또는 되묻는 질문).")
        elif field == "code":
            lines.append(f"code는 필수이며 설명 문장이 아니라 다음 코드 중 하나입니다: {codes}. 설명은 message에 씁니다.")
        elif field == "options":
            lines.append("options는 서로 다른 선택지 문자열 2개 이상의 배열입니다. null은 쓸 수 없습니다.")
        else:
            lines.append(f"{field or '$'}: 위 형태에 맞게 고치세요.")
    if not errors:
        lines.append("앞뒤 설명 없이 큰따옴표를 쓴 JSON 객체 하나만 출력하세요.")
    return "\n".join(lines)[:1600]
