"""Small, strict model-facing answer contract; shared result schemas stay unchanged."""

from __future__ import annotations

import sys

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kosis_chat.config import PROJECT_ROOT

try:
    from kosis_ai.schemas import Outcome, RefusalReason
except ModuleNotFoundError as exc:
    if exc.name != "kosis_ai":
        raise
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from kosis_ai.schemas import Outcome, RefusalReason

PLAIN_TEXT = "일반 문장으로만. 마크다운 머리말(#), 굵게(**), 표, LaTeX 수식, 이모지, 원인 추측, 맺음 인사 금지."


class LLMAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    outcome: Outcome
    answer: str = Field(default="", description=PLAIN_TEXT + " 답할 때만 사용하며 출처 줄은 쓰지 않는다.")
    refusal_reason: RefusalReason | None = Field(default=None, description=(
        "no_period=요청 시점 없음; no_item=항목 없음; class_mismatch=분류 없음; "
        "unit_mismatch=단위 다른 값 결합; not_aggregatable=비율·평균 합산 또는 저량의 시점 간 합산; "
        "causal=원인 질문; missing_value=셀이 '-'/'X' 등 기호; out_of_scope=제공 표 밖."
    ))
    refusal_message: str | None = Field(default=None, description=PLAIN_TEXT + " 거절 이유.")
    available: str | None = Field(default=None, description="거절할 때 대신 확인 가능한 범위.")
    clarify_question: str | None = Field(default=None, description=PLAIN_TEXT + " 필요한 조건을 묻는 질문.")
    clarify_options: list[str] = Field(default_factory=list, description="되묻기 선택지 2개 이상, 서로 다르게.")
    default_applied: dict[str, str] = Field(default_factory=dict, description="답할 때 적용한 기본 조건만 기록.")

    @model_validator(mode="after")
    def coherent_outcome(self) -> LLMAnswer:
        if self.available is not None and not self.available:
            raise ValueError("available cannot be blank")
        if any(not key.strip() or not value for key, value in self.default_applied.items()):
            raise ValueError("default_applied keys and values cannot be blank")
        if any(not option for option in self.clarify_options):
            raise ValueError("clarify_options cannot contain blank choices")
        if len(set(self.clarify_options)) != len(self.clarify_options):
            raise ValueError("clarify_options must be distinct")
        refusal_fields = self.refusal_reason is not None or self.refusal_message is not None or self.available is not None
        clarify_fields = self.clarify_question is not None or bool(self.clarify_options)
        if self.outcome == "answered":
            if not self.answer or refusal_fields or clarify_fields:
                raise ValueError("answered requires answer only, with optional default_applied")
        elif self.outcome == "refused":
            if self.refusal_reason is None or not self.refusal_message:
                raise ValueError("refused requires refusal_reason and nonblank refusal_message")
            if self.answer or clarify_fields or self.default_applied:
                raise ValueError("refused cannot contain answer, clarification or defaults")
        else:
            if not self.clarify_question or len(self.clarify_options) < 2:
                raise ValueError("clarify requires a question and at least two distinct choices")
            if self.answer or refusal_fields or self.default_applied:
                raise ValueError("clarify cannot contain answer, refusal or defaults")
        return self


class CompactAnswer(BaseModel):
    """Opt-in four-field model output; no model-authored default selection metadata."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    outcome: Outcome
    message: str = Field(min_length=1, description="답변·거절 이유·되묻기 문장. 출처 없이 일반 문장으로 작성.")
    reason: RefusalReason | None = Field(default=None, description="refused일 때만 거절 코드 하나.")
    options: list[str] = Field(default_factory=list, description="clarify일 때만 서로 다른 선택지 2개 이상. 그 외 [] 또는 생략.")

    @model_validator(mode="after")
    def coherent_compact_outcome(self) -> CompactAnswer:
        if not self.message:
            raise ValueError("message must be nonblank")
        if any(not option for option in self.options) or len(set(self.options)) != len(self.options):
            raise ValueError("options must contain distinct nonblank strings")
        if self.outcome == "answered" and (self.reason is not None or self.options):
            raise ValueError("answered requires reason=null and options=[] or omitted")
        if self.outcome == "refused" and (self.reason is None or self.options):
            raise ValueError("refused requires reason code and options=[] or omitted")
        if self.outcome == "clarify" and (self.reason is not None or len(self.options) < 2):
            raise ValueError("clarify requires reason=null and at least two options")
        return self

    def to_legacy(self) -> LLMAnswer:
        if self.outcome == "answered":
            return LLMAnswer(outcome="answered", answer=self.message)
        if self.outcome == "refused":
            return LLMAnswer(outcome="refused", refusal_reason=self.reason, refusal_message=self.message)
        return LLMAnswer(outcome="clarify", clarify_question=self.message, clarify_options=self.options)


def answer_model(version: str) -> type[BaseModel]:
    if version == "compact-v2":
        from kosis_chat.llm.compact_v2 import CompactAnswerV2  # imported here: compact_v2 builds on LLMAnswer
        return CompactAnswerV2
    return CompactAnswer if version == "compact-v1" else LLMAnswer
