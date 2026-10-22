"""Versioned system prompts and the user prompt template; eval results record the version used."""

from __future__ import annotations

# Shared numeric rule wording used across all docs and prompt versions.
NUMBER_RULE = "원본 셀 또는 검증 가능한 계산식과 입력 셀로 추적되는 수치만 사용한다."

SYSTEM_PROMPTS: dict[str, str] = {
    "baseline-v1": f"""당신은 국가통계(KOSIS) 통계표를 근거로 질문에 답하는 도우미입니다.
아래 규칙을 반드시 지키세요.

1. {NUMBER_RULE}
2. 계산한 수치(증감률, 차이, 구성비 등)는 계산식과 입력값을 함께 씁니다.
   예: 2023년 대비 2024년 증가율 2.1% ((51,300 − 50,240) ÷ 50,240 × 100)
3. 표에 없는 값을 추정하거나 외삽하지 않습니다. 입력이 표에 없는 계산도 하지 않습니다.
4. 원인, 정책 효과, 정치적 판단은 말하지 않습니다.
5. 단위와 시점, 분류(성별·연령 등)를 수치와 함께 정확히 씁니다.
6. 답변 끝에 출처를 씁니다: "출처: KOSIS / <조사명> / <통계표명>"
7. 표만으로 답할 수 없으면 통계 수치를 말하지 말고, 답할 수 없는 이유와 표에서 확인 가능한 범위를 안내합니다.
8. 질문의 조건이 모호해 답이 달라질 수 있으면, 어떤 조건인지 되묻습니다.""",
}

USER_TEMPLATE = """[통계표]
{table_text}

[질문]
{question}"""

# Optional answer-format instructions (e.g. a form.txt given with --form), placed after the question.
FORM_TEMPLATE = """

[답변 형식]
{form}"""


def get_system_prompt(version: str) -> str:
    try:
        return SYSTEM_PROMPTS[version]
    except KeyError:
        raise ValueError(f"unknown prompt version {version!r}; known: {sorted(SYSTEM_PROMPTS)}") from None


def build_user_prompt(table_text: str, question: str, form: str | None = None) -> str:
    prompt = USER_TEMPLATE.format(table_text=table_text, question=question.strip())
    if form and form.strip():
        prompt += FORM_TEMPLATE.format(form=form.strip())
    return prompt
