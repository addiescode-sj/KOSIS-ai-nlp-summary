"""compose-v1: the model writes the answer sentence around computed facts, citing them as {f1}, {f2} (T1-09).

The model sees each fact's value (to choose words such as 늘었다) but never types a number: code puts the numbers in
(kosis_chat/answer/render.py). Its reply {"answer": "...{f1}..."} is split into TextPart / FactPart.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from kosis_chat.answer.render import direction, fact_label, format_fact, is_change
from kosis_chat.compute import Computed

from kosis_ai.schemas import Fact, FactPart, TextPart  # isort: skip

VERSION = "compose-v1"
SYSTEM_PROMPT = """당신은 KOSIS 통계표의 계산 결과를 한국어 답변 문장으로 쓰는 도우미입니다.
수치는 코드가 이미 계산했습니다. 당신은 [사실 목록]의 사실을 {f번호}로 넣어 문장만 씁니다.

규칙
1. 숫자는 직접 쓰지 말고 반드시 {f번호}로 넣습니다. 연도(예: 2023년)와 이름 속 숫자(예: 1인 가구)만 그대로 씁니다.
2. [반드시 인용] 표시가 있는 사실은 모두 넣습니다. [참고] 사실은 도움이 될 때만 넣습니다.
3. {f번호} 자리에는 단위까지 들어갑니다. 뒤에 단위를 다시 쓰지 않습니다.
4. 차이와 증감률은 부호 없이 표시됩니다. 사실 목록의 (증가)/(감소)에 맞게 '늘었습니다/줄었습니다', '많습니다/적습니다'로 방향을 씁니다.
5. 표에 없는 원인, 전망, 평가는 쓰지 않습니다. 출처와 기본 조건 안내는 코드가 붙이므로 쓰지 않습니다.
6. 존댓말로 1~3문장. 질문에 바로 답하는 문장으로 시작합니다.

출력: 설명 없이 JSON 객체 하나 {"answer": "<{f번호}를 넣은 답변 문장>"}"""
EXAMPLE = '{"answer": "2023년 서울특별시의 1인 가구는 {f1}입니다."}'
RETRY = "[이전 답변의 문제]\n{problems}\n위 문제를 고쳐 같은 형식의 JSON 객체 하나만 다시 출력하세요."
_PLACEHOLDER = re.compile(r"\{\s*(f\d+)\s*\}")
_KIND_WORDS = {"growth": "증감률", "difference": "차이", "share": "비중", "ratio": "배수"}


class ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    answer: str = Field(min_length=1, description="{f번호}를 넣은 답변 문장")


def _kind(fact: Fact) -> str:
    expr = fact.derivation or ""
    if "* 100" in expr and "- 1)" in expr:
        return _KIND_WORDS["growth"]
    if "* 100" in expr:
        return _KIND_WORDS["share"]
    if "/" in expr:
        return _KIND_WORDS["ratio"]
    return _KIND_WORDS["difference"]


def fact_line(fact: Fact, main: bool) -> str:
    tag = "[반드시 인용]" if main else "[참고]"
    if fact.derivation:
        what = f"{_kind(fact)} ({fact.derivation})"
        way = f" ({direction(fact)})" if is_change(fact) else ""
    else:
        what = f"{fact.period}년 {fact_label(fact, skip_totals=False)}"
        way = ""
    note = f" — {fact.note}" if fact.note else ""
    return f"{{{fact.id}}} {tag} {what} = {format_fact(fact)}{way}{note}"


def user_prompt(question: str, computed: Computed, note: str | None = None) -> str:
    lines = [fact_line(f, f.id in computed.main) for f in computed.result.facts]
    extra = f"\n\n[참고 사항]\n{note}" if note else ""
    return f"[질문]\n{question}\n\n[사실 목록]\n" + "\n".join(lines) + extra + f"\n\n[출력 예]\n{EXAMPLE}"


def retry_prompt(user: str, problems: list[str]) -> str:
    return user + "\n\n" + RETRY.format(problems="\n".join(f"- {p}" for p in problems))


def to_parts(answer: str, computed: Computed) -> list:
    """'...{f1}가구...' -> [TextPart, FactPart, TextPart]; a unit the model repeats after a placeholder is dropped."""
    by_id = {f.id: f for f in computed.result.facts}
    parts: list = []
    pieces = _PLACEHOLDER.split(answer)
    for n, piece in enumerate(pieces):
        if n % 2:
            parts.append(FactPart(fact_id=piece))
            continue
        if parts and isinstance(parts[-1], FactPart) and (fact := by_id.get(parts[-1].fact_id)) and fact.unit:
            if piece.startswith(fact.unit):
                piece = piece[len(fact.unit):]
        if piece:
            parts.append(TextPart(text=piece))
    return parts
