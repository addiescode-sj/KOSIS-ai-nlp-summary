"""raw-v2: one prompt with the table and the question, and the model answers in its own words (eval baseline).

No rules, analysis, computation, retries, format checks or source clean-up run before or after the call
(--pipeline raw). The model decides by itself whether to answer, refuse or ask back. The reply is kept as written
and read by people (eval/view_raw.py, eval/compare_runs.py); it is not graded.
"""

from __future__ import annotations

VERSION = "raw-v2"
SYSTEM_PROMPT = """당신은 KOSIS 통계표 하나를 보고 질문에 답하는 도우미입니다.
[통계표]에 있는 값만 근거로 한국어 존댓말로 답합니다. 표에 없는 원인, 전망, 평가는 쓰지 않습니다.
표로 답할 수 없으면 그 이유와 표에서 확인할 수 있는 범위를 말합니다.
질문의 뜻에 따라 답이 달라지면 무엇을 원하는지 되묻습니다.
질문에 없는 조건(시점, 분류 등)을 정해서 답했다면 답변에 밝힙니다."""
USER_TEMPLATE = """[통계표]
{table_text}

[질문]
{question}"""


def user_prompt(table_text: str, question: str) -> str:
    return USER_TEMPLATE.format(table_text=table_text, question=question.strip())
