"""The Answer one question returns (StatPipeline.ask, and the B0 Chatbot.ask): the run record eval/run_eval.py and
Phase 2 read (status, attempts, usage, raw_responses, format_errors, decision_metadata with its spans, the finished
StatAnswer and the ChartSpec).

T1-28 S6 (legacy isolation): moved here from kosis_chat/legacy/chatbot.py, so the module entry point (api.py) and the
stat pipeline (pipeline.py) use it without importing kosis_chat/legacy/. legacy/chatbot.py imports it from here: the
same class under every path (kosis_chat.chatbot.Answer, kosis_chat.legacy.chatbot.Answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kosis_chat.llm.model import STATUS_OK
from kosis_chat.prepare.schemas import TableMeta


@dataclass(frozen=True)
class Answer:
    text: str
    status: str
    table_meta: TableMeta
    prompt_version: str
    error: str | None = None
    input_tokens_estimate: int = 0
    latency_seconds: float = 0.0  # call/validation wall time including HTTP retries, excluding format-retry interval
    attempts: int = 0  # HTTP requests made (0 when no call was sent)
    usage: dict[str, Any] = field(default_factory=dict)  # token counts reported by the API
    scope: dict[str, Any] = field(default_factory=dict)  # table part sent: mode, scope, cells, table_chars
    parsed: Any = None  # the B0 chatbot's validated model answer (legacy/answer_schema.py LLMAnswer); None otherwise
    format_errors: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    answer_format: str = "text"
    retry_wait_seconds: float = 0.0
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)  # rule verdict (T1-08); empty when judge_mode is off
    stat_answer: Any = None  # the finished StatAnswer when the C pipeline built it (kosis_chat/pipeline.py)
    chart: Any = None  # kosis_ai.schemas.ChartSpec built from the computed facts (T1-28 S5; a Vega-Lite dict before)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


__all__ = ["Answer"]
