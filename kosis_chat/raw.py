"""Raw mode (--pipeline raw): table text + question -> one model call -> the reply as written (eval baseline).

The model gets the same table part and serialization as the chatbot (scope mode, --serialization) and nothing else:
no judge, no analysis or computation, no format check or retry, no source clean-up, no default note. RawRunner.ask
returns the same Answer as Chatbot.ask so run_eval can record it; the reply is never parsed, repaired or graded.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from kosis_chat.chatbot import Answer, estimate_tokens
from kosis_chat.config import Settings, load_settings
from kosis_chat.data import scope, serializer
from kosis_chat.data.schema import TableInput
from kosis_chat.llm import raw_prompt
from kosis_chat.llm.model import (
    STATUS_API_ERROR,
    STATUS_INPUT_TOO_LONG,
    STATUS_TIMEOUT,
    LLMClient,
    TraceWriteError,
    build_client,
)

DECISION_METADATA = {"pipeline": "raw", "route": "raw", "compose": "model", "outcome_origin": "model",
                     "numbers_origin": "model", "chart_origin": "none", "defaults_origin": "model"}


class RawRunner:
    """Answers questions about one table with one model call each; same attributes run_eval reads from a Chatbot."""

    def __init__(
        self,
        table: TableInput,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        scope_mode: str = "oracle",
    ) -> None:
        if scope_mode not in scope.MODES:
            raise ValueError(f"unknown scope mode {scope_mode!r}; expected one of {scope.MODES}")
        self.settings = settings or load_settings()
        self.client = client or build_client(self.settings)
        self.table = table  # as loaded: preprocessing only fills parsed values, which the table text does not use
        self.scope_mode = scope_mode
        self.prompt_version = raw_prompt.VERSION
        self.system_prompt = raw_prompt.SYSTEM_PROMPT
        self.answer_model = None  # free text, no response schema
        self.decision_metadata = dict(DECISION_METADATA)

    def prepare(self, question: str, golden_scope: dict | None = None) -> tuple[str, dict[str, Any]]:
        """Table text for one question (same cut and style as Chatbot.prepare) and a record with its hash."""
        part, record = scope.select(self.table, self.scope_mode, question, golden_scope, self.settings.serialization)
        text = serializer.serialize(part, self.settings.serialization)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, {**record, "table_chars": len(text), "table_text_sha256": digest}

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        table_text, record = self.prepare(question, golden_scope)
        user = raw_prompt.user_prompt(table_text, question)
        tokens = estimate_tokens(self.system_prompt + user)
        limit = self.settings.max_input_tokens
        if limit is not None and tokens > limit:
            return self._answer("", STATUS_INPUT_TOO_LONG, f"estimated {tokens} tokens > limit {limit}", tokens, record)
        started = time.perf_counter()
        try:
            result = self.client.generate(self.system_prompt, user)
        except TraceWriteError:
            raise
        except TimeoutError as exc:
            return self._answer("", STATUS_TIMEOUT, str(exc) or "timeout", tokens, record, time.perf_counter() - started, 1)
        except Exception as exc:  # recorded as a run failure, never dropped from evaluation
            error = f"{type(exc).__name__}: {exc}"
            return self._answer("", STATUS_API_ERROR, error, tokens, record, time.perf_counter() - started, 1)
        return self._answer(result.text, result.status, result.error, tokens, record, time.perf_counter() - started,
                            result.attempts, result.usage)

    def _answer(
        self,
        text: str,
        status: str,
        error: str | None,
        tokens: int,
        record: dict[str, Any],
        latency: float = 0.0,
        attempts: int = 0,
        usage: dict[str, Any] | None = None,
    ) -> Answer:
        return Answer(
            text=text, status=status, table_meta=self.table.meta, prompt_version=self.prompt_version, error=error,
            input_tokens_estimate=tokens, latency_seconds=latency, attempts=attempts, usage=usage or {},
            scope=record, raw_responses=[text] if attempts else [], answer_format="text",
            decision_metadata=dict(self.decision_metadata),
        )
