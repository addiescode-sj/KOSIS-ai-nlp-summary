"""Assemble the answer flow: table -> clean -> serialize -> prompt -> model -> Answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kosis_chat.config import Settings, load_settings
from kosis_chat.data import preprocessing, serializer
from kosis_chat.data.schema import TableInput, TableMeta
from kosis_chat.llm import prompts
from kosis_chat.llm.model import (
    STATUS_API_ERROR,
    STATUS_INPUT_TOO_LONG,
    STATUS_OK,
    STATUS_TIMEOUT,
    LLMClient,
    build_client,
)


@dataclass(frozen=True)
class Answer:
    text: str
    status: str
    table_meta: TableMeta
    prompt_version: str
    error: str | None = None
    input_tokens_estimate: int = 0
    latency_seconds: float = 0.0  # time spent in the model call, including retries
    attempts: int = 0  # HTTP requests made (0 when no call was sent)
    usage: dict[str, Any] = field(default_factory=dict)  # token counts reported by the API

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (one token per character). Replace with the HCX tokenizer count once known."""
    return len(text)


class Chatbot:
    """Answers questions about one table given at construction (stage 1 input is the data itself)."""

    def __init__(
        self,
        table: TableInput,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        prompt_version: str | None = None,
        form: str | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.client = client or build_client(self.settings)
        self.prompt_version = prompt_version or self.settings.prompt_version
        self.system_prompt = prompts.get_system_prompt(self.prompt_version)
        self.form = form  # answer-format instructions appended to every question
        # Each stage adds one step here (B1 rules, B2 analysis, B3 calculation, B4 validation).
        self.table = preprocessing.clean(table)
        self.table_text = serializer.serialize(self.table, self.settings.serialization)

    def ask(self, question: str) -> Answer:
        user_prompt = prompts.build_user_prompt(self.table_text, question, self.form)
        tokens = estimate_tokens(self.system_prompt + user_prompt)
        limit = self.settings.max_input_tokens
        if limit is not None and tokens > limit:
            # Never truncate the table silently.
            return self._answer("", STATUS_INPUT_TOO_LONG, f"estimated {tokens} tokens > limit {limit}", tokens)
        started = time.perf_counter()
        try:
            result = self.client.generate(self.system_prompt, user_prompt)
        except TimeoutError as exc:
            return self._answer("", STATUS_TIMEOUT, str(exc) or "timeout", tokens, time.perf_counter() - started, 1)
        except Exception as exc:  # recorded as a run failure, never dropped from evaluation
            error = f"{type(exc).__name__}: {exc}"
            return self._answer("", STATUS_API_ERROR, error, tokens, time.perf_counter() - started, 1)
        latency = time.perf_counter() - started
        return self._answer(result.text, result.status, result.error, tokens, latency, result.attempts, result.usage)

    def _answer(
        self,
        text: str,
        status: str,
        error: str | None,
        tokens: int,
        latency: float = 0.0,
        attempts: int = 0,
        usage: dict[str, Any] | None = None,
    ) -> Answer:
        return Answer(
            text=text,
            status=status,
            table_meta=self.table.meta,
            prompt_version=self.prompt_version,
            error=error,
            input_tokens_estimate=tokens,
            latency_seconds=latency,
            attempts=attempts,
            usage=usage or {},
        )
