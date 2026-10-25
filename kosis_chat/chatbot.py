"""Assemble the answer flow: table -> clean -> scope (question part) -> serialize -> prompt -> model -> Answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from kosis_chat.config import Settings, load_settings
from kosis_chat.data import preprocessing, scope, serializer
from kosis_chat.data.schema import TableInput, TableMeta
from kosis_chat.judge import messages as judge_messages
from kosis_chat.judge.answerability import Verdict, judge
from kosis_chat.judge.question import build_index
from kosis_chat.llm import prompts
from kosis_chat.llm.answer_schema import CompactAnswer, LLMAnswer, answer_model
from kosis_chat.llm.compact_v2 import CompactAnswerV2
from kosis_chat.llm.structured import ask_structured, structured_prompt
from kosis_chat.llm.model import (
    STATUS_API_ERROR,
    STATUS_INPUT_TOO_LONG,
    STATUS_OK,
    STATUS_TIMEOUT,
    LLMClient, TraceWriteError,
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
    latency_seconds: float = 0.0  # call/validation wall time including HTTP retries, excluding format-retry interval
    attempts: int = 0  # HTTP requests made (0 when no call was sent)
    usage: dict[str, Any] = field(default_factory=dict)  # token counts reported by the API
    scope: dict[str, Any] = field(default_factory=dict)  # table part sent: mode, scope, cells, table_chars
    parsed: LLMAnswer | None = None
    format_errors: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    answer_format: str = "text"
    retry_wait_seconds: float = 0.0
    decision_metadata: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)  # rule verdict (T1-08); empty when judge_mode is off
    stat_answer: Any = None  # the finished StatAnswer when the C pipeline built it (kosis_chat/pipeline.py)
    chart: dict[str, Any] | None = None  # Vega-Lite spec built from the computed facts (kosis_chat/answer/chart.py)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


JUDGE_MODES = ("off", "rules")


def judge_record(verdict: Verdict) -> dict[str, Any]:
    return {
        "rule": verdict.rule, "outcome": verdict.outcome, "final": verdict.final,
        "reason": verdict.refusal.reason if verdict.refusal else None, "default_applied": dict(verdict.default_applied),
    }


def verdict_answer(verdict: Verdict) -> LLMAnswer:
    """A final rule verdict in the model's answer shape, so contract.to_stat_answer handles both the same way."""
    if verdict.outcome == "refused":
        refusal = verdict.refusal
        return LLMAnswer(outcome="refused", refusal_reason=refusal.reason, refusal_message=refusal.message,
                         available=refusal.available)
    return LLMAnswer(outcome="clarify", clarify_question=verdict.clarify_question,
                     clarify_options=list(verdict.clarify_options))


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (one token per character). Replace with the HCX tokenizer count once known."""
    return len(text)


class Chatbot:
    """Answers questions about one table given at construction (stage 1 input is the data itself).
    Each question gets only the part of the table its scope mode picks (kosis_chat/data/scope.py)."""

    def __init__(
        self,
        table: TableInput,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        prompt_version: str | None = None,
        form: str | None = None,
        scope_mode: str = "full",
        answer_format: str = "text",
        retry_interval: float = 0.0,
        judge_mode: str = "off",
    ) -> None:
        if scope_mode not in scope.MODES:
            raise ValueError(f"unknown scope mode {scope_mode!r}; expected one of {scope.MODES}")
        if judge_mode not in JUDGE_MODES:
            raise ValueError(f"unknown judge mode {judge_mode!r}; expected one of {JUDGE_MODES}")
        self.settings = settings or load_settings()
        self.prompt_version = prompts.resolve_answer_prompt(answer_format, prompt_version, self.settings.prompt_version)
        if retry_interval < 0:
            raise ValueError("retry_interval cannot be negative")
        self.answer_format = answer_format
        self.retry_interval = retry_interval
        self.client = client or build_client(self.settings)
        self.system_prompt = prompts.get_system_prompt(self.prompt_version)
        self.answer_model = answer_model(self.prompt_version)
        self.decision_metadata = ({
            "default_selection": "not_implemented", "defaults_origin": "none",
            "outcome_origin": "model", "semantic_validation": "not_implemented",
        } if self.prompt_version in ("compact-v1", "compact-v2") else {})
        self.form = form  # answer-format instructions appended to every question
        # Each stage adds one step here (B1 rules, B2 analysis, B3 calculation, B4 validation).
        self.scope_mode = scope_mode
        self.table = preprocessing.clean(table)
        # rules: kosis_chat/judge looks at the question and the whole table first (T1-08).
        self.judge_mode = judge_mode
        self.judge_index = build_index(self.table) if judge_mode == "rules" else None

    def prepare(self, question: str, golden_scope: dict | None = None) -> tuple[str, dict[str, Any]]:
        """Table text for one question and a record of the part chosen."""
        part, record = scope.select(self.table, self.scope_mode, question, golden_scope, self.settings.serialization)
        text = serializer.serialize(part, self.settings.serialization)
        return text, {**record, "table_chars": len(text)}

    def ask(self, question: str, golden_scope: dict | None = None) -> Answer:
        if self.judge_index is None:
            return self._ask_model(question, golden_scope)
        verdict = judge(question, self.table, self.judge_index)
        if verdict.final:  # refused or clarify by rule: the model is not called
            parsed = verdict_answer(verdict)
            text = parsed.refusal_message or "\n".join([parsed.clarify_question, *parsed.clarify_options])
            return replace(self._answer(text, STATUS_OK, None, 0), parsed=parsed, judge=judge_record(verdict),
                           decision_metadata={**self.decision_metadata, "outcome_origin": "rules", "default_selection": "not_applicable"})
        answer = self._ask_model(question, golden_scope, judge_messages.default_note(verdict.default_applied))
        metadata = {**answer.decision_metadata, "rule_default_plan": verdict.default_applied,
                    "rule_default_plan_validation": "not_verified_as_applied"}
        if self.prompt_version == "compact-v2" and verdict.default_applied:
            metadata["defaults_origin"] = "rules"  # recorded in StatAnswer.default_applied by contract.to_stat_answer
        return replace(answer, judge=judge_record(verdict), decision_metadata=metadata)

    def _ask_model(self, question: str, golden_scope: dict | None = None, note: str = "") -> Answer:
        table_text, record = self.prepare(question, golden_scope)
        user_prompt = prompts.build_user_prompt(table_text, question + note, self.form)
        if self.answer_format == "json":
            user_prompt = structured_prompt(user_prompt, self.answer_model)
        tokens = estimate_tokens(self.system_prompt + user_prompt)
        limit = self.settings.max_input_tokens
        if limit is not None and tokens > limit:
            # Never truncate the table silently.
            error = f"estimated {tokens} tokens > limit {limit}"
            return self._answer("", STATUS_INPUT_TOO_LONG, error, tokens, scope=record)
        started = time.perf_counter()
        if self.answer_format == "json":
            result = ask_structured(
                self.client, self.system_prompt, user_prompt, self.answer_model,
                retry_interval=self.retry_interval, max_input_tokens=limit,
            )
            normalized = list(getattr(result.value, "absent_normalized", ()))
            return Answer(
                text=result.raw[-1] if result.raw else "", status=result.status,
                table_meta=self.table.meta, prompt_version=self.prompt_version, error=result.error,
                input_tokens_estimate=tokens,
                latency_seconds=max(0.0, time.perf_counter() - started - result.retry_wait_seconds),
                attempts=result.attempts, usage=result.usage, scope=record, parsed=result.value.to_legacy() if isinstance(result.value, (CompactAnswer, CompactAnswerV2)) else result.value,
                format_errors=result.errors, raw_responses=result.raw, answer_format="json",
                retry_wait_seconds=result.retry_wait_seconds,
                decision_metadata={**self.decision_metadata, **({"absent_fields_normalized": normalized} if normalized else {})},
            )
        try:
            result = self.client.generate(self.system_prompt, user_prompt)
        except TimeoutError as exc:
            latency = time.perf_counter() - started
            return self._answer("", STATUS_TIMEOUT, str(exc) or "timeout", tokens, latency, 1, scope=record)
        except TraceWriteError:
            raise
        except Exception as exc:  # recorded as a run failure, never dropped from evaluation
            error = f"{type(exc).__name__}: {exc}"
            return self._answer("", STATUS_API_ERROR, error, tokens, time.perf_counter() - started, 1, scope=record)
        latency = time.perf_counter() - started
        return self._answer(
            result.text, result.status, result.error, tokens, latency, result.attempts, result.usage, scope=record
        )

    def _answer(
        self,
        text: str,
        status: str,
        error: str | None,
        tokens: int,
        latency: float = 0.0,
        attempts: int = 0,
        usage: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
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
            scope=scope or {},
            answer_format=self.answer_format, decision_metadata=self.decision_metadata,
        )
