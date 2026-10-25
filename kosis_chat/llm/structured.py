"""Validated JSON replies with one format retry, preserving all attempts and usage."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from kosis_chat.llm.model import (
    STATUS_API_ERROR, STATUS_FORMAT_ERROR, STATUS_INPUT_TOO_LONG, STATUS_OK, STATUS_TIMEOUT, LLMClient, LLMResult, TraceWriteError,
)

JSON_INSTRUCTION = "[출력 형식]\n설명 없이 아래 JSON 스키마를 따르는 JSON 객체 하나만 출력하세요.\n{schema}"
RETRY_INSTRUCTION = "[이전 출력 오류]\n{errors}\n같은 질문에 대해 스키마에 맞는 JSON 객체 하나만 다시 출력하세요."


@dataclass(frozen=True)
class StructuredResult:
    value: BaseModel | None = None
    status: str = STATUS_FORMAT_ERROR
    error: str | None = None
    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    retry_wait_seconds: float = 0.0


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def extract_json(text: str) -> dict:
    """Accept exactly one JSON object, optionally enclosed in a complete JSON fence."""
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1)
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    _finite_numbers(value)
    return value


def _finite_numbers(value: Any) -> None:
    # JSON exponents such as 1e999 can overflow despite not using a NaN/Infinity token.
    import math
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _finite_numbers(child)
    elif isinstance(value, list):
        for child in value:
            _finite_numbers(child)


def schema_text(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


def structured_prompt(user: str, model: type[BaseModel]) -> str:
    return user + "\n\n" + JSON_INSTRUCTION.format(schema=schema_text(model))


def _merge_usage(total: dict, incoming: dict) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict):
            if not isinstance(total.get(key), dict):
                total[key] = {}
            _merge_usage(total[key], value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
        else:
            total[key] = value


def _repair_hint(exc: Exception) -> str:
    """Give field/type guidance without echoing invalid input, error URLs or source data."""
    if not isinstance(exc, ValidationError):
        return "field=$; type=json_object; expected=one strict JSON object, double-quoted strings, no surrounding text or non-finite numbers"
    expectations = {
        "message": "nonblank string with the answer, refusal explanation or clarification question",
        "reason": "one refusal code only when refused; null or omit otherwise; explanation belongs in message",
        "options": "array of distinct nonblank strings; at least two for clarify, [] or omit otherwise; never null",
        "default_applied": 'object mapping strings to strings, e.g. {"period":"latest (2099)"}; use {} or omit when unused; never a string or null',
        "clarify_options": 'array of distinct nonblank strings; at least two for clarify; [] or omit otherwise; never null',
        "outcome": "one code: answered, refused, clarify",
        "refusal_reason": "one schema refusal code (out_of_scope belongs here, not in outcome); put explanation in refusal_message",
        "refusal_message": "nonblank string when refused; reason code is a separate required field",
        "clarify_question": "nonblank string when clarify, with at least two distinct choices",
    }
    hints = []
    for error in exc.errors(include_url=False, include_input=False, include_context=False)[:8]:
        location = error.get("loc", ())
        safe_location = [str(part) if isinstance(part, int) or re.fullmatch(r"[A-Za-z_]\w*", str(part)) else "<key>" for part in location]
        field = ".".join(safe_location) or "$"
        kind = error["type"]
        expected = expectations.get(str(location[0]) if location else "", {
            "string_type": "string", "list_type": "array", "dict_type": "object",
            "missing": "required field", "extra_forbidden": "remove unknown field",
        }.get(kind, "schema-valid fields for this outcome; do not change the outcome to hide a type error"))
        hints.append(f"field={field}; type={kind}; expected={expected}")
    return "\n".join(hints)[:1600]


def ask_structured(
    client: LLMClient, system: str, user: str, model: type[BaseModel], *,
    retry_interval: float = 0.0, max_input_tokens: int | None = None,
    sleep: Callable[[float], None] | None = None,
) -> StructuredResult:
    """User already includes structured_prompt; cap format calls at two, independently of HTTP retries."""
    if retry_interval < 0:
        raise ValueError("retry_interval cannot be negative")
    sleep = sleep or time.sleep
    raw, errors, usage = [], [], {}
    attempts = 0
    waited = 0.0
    original_user = user
    locked_outcome = None  # compact-v2: the first attempt's outcome must survive the format retry

    def done(status, error=None, value=None):
        return StructuredResult(value, status, error, attempts, usage, errors, raw, waited)

    for call in range(2):
        if max_input_tokens is not None and len(system + user) > max_input_tokens:
            return done(STATUS_INPUT_TOO_LONG, "structured prompt exceeds input token estimate limit")
        if call and retry_interval:
            started = time.perf_counter()
            sleep(retry_interval)
            waited += time.perf_counter() - started
        try:
            result = client.generate(system, user)
        except TimeoutError as exc:
            result = LLMResult("", STATUS_TIMEOUT, error=str(exc) or "timeout")
        except TraceWriteError:
            raise
        except Exception as exc:
            result = LLMResult("", STATUS_API_ERROR, error=f"{type(exc).__name__}: {exc}")
        attempts += result.attempts
        _merge_usage(usage, result.usage)
        raw.append(result.text)
        if result.status != STATUS_OK:
            return done(result.status, result.error)
        try:
            value = model.model_validate(extract_json(result.text))
        except (ValueError, ValidationError, RecursionError) as exc:
            errors.append(f"format attempt {call + 1}: {exc}")
            client.record_event({"type": "validation_error", "format_attempt": call + 1, "error": str(exc), "schema": model.__name__})
            if call == 0 and hasattr(model, "outcome_of"):
                locked_outcome = model.outcome_of(result.text)
            hint = model.repair_hint(result.text, exc) if hasattr(model, "repair_hint") else _repair_hint(exc)
            user = original_user + "\n\n" + RETRY_INSTRUCTION.format(errors=hint)
        else:
            if locked_outcome is not None and value.outcome != locked_outcome:
                error = f"format attempt {call + 1}: outcome changed from {locked_outcome} to {value.outcome} during format retry"
                errors.append(error)
                client.record_event({"type": "validation_error", "format_attempt": call + 1, "error": error, "schema": model.__name__})
                break
            normalized = list(getattr(value, "absent_normalized", ()))
            client.record_event({"type": "validation_ok", "format_attempt": call + 1, "schema": model.__name__,
                                 **({"absent_fields_normalized": normalized} if normalized else {})})
            return done(STATUS_OK, value=value)
    return done(STATUS_FORMAT_ERROR, errors[-1])
