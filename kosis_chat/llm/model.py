"""LLM clients behind one interface: MockClient for offline tests, HCXClient for HyperCLOVA X."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlsplit, urlunsplit
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from kosis_chat.config import Settings, get_secret

# Call status values recorded with every answer and eval result.
STATUS_OK = "ok"
STATUS_FORMAT_ERROR = "format_error"
STATUS_TIMEOUT = "timeout"
STATUS_API_ERROR = "api_error"
STATUS_INPUT_TOO_LONG = "input_too_long"

log = logging.getLogger("kosis.llm")


@dataclass(frozen=True)
class LLMResult:
    text: str
    status: str = STATUS_OK
    usage: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 1  # HTTP requests made, including retries


class TraceWriteError(RuntimeError):
    """Audit persistence failed; never downgrade this to a model or API error."""


class LLMClient(ABC):
    """Common interface; implementations must not raise for API failures, they return a status."""

    name: str = "base"

    def record_event(self, event: dict[str, Any]) -> None:
        """Optional audit hook; bare clients do not claim transport trace availability."""

    @abstractmethod
    def generate(self, system: str, user: str) -> LLMResult:
        """Return the model's reply for one system + user prompt pair."""


class MockClient(LLMClient):
    """Deterministic client for tests: fixed text, a queue of replies, or a function of the prompt."""

    name = "mock"

    def __init__(
        self,
        response: str | Sequence[LLMResult | str] | Callable[[str, str], LLMResult | str] = "[mock] answer",
    ) -> None:
        self._response = response
        self._queue = list(response) if isinstance(response, (list, tuple)) else None
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> LLMResult:
        self.calls.append((system, user))
        if callable(self._response):
            reply = self._response(system, user)
        elif self._queue is not None:
            if not self._queue:
                return LLMResult(text="", status=STATUS_API_ERROR, error="mock queue exhausted")
            reply = self._queue.pop(0)
        else:
            reply = self._response
        return reply if isinstance(reply, LLMResult) else LLMResult(text=str(reply))


class HCXClient(LLMClient):
    """HyperCLOVA X through the CLOVA Studio v3 chat-completions API (non-streaming).

    POST {base_url}/v3/chat-completions/{model} with "Authorization: Bearer <HCX_API_KEY>".
    Reasoning models (HCX-007) take maxCompletionTokens and thinking.effort; the others take maxTokens.
    Rate limits (429) and server errors (5xx) are retried after each delay in RETRY_DELAYS,
    or after the server's Retry-After when that is longer (capped at MAX_RETRY_WAIT).
    """

    name = "hcx"
    REASONING_MODELS = ("HCX-007",)
    RETRY_DELAYS = (5.0, 15.0, 30.0, 60.0)
    MAX_RETRY_WAIT = 120.0
    RETRY_HTTP_CODES = (429, 500, 502, 503, 504)
    CONTEXT_EXCEEDED_CODE = "40003"

    def __init__(
        self, settings: Settings, sleep: Callable[[float], None] = time.sleep,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not settings.model_name or settings.model_name == "TBD":
            raise ValueError("model.name (HCX model) is not set in configs/model.yaml (or pass --model)")
        key = get_secret(settings.api_key_env)
        if not key:
            raise ValueError(f"{settings.api_key_env} is not set in .env")
        self.settings = settings
        self.url = f"{settings.base_url.rstrip('/')}/v3/chat-completions/{settings.model_name}"
        self._key = key
        self._sleep = sleep
        self.observer = observer

    @property
    def is_reasoning(self) -> bool:
        return self.settings.model_name.upper().startswith(self.REASONING_MODELS)

    def generate(self, system: str, user: str) -> LLMResult:
        result, http_code, retry_after = self._observed_call(system, user, 1)
        attempts = 1
        for retry, delay in enumerate(self.RETRY_DELAYS, start=1):
            if http_code not in self.RETRY_HTTP_CODES:
                break
            wait = min(max(delay, retry_after or 0.0), self.MAX_RETRY_WAIT)
            log.warning("HTTP %s, retry %d/%d after %.0fs", http_code, retry, len(self.RETRY_DELAYS), wait)
            self._emit({"type": "http_backoff", "http_attempt": attempts, "scheduled_wait_seconds": wait})
            self._sleep(wait)
            result, http_code, retry_after = self._observed_call(system, user, attempts + 1)
            attempts += 1
        return replace(result, attempts=attempts)

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._redact(key): self._redact(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self._redact(child) for child in value]
        return value

    def _emit(self, event: dict[str, Any]) -> None:
        if self.observer is not None:
            try:
                self.observer(self._redact(event))
            except TraceWriteError:
                raise
            except Exception as exc:
                raise TraceWriteError("HTTP audit observer failed") from exc

    def _observed_call(self, system: str, user: str, attempt: int):
        endpoint = urlsplit(self.url)
        host = endpoint.hostname or ""
        host = f"[{host}]" if ":" in host else host
        authority = f"{host}:{endpoint.port}" if endpoint.port is not None else host
        safe_url = urlunsplit((endpoint.scheme, authority, endpoint.path, "", ""))
        self._emit({"type": "http_request", "http_attempt": attempt, "method": "POST", "url": safe_url,
                    "body": self._body(system, user), "transport": "hcx_http"})
        started = time.perf_counter()
        try:
            result, http_code, retry_after = self._call(system, user)
        except BaseException as exc:
            self._emit({"type": "http_raised", "http_attempt": attempt, "exception_type": type(exc).__name__})
            raise
        result = replace(result, text=self._redact(result.text), error=self._redact(result.error), usage=self._redact(result.usage))
        self._emit({"type": "http_response", "http_attempt": attempt, "http_status": http_code,
                    "call_status": result.status, "text": result.text, "error": result.error,
                    "usage": result.usage, "latency_seconds": time.perf_counter() - started})
        return result, http_code, retry_after

    def _body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.settings.temperature,
        }
        if self.is_reasoning:
            body["maxCompletionTokens"] = self.settings.max_tokens  # includes thinking tokens
            body["thinking"] = {"effort": self.settings.thinking_level or "none"}
        else:
            body["maxTokens"] = self.settings.max_tokens
        return body

    def _request(self, system: str, user: str) -> urllib.request.Request:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._key}",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": uuid.uuid4().hex,
        }
        data = json.dumps(self._body(system, user)).encode("utf-8")
        return urllib.request.Request(self.url, data=data, headers=headers, method="POST")

    def _call(self, system: str, user: str) -> tuple[LLMResult, int | None, float | None]:
        """One request -> (result, HTTP error code or None, Retry-After seconds or None)."""
        try:
            with urllib.request.urlopen(self._request(system, user), timeout=self.settings.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_status = getattr(resp, "status", None)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            too_long = self.CONTEXT_EXCEEDED_CODE in detail or "context length" in detail.lower()
            status = STATUS_INPUT_TOO_LONG if too_long else STATUS_API_ERROR
            return LLMResult(text="", status=status, error=f"HTTP {exc.code}: {detail}"), exc.code, _retry_after(exc)
        except TimeoutError:
            return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out"), None, None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                return LLMResult(text="", status=STATUS_TIMEOUT, error="request timed out"), None, None
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"connection failed: {exc.reason}"), None, None
        except json.JSONDecodeError as exc:
            return LLMResult(text="", status=STATUS_API_ERROR, error=f"invalid JSON response: {exc}"), None, None
        return _parse_response(data), response_status, None


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    try:
        return float(exc.headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None


def _parse_response(data: dict[str, Any]) -> LLMResult:
    result = data.get("result") or {}
    if str((data.get("status") or {}).get("code")) != "20000" or not isinstance(result.get("message"), dict):
        return LLMResult(text="", status=STATUS_API_ERROR, error=f"unexpected response: {str(data)[:300]}")
    usage = result.get("usage") or {}
    text = result["message"].get("content") or ""
    if not text:
        # With thinking on, a small maxCompletionTokens can be used up before any answer text.
        return LLMResult(text="", status=STATUS_API_ERROR, usage=usage, error=f"empty response (finishReason={result.get('finishReason')})")
    return LLMResult(text=text, usage=usage)


BACKENDS = ("mock", "hcx")


def build_client(settings: Settings) -> LLMClient:
    """Create the client named by settings.backend."""
    if settings.backend == "mock":
        return MockClient(settings.mock_response)
    if settings.backend == "hcx":
        return HCXClient(settings)
    raise ValueError(f"unknown backend: {settings.backend!r} (expected one of {BACKENDS})")
