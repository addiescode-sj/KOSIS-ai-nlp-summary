"""How an LLM step reaches the model (T1-28 S3b): one structured call, recorded as an LLMCall.

The client is not part of StatResources (resources are per table and frozen; run_eval swaps the client of a built
pipeline), so a step gets a ModelAccess: a way to the owner's client, read when the step first calls the model. A
StatPipeline built from an LLMConfig without a key raises LLMKeyMissing there, so rule-only answers need no key.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from kosis_chat.llm.model import LLMClient
from kosis_chat.llm.structured import StructuredResult, ask_structured
from kosis_chat.schemas.state import LLMCall


@dataclass(frozen=True)
class ModelAccess:
    """client(): the client (built on first use by its owner); model_name(): the name a span records (gen_ai.request.
    model), read after the call. retry_interval: seconds to wait before a second model call for the same question
    (HCX 429), also passed to ask_structured for its format retry; max_input_tokens: checked before calling."""

    client: Callable[[], LLMClient]
    model_name: Callable[[], str]
    retry_interval: float = 0.0
    max_input_tokens: int | None = None

    def ask(self, name: str, system: str, user: str, model: type[BaseModel], *, called: bool,
            **attributes) -> tuple[StructuredResult, LLMCall]:
        """One model step: wait first when a model answer came back already for this question (called), call, and
        record the call as span `name` with the step's own attributes."""
        started = time.perf_counter()
        slept = 0.0
        if called and self.retry_interval:  # a second call for the same question: stay under the rate limit
            time.sleep(self.retry_interval)
            slept = time.perf_counter() - started
        result = ask_structured(self.client(), system, user, model, retry_interval=self.retry_interval,
                                max_input_tokens=self.max_input_tokens)
        call = LLMCall(
            name=name, started=started, ended=time.perf_counter(), model=self.model_name(), status=result.status,
            attempts=result.attempts, usage=result.usage, raw=result.raw, errors=result.errors, slept=slept,
            retry_wait_seconds=result.retry_wait_seconds, prompt_chars=len(system) + len(user), attributes=attributes,
        )
        return result, call
