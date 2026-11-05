"""The fallback node (T1-28 S3b): the compact-v2 chatbot answers what neither the rules nor the model's plan placed.

It lives in legacy/ because it runs the B0 chatbot (kosis_chat/chatbot.py), which the stage packages must not import.
S6 decides whether it stays (D2: StatOptions.use_fallback_chatbot).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import ConfigDict, SkipValidation

from kosis_chat.chatbot import Answer, Chatbot
from kosis_chat.llm.calls import ModelAccess
from kosis_chat.schemas.resources import StatResources
from kosis_chat.schemas.state import LLMCall
from kosis_chat.shared.boundary import BoundaryModel

FALLBACK_PROMPT = "compact-v2"


class FallbackIn(BoundaryModel):
    """step_fallback_chatbot's input: the question and the golden scope run_eval passes to a chatbot (None otherwise)."""

    question: str
    golden_scope: SkipValidation[dict[str, Any] | None] = None


class FallbackOut(BoundaryModel):
    """step_fallback_chatbot's output: the chatbot's Answer, and its calls as one LLMCall (span "fallback_chatbot",
    summed=False: the Answer reports them, graph/legacy.py fallback_answer adds the analysis calls to it)."""

    model_config = ConfigDict(frozen=True)

    answer: SkipValidation[Answer]
    call: SkipValidation[LLMCall]


def step_fallback_chatbot(inp: FallbackIn, res: StatResources, *, chatbot: Callable[[], Chatbot],
                          llm: ModelAccess) -> FallbackOut:
    """The compact-v2 chatbot's answer (its own model calls: at most 2, the answer and its format retry).

    In: FallbackIn(question, golden_scope)
    Out: FallbackOut(answer, call)
    Side effects: the chatbot's model calls
    Node: fallback_chatbot (span "fallback_chatbot")

    chatbot: the pipeline's fallback Chatbot (StatPipeline.fallback, built on first use, so LLMKeyMissing is raised
    here); llm: for the model name the span records.
    """
    started = time.perf_counter()
    answer = chatbot().ask(inp.question, inp.golden_scope)
    # ponytail: the chatbot's own steps stay one span; split it if the fallback route gets tuned
    call = LLMCall(name="fallback_chatbot", started=started, ended=time.perf_counter(), model=llm.model_name(),
                   status=answer.status, attempts=answer.attempts, usage=answer.usage,
                   retry_wait_seconds=answer.retry_wait_seconds, summed=False)
    return FallbackOut(answer=answer, call=call)

