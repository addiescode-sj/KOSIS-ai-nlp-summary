"""What the compose stage reads and passes on (T1-28 S1b, S3a).

ComposeOutput moved here unchanged from llm/compose.py (it was Pydantic already: the structured-output schema the
model is asked for); llm/compose.py re-exports it. The class name is the schema title (model_json_schema) and the
name in validation events (llm/structured.py), so it stays ComposeOutput. Caveat is the shared contract model
(kosis_ai.schemas.Caveat) and stays there.

S3a: the In/Out models of the pure compose steps (compose/node.py): caveats, verify, template. S3b: the model's
sentence (compose_llm).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from kosis_chat.analyze.schemas import Built
from kosis_chat.compute.schemas import Alternatives, Computed
from kosis_chat.schemas.state import LLMCall
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import Caveat, QuestionType  # isort: skip

# Model sentences tried before the code template (the first answer and one retry): StatPipeline's compose loop runs
# this many compose_llm attempts, and step_verify routes the last one's problems to the template.
COMPOSE_MAX_ATTEMPTS = 2


class ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    answer: str = Field(min_length=1, description="{f번호}를 넣은 답변 문장")


class CaveatsIn(BoundaryModel):
    """step_caveats's input: the computed facts, the built plan (its context cells), the other scenarios shown and
    the defaults the judge applied (their sentence already names them, so a default_applied caveat is not shown)."""

    computed: SkipValidation[Computed]
    built: SkipValidation[Built]
    alternatives: SkipValidation[list[Alternatives]] = Field(default_factory=list)
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)


class CaveatsOut(BoundaryModel):
    """context_ids: facts shown beside the answer (context cells that are not main facts); caveats: every Caveat
    (StatAnswer.caveats); shown: the caveat texts given to compose and written as '참고:' lines."""

    context_ids: SkipValidation[list[str]]
    caveats: SkipValidation[list[Caveat]]
    shown: SkipValidation[list[str]]


class VerifyIn(BoundaryModel):
    """step_verify's input: answer parts (the model's or a template's), the facts they cite, the 0-based attempt."""

    parts: SkipValidation[list]
    computed: SkipValidation[Computed]
    attempt: int = 0


class VerifyOut(BoundaryModel):
    """problems: what verify() found ([] = the parts pass). route: "ok" (no problems), "retry" (problems and another
    model attempt is left), "template" (problems on the last attempt: the code template answers)."""

    problems: SkipValidation[list[str]]
    route: Literal["ok", "retry", "template"]


class TemplateIn(BoundaryModel):
    """step_template's input: the plan's question type and the computed facts."""

    type: QuestionType | None
    computed: SkipValidation[Computed]


class TemplateOut(BoundaryModel):
    """parts written by code, what verify() found in them (a template cites every main fact by construction, so this
    is recorded, not assumed), and verified = no problems."""

    parts: SkipValidation[list]
    problems: SkipValidation[list[str]]
    verified: bool


class ComposeLLMIn(BoundaryModel):
    """step_compose_llm's input: the question, the computed facts, the caveat texts shown ('참고:' lines) and the facts
    shown beside the answer (CaveatsOut.shown, context_ids); attempt (0-based); problems: what verify() found in the
    previous attempt's parts (said back to the model), prompt: the previous attempt's user prompt (None on attempt 0);
    called: a model answer came back already for this question (then the call waits retry_interval first)."""

    question: str
    computed: SkipValidation[Computed]
    cautions: SkipValidation[list[str]] = Field(default_factory=list)
    context_ids: SkipValidation[list[str]] = Field(default_factory=list)
    attempt: int = 0
    problems: SkipValidation[list[str]] = Field(default_factory=list)
    prompt: str | None = None
    called: bool = False


class ComposeLLMOut(BoundaryModel):
    """step_compose_llm's output. route "verify": parts (the model's sentence split into TextPart / FactPart) go to
    verify. route "template": the call failed (meta compose_status), the code template answers. prompt: the user
    prompt sent (the next attempt's retry prompt builds on it); call: the model call (span "compose")."""

    parts: SkipValidation[list | None] = None
    prompt: str
    call: SkipValidation[LLMCall]
    meta: SkipValidation[dict[str, Any]] = Field(default_factory=dict)
    route: Literal["verify", "template"]
