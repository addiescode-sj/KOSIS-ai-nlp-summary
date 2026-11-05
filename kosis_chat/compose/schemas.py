"""What the compose stage reads and passes on (T1-28 S1b, S3a).

ComposeOutput moved here unchanged from llm/compose.py (it was Pydantic already: the structured-output schema the
model is asked for); llm/compose.py re-exports it. The class name is the schema title (model_json_schema) and the
name in validation events (llm/structured.py), so it stays ComposeOutput. Caveat is the shared contract model
(kosis_ai.schemas.Caveat) and stays there.

S3a: the In/Out models of the pure compose steps (compose/node.py): caveats, verify, template. The model's sentence
(compose_llm) comes with S3b.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from kosis_chat.analyze.schemas import Built
from kosis_chat.compute.schemas import Alternatives, Computed
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import Caveat, QuestionType  # isort: skip

# Model sentences tried before the code template (pipeline._compose: the first answer and one retry).
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
