"""What the render stage reads and returns (T1-28 S3a): everything the finished StatAnswer is made of.

RenderIn: an answered question (facts, parts, caveats, chart). AnswerFinalIn: an answer without facts: the judge's
refusal or ask-back, the calculation's refusal (CalcRefused), or the model plan's ask-back (NeedsClarify).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SkipValidation

from kosis_chat.compute.schemas import Computed
from kosis_chat.judge.schemas import Verdict
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import Caveat, Chart, Refusal, Source, StatAnswer, StatQuery  # isort: skip


class RenderIn(BoundaryModel):
    """step_render's input. source is made once per question before compute (to_source), as the pipeline did."""

    question: str
    query: SkipValidation[StatQuery]
    computed: SkipValidation[Computed]
    parts: SkipValidation[list]
    verified: bool  # the parts shown passed verify() (StatAnswer.numbers_verified)
    source: SkipValidation[Source]
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)
    caveats: SkipValidation[list[Caveat]] = Field(default_factory=list)
    shown: SkipValidation[list[str]] = Field(default_factory=list)  # caveat texts written as '참고:' lines
    context_ids: SkipValidation[list[str]] = Field(default_factory=list)
    chart: SkipValidation[Chart | None] = None


class AnswerFinalIn(BoundaryModel):
    """step_answer_final's input; kind says which field holds the answer:
    "verdict": the judge's final verdict (refused or clarify); source is made here (none for out_of_scope).
    "refused": the calculation's refusal (ComputeOut.refusal) with its query and the source made before compute.
    "clarify": the model plan's ask-back (compute.resolve.NeedsClarify: axis, options as (code, label), message)."""

    kind: Literal["verdict", "refused", "clarify"]
    question: str
    verdict: SkipValidation[Verdict | None] = None
    refusal: SkipValidation[Refusal | None] = None
    query: SkipValidation[StatQuery | None] = None
    source: SkipValidation[Source | None] = None
    clarify: Any = None  # NeedsClarify (an exception of the compute stage)


class RenderOut(BoundaryModel):
    """The finished answer (StatAnswer, its text included)."""

    stat: SkipValidation[StatAnswer]
