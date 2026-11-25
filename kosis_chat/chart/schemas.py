"""What the chart stage reads and passes on (T1-28 S3a): the computed facts in, the shared ChartSpec out (T1-28 S5:
data + kind, Vega-Lite optional; it is StatOutput.chart, no longer a StatAnswer field).
T1-32: the question comes in too (ChartIn.question: asked kind, "그래프는 빼고", time words), and chart/module.py
make_chart / redraw return a ChartResult (what was drawn, why, a note for the user)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from kosis_chat.compute.schemas import Computed
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import ChartSpec, QuestionType  # isort: skip


class ChartIn(BoundaryModel):
    """step_chart's input: the plan's question type (the builder when the pattern names none) and the facts."""

    type: QuestionType | None
    computed: SkipValidation[Computed]
    question: str | None = None  # the user's sentence (None: no asked kind, drawn by the data's shape alone)


class ChartOut(BoundaryModel):
    """chart: kind, the drawn points and the facts drawn (no Vega-Lite); None when there is nothing to draw."""

    chart: SkipValidation[ChartSpec | None] = None
    note: str | None = None  # a sentence for the user when the chart is not what was asked or was cut (T1-32)


# The kinds a user can ask for in words (stacked = 누적 막대)
RequestKind = Literal["bar", "line", "pie", "stacked"]
# How the kind was chosen: by the table below / as asked / another kind after the check / not drawn
Decision = Literal["auto", "requested", "fallback", "skipped"]


class ChartIntent(BaseModel):
    """What the question says about a chart (chart/intent.py read_intent; rules, no model)."""

    model_config = ConfigDict(frozen=True)

    requested: bool = False  # 그래프·차트·그려 ...
    kind: RequestKind | None = None  # 막대·꺾은선·원형·누적
    suppress: bool = False  # "그래프는 빼고", "표로만", "숫자만"
    cue: Literal["time", "composition", "ranking", "compare"] | None = None  # 추이·비중·순위·비교


class ChartResult(BaseModel):
    """What make_chart / redraw drew: spec is the ChartSpec drawn (kind, points; no Vega-Lite, KOS-202: a screen draws
    the ChartSpec itself, chart_data), reasons the code's decisions (logs, grading), note one sentence for the user."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    drawn: bool
    decision: Decision
    kind: str | None = None
    spec: SkipValidation[ChartSpec | None] = None
    intent: ChartIntent = Field(default_factory=ChartIntent)
    reasons: list[str] = Field(default_factory=list)
    note: str | None = None
