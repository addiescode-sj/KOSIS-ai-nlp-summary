"""What the chart stage reads and passes on (T1-28 S3a): the computed facts in, the shared ChartSpec out (T1-28 S5:
data + kind, Vega-Lite optional; it is StatOutput.chart, no longer a StatAnswer field)."""

from __future__ import annotations

from pydantic import SkipValidation

from kosis_chat.compute.schemas import Computed
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import ChartSpec, QuestionType  # isort: skip


class ChartIn(BoundaryModel):
    """step_chart's input: the plan's question type (the builder when the pattern names none) and the facts."""

    type: QuestionType | None
    computed: SkipValidation[Computed]


class ChartOut(BoundaryModel):
    """chart: kind, the drawn points and the facts drawn (no Vega-Lite); None when there is nothing to draw."""

    chart: SkipValidation[ChartSpec | None] = None
