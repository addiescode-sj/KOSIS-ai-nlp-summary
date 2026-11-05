"""What the chart stage reads and passes on (T1-28 S3a): the computed facts in, the shared Chart out.

ChartSpec (data + kind, Vega-Lite optional) replaces Chart in StatOutput with T1-28 S5; until then the step returns
the kosis_ai.schemas.Chart that StatAnswer.chart carries.
"""

from __future__ import annotations

from pydantic import SkipValidation

from kosis_chat.compute.schemas import Computed
from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import Chart, QuestionType  # isort: skip


class ChartIn(BoundaryModel):
    """step_chart's input: the plan's question type (the builder when the pattern names none) and the facts."""

    type: QuestionType | None
    computed: SkipValidation[Computed]


class ChartOut(BoundaryModel):
    """chart: kind, Vega-Lite v5 spec and the facts drawn; None when there is nothing to draw."""

    chart: SkipValidation[Chart | None] = None
