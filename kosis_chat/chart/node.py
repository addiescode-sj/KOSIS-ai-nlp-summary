"""chart node (T1-28 S3a): the answer's chart, built by code from the computed facts.

A pure step: (ChartIn, StatResources) -> ChartOut. StatPipeline.ask calls it (no span, as before).
"""

from __future__ import annotations

from kosis_chat.chart.module import make_chart
from kosis_chat.chart.schemas import ChartIn, ChartOut
from kosis_chat.schemas.resources import StatResources


def step_chart(inp: ChartIn, res: StatResources) -> ChartOut:
    """The chart for the computed facts and the question (chart/module.py make_chart, T1-32: the builders' data, the
    kind by the data's shape or as asked, the shape check).

    In: ChartIn(type, computed, question)
    Out: ChartOut(chart: ChartSpec (kind, points, fact ids; no Vega-Lite); None when nothing is drawn, note)
    Side effects: none
    Node: chart (no span)
    """
    out = make_chart(inp.question, inp.computed, inp.type)
    return ChartOut(chart=out.spec, note=out.note)
