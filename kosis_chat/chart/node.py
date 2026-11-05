"""chart node (T1-28 S3a): the answer's chart, built by code from the computed facts.

A pure step: (ChartIn, StatResources) -> ChartOut. StatPipeline.ask calls it (no span, as before).
"""

from __future__ import annotations

from kosis_chat.chart.builders import chart_spec
from kosis_chat.chart.schemas import ChartIn, ChartOut
from kosis_chat.schemas.resources import StatResources


def step_chart(inp: ChartIn, res: StatResources) -> ChartOut:
    """The chart for the computed facts (chart/builders.py: a grid when the cells form one, else by pattern).

    In: ChartIn(type, computed)
    Out: ChartOut(chart: kind, Vega-Lite spec, fact ids; None when there is nothing to draw)
    Side effects: none
    Node: chart (no span)
    """
    return ChartOut(chart=chart_spec(inp.type, inp.computed))
