"""Which kind (T1-32): the ChartSpec the builders made, its shape and the question's cue pick the kind; the points
are cut to a readable number and tied winners are all highlighted. Numbers are never changed.

The automatic table (no kind asked)
| the data (builder kind, pattern)              | drawn as                                                         |
|-----------------------------------------------|------------------------------------------------------------------|
| one target over periods (timeline)            | extreme / avg / sum: line; value: bars (the year highlighted); a time cue: line |
| one target over periods (line)                | line                                                             |
| 2+ targets x 2+ periods (the grid)            | more than GROUPED_MAX_PERIODS periods or a time cue: multi_line, else grouped_bar |
| the parts of one whole (share, pie)           | pie                                                              |
| a share at two periods (shr, %)               | stacked_bar (the share and the rest to 100%)                     |
| targets compared or ranked (bar)              | bar                                                              |
"""

from __future__ import annotations

from kosis_chat.chart.builders import BUILDERS, GROUPED_MAX_PERIODS, chart_spec

from kosis_ai.schemas import ChartSpec  # isort: skip

MAX_BARS = 10  # bars (a ranking) kept from the front; a highlighted or fact point is always kept
MAX_TIME_POINTS = 40  # periods kept from the most recent
TIMELINE_TO_LINE = ("extreme", "avg", "sum")
OWN_BUILDER = ("grw", "gap")  # draw the answer's numbers (the changes, the gaps), not the grid of levels


def base_spec(question_type, computed) -> tuple[ChartSpec | None, str | None]:
    """The builders' ChartSpec, or (None, why): no error in a builder stops the answer."""
    try:
        if computed.pattern in OWN_BUILDER:
            by_id = {f.id: f for f in computed.result.facts}
            spec = BUILDERS[computed.pattern](computed, by_id)
        else:
            spec = chart_spec(question_type or computed.pattern, computed)
    except Exception as e:  # noqa: BLE001 - the chart is an extra of the answer
        return None, f"builder_error:{type(e).__name__}"
    return spec, None if spec is not None else "builder_none"


def periods_of(spec: ChartSpec) -> list[str]:
    return list(dict.fromkeys(p.period or p.label for s in spec.series for p in s.points))


def auto_kind(spec: ChartSpec, cue: str | None) -> str:
    kind, pattern = spec.kind, spec.pattern or ""
    if kind == "timeline":
        return "line" if pattern in TIMELINE_TO_LINE or cue == "time" else "timeline"
    if kind in ("grouped_bar", "multi_line"):
        return "multi_line" if len(periods_of(spec)) > GROUPED_MAX_PERIODS or cue == "time" else "grouped_bar"
    if pattern == "shr" and kind == "bar":
        return "stacked_bar"
    return kind


def _keep(points, limit: int, from_end: bool):
    """limit points from the front (a ranking) or the end (recent periods), and every highlighted or fact point."""
    if len(points) <= limit:
        return points, False
    window = set(range(len(points) - limit, len(points))) if from_end else set(range(limit))
    kept = [p for n, p in enumerate(points) if n in window or p.highlight or p.fact_id is not None]
    return kept, True


def trim(spec: ChartSpec) -> tuple[ChartSpec, str | None]:
    """Cut a one-series chart with too many points (a grid is not cut); the note says what was cut."""
    if len(spec.series) != 1:
        return spec, None
    s = spec.series[0]
    if spec.kind in ("bar",):
        kept, cut = _keep(s.points, MAX_BARS, from_end=False)
        note = f"상위 {MAX_BARS}개만 그렸어요(전체 {len(s.points)}개)." if cut else None
    elif spec.kind in ("timeline", "line"):
        kept, cut = _keep(s.points, MAX_TIME_POINTS, from_end=True)
        note = f"최근 {MAX_TIME_POINTS}개 시점만 그렸어요(전체 {len(s.points)}개)." if cut else None
    else:
        return spec, None
    if not cut:
        return spec, None
    sort = [p.label for p in kept] if isinstance(spec.sort, list) else spec.sort
    return spec.model_copy(update={"series": [s.model_copy(update={"points": kept})], "sort": sort}), note


def mark_ties(spec: ChartSpec) -> ChartSpec:
    """In a ranking every point with the winner's value is highlighted (tied winners)."""
    if spec.pattern != "extreme" or len(spec.series) != 1:
        return spec
    pts = spec.series[0].points
    top = [p.value for p in pts if p.highlight]
    if not top or top[0] is None:
        return spec
    new = [p.model_copy(update={"highlight": True}) if p.value == top[0] and not p.highlight else p for p in pts]
    return spec.model_copy(update={"series": [spec.series[0].model_copy(update={"points": new})]})


def with_kind(spec: ChartSpec, kind: str) -> ChartSpec:
    if kind == spec.kind:
        return spec
    if kind == "line":  # a line reads x from period: bars labelled by period only (two periods compared) get it
        series = [s.model_copy(update={"points": [p if p.period else p.model_copy(update={"period": p.label})
                                                  for p in s.points]}) for s in spec.series]
        spec = spec.model_copy(update={"series": series})
    return spec.model_copy(update={"kind": kind, "vega_lite": None})


# chart_type values a caller may pass for the four kinds a user can ask for
ASKED = {"stacked_bar": "stacked", "timeline": "bar", "grouped_bar": "bar", "multi_line": "line"}


# The kind asked in words -> the kind of this shape (bars over periods stay a timeline, a grid stays a grid)
def requested_kind(requested: str, current: str) -> str:
    requested = ASKED.get(requested, requested)
    if requested == "bar":
        return {"line": "timeline", "multi_line": "grouped_bar", "stacked_bar": "bar", "pie": "bar"}.get(current, current if current in ("bar", "timeline", "grouped_bar") else "bar")
    if requested == "line":
        return {"timeline": "line", "grouped_bar": "multi_line", "bar": "line", "stacked_bar": "line"}.get(current, current if current in ("line", "multi_line") else "line")
    if requested == "stacked":
        return "stacked_bar"
    return "pie"
