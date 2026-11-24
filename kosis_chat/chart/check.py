"""The shape check (T1-32): may this data be drawn as this kind? ok / skip (no chart) / fallback (another kind).

The automatic choice, a kind the user asked for and the grader all use check()."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from kosis_ai.schemas import ChartSpec  # isort: skip

TIME_KINDS = ("line", "timeline", "multi_line", "grouped_bar", "stacked_bar")
PERIOD_LABEL = re.compile(r"^(\d{4}(\d{2})?|\d{4}년( \d{1,2}월| [상하]반기| \d분기)?(~(\d{4}년)?( ?\d{1,2}월)?)?( 평균| 합계)?)$")
PIE_TOLERANCE = 1.5  # rounding (%p) a pie's shares may miss 100 by


@dataclass(frozen=True)
class Verdict:
    action: Literal["ok", "skip", "fallback"]
    reason: str | None = None
    kind: str | None = None  # the kind to draw instead (fallback)

    @staticmethod
    def ok() -> "Verdict":
        return Verdict("ok")


def has_time_axis(spec: ChartSpec) -> bool:
    return spec.kind in TIME_KINDS or _time_ordered(spec)


def _time_ordered(spec: ChartSpec) -> bool:
    """Every point is a period and the periods of a series differ (targets compared at one period are not a time axis
    though each point carries that period)."""
    for s in spec.series:
        periods = [p.period or (p.label if PERIOD_LABEL.match(p.label) else None) for p in s.points]
        if None in periods or len(set(periods)) != len(periods):
            return False
    return True


def _as_drawn(spec: ChartSpec) -> str:
    """The kind to fall back to from a pie or a stacked bar: the builders' own kind, else bars."""
    return spec.kind if spec.kind not in ("pie", "stacked_bar") else "bar"


def check(spec: ChartSpec, kind: str) -> Verdict:
    """May spec's data be drawn as kind."""
    pts = spec.points()
    # 1) no chart: nothing to show as a picture, or a number missing
    if len(pts) < 2:
        return Verdict("skip", "single_point")
    if any(p.value is None for p in pts):
        return Verdict("skip", "missing_value")
    if not any(p.fact_id for p in pts):
        return Verdict("skip", "no_fact_point")  # a picture without any of the answer's facts
    if len({p.label for p in pts}) == 1 and len(spec.series) == 1:
        return Verdict("skip", "same_label")
    # 2) what each kind needs
    if kind == "pie":
        if spec.pattern != "share":
            return Verdict("fallback", "pie_not_composition", _as_drawn(spec))
        if any(p.value < 0 for p in pts):
            return Verdict("fallback", "pie_negative", _as_drawn(spec))
        total = sum(p.share_pct or 0 for p in pts)
        if abs(total - 100) > PIE_TOLERANCE:
            return Verdict("fallback", f"pie_sum_{total:.1f}", _as_drawn(spec))
    if kind == "stacked_bar":
        if spec.unit != "%" or any(not 0 <= p.value <= 100 for p in pts):
            return Verdict("fallback", "stacked_not_percent", _as_drawn(spec))
    if kind in ("line", "multi_line"):
        instead = spec.kind if spec.kind not in ("line", "multi_line") else "bar" if kind == "line" else "grouped_bar"
        if not _time_ordered(spec):
            return Verdict("fallback", "line_without_time_axis", instead)
        per_series = min(len(s.points) for s in spec.series)
        if per_series < 2:
            return Verdict("fallback", "line_single_period", instead)
    if kind in ("grouped_bar", "multi_line") and len(spec.series) < 2:
        return Verdict("fallback", "grid_single_series", "timeline" if _time_ordered(spec) else "bar")
    return Verdict.ok()
