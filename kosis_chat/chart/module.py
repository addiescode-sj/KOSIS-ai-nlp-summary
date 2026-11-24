"""The chart module's entry (T1-32): the question + the computed facts -> whether to draw, what, and why.

make_chart(question, computed, question_type) is what the chart node runs (node.py step_chart); redraw(spec,
chart_type) draws a chart the answer already has as another kind ("그 표로 막대그래프"). Both return a ChartResult.
chart_data(spec) is the drawn chart as JSON for the screen (Recharts draws it directly; KOS-202: no Vega-Lite).

make_chart, in order
1. the question (intent.read_intent): a chart asked, the kind asked, "그래프는 빼고", time / share words
2. "그래프는 빼고": no chart; nothing asked and auto_draw off: no chart
3. the data (decide.base_spec: the builders; an error in a builder is no chart, never a failed answer)
4. tied winners highlighted (decide.mark_ties)
5. the kind: the one asked (decide.requested_kind), else the table (decide.auto_kind)
6. the check (check.check): skip is no chart, fallback is another kind, checked once more
7. too many points cut (decide.trim)
The numbers are the facts' own (the ChartSpec points): this module changes the kind, the number of points and the
highlight only.
"""

from __future__ import annotations

from kosis_chat.chart.check import check
from kosis_chat.chart.decide import auto_kind, base_spec, mark_ties, requested_kind, trim, with_kind
from kosis_chat.chart.intent import read_intent
from kosis_chat.chart.schemas import ChartIntent, ChartResult
from kosis_chat.compute import Computed
from kosis_chat.shared.numbers import format_number

from kosis_ai.schemas import CHART_KINDS, ChartPoint, ChartSeries, ChartSpec, QuestionType  # isort: skip

KIND_NAMES = {"bar": "막대그래프", "timeline": "막대그래프", "line": "꺾은선 그래프", "multi_line": "꺾은선 그래프",
              "grouped_bar": "묶음 막대그래프", "pie": "원형 차트", "stacked_bar": "누적 막대그래프"}
WHY = {  # why the asked kind was not drawn (a sentence for the user)
    "pie_not_composition": "원형 차트는 한 전체를 나눈 부분(구성비)일 때만 그려요",
    "pie_negative": "음수가 있어 원형 차트로 그릴 수 없어요",
    "stacked_not_percent": "누적 막대그래프는 비중(%)일 때만 그려요",
    "line_without_time_axis": "꺾은선 그래프는 시점이 이어지는 값에만 그려요",
    "line_single_period": "시점이 하나뿐이라 꺾은선으로 이을 수 없어요",
    "grid_single_series": "비교할 대상이 하나뿐이에요",
}
SKIP_TEXT = {
    "single_point": "값이 하나뿐이라 그래프 대신 숫자로 보여 드려요.",
    "missing_value": "비어 있는 값이 있어 그래프를 그리지 않았어요.",
    "no_fact_point": "답에 쓴 숫자가 그래프에 없어 그리지 않았어요.",
    "same_label": "같은 항목 하나뿐이라 그래프를 그리지 않았어요.",
}
NO_DATA_TEXT = "이 답에는 그릴 수 있는 차트 데이터가 없어요."
REST_NAME = "그 외"  # the rest of a whole: the pie's rest slice (builders._share) and a stacked bar's upper part


def _skipped(intent: ChartIntent, reasons: list[str], note: str | None = None) -> ChartResult:
    return ChartResult(drawn=False, decision="skipped", intent=intent, reasons=reasons, note=note)


def make_chart(question: str | None, computed: Computed, question_type: QuestionType | None = None, *,
               auto_draw: bool = True) -> ChartResult:
    """The chart of an answer. auto_draw: draw when the data can be drawn though no chart was asked (the default)."""
    intent = read_intent(question)
    if intent.suppress:
        return _skipped(intent, ["user_suppressed"])
    if not intent.requested and not auto_draw:
        return _skipped(intent, ["not_requested"])
    spec, why = base_spec(question_type, computed)
    if spec is None:
        return _skipped(intent, [why or "no_spec"], NO_DATA_TEXT if intent.requested else None)
    return _decide(mark_ties(spec), intent, intent.kind)


def redraw(spec: ChartSpec | None, chart_type: str | None = None, question: str | None = None) -> ChartResult:
    """A chart the answer already has, as the kind asked (bar | line | pie | stacked, or a ChartKind); None keeps the
    chart's kind. Same check and notes as make_chart."""
    intent = read_intent(question)
    if spec is None:
        return _skipped(intent, ["no_spec"], NO_DATA_TEXT)
    spec = spec.model_copy(update={"vega_lite": None})
    if chart_type is None:
        return _decide(spec, intent, None, keep=True)
    return _decide(spec, intent, chart_type)


def _decide(spec: ChartSpec, intent: ChartIntent, asked: str | None, *, keep: bool = False) -> ChartResult:
    reasons: list[str] = []
    notes: list[str] = []
    if asked is not None:
        kind, decision = requested_kind(asked, auto_kind(spec, intent.cue)), "requested"
    else:
        kind, decision = (spec.kind if keep else auto_kind(spec, intent.cue)), "auto"
    reasons.append(f"{decision}:{spec.kind}->{kind}")

    verdict = check(spec, kind)
    if verdict.action == "fallback":  # one other kind, checked once more
        reasons.append(f"fallback:{verdict.reason}")
        if decision == "requested":
            notes.append(f"{WHY.get(verdict.reason, '요청한 종류로 그릴 수 없어요')}. 그래서 "
                         f"{KIND_NAMES.get(verdict.kind, verdict.kind)}로 그렸어요.")
        kind, decision = verdict.kind, "fallback"
        verdict = check(spec, kind)
    if verdict.action != "ok":
        reasons.append(f"skip:{verdict.reason}")
        return _skipped(intent, reasons, SKIP_TEXT.get(verdict.reason) if intent.requested or asked else None)

    drawn, cut = trim(with_kind(spec, kind))
    if cut:
        reasons.append("trimmed")
        notes.append(cut)
    if kind not in CHART_KINDS:  # a kind no screen draws
        reasons.append(f"no_drawing:{kind}")
        return _skipped(intent, reasons, NO_DATA_TEXT if intent.requested or asked else None)
    return ChartResult(drawn=True, decision=decision, kind=kind, spec=drawn, intent=intent, reasons=reasons,
                       note=" ".join(notes) or None)


def chart_data(spec: ChartSpec | None) -> dict | None:
    """The drawn chart as JSON (the ChartSpec without vega_lite) for the screen, which draws it directly (Recharts).
    A stacked bar gets its rest (100 - share) as a second series here, so the screen computes no number."""
    if spec is None:
        return None
    if spec.kind == "stacked_bar":
        part = spec.title.removesuffix(" 비교").removesuffix(" 비중").strip() or "해당"
        rest = [ChartPoint(label=p.label, value=r, display=format_number(r, "%", 1), period=p.period)
                for p in spec.points() for r in [round(100 - p.value, 6)]]
        spec = spec.model_copy(update={"series": [ChartSeries(name=part, points=list(spec.points())),
                                                  ChartSeries(name=REST_NAME, points=rest)]})
    return spec.model_dump(mode="json", exclude={"vega_lite"})
