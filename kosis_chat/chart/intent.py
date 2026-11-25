"""The question -> what it says about a chart (T1-32). Rules only, no model. The chart words are Phase 3
graph/context.py CHART_WORDS / CHART_KINDS, with 누적 (stacked), a refusal ("그래프는 빼고") and the question's cues added."""

from __future__ import annotations

import re

from kosis_chat.chart.schemas import ChartIntent

CHART_WORDS = re.compile(r"그래프|차트|시각화|그려|그림으로|막대|꺾은선|선\s*그래프|원형|파이|도표")
# 먼저 맞는 것이 이긴다 ("누적 막대"는 누적)
KIND_WORDS = [
    (re.compile(r"누적\s*(막대|바|그래프|차트)"), "stacked"),  # "누적 확진자" is a number, not a chart
    (re.compile(r"막대|바\s*차트|바\s*그래프"), "bar"),
    (re.compile(r"꺾은선|선\s*그래프|라인|(?<![가-힣])선(?=으로|\s*차트)"), "line"),
    (re.compile(r"원형|파이|도넛"), "pie"),
]
SUPPRESS = re.compile(
    r"(그래프|차트|그림|시각화)\s*(는|은|도)?\s*(빼|없이|말고|필요\s*없|안\s*그려|그리지\s*마)"
    r"|표로만|숫자만|글로만|텍스트로만|수치만"
)
CUES = [  # 질문 단서: 종류를 직접 말하지 않았을 때 데이터 모양이 애매하면 쓴다
    (re.compile(r"추이|변화|흐름|추세|증감|늘었|줄었|최근\s*\d+\s*년|연도별|해마다|매년|월별|분기별"), "time"),
    (re.compile(r"비중|구성|차지|비율\s*중|점유"), "composition"),
    (re.compile(r"순위|가장|제일|최고|최저|상위|하위|몇\s*위"), "ranking"),
    (re.compile(r"비교|차이|보다|격차|와\s*.+\s*중"), "compare"),
]


def read_intent(question: str | None) -> ChartIntent:
    q = question or ""
    suppress = bool(SUPPRESS.search(q))
    requested = bool(CHART_WORDS.search(q)) and not suppress
    kind = next((k for pattern, k in KIND_WORDS if pattern.search(q)), None) if requested else None
    cue = next((c for pattern, c in CUES if pattern.search(q)), None)
    return ChartIntent(requested=requested, kind=kind, suppress=suppress, cue=cue)
