"""The caution protocol (T1-28 S2b, D9 (a)): 'kind: detail' strings in Computed.cautions, written while a plan is
computed (compute/resolve.py, compute/semantics.py) and read into Caveat records (answer/caveats.py). The strings stay
as they are; making and parsing them is here.

    event_source: model (코로나19 = 2020-01)     event_source(name, start)
    default_applied: 가정별 = 중위 추계           default_applied(axis, label); DEFAULT_LATEST for '시점 = 최신'
    missing_years: 2019, 2020                   missing_years(periods)
    recent: 5개 중 3개 시점만 숫자가 있음          recent_short(n, found)
    simple_average: <items>의 시점 평균은 ...     simple_average(items)
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def make_caution(kind: str, detail: str) -> str:
    return f"{kind}: {detail}"


def event_source(name: str, start: str) -> str:
    """An event start the model gave (resolve.event_start)."""
    return make_caution("event_source", f"model ({name} = {start})")


def default_applied(axis: str, label: str) -> str:
    """A category (or the period) the plan left out and code filled in (resolve._default, resolve_select)."""
    return make_caution("default_applied", f"{axis} = {label}")


DEFAULT_LATEST = default_applied("시점", "최신")  # 'default_applied: 시점 = 최신': a select without periods


def missing_years(periods: Iterable[str]) -> str:
    """Asked periods the table lacks (resolve.resolve_periods)."""
    return make_caution("missing_years", ", ".join(periods))


def recent_short(n: int, found: int) -> str:
    """Fewer of the last n periods hold a number than asked (resolve_select)."""
    return make_caution("recent", f"{n}개 중 {found}개 시점만 숫자가 있음")


def simple_average(items: str) -> str:
    """An average over periods of a rate (semantics.average_caution)."""
    return make_caution("simple_average", f"{items}의 시점 평균은 가중치 없는 단순 평균입니다.")


# Readers (answer/caveats._from_caution).
EVENT_MODEL = re.compile(r"event_source: model \((?P<name>.+) = (?P<start>\d{4})(?:-(?P<month>\d{2}))?\)")
PREFIX = re.compile(r"^(\w+): (.*)$")


def parse_event_source(text: str) -> re.Match | None:
    """The whole caution as an event_source one: groups name, start (YYYY) and month (MM or None)."""
    return EVENT_MODEL.fullmatch(text)


def parse_caution(text: str) -> tuple[str, str] | None:
    """(kind, detail), or None when the text is not 'kind: detail'."""
    m = PREFIX.match(text)
    return m.groups() if m else None


def parse_default_applied(detail: str) -> tuple[str, str]:
    """(axis, label) of a default_applied detail."""
    axis, _, label = detail.partition(" = ")
    return axis, label


def parse_missing_years(detail: str) -> list[str]:
    """The periods of a missing_years detail."""
    return [y.strip() for y in detail.split(",")]
