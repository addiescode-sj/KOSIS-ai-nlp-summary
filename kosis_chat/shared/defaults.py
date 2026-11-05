"""The default_applied protocol (T1-28 S2b, D9 (a)): the strings stay as they are; writing and reading them is here.

The judge (judge/answerability.defaults) records the period a question left open as one of three strings under the
key "period" (other keys hold a label as it is, e.g. the projection scenario '중위 추계'):

    latest (2024)                   make_latest("2024")
    latest (2024) vs previous (2023)  make_previous("2024", "2023")
    recent 5 points (2020~2024)     make_recent(["2020", ..., "2024"])

Two readers with different needs, kept apart (their patterns capture differently, D16 (d)):
- parse_period: the whole value, any text in the parentheses (judge/messages._default_text, answer/render
  .default_sentence: the sentence that states the condition). Was judge/messages._LATEST/_RECENT/_PREVIOUS.
- parse_period_codes: period codes only, found anywhere in the value (compute/analyze._periods: which table periods
  the plan reads). Was compute/analyze._LATEST/_RECENT/_PREVIOUS.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

PERIOD_KEY = "period"
# How a key is named in a sentence (judge/messages._DEFAULT_KEYS).
DEFAULT_KEYS = {PERIOD_KEY: "시점"}


def make_latest(period: str) -> str:
    return f"latest ({period})"


def make_previous(latest: str, previous: str) -> str:
    return f"latest ({latest}) vs previous ({previous})"


def make_recent(points: Sequence[str]) -> str:
    return f"recent {len(points)} points ({points[0]}~{points[-1]})"


# Whole-value forms (fullmatch).
LATEST = re.compile(r"latest \((.+)\)")
RECENT = re.compile(r"recent (\d+) points \((.+)\)")
PREVIOUS = re.compile(r"latest \((.+)\) vs previous \((.+)\)")

# Period-code forms (search).
LATEST_CODES = re.compile(r"latest[^(]*\((\d{4,6})")
RECENT_CODES = re.compile(r"recent (\d+) points \((\d{4,6})~(\d{4,6})\)")
PREVIOUS_CODES = re.compile(r"latest \((\d{4,6})\) vs previous \((\d{4,6})\)")


def parse_period(value: str) -> tuple[str, tuple[str, ...]] | None:
    """("previous", (latest, previous)) | ("latest", (period,)) | ("recent", (n, "first~last")) when the whole value
    is one of the three forms, tried in that order; else None."""
    for kind, pattern in (("previous", PREVIOUS), ("latest", LATEST), ("recent", RECENT)):
        if m := pattern.fullmatch(value):
            return kind, m.groups()
    return None


def parse_period_codes(value: str) -> tuple[str, tuple[str, ...]] | None:
    """("previous", (latest, previous)) | ("recent", (n, first, last)) | ("latest", (period,)) for the first form
    found in the value, tried in that order; else None."""
    for kind, pattern in (("previous", PREVIOUS_CODES), ("recent", RECENT_CODES), ("latest", LATEST_CODES)):
        if m := pattern.search(value):
            return kind, m.groups()
    return None
