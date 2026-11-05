"""What the judge stage passes on (T1-28 S1b): the parsed question and the verdict, as Pydantic models.

Moved from judge/question.py (Periods, Match, Parsed) and judge/answerability.py (Verdict, Resolved); the old modules
re-export them. Same fields, order, defaults and frozenness as the dataclasses; see shared/boundary.py for how
validation is kept from changing a value. TableIndex and Axis stay dataclasses (D5 a: large table data is not a
node boundary); since T1-28 S2 they live in prepare/index.py, re-exported by judge/question.py.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field, SkipValidation

from kosis_chat.shared.boundary import BoundaryModel

from kosis_ai.schemas import Outcome, Refusal  # isort: skip


def in_range(period: str, start: str, end: str) -> bool:
    """Whether a table period ('2024', '202406') falls in a year range ('2020', '2024') or a month range
    ('202401', '202406'); a yearly period never falls in a month range."""
    if len(start) == 6:
        return len(period) == 6 and start <= period <= end
    return start <= period[:4] <= end


# ---------------------------------------------------------------- judge/question.py


class Periods(BoundaryModel):
    """Periods named in the question: single years ('2024') or months ('202401'), year or month ranges, or "최근".

    spans: ranges that name one aggregate period (a half or a quarter of a year), in question order. On a monthly
    table a value asked for such a span is the sum of its months. previous: "작년 대비" (the latest period against
    the one before it). before: "10년 전", years before the named year, already turned into a single."""

    model_config = ConfigDict(frozen=True)

    singles: SkipValidation[tuple[str, ...]] = ()
    ranges: SkipValidation[tuple[tuple[str, str], ...]] = ()
    recent_n: int | None = None
    recent: bool = False
    spans: SkipValidation[tuple[tuple[str, str], ...]] = ()
    previous: bool = False
    before: int | None = None
    events: SkipValidation[tuple[tuple[str, str], ...]] = ()  # (event phrase, event start) a period was read from

    @property
    def explicit(self) -> bool:
        return bool(self.singles or self.ranges)

    def contains(self, period: str) -> bool:
        if not self.explicit:
            return True
        if any(period == s or (len(s) == 4 and period[:4] == s) for s in self.singles):
            return True
        return any(in_range(period, a, b) for a, b in self.ranges)


class Match(BoundaryModel):
    model_config = ConfigDict(frozen=True)

    axis: str
    code: str
    label: str
    form: str  # the squashed text found in the question
    exact: bool  # the whole label (minus a unit suffix) is in the question


class Parsed(BoundaryModel):
    question: str
    periods: SkipValidation[Periods]
    matches: SkipValidation[list[Match]]
    rest: str  # the question with matched label text removed, for operation words
    words: SkipValidation[tuple[str, ...]] = ()  # question words that are not matched labels, particles cut off

    def by_axis(self) -> dict[str, list[Match]]:
        out: dict[str, list[Match]] = {}
        for m in self.matches:
            out.setdefault(m.axis, []).append(m)
        return out


# ---------------------------------------------------------------- judge/answerability.py


class Verdict(BoundaryModel):
    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    rule: str | None = None  # the rule that decided; None when no rule fired
    refusal: SkipValidation[Refusal | None] = None
    clarify_question: str = ""
    clarify_options: SkipValidation[tuple[str, ...]] = ()
    default_applied: SkipValidation[dict[str, str]] = Field(default_factory=dict)
    final: bool = False  # True: use this outcome and skip the LLM
    # Verdict.refuse(rule, reason, ...) of the dataclass is judge/answerability.refuse(...) now: it writes the refusal
    # sentence with judge/messages.py, an old path this module must not import (tests/test_layout.py OLD_PATHS).


class Resolved(BoundaryModel):
    codes: SkipValidation[dict[str, set[str]]]  # axis -> codes the question names
    ambiguous: SkipValidation[list[tuple[str, list[str]]]]  # (label, options) for a label several parents share
