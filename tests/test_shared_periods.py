"""kosis_chat/shared/periods.py (T1-28 S2b): period_text and span_text moved from answer/render.py, unchanged."""

from __future__ import annotations

import pytest

from kosis_chat.answer import render
from kosis_chat.judge import answerability
from kosis_chat.shared.periods import period_text, span_text


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        ("2024", "2024년"),
        ("202406", "2024년 6월"),
        ("202412", "2024년 12월"),
        ("2024A", "2024A년"),  # not a month: as it is, with 년
        ("abcdef", "abcdef년"),  # six characters but not digits
        ("", "년"),
    ],
)
def test_period_text(period, expected):
    assert period_text(period) == expected


@pytest.mark.parametrize(
    ("first", "last", "expected"),
    [
        ("2024", "2024", "2024년"),
        ("2019", "2023", "2019년~2023년"),
        ("202401", "202406", "2024년 상반기"),
        ("202407", "202412", "2024년 하반기"),
        ("202501", "202503", "2025년 1분기"),
        ("202504", "202506", "2025년 2분기"),
        ("202410", "202412", "2024년 4분기"),
        ("202402", "202404", "2024년 2월~4월"),  # three months, not a quarter
        ("202401", "202405", "2024년 1월~5월"),
        ("202311", "202402", "2023년 11월~2024년 2월"),
        ("2023", "202401", "2023년~2024년 1월"),
    ],
)
def test_span_text(first, last, expected):
    assert span_text(first, last) == expected


def test_old_names_are_the_same_functions():
    assert render.period_text is period_text
    assert render.span_text is span_text
    assert answerability._period_text is period_text  # was a second copy of the same body
