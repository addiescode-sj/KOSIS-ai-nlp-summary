"""kosis_chat/shared/numbers.py (T1-28 S2b): the number helpers moved from answer/render.py and data/serializer.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kosis_chat.answer import render
from kosis_chat.compute import engine
from kosis_chat.data import serializer
from kosis_chat.shared.numbers import (
    SYMBOL_MEANING,
    derived_kind,
    direction,
    display_value,
    format_fact,
    format_number,
    is_change,
)


def fact(value=None, raw=None, derivation=None, unit=None, round=None):  # noqa: A002  (the Fact field name)
    return SimpleNamespace(value=value, raw=raw, derivation=derivation, unit=unit, round=round)


@pytest.mark.parametrize(
    ("value", "unit", "digits", "expected"),
    [
        (123456, "가구", None, "123,456가구"),
        (3.0, "명", None, "3명"),
        (1.5, None, None, "1.5"),
        (0.123456789, "", None, "0.1235"),  # a cell value keeps at most 4 decimals
        (12.3, "%", 1, "12.3%"),
        (1234.5, "원", 0, "1,234원"),
        (7, "배", 2, "7.00배"),
    ],
)
def test_format_number(value, unit, digits, expected):
    assert format_number(value, unit, digits) == expected


@pytest.mark.parametrize(
    ("f", "expected"),
    [
        (fact(value=None, raw="x"), "'x'(비공개)"),
        (fact(value=None, raw=None), "'-'(자료 없음)"),
        (fact(value=None, raw="?"), "'?'(수치 아님)"),
        (fact(value=-5.0, derivation="f4 - f1", unit="명", round=0), "5명"),
        (fact(value=-2.36, derivation="(f4 / f1 - 1) * 100", unit="%", round=1), "2.4%"),
        (fact(value=-5.0, derivation="f1 + f2", unit="명", round=0), "-5명"),  # not a change: the sign stays
        (fact(value=65975, unit="가구"), "65,975가구"),
    ],
)
def test_format_fact(f, expected):
    assert format_fact(f) == expected


@pytest.mark.parametrize(
    ("derivation", "change", "kind"),
    [
        ("f4 - f1", True, "차이"),
        (" f4 - f1 ", True, "차이"),
        ("(f4 / f1 - 1) * 100", True, "증감률"),
        ("f1 + f2 + f3", False, "합계"),
        ("(f1 + f2) / 2", False, "평균"),
        ("f1 / f2 * 100", False, "비중"),
        ("f1 / f2", False, "배수"),
        (None, False, "차이"),
    ],
)
def test_change_and_kind(derivation, change, kind):
    assert is_change(fact(derivation=derivation)) is change
    assert derived_kind(fact(derivation=derivation)) == kind


@pytest.mark.parametrize(("value", "expected"), [(None, "같음"), (0, "같음"), (3, "증가"), (-1.5, "감소")])
def test_direction(value, expected):
    assert direction(fact(value=value)) == expected


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("65975.0000", "65975"),
        ("12.5000", "12.5"),
        (" 1,234.50 ", "1,234.5"),
        ("-3.000", "-3"),
        ("10.0", "10"),
        ("100", "100"),
        ("12.3p", "12.3p"),
        ("-", "-"),
        ("X", "X"),
    ],
)
def test_display_value(raw, shown):
    assert display_value(raw) == shown


def test_old_names_are_the_same_objects():
    assert render.format_number is format_number and render.format_fact is format_fact
    assert render.is_change is is_change and render.direction is direction and render.derived_kind is derived_kind
    assert render.SYMBOL_MEANING is SYMBOL_MEANING
    assert serializer.display_value is engine.display_value is display_value
