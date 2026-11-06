"""kosis_chat/shared/labels.py (T1-28 S2b): squash (two copies before), LABEL_ALIASES, item_label, fact_label."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kosis_chat.answer import render
from kosis_chat.data import scope
from kosis_chat.judge import question
from kosis_chat.shared.labels import LABEL_ALIASES, fact_label, item_label, squash


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 서울 특별시\t", "서울특별시"),
        ("1인 (가구)", "1인(가구)"),
        ("a\nb  c", "abc"),
        ("", ""),
        (None, ""),
    ],
)
def test_squash(text, expected):
    assert squash(text) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("1인 (가구)", "1인 가구"),
        ("5인 이상 (가구)", "5인 이상 가구"),
        ("1인", "1인"),
        ("출생건수 (명)", "출생건수"),
        ("조출생률(천명당) ", "조출생률"),
        ("일반가구_계", "일반가구 전체"),
        ("65세 이상", "65세 이상"),
        ("(명)", "(명)"),  # nothing left without the suffix: the label as it is
    ],
)
def test_item_label(label, expected):
    assert item_label(label) == expected


@pytest.mark.parametrize(
    ("dims", "skip_totals", "expected"),
    [
        ({"행정구역별": "서울특별시", "가구원수별": "1인 (가구)"}, True, "서울특별시 1인 가구"),
        ({"행정구역별": "계", "성별": "여자"}, True, "여자"),
        ({"행정구역별": "계", "성별": "여자"}, False, "계 여자"),
        ({"a": "합계 (명)", "b": "소계"}, True, ""),
        ({"a": "서울", "b": "서울"}, True, "서울"),
        ({}, True, ""),
        (None, True, ""),
    ],
)
def test_fact_label(dims, skip_totals, expected):
    assert fact_label(SimpleNamespace(dims=dims), skip_totals=skip_totals) == expected


def test_aliases_and_old_names():
    assert LABEL_ALIASES["전북특별자치도"] == ("전북", "전라북도")
    assert len(LABEL_ALIASES) == 10
    assert question.LABEL_ALIASES is scope.LABEL_ALIASES is LABEL_ALIASES
    assert question.squash is scope._squash is squash
    assert render.item_label is item_label and render.fact_label is fact_label
