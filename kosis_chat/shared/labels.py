"""Item and category labels: squashing for matching, the words people use for a label, and labels as shown (T1-28 S2b).

- squash: was judge/question.squash and data/scope._squash (the same body); both names are aliases now.
- LABEL_ALIASES: was data/scope.py; judge/question.py read it from there (the judge importing the B0 scope module).
- item_label, fact_label: were answer/render.py (re-exported there); compose, chart and render all show labels so.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from kosis_chat.shared.patterns import UNIT_PAREN

if TYPE_CHECKING:
    from kosis_ai.schemas import Fact

# Words people use for a label; matched without spaces.
LABEL_ALIASES = {
    "여자": ("여성",),
    "남자": ("남성",),
    "충청북도": ("충북",),
    "충청남도": ("충남",),
    "전라북도": ("전북",),
    "전북특별자치도": ("전북", "전라북도"),
    "전라남도": ("전남",),
    "경상북도": ("경북",),
    "경상남도": ("경남",),
    "세종특별자치시": ("세종시",),
}


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


# Loop 6: an interval name, a low and a high bound joined by a mark and followed by a counting word ('20-24세',
# '5~9인', '100∼199명'), and the other ways people join the same two bounds: another mark (-, ~, ∼), the counting
# word after each bound ('20세~24세'), "A에서 B까지" / "A부터 B까지" ('20세부터 24세까지').
_INTERVAL = re.compile(r"^(.*?)(\d+)\s*[-~∼]\s*(\d+)\s*([가-힣]{1,3})(.*)$")
_JOINS = ("-", "~", "∼")
_FROM_TO = ("에서", "부터")
# The halves of a ten-year group ("20대 초반" = 20-24, "30대 후반" = 35-39): the first half's words and the second's.
HALF_OF_TEN = {"초반": 0, "전반": 0, "후반": 5}


def interval_forms(core: str) -> set[str]:
    """Other squashed forms of an interval name ('20-24세' -> '20~24세', '20세~24세', '20세에서24세', ..., '20대초반'),
    none for another name."""
    m = _INTERVAL.match(squash(core))
    if m is None:
        return set()
    head, low, high, unit, tail = m.groups()
    out = {f"{head}{low}{j}{high}{unit}{tail}" for j in _JOINS}
    out |= {f"{head}{low}{unit}{j}{high}{unit}{tail}" for j in (*_JOINS, *_FROM_TO)}
    a, b = int(low), int(high)
    if b - a == 4 and a % 5 == 0 and a >= 10:  # a half of a ten-year group: 20-24 (초반, 전반), 25-29 (후반)
        out |= {f"{head}{a - a % 10}대{word}{tail}" for word, start in HALF_OF_TEN.items() if start == a % 10}
    return out


_HOUSEHOLD = re.compile(r"\d+인(?: 이상)?")


def item_label(label: str) -> str:
    """'1인 (가구)' -> '1인 가구', '출생건수 (명)' -> '출생건수', '일반가구_계' -> '일반가구 전체' (T1-14)."""
    core = UNIT_PAREN.sub("", label).strip() or label.strip()
    if core.endswith("_계"):
        return core[:-2] + " 전체"
    if _HOUSEHOLD.fullmatch(core) and "(가구)" in label:
        return f"{core} 가구"
    return core


def fact_label(fact: Fact, skip_totals: bool = True) -> str:
    """'서울특별시 1인' for a cell fact (categories, then the item); totals such as '계' are left out."""
    if not fact.dims:
        return ""
    words = []
    for name, label in fact.dims.items():
        label = item_label(label)
        if skip_totals and re.fullmatch(r"(?:전체|계|합계|총계|소계)", label):
            continue
        words.append(label)
    return " ".join(dict.fromkeys(words))
