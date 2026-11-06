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
