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

# Loop 15: a role in a family written as one Hanja character (a parent of a child, a spouse, a child of a parent) and
# the everyday words for it: the plain word, the familiar one and the two-character Hanja word. The one character
# itself is not matched (judge/question.py _found takes two characters or more); a word for both roles ('부모') is
# neither's.
FAMILY_ROLES = {
    "부": ("아버지", "아빠", "부친"),
    "모": ("어머니", "엄마", "모친"),
    "처": ("아내", "부인"),
    "자": ("아들",),
}
LABEL_ALIASES.update(FAMILY_ROLES)

# Loop 15: the describing endings of a 하다 verb ('취업한', '취업하는', '취업했던'): a name for the people who did a
# thing ('취업자', '귀농인') is also said by its stem with one of them (judge/question.py doing_forms).
DOING_ENDINGS = ("한", "하는", "하던", "했던")

# Loop 15: a category of nations, read on its name ('아내 국적별', '국가별', '출신 나라별'), and a word for a nation or
# its people in a category's name or a table's title ('한국인', '한국', '한국인과의'): group 1 the nation. The words
# for the nations other than one ('외국인', '타국') name no nation.
NATION_AXIS = re.compile(r"국적|국가|나라")
NATION_WORD = re.compile(r"^(?!외국|타국)([가-힣]+국)인?(?:과의|와의|의|과|와|은|는|이|가)?$")


# Loop 14: the tail of a name of a place people attend, and the word for the people who attend it: a name ending
# with the tail is also said by its members ('중학교' as '중학생', '대학교' as '대학생', '대학원' as '대학원생'). The
# last character of the member tail is the person suffix of the three-character shortenings ('남고생': a sex, the
# first character of the name, the suffix). Longest tail first.
MEMBER_TAILS = (
    ("대학원", "대학원생"),
    ("유치원", "유치원생"),
    ("학교", "학생"),
    ("대학", "대학생"),
    ("학원", "학원생"),
)
# Loop 14: names of the two sexes; their first character is put before a noun for people ('남학생', '여고생').
SEX_NAMES = ("남자", "여자", "남성", "여성", "남", "여")


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


def interval_bounds(core: str) -> tuple[int, int, str] | None:
    """Loop 14: (low, high, counting word) of an interval name with nothing before or after it ('20-24세', '20 - 24세'
    -> (20, 24, '세')), None for another name. The one reading of an interval's bounds interval_forms uses too."""
    m = _INTERVAL.match(squash(core))
    if m is None or m.group(1) or m.group(5):
        return None
    return int(m.group(2)), int(m.group(3)), m.group(4)


# Loop 14: a ten group said as one word ('20대', '30 대'): not a decade of years ('2010년대'), not a half of one ('20대
# 초반', an interval name's form above), not a part of a larger number.
TEN_GROUP = re.compile(r"(?<![\d.])([1-9]0)\s*대(?!\s*(?:초반|전반|중반|후반))")


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
