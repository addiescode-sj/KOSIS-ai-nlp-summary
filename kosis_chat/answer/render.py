"""Numbers in the answer are written by code (T1-11): a FactPart becomes '123,456가구', '12.3%', '1.5배' or a symbol.

render_parts joins the parts; final_text adds the lines code always writes (default conditions, notes, source).
Those code lines are not parts: parts are what the model chose and what the grader checks for unsupported numbers.
"""

from __future__ import annotations

import re

from kosis_chat.judge import messages

from kosis_ai.schemas import Fact, FactPart, Source, StatResult, TextPart  # isort: skip

SYMBOL_MEANING = {"-": "자료 없음", "x": "비공개", "X": "비공개", "…": "미상", "...": "미상", "*": "통계 미작성"}
_UNIT_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")


def _number(value: float, digits: int | None) -> str:
    if digits is None:  # a cell value: as many decimals as it has (stored values never carry float noise)
        digits = 0 if float(value).is_integer() else min(len(repr(float(value)).split(".")[1]), 4)
    return f"{value:,.{digits}f}"


def format_fact(fact: Fact) -> str:
    """'123,456가구'; a change without its sign (the sentence says 늘었다/줄었다); a symbol as "'-'(자료 없음)"."""
    if fact.value is None:
        raw = fact.raw or "-"
        return f"'{raw}'({SYMBOL_MEANING.get(raw, '수치 아님')})"
    value = abs(fact.value) if is_change(fact) else fact.value
    return f"{_number(value, fact.round)}{fact.unit or ''}"


_CHANGE = re.compile(r"\s*(?:f\d+\s*-\s*f\d+|\(f\d+\s*/\s*f\d+\s*-\s*1\)\s*\*\s*100)\s*")


def is_change(fact: Fact) -> bool:
    """A difference ('f4 - f1') or a growth rate ('(f4 / f1 - 1) * 100'): shown without its sign."""
    return bool(fact.derivation) and _CHANGE.fullmatch(fact.derivation) is not None


def direction(fact: Fact) -> str:
    """'증가' / '감소' / '같음' for a change fact (what the sentence must say, since the number has no sign)."""
    if fact.value is None or fact.value == 0:
        return "같음"
    return "증가" if fact.value > 0 else "감소"


def item_label(label: str) -> str:
    """'1인 (가구)' -> '1인'; the unit is written with the number."""
    return _UNIT_SUFFIX.sub("", label).strip() or label.strip()


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


def render_parts(parts: list, result: StatResult) -> str:
    out = []
    for part in parts:
        if isinstance(part, FactPart):
            out.append(format_fact(result.get(part.fact_id)))
        elif isinstance(part, TextPart):
            out.append(part.text)
    return "".join(out).strip()


# Final consonant of the sound a word ends with: digits and unit symbols as they are read (3 = 삼, % = 퍼센트).
_READ_LAST = {"0": "영", "1": "일", "2": "이", "3": "삼", "4": "사", "5": "오", "6": "육", "7": "칠", "8": "팔", "9": "구",
              "%": "트", "p": "트", "㎢": "터"}
_PAIRS = {"이": ("이", "가"), "은": ("은", "는"), "을": ("을", "를"), "과": ("과", "와"), "으로": ("으로", "로")}


def josa(word: str, particle: str) -> str:
    """word + the particle form its last sound takes: josa('서울특별시', '이') -> '서울특별시가'."""
    with_final, without = _PAIRS[particle]
    last = next((ch for ch in reversed(word.strip()) if not ch.isspace() and ch not in ")'\""), "")
    last = _READ_LAST.get(last, last)
    code = ord(last) - 0xAC00 if last else -1
    if not 0 <= code < 11172:
        return word + without
    final = code % 28
    if particle == "으로" and final == 8:  # ㄹ: 서울로
        return word + without
    return word + (with_final if final else without)


def default_sentence(applied: dict[str, str]) -> str:
    """The conditions the rules chose, in one sentence written by code."""
    out = []
    for key, value in applied.items():
        if key == "period" and (m := messages._LATEST.fullmatch(value)):
            out.append(f"질문에 시점이 없어 가장 최근 시점인 {m.group(1)}년 기준으로 답합니다.")
        elif key == "period" and (m := messages._RECENT.fullmatch(value)):
            out.append(f"질문에 시점이 없어 최근 {m.group(1)}개 시점({m.group(2)}년)으로 답합니다.")
        else:
            out.append(f"질문에 없는 조건은 {messages._DEFAULT_KEYS.get(key, key)} {value} 기준으로 답합니다.")
    return " ".join(out)


def final_text(body: str, source: Source | None, applied: dict[str, str] | None = None, notes: list[str] | None = None) -> str:
    lines = [s for s in (default_sentence(applied or {}), body) if s]
    lines += [f"참고: {n}" for n in notes or [] if n]
    text = "\n".join(lines)
    return f"{text}\n\n출처: {source.citation()}" if source else text
