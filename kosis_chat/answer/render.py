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


def format_number(value: float, unit: str | None, digits: int | None = None) -> str:
    """'123,456가구' for a plain number (charts use it for values that are not facts of the answer)."""
    return f"{_number(value, digits)}{unit or ''}"


def format_fact(fact: Fact) -> str:
    """'123,456가구'; a change without its sign (the sentence says 늘었다/줄었다); a symbol as "'-'(자료 없음)"."""
    if fact.value is None:
        raw = fact.raw or "-"
        return f"'{raw}'({SYMBOL_MEANING.get(raw, '수치 아님')})"
    value = abs(fact.value) if is_change(fact) else fact.value
    return format_number(value, fact.unit, fact.round)


_CHANGE = re.compile(r"\s*(?:f\d+\s*-\s*f\d+|\(f\d+\s*/\s*f\d+\s*-\s*1\)\s*\*\s*100)\s*")


def is_change(fact: Fact) -> bool:
    """A difference ('f4 - f1') or a growth rate ('(f4 / f1 - 1) * 100'): shown without its sign."""
    return bool(fact.derivation) and _CHANGE.fullmatch(fact.derivation) is not None


def direction(fact: Fact) -> str:
    """'증가' / '감소' / '같음' for a change fact (what the sentence must say, since the number has no sign)."""
    if fact.value is None or fact.value == 0:
        return "같음"
    return "증가" if fact.value > 0 else "감소"


_KIND_WORDS = {"growth": "증감률", "difference": "차이", "share": "비중", "ratio": "배수", "avg": "평균", "sum": "합계"}
_SUM = re.compile(r"^f\d+(?:\s*\+\s*f\d+)+$")
_AVG = re.compile(r"^\(f\d+(?:\s*\+\s*f\d+)+\)\s*/\s*\d+$")


def derived_kind(fact: Fact) -> str:
    """What a computed fact is, read from its derivation: 차이, 증감률, 비중, 배수, 평균 or 합계."""
    expr = fact.derivation or ""
    if _SUM.match(expr):
        return _KIND_WORDS["sum"]
    if _AVG.match(expr):
        return _KIND_WORDS["avg"]
    if "* 100" in expr and "- 1)" in expr:
        return _KIND_WORDS["growth"]
    if "* 100" in expr:
        return _KIND_WORDS["share"]
    if "/" in expr:
        return _KIND_WORDS["ratio"]
    return _KIND_WORDS["difference"]


def period_text(period: str) -> str:
    """'2024' -> '2024년'; '202406' -> '2024년 6월'."""
    return f"{period[:4]}년 {int(period[4:])}월" if len(period) == 6 and period.isdigit() else f"{period}년"


def span_text(first: str, last: str) -> str:
    """'2019년~2023년'; a half or a quarter of a year by name ('2024년 상반기', '2025년 1분기'), else '2024년 1월~6월'."""
    if first == last:
        return period_text(first)
    if len(first) == 6 and len(last) == 6 and first[:4] == last[:4]:
        a, b = int(first[4:]), int(last[4:])
        if (a, b) == (1, 6):
            return f"{first[:4]}년 상반기"
        if (a, b) == (7, 12):
            return f"{first[:4]}년 하반기"
        if b - a == 2 and a % 3 == 1:
            return f"{first[:4]}년 {a // 3 + 1}분기"
        return f"{first[:4]}년 {a}월~{b}월"
    return f"{period_text(first)}~{period_text(last)}"


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
        if key == "period" and (m := messages._PREVIOUS.fullmatch(value)):
            out.append(f"질문에 시점이 없어 가장 최근 시점인 {m.group(1)}년을 그 직전 시점 {m.group(2)}년과 비교합니다.")
        elif key == "period" and (m := messages._LATEST.fullmatch(value)):
            out.append(f"질문에 시점이 없어 가장 최근 시점인 {m.group(1)}년 기준으로 답합니다.")
        elif key == "period" and (m := messages._RECENT.fullmatch(value)):
            out.append(f"질문에 시점이 없어 최근 {m.group(1)}개 시점({m.group(2)}년)으로 답합니다.")
        else:
            out.append(f"질문에 없는 조건은 {messages._DEFAULT_KEYS.get(key, key)} {value} 기준으로 답합니다.")
    return " ".join(out)


def final_text(body: str, source: Source | None, applied: dict[str, str] | None = None, notes: list[str] | None = None) -> str:
    """The default sentence, the body, one '참고:' line per note (S6: the caveats' texts), then the source."""
    lines = [s for s in (default_sentence(applied or {}), body) if s]
    lines += [f"참고: {n}" for n in dict.fromkeys(notes or []) if n]
    text = "\n".join(lines)
    return f"{text}\n\n출처: {source.citation()}" if source else text
