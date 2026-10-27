"""Checks on the model's answer parts before they are shown (T1-10). An empty list means the parts pass.

- every cited fact exists (V1) and every main fact is cited;
- connecting text has no numbers of its own: only a fact's period ('2023년') or a digit inside a fact's labels ('1인');
- a change over time cited as a fact is described in the right direction (늘었다 / 줄었다).
eval/grade.py allows periods only, so a label digit passes here but counts there; grade_run reports both.
"""

from __future__ import annotations

import re

from kosis_chat.answer.render import direction, is_change
from kosis_chat.compute import Computed

from kosis_ai.schemas import FactPart, TextPart  # isort: skip

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")  # the grader's pattern (eval/grade.py)
_UP = re.compile(r"늘|증가|많아|커졌|높아|상승")
_DOWN = re.compile(r"줄|감소|적어|작아|낮아|하락")


def allowed_numbers(computed: Computed) -> tuple[set[str], set[str]]:
    """(periods, digits in labels) that connecting text may contain."""
    periods: set[str] = set()
    labels: set[str] = set()
    for f in computed.result.facts:
        if f.period:
            periods |= {f.period, f.period[:4]} | ({str(int(f.period[4:]))} if len(f.period) == 6 else set())
        for label in (f.dims or {}).values():
            labels |= set(NUMBER.findall(label))
    return periods, labels


def label_digits(parts: list, computed: Computed) -> int:
    """Numbers in text that only a label explains (passes here, counts in eval/grade.py)."""
    periods, labels = allowed_numbers(computed)
    found = [n for p in parts if isinstance(p, TextPart) for n in NUMBER.findall(p.text)]
    return sum(n not in periods and n in labels for n in found)


def _over_time(fact, by_id: dict) -> bool:
    """A difference or growth rate between two periods of one series (a compare can be worded either way round)."""
    if not is_change(fact):
        return False
    a, b = (by_id.get(i) for i in re.findall(r"f\d+", fact.derivation)[:2])
    return a is not None and b is not None and a.dims == b.dims and a.period != b.period


def verify(parts: list, computed: Computed) -> list[str]:
    problems: list[str] = []
    by_id = {f.id: f for f in computed.result.facts}
    cited = [p.fact_id for p in parts if isinstance(p, FactPart)]
    unknown = [i for i in cited if i not in by_id]
    if unknown:
        problems.append(f"없는 사실 번호: {', '.join(unknown)}")
    missing = [i for i in computed.main if i not in cited]
    if missing:
        problems.append(f"빠진 사실: {', '.join('{' + i + '}' for i in missing)}")
    periods, labels = allowed_numbers(computed)
    text = "".join(p.text for p in parts if isinstance(p, TextPart))
    stray = [n for n in NUMBER.findall(text) if n not in periods and n not in labels]
    if stray:
        problems.append(f"문장에 직접 쓴 숫자: {', '.join(dict.fromkeys(stray))} (숫자는 {{f번호}}로만)")
    if not cited:
        problems.append("인용한 사실이 없음")
    for i in dict.fromkeys(cited):
        fact = by_id.get(i)
        if fact is None or not _over_time(fact, by_id):
            continue
        way = direction(fact)
        if way == "증가" and _DOWN.search(text) and not _UP.search(text):
            problems.append(f"{{{i}}}는 증가인데 감소로 씀")
        if way == "감소" and _UP.search(text) and not _DOWN.search(text):
            problems.append(f"{{{i}}}는 감소인데 증가로 씀")
    return problems
