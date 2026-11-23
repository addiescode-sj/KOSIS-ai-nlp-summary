"""Question words that decide how a calculation is framed: which compared value is the reference, max or min."""

from __future__ import annotations

import re

from kosis_chat.judge.question import label_forms, squash
from kosis_chat.shared.patterns import COUNT_WORD, DIVIDE, DIVISOR_AFTER, REGION_SUFFIX

# "전국보다", "전국 대비", "전국 평균의 몇 배", "전국에 비해": the label before these is the reference.
_REFERENCE_AFTER = re.compile(r"^(?:평균)?(?:보다|대비|에비해|의몇배|의)")
_OF_PERCENT = re.compile(r"^의몇(?:퍼센트|%|프로)")  # loop 8: "서울의 몇 퍼센트"
# loop 3: also the adverbs (적게, 작게), 최솟값, and the bottom of a ranking (하위, 아래·밑·뒤에서 세어).
_MIN_WORDS = re.compile(r"낮|적은|적었|적게|최저|최소|최솟|작은|작았|작게|꼴찌|하위|(?:아래|밑|뒤)에서")
# T1-28 S2b: was r"(?:특별시|광역시|특별자치시|특별자치도|도)$", the same literal suffixes anchored at the end in another
# order and without the group; is_province only asks whether one matches, which neither changes.
_PROVINCE = REGION_SUFFIX


def _skip_names(q: str, end: int, between: tuple[str, ...]) -> int:
    """Loop 8: past the other names and the counting words that follow a target's name ("서울 출생아 수의", "부산
    가구수로"): between holds the forms of the other labels the question names."""
    forms = sorted({f for f in between if f}, key=len, reverse=True)
    while True:
        nxt = next((f for f in forms if q.startswith(f, end)), None)
        if nxt is None and (m := COUNT_WORD.match(q[end:])):
            nxt = m.group(0)
        if nxt is None:
            return end
        end += len(nxt)


def _refers(rest: str, q: str) -> bool:
    if _REFERENCE_AFTER.match(rest):
        # "의" alone is a possessive ("서울의 출생아"); count it only before 몇 배 (loop 8: or 몇 퍼센트).
        return not rest.startswith("의") or "배" in rest[:6] or bool(_OF_PERCENT.match(rest))
    # loop 8: the divisor of a division said with its verb ("B로 나누면")
    return bool(DIVISOR_AFTER.match(rest)) and bool(DIVIDE.search(rest))


def reference_index(question: str, labels: list[str], between: tuple[str, ...] | list[str] = ()) -> int | None:
    """Index of the label the question compares against, or None when it only names both ('서울이랑 경기도').

    Loop 8: also the label a division names as its divisor (shared DIVISOR_AFTER, DIVIDE), and either after the other
    names and the counting words between the label and the word (_skip_names: "서울 출생아 수로 나누면")."""
    q = squash(question)
    between = tuple(between)
    for skip in (False, True):
        for i, label in enumerate(labels):
            exact, _, _ = label_forms(label, regional_total=True)
            for form in sorted(exact, key=len, reverse=True):
                at = q.find(form)
                while at >= 0:
                    end = at + len(form)
                    end = _skip_names(q, end, between) if skip else end
                    if (not skip or end > at + len(form)) and _refers(q[end:], q):
                        return i
                    at = q.find(form, at + 1)
    return None


def pick_of(question: str) -> str:
    return "min" if _MIN_WORDS.search(question) else "max"


def is_province(label: str) -> bool:
    """A 시도 name ('부산광역시', '경기도'), not 전국, 동부·읍부·면부 or other 2-digit regions."""
    return bool(_PROVINCE.search(label.strip()))
