"""Question words that decide how a calculation is framed: which compared value is the reference, max or min."""

from __future__ import annotations

import re

from kosis_chat.judge.question import label_forms, squash

# "전국보다", "전국 대비", "전국 평균의 몇 배", "전국에 비해": the label before these is the reference.
_REFERENCE_AFTER = re.compile(r"^(?:평균)?(?:보다|대비|에비해|의몇배|의)")
_MIN_WORDS = re.compile(r"낮|적은|적었|최저|최소|작은|작았|꼴찌")
_PROVINCE = re.compile(r"(?:특별시|광역시|특별자치시|특별자치도|도)$")


def reference_index(question: str, labels: list[str]) -> int | None:
    """Index of the label the question compares against, or None when it only names both ('서울이랑 경기도')."""
    q = squash(question)
    for i, label in enumerate(labels):
        exact, _, _ = label_forms(label, regional_total=True)
        for form in sorted(exact, key=len, reverse=True):
            at = q.find(form)
            if at >= 0 and _REFERENCE_AFTER.match(q[at + len(form) :]):
                # "의" alone is a possessive ("서울의 출생아"); count it only before 몇 배.
                rest = q[at + len(form) :]
                if rest.startswith("의") and "배" not in rest[:6]:
                    continue
                return i
    return None


def pick_of(question: str) -> str:
    return "min" if _MIN_WORDS.search(question) else "max"


def is_province(label: str) -> bool:
    """A 시도 name ('부산광역시', '경기도'), not 전국, 동부·읍부·면부 or other 2-digit regions."""
    return bool(_PROVINCE.search(label.strip()))
