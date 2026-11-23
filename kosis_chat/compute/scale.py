"""Units that differ only by a power-of-ten prefix ('천명' and '명', '백만원' and '원'), for numbers from two tables (T1-30).

Two tables can count the same thing at different scales (one in 천명, the other in 명). A calculation across them
first turns one number into the other's unit with a conversion fact whose derivation names the factor
('f1 * 1000'), so eval/grade.py recomputes it like any derived fact. Units that differ in anything else ('명' and
'가구', a rate '천명당') are not converted.
"""

from __future__ import annotations

# Longest prefix first: '십만' before '만', '천만' before '천'.
PREFIXES = (("십억", 10 ** 9), ("천만", 10 ** 7), ("백만", 10 ** 6), ("십만", 10 ** 5), ("천", 10 ** 3), ("만", 10 ** 4),
            ("억", 10 ** 8), ("조", 10 ** 12))
PREFIXES = tuple(sorted(PREFIXES, key=lambda p: -len(p[0])))


def scale_of(unit: str | None) -> tuple[str, int] | None:
    """(base unit, factor): '천명' -> ('명', 1000), '명' -> ('명', 1). None for no unit, a rate ('천명당'), a
    percentage or a unit that is only a prefix."""
    if not unit:
        return None
    text = unit.strip()
    if not text or text.endswith("당") or "%" in text or "/" in text:
        return None
    for prefix, factor in PREFIXES:
        if text.startswith(prefix) and len(text) > len(prefix):
            return text[len(prefix):], factor
    return text, 1


def conversion(src: str | None, dst: str | None) -> tuple[str, int] | None:
    """How to turn a number in src into dst: ('*', n) or ('/', n); None when they are not the same base unit with
    different prefixes (equal units need no conversion and also give None)."""
    a, b = scale_of(src), scale_of(dst)
    if a is None or b is None or a[0] != b[0] or a[1] == b[1]:
        return None
    return ("*", a[1] // b[1]) if a[1] > b[1] else ("/", b[1] // a[1])


def common_unit(a: str | None, b: str | None) -> str | None:
    """The finer of two units of one base ('천명', '명' -> '명'); None when they cannot be converted."""
    sa, sb = scale_of(a), scale_of(b)
    if sa is None or sb is None or sa[0] != sb[0]:
        return None
    return a if sa[1] <= sb[1] else b


__all__ = ["PREFIXES", "common_unit", "conversion", "scale_of"]
