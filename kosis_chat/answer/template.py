"""Answer sentences written by code: the fallback when the model's sentence fails verification twice (T1-10).

Each template cites the facts the calculation marked as main, so a template answer carries the same numbers.
"""

from __future__ import annotations

from kosis_chat.answer.render import direction, fact_label, format_fact, item_label, josa
from kosis_chat.compute import Computed

from kosis_ai.schemas import Fact, FactPart, QuestionType, TextPart  # isort: skip


def _t(text: str) -> TextPart:
    return TextPart(text=text)


def _f(fact: Fact) -> FactPart:
    return FactPart(fact_id=fact.id)


def _after(fact: Fact, particle: str) -> TextPart:
    """The particle a rendered number takes ('1,234명' + 으로 -> '으로')."""
    shown = format_fact(fact)
    return _t(josa(shown, particle)[len(shown):])


def _year(fact: Fact) -> str:
    return f"{fact.period}년 " if fact.period else ""


def _side(a: Fact, b: Fact) -> str:
    """What tells a apart from b: the category label they differ on, else the period."""
    for name, label in (a.dims or {}).items():
        if (b.dims or {}).get(name) != label:
            return item_label(label)
    return f"{a.period}년"


def _common(a: Fact, b: Fact) -> str:
    """Labels a and b share ('1인' when two regions are compared)."""
    shared = {k: v for k, v in (a.dims or {}).items() if (b.dims or {}).get(k) == v}
    return fact_label(a.model_copy(update={"dims": shared}))


def _subject(text: str) -> str:
    return josa(text, "은") + " " if text else ""


def _value(facts: list[Fact], computed: Computed) -> list:
    f1 = facts[0]
    return [_t(f"{_year(f1)}{_subject(fact_label(f1))}"), _f(f1), _t("입니다.")]


def _trend(facts: list[Fact], computed: Computed) -> list:
    first, last, diff, *growth = facts
    parts = [_t(f"{_subject(fact_label(first))}{first.period}년 "), _f(first), _t("에서 "), _t(f"{last.period}년 "),
             _f(last), _after(last, "으로"), _t(" ")]
    way = direction(diff)
    if way == "같음":
        return parts + [_t("변함이 없습니다.")]
    parts += [_f(diff), _t(" 늘었습니다" if way == "증가" else " 줄었습니다")]
    if growth:
        parts += [_t(f"({'증가율' if way == '증가' else '감소율'} "), _f(growth[0]), _t(")")]
    return parts + [_t(".")]


def _compare(facts: list[Fact], computed: Computed) -> list:
    a, b, diff, *ratio = facts
    subject_id = diff.derivation.split("-")[0].strip()
    subject, reference = (a, b) if subject_id == a.id else (b, a)
    common = _common(a, b)
    parts = [_t(f"{_year(a)}{common + ' ' if common else ''}값은 {_side(a, b)} "), _f(a), _t(f", {_side(b, a)} "), _f(b),
             _t("입니다. ")]
    way = direction(diff)
    if way == "같음":
        return parts + [_t("두 값은 같습니다.")]
    parts += [_t(f"{josa(_side(subject, reference), '이')} {_side(reference, subject)}보다 "), _f(diff),
              _t(" 많습니다" if way == "증가" else " 적습니다")]
    if ratio:
        parts += [_t(f"({_side(reference, subject)}의 "), _f(ratio[0]), _t(")")]
    return parts + [_t(".")]


def _share(facts: list[Fact], computed: Computed) -> list:
    *cells, total, share = facts
    names = "·".join(_side(c, total) for c in cells)
    whole = fact_label(total)
    return [_t(f"{_year(total)}{whole + ' 가운데 ' if whole else ''}{names} 비중은 "), _f(share), _t("입니다("), *_sum(cells),
            _t(" / 전체 "), _f(total), _t(").")]


def _sum(cells: list[Fact]) -> list:
    out: list = []
    for n, c in enumerate(cells):
        out += ([_t(" + ")] if n else []) + [_f(c)]
    return out


def _extreme(facts: list[Fact], computed: Computed) -> list:
    by_id = {f.id: f for f in computed.result.facts}
    winner, runner = by_id[computed.picked["winner"]], by_id.get(computed.picked.get("runner_up", ""))

    def cell_of(f: Fact) -> Fact:  # a share ranking scores a derived fact; its label comes from the part cell
        return f if f.source_ref else by_id[f.derivation.split("/")[0].strip(" ()+")]

    w, r = cell_of(winner), cell_of(runner) if runner else None
    word = "가장 높습니다" if "높음" in (winner.note or "") else "가장 낮습니다"
    name = _side(w, r) if r else fact_label(w)
    lead = f"{'' if name.endswith('년') else _year(w)}{_common(w, r) if r else ''}".strip()
    what = "비중은" if winner.derivation else "값은"
    parts = [_t(f"{lead} {what} " if lead else ""), _t(f"{josa(name, '이')} "), _f(winner), _after(winner, "으로"),
             _t(f" {word}.")]
    if runner and r:
        parts += [_t(f" 다음은 {_side(r, w)}("), _f(runner), _t(")입니다.")]
    return parts


_TEMPLATES = {"value": _value, "trend": _trend, "compare": _compare, "share": _share, "extreme": _extreme}


def template_parts(kind: QuestionType, computed: Computed) -> list:
    by_id = {f.id: f for f in computed.result.facts}
    return [p for p in _TEMPLATES[kind]([by_id[i] for i in computed.main], computed) if not isinstance(p, TextPart) or p.text]
