"""Answer sentences written by code: the fallback when the model's sentence fails verification twice (T1-10).

Each template cites the facts the calculation marked as main, so a template answer carries the same numbers.
"""

from __future__ import annotations

from kosis_chat.answer.render import direction, fact_label, format_fact, item_label, josa, period_text, span_text
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
    if way == "같음":  # the difference is a main fact: cite it even when it is zero
        return parts + [_t("변함이 없습니다(차이 "), _f(diff), _t(").")]
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
        return parts + [_t("두 값은 같습니다(차이 "), _f(diff), _t(").")]
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


# ---------------------------------------------------------------- patterns over periods


def _span(term) -> str:
    return span_text(*term.periods)


def _agg_word(term) -> str:
    return {"avg": "평균", "sum": "합계"}.get(term.agg or "", "값")


def _one_term(facts: list[Fact], computed: Computed) -> list:
    """avg / sum: '2019년~2023년 서울특별시 출생건수 평균은 45,741.4명입니다.'"""
    term = computed.terms[0]
    by_id = {f.id: f for f in computed.result.facts}
    cell, value = by_id[term.cell_ids[0]], by_id[term.value_id]
    return [_t(f"{_span(term)} {_subject(fact_label(cell) + ' ' + _agg_word(term))}"), _f(value), _t("입니다.")]


def _two_terms(facts: list[Fact], computed: Computed) -> list:
    """pavg / psum / vsa: the later term against the earlier one, with the difference and the growth rate."""
    earlier, later = computed.terms
    by_id = {f.id: f for f in computed.result.facts}
    a, b, diff, *growth = facts
    label = fact_label(by_id[earlier.cell_ids[0]])
    if computed.pattern == "vsa":
        lead = f"{_span(later)} {_subject(label)}"
        parts = [_t(lead), _f(b), _after(b, "으로"), _t(f" {_span(earlier)} 평균 "), _f(a), _t("보다 ")]
    else:
        parts = [_t(f"{_subject(label)}{_span(earlier)} {_agg_word(earlier)} "), _f(a), _t(f"에서 {_span(later)} {_agg_word(later)} "),
                 _f(b), _after(b, "으로"), _t(" ")]
    way = direction(diff)
    if way == "같음":  # the difference is a main fact: cite it even when it is zero
        return parts + [_t("같습니다(차이 " if computed.pattern == "vsa" else "변함이 없습니다(차이 "), _f(diff), _t(")."), ]
    parts += [_f(diff)]
    parts += [_t(" 많습니다" if way == "증가" else " 적습니다")] if computed.pattern == "vsa" else [_t(" 늘었습니다" if way == "증가" else " 줄었습니다")]
    if growth:
        parts += [_t(f"({'증가율' if way == '증가' else '감소율'} "), _f(growth[0]), _t(")")]
    return parts + [_t(".")]


def _shr(facts: list[Fact], computed: Computed) -> list:
    """shr: '전국 일반가구 가운데 1인 비중은 2015년 27.2%에서 2024년 36.1%로 8.9%p 늘었습니다.'"""
    earlier, later = computed.terms
    by_id = {f.id: f for f in computed.result.facts}
    a, b, diff = facts
    cells = [by_id[i] for i in earlier.cell_ids]
    whole = next((f for f in computed.result.facts if f.source_ref and f.id not in earlier.cell_ids and f.period == cells[0].period), None)
    names = "·".join(_side(c, whole) if whole else fact_label(c) for c in cells)
    lead = f"{fact_label(whole) + ' 가운데 ' if whole and fact_label(whole) else ''}{names} 비중은 {_span(earlier)} "
    parts = [_t(lead), _f(a), _t(f"에서 {_span(later)} "), _f(b), _after(b, "으로"), _t(" ")]
    way = direction(diff)
    if way == "같음":
        return parts + [_t("변함이 없습니다(차이 "), _f(diff), _t(").")]
    return parts + [_f(diff), _t(" 늘었습니다." if way == "증가" else " 줄었습니다.")]


# ---------------------------------------------------------------- two targets at two periods


def _changed(fact: Fact, up: str, down: str, same: str) -> str:
    return {"증가": up, "감소": down}.get(direction(fact), same)


def _gap(facts: list[Fact], computed: Computed) -> list:
    """gap: '서울특별시와 경기도의 인구밀도 격차는 2015년 15,137.6명/㎢(...)에서 2025년 14,022.1명/㎢(...)로 1,115.5명/㎢
    줄었습니다.' (each gap with the two values it is made of)"""
    x1, y1, x2, y2, g1, g2, d = facts
    common = _common(x1, y2)
    lead = f"{josa(_side(x1, y1), '과')} {_side(y1, x1)}의 {common + ' ' if common else ''}격차는 {period_text(x1.period or '')} "
    parts = [_t(lead), _f(g1), _t(f"({_side(x1, y1)} "), _f(x1), _t(f", {_side(y1, x1)} "), _f(y1),
             _t(f")에서 {period_text(x2.period or '')} "), _f(g2), _t(f"({_side(x2, y2)} "), _f(x2), _t(f", {_side(y2, x2)} "),
             _f(y2), _t(")"), _after(g2, "으로")]
    if direction(d) == "같음":
        return parts + [_t(" 변함이 없습니다(차이 "), _f(d), _t(").")]
    return parts + [_t(" "), _f(d), _t(_changed(d, " 벌어졌습니다.", " 줄었습니다.", ""))]


def _grw(facts: list[Fact], computed: Computed) -> list:
    """grw: each target's change, then which one changed more and by how much."""
    a1, b1, a2, b2, v1, v2, d = facts
    rate = v1.unit == "%"
    parts: list = []
    for n, (a, b, v, other) in enumerate(((a1, b1, v1, a2), (a2, b2, v2, a1))):
        lead = f"{_subject(_side(a, other))}" if n else f"{span_text(a.period or '', b.period or '')} {_subject(_side(a, other))}"
        parts += [_t(lead), _f(a), _t("에서 "), _f(b), _after(b, "으로"), _t(" "), _f(v)]
        word = _changed(v, "증가", "감소", "변화 없음")
        parts += [_t(f" {word}" + ("했고, " if n == 0 else "했습니다. ") if word != "변화 없음" else (" 변화가 없고, " if n == 0 else " 변화가 없습니다. "))]
    subject_id = d.derivation.split("-")[0].strip()
    winner, loser = (a1, a2) if subject_id == v1.id else (a2, a1)
    winner_v = v1 if subject_id == v1.id else v2
    if direction(d) == "같음":
        return parts + [_t("두 곳의 " + ("증감률" if rate else "증감") + "은 같습니다(차이 "), _f(d), _t(").")]
    if direction(winner_v) == "감소" and direction(d) == "감소":  # the more negative change first: a larger decline
        what = "감소율이" if rate else "감소 폭이"
        return parts + [_t(f"{_side(winner, loser)}의 {what} "), _f(d), _t(" 더 큽니다.")]
    what = "증감률이" if rate else "증감이"
    return parts + [_t(f"{_side(winner, loser)}의 {what} "), _f(d), _t(" 더 높습니다.")]


_TEMPLATES = {"value": _value, "trend": _trend, "compare": _compare, "share": _share, "extreme": _extreme,
              "avg": _one_term, "sum": _one_term, "pavg": _two_terms, "psum": _two_terms, "vsa": _two_terms, "shr": _shr,
              "gap": _gap, "grw": _grw}


def template_parts(kind: QuestionType, computed: Computed) -> list:
    """The sentence for the pattern the rules built, else for the question type (a plan with no pattern)."""
    by_id = {f.id: f for f in computed.result.facts}
    template = _TEMPLATES[computed.pattern or kind]
    return [p for p in template([by_id[i] for i in computed.main], computed) if not isinstance(p, TextPart) or p.text]
