"""Meaning checks for v3 plans (T0-04 S3): numbers that can be computed but would mean nothing are refused.

Each block of plan_engine.py calls the check for what it is about to compute, with the cells it read, so the
check sees the real units and measure kinds (normalize.measure_kind via judge.question.cell_measure). Refusals use
the shared reasons; a caution is kept when the number is allowed but needs a word of care (a simple average of rates).
"""

from __future__ import annotations

from kosis_chat.compute.engine import CalcRefused, _Read
from kosis_chat.data.schema import TableInput
from kosis_chat.judge import messages
from kosis_chat.judge.question import COUNT_KINDS, ITEM_AXIS, TableIndex, build_index, is_total

RATE_KINDS = ("ratio", "average", "index")
_INDEX: dict[int, tuple[TableInput, TableIndex]] = {}


def _items(reads: list[_Read]) -> str:
    return "·".join(dict.fromkeys(r.cell.item for r in reads))


def check_sum(reads: list[_Read]) -> None:
    """R1: a sum over periods needs a flow (births); a stock counts the same people again, a rate is meaningless,
    and an unknown measure is refused rather than guessed."""
    kinds = {r.kind for r in reads}
    if kinds != {"count_flow"}:
        raise CalcRefused("not_aggregatable", messages.refusal_message("not_aggregatable", what=_items(reads)))


def average_caution(reads: list[_Read]) -> str | None:
    """R2: an average over periods of a rate is a simple average (not weighted); allowed, with a caution."""
    if any(r.kind in RATE_KINDS for r in reads):
        return f"simple_average: {_items(reads)}의 시점 평균은 가중치 없는 단순 평균입니다."
    return None


def check_same_unit(fn: str, a: str | None, b: str | None, what: str) -> None:
    """R3/R4/R9: a difference, a growth rate or a threshold compares like with like."""
    if a and b and a != b:
        raise CalcRefused("unit_mismatch", messages.refusal_message("unit_mismatch", what=what, units=f"{a} / {b}"))
    if fn == "pp_diff" and not (a == "%" and b == "%"):
        raise CalcRefused("unit_mismatch", f"{what}: %p 차이는 두 값이 모두 %일 때만 구할 수 있습니다(단위 {a} / {b}).")


def check_share_parts(parts: list[_Read]) -> None:
    """R7: several parts are summed for a share, so they must be counts."""
    if len(parts) > 1 and any(r.kind not in COUNT_KINDS for r in parts if r.kind):
        raise CalcRefused("not_aggregatable", messages.refusal_message("not_aggregatable", what=_items(parts)))


def index_of(table: TableInput) -> TableIndex:
    cached = _INDEX.get(id(table))
    if cached is None or cached[0] is not table:
        if len(_INDEX) > 8:
            _INDEX.clear()
        cached = (table, build_index(table))
        _INDEX[id(table)] = cached
    return cached[1]


def _whole_label(label: str) -> bool:
    """A label naming everything of its category: 계·합계·전국 (is_total), or 총인구·총계·총가구 ('총...')."""
    return is_total(label) or label.strip().startswith("총")


def _covers(index: TableIndex, axis: str, whole: str, part: str, whole_label: str) -> bool:
    if whole == part:
        return True
    found = index.axes.get(axis)
    return _whole_label(whole_label) or (found is not None and whole in found.ancestors(part))


def _period_covers(whole: str, part: str) -> bool:
    """The same period, or a year holding the part's month or quarter (12월 혼인 / 2025년 혼인)."""
    return whole == part or (len(whole) == 4 and len(part) > 4 and part.startswith(whole))


def check_share_scope(table: TableInput, parts: list[_Read], total: _Read) -> None:
    """R5: the denominator holds the numerator: on every category where they differ, the total's code is an
    ancestor of the part's code (KOSIS codes nest by prefix) or the whole of that category; the total's period
    is the part's or the year that holds it."""
    index = index_of(table)
    t = total.cell
    for p in (r.cell for r in parts):
        wrong = []
        if not _period_covers(t.period_raw, p.period_raw):
            wrong.append(f"시점 {p.period_raw}≠{t.period_raw}")
        if not _covers(index, ITEM_AXIS, t.item_code, p.item_code, t.item):
            wrong.append(f"항목 {p.item}⊄{t.item}")
        for name in p.dim_codes.keys() | t.dim_codes.keys():
            pc, tc = p.dim_codes.get(name), t.dim_codes.get(name)
            if pc is None or tc is None or not _covers(index, name, tc, pc, t.dims.get(name, "")):
                wrong.append(f"{name} {p.dims.get(name)}⊄{t.dims.get(name)}")
        if wrong:
            raise CalcRefused("class_mismatch", f"비중의 분모가 분자를 포함하지 않습니다({', '.join(wrong)}).")


def check_share_bound(value: float, what: str) -> None:
    """R6: a share of a whole lies in 0..100%."""
    if not 0 <= value <= 100:
        raise CalcRefused("class_mismatch", f"{what} 비중이 {value:.1f}%로 0~100% 밖이라 분모가 분자의 전체가 아닙니다.")


def check_rank_units(units: list[str | None]) -> None:
    """R8: candidates of one ranking share a unit."""
    known = sorted({u for u in units if u})
    if len(known) > 1:
        raise CalcRefused("unit_mismatch", messages.refusal_message("unit_mismatch", what="순위 후보", units=" / ".join(known)))
