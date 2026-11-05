"""Words of care shown with an answer (S6): Caveat records built by code from what the calculation noted.

Sources: Computed.cautions (resolve.py and semantics.py write 'kind: detail'), Computed.note (the rules' note on
missing years and calendar events), the facts themselves (a symbol cell, a provisional value) and the scenario
alternatives shown beside the answer (compute/context.py). A caveat's text names periods but no other numbers.
"""

from __future__ import annotations

from kosis_chat.shared import cautions  # isort: skip  # first, as answer.render was: shared puts src/ on the path
from kosis_chat.shared.labels import fact_label  # isort: skip
from kosis_chat.shared.numbers import SYMBOL_MEANING  # isort: skip
from kosis_chat.shared.periods import period_text  # isort: skip
from kosis_chat.compute import Computed  # isort: skip
from kosis_chat.compute.context import Alternatives  # isort: skip
from kosis_chat.prepare.schemas import TableInput  # isort: skip

from kosis_ai.schemas import Caveat  # isort: skip

# The caution strings are parsed in kosis_chat/shared/cautions.py (T1-28 S2b); the old names are the same objects.
_EVENT_MODEL = cautions.EVENT_MODEL
_PREFIX = cautions.PREFIX


def _when(year: str, month: str | None) -> str:
    return f"{year}년" + (f" {int(month)}월" if month else "")


def _from_caution(text: str) -> Caveat | None:
    if m := cautions.parse_event_source(text):
        when = _when(m.group("start"), m.group("month"))
        return Caveat(kind="event_source", text=f"'{m.group('name')}'의 시작 시점({when})은 사건 달력에 없어 모델이 알려 준 값입니다.")
    parsed = cautions.parse_caution(text)
    if not parsed:
        return None
    kind, detail = parsed
    if kind == "missing_years":
        years = ", ".join(period_text(y) for y in cautions.parse_missing_years(detail))
        return Caveat(kind="missing_years", text=f"{years}은(는) 표에 없습니다.")
    if kind == "recent":
        return Caveat(kind="missing_years", text="요청한 최근 시점 가운데 일부만 숫자가 있습니다.")
    if kind == "default_applied":
        axis, label = cautions.parse_default_applied(detail)
        return Caveat(kind="default_applied", text=f"질문에 없는 {axis}은(는) '{label}' 기준입니다.")
    if kind == "simple_average":
        return Caveat(kind="simple_average", text=detail)
    return None


def _from_note(note: str) -> list[Caveat]:
    out = []
    for piece in (p.strip() for p in note.split(";")):
        if not piece:
            continue
        if "표에 없음" in piece:
            out.append(Caveat(kind="missing_years", text=piece.replace("년은 표에 없음", "년은 표에 없습니다.")))
        elif "사건 달력" in piece:
            out.append(Caveat(kind="event_source", text=piece + "입니다."))
    return out


def caveats_of(computed: Computed, table: TableInput, alternatives: list[Alternatives] | None = None,
               context_ids: list[str] | None = None) -> list[Caveat]:
    out: list[Caveat] = []
    for text in computed.cautions:
        if (c := _from_caution(text)) is not None:
            out.append(c)
    if computed.note:
        out += _from_note(computed.note)
    by_ref = {c.source_ref: c for c in table.cells}
    shown = set(computed.main) | set(context_ids or [])
    for f in computed.result.facts:
        if f.id not in shown or not f.source_ref:
            continue
        name = f"{period_text(f.period or '')} {fact_label(f)}".strip()
        if f.value is None and f.raw:
            meaning = SYMBOL_MEANING.get(f.raw, "수치 아님")
            out.append(Caveat(kind="symbol_cell", text=f"{name} 값은 기호 '{f.raw}'({meaning})입니다.", fact_ids=[f.id]))
        elif (cell := by_ref.get(f.source_ref)) is not None and cell.value_flag == "provisional":
            out.append(Caveat(kind="provisional", text=f"{name} 값은 잠정치입니다.", fact_ids=[f.id]))
    ref_ids = {f.source_ref: f.id for f in computed.result.facts if f.source_ref}
    for alt in alternatives or []:
        ids = [ref_ids[r] for r in alt.refs if r in ref_ids]
        if ids:
            chosen = alt.chosen.split("(")[0].strip()
            out.append(Caveat(kind="alternative", fact_ids=ids,
                              text=f"{alt.axis}은(는) 질문에 없어 '{chosen}' 기준으로 답했습니다. 다른 가정의 값도 함께 보여 줍니다."))
    seen: set[tuple[str, str]] = set()
    return [c for c in out if (c.kind, c.text) not in seen and not seen.add((c.kind, c.text))]
