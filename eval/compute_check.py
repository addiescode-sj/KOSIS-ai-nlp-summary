"""M1: does the calculation engine reproduce the golden answers when it is given the right cells? (no LLM)

For every answerable / default_allowed golden item the cells come from its required_facts (an oracle request:
cell choice is not tested here, T1-06 is); kosis_chat.compute does the arithmetic and eval.grade grades the result
exactly as it grades a model answer. Reports go to eval/results/_checks/compute/ (data values: not committed).

    python -m eval.compute_check --golden eval/golden/stat_golden.jsonl
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

from eval.golden import GoldenItem, load_items, read_cells, read_source
from eval.grade import grade
from kosis_chat.compute import CalcRefused, CalcRequest, compute
from kosis_chat.compute.request import Term
from kosis_chat.compute.roles import is_province, pick_of, reference_index
from kosis_chat.config import PROJECT_ROOT
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.judge.question import is_total

from kosis_ai.schemas import FactPart, StatAnswer, StatQuery  # isort: skip

OUT_DIR = PROJECT_ROOT / "eval" / "results" / "_checks" / "compute"
COMPUTED_KINDS = ("answerable", "default_allowed")
_DENOMINATOR = re.compile(r"/\s*(f\d+)\s*\*\s*100")
# T1-25 complex items carry their operation in the id (S-CX-AVG-01); the engine's op for each (v2).
_OP = re.compile(r"^S-CX-([A-Z]+)-\d+$")
_OPS = {"AVG": "avg", "SUM": "sum", "PAVG": "pavg", "VSA": "vsa", "SHR": "shr", "MON": None}  # MON: sum or psum by group count


def _cells(table: TableInput) -> dict[str, Cell]:
    return {c.source_ref: c for c in table.cells}


def _same_except(cell: Cell, other: Cell, axis: str | None) -> bool:
    """Same item and categories, differing at most on `axis` (a category name, or None for the period)."""
    if cell.item_code != other.item_code:
        return False
    if axis is None:
        return cell.dim_codes == other.dim_codes
    return cell.period_raw == other.period_raw and all(
        other.dim_codes.get(k) == v for k, v in cell.dim_codes.items() if k != axis
    )


def _varying_axis(item: GoldenItem) -> str | None:
    """The category an extreme question ranks over (None: the periods)."""
    dims = (item.scope or {}).get("dims") or {}
    periods = (item.scope or {}).get("periods") or []
    if len(set(periods)) > 1 and not dims:
        return None
    return next((name for name, codes in dims.items() if len(codes) > 1), None)


def _op_request(item: GoldenItem, op: str | None, refs: list[str], table: TableInput) -> CalcRequest | None:
    """Terms from the golden cells: the derived facts' inputs say which cells form each group."""
    by_ref = _cells(table)
    by_id = {f.id: f for f in item.required_facts}
    groups: list[list[str]] = []
    for f in item.required_facts:  # an average / sum over cells only (not over other derived facts)
        if f.derivation and all(by_id.get(n) is not None and by_id[n].source_ref for n in re.findall(r"f\d+", f.derivation)):
            cells = [by_id[n].source_ref for n in re.findall(r"f\d+", f.derivation)]
            if len(cells) >= 2 and cells not in groups:
                groups.append(cells)
    if op is None:  # MON: one span summed, or two spans summed and compared
        op = "psum" if len(groups) == 2 else "sum"
    if op == "shr":
        shares = [f for f in item.required_facts if f.derivation and _DENOMINATOR.search(f.derivation)]
        terms = []
        for f in sorted(shares, key=lambda f: by_id[re.findall(r"f\d+", f.derivation)[0]].period or ""):
            ids = re.findall(r"f\d+", f.derivation)
            terms.append(Term(cells=[by_id[i].source_ref for i in ids[:-1]], total=by_id[ids[-1]].source_ref))
        return CalcRequest(type="share", op="shr", terms=terms)
    if op == "vsa":
        rest = [r for r in refs if r not in groups[0]]
        terms = [Term(cells=groups[0], agg="avg"), Term(cells=rest[:1])]
        return CalcRequest(type=item.type, op="vsa", terms=terms)
    agg = "sum" if op in ("sum", "psum") else "avg"
    ordered = sorted(groups, key=lambda g: by_ref[g[0]].period_raw)
    return CalcRequest(type=item.type, op=op, terms=[Term(cells=g, agg=agg) for g in ordered])


def oracle_request(item: GoldenItem, table: TableInput) -> CalcRequest:
    """The request a perfect question analyzer would send, built from the golden cells (not their derivations)."""
    by_ref = _cells(table)
    refs = [f.source_ref for f in item.required_facts if f.source_ref]
    derived = next((f for f in item.required_facts if f.derivation), None)
    kind = item.type
    if (m := _OP.match(item.id)) and m.group(1) in _OPS:
        return _op_request(item, _OPS[m.group(1)], refs, table)
    if kind == "trend":
        first = by_ref[refs[0]]
        periods = sorted(by_ref[r].period_raw for r in refs)
        series = [c for c in table.cells if _same_except(first, c, None) and periods[0] <= c.period_raw <= periods[-1]]
        return CalcRequest(type="trend", cells=[c.source_ref for c in sorted(series, key=lambda c: c.period_raw)])
    if kind == "compare":
        a, b = by_ref[refs[0]], by_ref[refs[1]]
        labels = [next((v for k, v in x.dims.items() if y.dims.get(k) != v), x.item) for x, y in ((a, b), (b, a))]
        return CalcRequest(type="compare", cells=refs[:2], reference=reference_index(item.question, labels))
    if kind == "share" or (kind == "extreme" and derived is not None):
        denominator = _DENOMINATOR.search(derived.derivation).group(1)
        by_id = {f.id: f.source_ref for f in item.required_facts if f.source_ref}
        total = by_id[denominator]
        parts = [r for r in refs if r != total]
        if kind == "share":
            return CalcRequest(type="share", cells=parts, total=total)
        return _extreme(item, table, parts[0], total)
    if kind == "extreme":
        return _extreme(item, table, refs[0], None)
    # A default "latest" value: newer periods of the same series that hold symbols are shown too ('2025 is -').
    cell = by_ref[refs[0]]
    newer = [c.source_ref for c in table.cells if _same_except(cell, c, None) and c.period_raw > cell.period_raw]
    return CalcRequest(type="value", cells=refs[:1], context=newer if item.default_applied else [])


def _extreme(item: GoldenItem, table: TableInput, winner_ref: str, total_ref: str | None) -> CalcRequest:
    by_ref = _cells(table)
    winner = by_ref[winner_ref]
    axis = _varying_axis(item)
    if axis is None:
        periods = sorted((item.scope or {}).get("periods") or [winner.period_raw])
        lo, hi = periods[0], periods[-1]
        pool = [c for c in table.cells if _same_except(winner, c, None) and lo <= c.period_raw[:4] <= hi]
        context: list[str] = []
    else:
        allowed = set((item.scope or {}).get("dims", {}).get(axis, []))
        same = [c for c in table.cells if _same_except(winner, c, axis) and c.dim_codes.get(axis) in allowed]
        pool = [c for c in same if not is_total(c.dims.get(axis, ""))]
        if "시도" in item.question:  # rank 시도 only, not 동부·읍부·면부 or other regions on the same level
            pool = [c for c in pool if is_province(c.dims.get(axis, ""))]
        context = [c.source_ref for c in same if is_total(c.dims.get(axis, ""))]
    candidates = [c.source_ref for c in pool]
    totals: list[str] = []
    if total_ref:
        total = by_ref[total_ref]
        twin = {(c.period_raw, tuple(sorted(c.dim_codes.items()))): c.source_ref for c in table.cells if c.item_code == total.item_code}
        totals = [twin[(c.period_raw, tuple(sorted(c.dim_codes.items())))] for c in pool]
    return CalcRequest(type="extreme", candidates=candidates, candidate_totals=totals, pick=pick_of(item.question),
                       context=context)


def check(item: GoldenItem) -> dict:
    table = clean(load_table(item.path))
    row = {"id": item.id, "kind": item.kind, "type": item.type, "question": item.question}
    try:
        request = oracle_request(item, table)
        computed = compute(table, request)
    except CalcRefused as exc:
        return row | {"verdict": "FAIL", "error_types": ["calc_refused"], "details": [f"{exc.reason}: {exc.message}"]}
    answer = StatAnswer(
        outcome="answered",
        query=StatQuery(question=item.question, type=item.type),
        result=computed.result,
        parts=[FactPart(fact_id=i) for i in computed.main],
        default_applied=item.default_applied,
        source=read_source(item.path),
        numbers_origin="code",
    )
    facts = [{"id": f.id, "value": f.value, "derivation": f.derivation, "source_ref": f.source_ref} for f in computed.result.facts]
    row |= {"request": request.model_dump(exclude_defaults=True), "facts": facts}
    try:
        graded = grade(item, answer, read_cells(item.path))
    except ValueError as exc:  # the grader cannot place an error type for this kind (eval/scorer.RULES)
        return row | {"verdict": "ERROR", "error_types": ["grader_error"], "details": [str(exc)]}
    return row | {"verdict": graded.verdict, "error_types": graded.error_types, "details": graded.details}


def render(rows: list[dict], meta: dict) -> str:
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("PASS", "PARTIAL", "FAIL", "ERROR")}
    lines = [
        f"# M1 계산 엔진 골든 대조 ({meta['started_at']})",
        "",
        f"정답 셀을 준 상태에서 계산 엔진이 골든 답을 재현하는지 (LLM 없음). 문항 {len(rows)}개: "
        f"PASS {counts['PASS']} / PARTIAL {counts['PARTIAL']} / FAIL {counts['FAIL']} / 채점기 오류 {counts['ERROR']}",
        "",
        "| 문항 | 유형 | 판정 | 오류 | 상세 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        detail = "; ".join(r["details"])[:200].replace("|", "/")
        lines.append(f"| {r['id']} | {r['type']} | {r['verdict']} | {', '.join(r['error_types'])} | {detail} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compute_check", description=__doc__.splitlines()[0])
    parser.add_argument("--golden", default="eval/golden/stat_golden.jsonl")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    version, items, golden_sha = load_items(Path(args.golden))
    rows = [check(i) for i in items if i.kind in COMPUTED_KINDS]
    meta = {"started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "golden": args.golden,
            "golden_version": version, "golden_sha256": golden_sha}
    out = Path(args.out) / f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    report = render(rows, meta)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(report)
    print(f"saved: {out}")
    return 0 if all(r["verdict"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
