"""Build golden items from a spec whose facts point at Parquet cells, so no golden number is typed by hand.

Data In:
  --spec eval/golden/edge_spec.json     list of items: GoldenItem fields, but `table` is the table key
                                        (e.g. "101_DT_1B83A35") and each fact is one of
                                          {"id": "f1", "ref": "<Cell.ref()>"}                     source cell
                                          (optional "unit" when the table has none but the label states it)
                                          {"id": "f3", "derivation": "f1 / f2 * 100", "round": 1, "unit": "%"}
                                          {"id": "a2", "note": "..."}
Data Out:
  --out eval/golden/stat_golden_edge.jsonl           GoldenItem lines (no `scope`; run scripts/golden_to_parquet.py next)
  --checklist eval/golden/verify_checklist_edge.md   one unchecked row per source fact for the KOSIS web check

Run from the repo root:
  python scripts/build_golden_edge.py
  python scripts/golden_to_parquet.py --golden eval/golden/stat_golden_edge.jsonl --report eval/golden/scope_report_edge.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.golden import GoldenItem, read_cells  # noqa: E402
from eval.grade import evaluate  # noqa: E402


def build_item(spec: dict) -> GoldenItem:
    table = f"data/values/{spec['table']}.parquet"
    cells = read_cells(ROOT / table)
    values: dict[str, float] = {}

    def fact(f: dict) -> dict:
        if "ref" in f:
            cell = cells.get(f["ref"])
            if cell is None:
                raise KeyError(f"{spec['id']}/{f['id']}: no cell {f['ref']}")
            out = cell.to_fact(f["id"]).model_dump(exclude_none=True)
            if "unit" in f:  # unit the table leaves out but the label states, e.g. "인구(명): 65세 이상"
                out["unit"] = f["unit"]
            if cell.value is not None:
                values[f["id"]] = cell.value
            return out
        if "derivation" in f:
            value = round(evaluate(f["derivation"], values), f["round"])
            values[f["id"]] = value
            return {**f, "value": int(value) if f["round"] == 0 else value}
        return f

    item = {**spec, "table": table, "table_sha256": hashlib.sha256((ROOT / table).read_bytes()).hexdigest()}
    item["required_facts"] = [fact(f) for f in spec.get("required_facts", [])]
    item["aux_facts"] = [fact(f) for f in spec.get("aux_facts", [])]
    return GoldenItem.model_validate(item)


def checklist(items: list[GoldenItem]) -> str:
    lines = ["# T1-19 신규 문항 웹 대조 체크리스트 (비커밋)", "",
             "KOSIS 웹 화면에서 셀 값을 확인하고 `[x]`. 다르면 `[!] 웹값=...`. 계산 fact는 대상 아님.", "",
             "| 확인 | 문항/fact | 표 | 시점 | 분류 | 골든 값 | 단위 |", "|---|---|---|---|---|---|---|"]
    for item in items:
        for f in item.required_facts + item.aux_facts:
            if f.source_ref:
                value = f.raw if f.value is None else f.value
                dims = ", ".join((f.dims or {}).values())
                lines.append(f"| [ ] | {item.id}/{f.id} | {Path(item.table).stem} | {f.period} | {dims} | {value} | {f.unit or ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="eval/golden/edge_spec.json")
    ap.add_argument("--out", default="eval/golden/stat_golden_edge.jsonl")
    ap.add_argument("--checklist", default="eval/golden/verify_checklist_edge.md")
    args = ap.parse_args(argv)
    specs = json.loads((ROOT / args.spec).read_text(encoding="utf-8"))
    items = [build_item(s) for s in specs]
    (ROOT / args.out).write_text("".join(i.model_dump_json(exclude_none=True, exclude_defaults=True) + "\n" for i in items),
                                 encoding="utf-8")
    (ROOT / args.checklist).write_text(checklist(items), encoding="utf-8")
    print(f"written: {args.out} ({len(items)} items), {args.checklist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
