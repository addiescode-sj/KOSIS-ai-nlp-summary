"""Human review: export a sheet of pending answers, then apply confirmed verdicts back to the results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from eval.scorer import ERROR_TYPES, VERDICTS, decide_verdict, read_jsonl, write_jsonl

SHEET_COLUMNS = (
    "question_id",
    "run_index",
    "kind",
    "question",
    "expected",
    "raw_answer",
    "automatic_checks",
    "error_types",
    "final_verdict",
    "review_note",
    "reviewer",
)


def _expected(item: dict) -> str:
    keys = ("required_facts", "aux_facts", "refusal_reason", "allowed_default", "clarify_conditions", "expected_handling")
    return json.dumps({k: item[k] for k in keys if k in item}, ensure_ascii=False)


def export_sheet(run_dir: Path, sheet: Path) -> int:
    """Write pending rows (with expected facts beside the answer) to a UTF-8 CSV; returns the row count."""
    results = read_jsonl(run_dir / "results.jsonl")
    golden = {i["id"]: i for i in read_jsonl(run_dir / "golden.jsonl")}
    rows = [r for r in results if r["review_status"] != "confirmed"]
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:  # BOM so Excel opens Korean correctly
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for r in rows:
            item = golden[r["question_id"]]
            writer.writerow(
                {
                    "question_id": r["question_id"],
                    "run_index": r["run_index"],
                    "kind": r["kind"],
                    "question": item["question"],
                    "expected": _expected(item),
                    "raw_answer": r["raw_answer"],
                    "automatic_checks": json.dumps(r["automatic_checks"], ensure_ascii=False),
                    "error_types": "",
                    "final_verdict": "",
                    "review_note": "",
                    "reviewer": "",
                }
            )
    return len(rows)


def apply_sheet(run_dir: Path, sheet: Path) -> tuple[int, list[str]]:
    """Confirm rows that have a reviewer; returns (confirmed count, problems). Nothing is written if any row is invalid."""
    results = read_jsonl(run_dir / "results.jsonl")
    index = {(r["question_id"], int(r["run_index"])): r for r in results}
    problems: list[str] = []
    updates: list[tuple[dict, dict]] = []
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            if not (row.get("reviewer") or "").strip():
                continue  # left pending on purpose
            key = (row["question_id"], int(row["run_index"]))
            target = index.get(key)
            where = f"line {line_no} ({key[0]}#{key[1]})"
            if target is None:
                problems.append(f"{where}: not in results")
                continue
            errors = [e.strip() for e in (row.get("error_types") or "").split(";") if e.strip()]
            bad = [e for e in errors if e not in ERROR_TYPES]
            if bad:
                problems.append(f"{where}: unknown error types {bad}; allowed: {ERROR_TYPES}")
                continue
            try:
                derived = decide_verdict(target["kind"], errors)
            except ValueError as exc:
                problems.append(f"{where}: {exc}")
                continue
            verdict = (row.get("final_verdict") or "").strip().upper() or derived
            if verdict not in VERDICTS:
                problems.append(f"{where}: verdict {verdict!r} not in {VERDICTS}")
            elif verdict != derived:
                problems.append(f"{where}: verdict {verdict} conflicts with error types (rules give {derived})")
            else:
                updates.append((target, {"verdict": verdict, "errors": errors, "row": row}))
    if problems:
        return 0, problems
    for target, u in updates:
        target.update(
            review_status="confirmed",
            final_verdict=u["verdict"],
            error_types=u["errors"],
            review_note=(u["row"].get("review_note") or "").strip(),
            reviewer=u["row"]["reviewer"].strip(),
        )
    write_jsonl(run_dir / "results.jsonl", results)
    return len(updates), []


def main(argv: list[str] | None = None) -> int:
    from kosis_chat.config import resolve_path

    parser = argparse.ArgumentParser(prog="python -m eval.review", description="Human review of eval answers.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="eval/results/<run_id>")
        p.add_argument("--sheet", default=None, help="review CSV (default: <run>/review.csv)")
    args = parser.parse_args(argv)

    run_dir = resolve_path(args.run)
    sheet = Path(args.sheet) if args.sheet else run_dir / "review.csv"
    if args.command == "export":
        count = export_sheet(run_dir, sheet)
        print(f"{count} pending rows -> {sheet}")
        print("Fill error_types (';'-separated) and reviewer; final_verdict may be left blank (derived from rules).")
        return 0
    confirmed, problems = apply_sheet(run_dir, sheet)
    if problems:
        print("Nothing applied. Fix these rows:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"confirmed {confirmed} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
