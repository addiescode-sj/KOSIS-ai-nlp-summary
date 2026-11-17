"""Run and grade the multi-table dev set (T1-30): eval/dev/multi_table_dev.jsonl (scripts/build_multi_dev.py).

    python -m eval.multi_dev --backend oracle      # the gold plans as the model's: checks the pipeline computes them
    python -m eval.multi_dev --backend hcx         # the real model (HCX_API_KEY in the shell environment, no .env)
    python -m eval.multi_dev --backend mock        # no model: every plan fails (a refusal), the floor
    python -m eval.multi_dev --backend hcx --only P01,T01 --interval 2

KOSIS_WORKSPACE names the folder whose data/ holds tables.csv and the Parquet files. A question is answered through
kosis_chat.api.summarize_stats_run with its table keys. Grading:
- answer: PASS when the answer is answered, every expected cell value is one of its facts and the expected final
  value (rounded as the row says) is one of its facts.
- refuse: PASS when the answer is refused (and with the row's reason, when it names one).
Results go to eval/results/<time>-multi-<backend>/ (results.jsonl, summary.json) and a summary is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from kosis_chat import api
from kosis_chat.analyze import multi
from kosis_chat.llm.model import MockClient
from kosis_chat.schemas.resources import DEFAULT_MODEL, LLMConfig

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "eval" / "dev" / "multi_table_dev.jsonl"


def load(path: Path, only: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if not only or r["qid"] in only]


def oracle_client(row: dict) -> MockClient:
    """The row's gold plan as the model's plan (twice: a retry gets it again); nothing usable for the sentence."""
    plan = json.dumps(row.get("gold_plan") or {"steps": [], "show": [], "present": "value"}, ensure_ascii=False)
    return MockClient(lambda system, user: plan if system == multi.SYSTEM_PROMPT else "no json")


def _same(value, expected, digits: int) -> bool:
    return value is not None and round(float(value), digits) == round(float(expected), digits)


def grade(row: dict, stat) -> tuple[bool, str]:
    expected = row["expected"]
    if row["kind"] == "refuse":
        if stat.outcome != "refused":
            return False, f"not refused ({stat.outcome})"
        reason = expected.get("reason")
        if reason and stat.refusal.reason != reason:
            return False, f"reason {stat.refusal.reason} != {reason}"
        return True, "refused"
    if stat.outcome != "answered":
        return False, f"{stat.outcome}: {stat.refusal.reason if stat.refusal else ''}"
    values = [f.value for f in stat.result.facts]
    missing = [k for k, v in expected["cells"].items() if not any(_same(x, v, 6) for x in values)]
    if missing:
        return False, f"cells not read: {missing}"
    if not any(_same(x, expected["value"], expected["round"]) for x in values):
        return False, f"final {expected['value']} not among {values}"
    return True, "answered"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="multi-table dev set (T1-30)")
    parser.add_argument("--backend", choices=("oracle", "mock", "hcx"), default="oracle")
    parser.add_argument("--dev", type=Path, default=DEV)
    parser.add_argument("--only", default="", help="comma-separated qids")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--interval", type=float, default=0.0, help="seconds before a second model call (HCX 429)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load(args.dev, {q.strip() for q in args.only.split(",") if q.strip()} or None)
    llm = None
    if args.backend == "hcx":
        key = os.environ.get("HCX_API_KEY", "")
        if not key:
            print("HCX_API_KEY is not set in the shell environment", file=sys.stderr)
            return 2
        llm = LLMConfig(api_key=key, model=args.model, provider="hcx")
    out_dir = args.out or ROOT / "eval" / "results" / f"{datetime.now():%Y%m%d-%H%M%S}-multi-{args.backend}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        client = oracle_client(row) if args.backend == "oracle" else MockClient() if args.backend == "mock" else None
        started = time.perf_counter()
        try:
            out = api.summarize_stats_run(row["question"], row["table_keys"], client=client, llm=llm,
                                          retry_interval=args.interval)
            ok, why = grade(row, out.answer)
            record = {"qid": row["qid"], "relation": row["relation"], "pass": ok, "why": why,
                      "outcome": out.answer.outcome, "text": out.answer.text,
                      "facts": [f.model_dump(exclude_none=True) for f in out.answer.result.facts],
                      "plan": out.answer.query.plan.model_dump(exclude_none=True) if out.answer.query and out.answer.query.plan else None,
                      "numbers_verified": out.answer.numbers_verified,
                      "sources": [s.tbl_id for s in out.answer.sources], "chart": out.chart.kind if out.chart else None}
        except Exception as exc:  # a failed model call is a failed question, not a stopped run
            record = {"qid": row["qid"], "relation": row["relation"], "pass": False, "why": f"error: {exc}"}
        record["seconds"] = round(time.perf_counter() - started, 2)
        results.append(record)
        print(f"{record['qid']:4} {'PASS' if record['pass'] else 'FAIL'}  {record['why']}", file=sys.stderr)

    by_relation: dict[str, Counter] = {}
    for r in results:
        by_relation.setdefault(r["relation"], Counter())[r["pass"]] += 1
    summary = {"backend": args.backend, "model": args.model if args.backend == "hcx" else None, "dev": str(args.dev),
               "questions": len(results), "pass": sum(r["pass"] for r in results),
               "by_relation": {k: f"{c[True]}/{c[True] + c[False]}" for k, c in by_relation.items()}}
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
