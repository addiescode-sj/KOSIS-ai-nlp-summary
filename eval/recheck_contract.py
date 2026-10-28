"""Re-validate the StatAnswers of a saved run under the current contract (src/kosis_ai/schemas.py).

    python -m eval.recheck_contract --run eval/results/<run folder>

Reads results.jsonl, loads every stat_answer with StatAnswer.model_validate and prints how many load, how many fail
and why, and how the answers are distributed over numbers_origin (None = written before T0-03 v2) and route.
Exit code 1 when any stored answer no longer satisfies the contract.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from eval.scorer import read_jsonl
from kosis_ai.schemas import StatAnswer


def recheck(rows: list[dict]) -> tuple[int, list[tuple[str, str]], Counter]:
    """(answers that load, [(question_id, first problem)] that do not, Counter of 'numbers_origin/route')."""
    ok, bad, origins = 0, [], Counter()
    for row in rows:
        answer = row.get("stat_answer")
        route = (row.get("decision_metadata") or {}).get("route", "-")
        if answer is None:
            origins[f"(no answer: {row.get('call_status')})"] += 1
            continue
        try:
            loaded = StatAnswer.model_validate(answer)
        except ValidationError as exc:
            bad.append((row["question_id"], str(exc).splitlines()[1].strip()[:160]))
            continue
        ok += 1
        origins[f"{loaded.numbers_origin}/{route}"] += 1
    return ok, bad, origins


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.recheck_contract", description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder written by eval.run_eval")
    args = parser.parse_args(argv)
    rows = read_jsonl(Path(args.run) / "results.jsonl")
    ok, bad, origins = recheck(rows)
    print(f"rows {len(rows)}: load ok {ok}, load fail {len(bad)}")
    for key, count in sorted(origins.items()):
        print(f"  numbers_origin/route {key}: {count}")
    for question_id, problem in bad:
        print(f"  FAIL {question_id}: {problem}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
