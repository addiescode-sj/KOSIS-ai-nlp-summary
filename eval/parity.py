"""Parity gate: check that two run_eval folders recorded the same answers, ignoring time- and host-dependent fields.

    python -m eval.parity --base eval/results/<run_a> --run eval/results/<run_b>
    python -m eval.parity --base <run_a> --run <run_b> --ignore decision_metadata.compose_status
    python -m eval.parity --base <run_a> --run <run_b> --check-calls

Place in the flow (T1-28): every refactor step runs the mock golden set and compares its results.jsonl with the
pinned S0 mock run. Rows are matched by (question_id, run_index). Every field is compared except the paths in
VOLATILE_PATHS (plus any --ignore paths); values must have the same JSON type and dict keys the same order, so
"identical" means the rows serialize to the same bytes once the volatile fields are dropped. Prints `identical` and
exits 0, or prints each differing path per question and exits 1. Exit 2 means the inputs could not be read (missing
file, bad JSON or encoding, bad run_index, duplicate rows) or the command line was wrong (argparse usage errors, such
as a missing --base or a bad --ignore path, also exit 2).

Run conditions: both folders must have manifest.json (run_eval writes it); the keys in CONDITION_KEYS (golden and
data hashes, question count, scope/judge mode, model settings, ...) must match before rows are compared; otherwise
the two runs did not answer the same thing and parity exits 2. --allow-condition KEY skips one key on purpose.

Completeness: both folders must have manifest.json with status "completed" and exactly `questions` question ids,
each once per run index 0..runs-1 (and, when the manifest lists `question_ids`, exactly those ids); otherwise exit 2.
Two runs that lost the same question would otherwise compare `identical`. --allow-missing-manifest compares bare
results.jsonl files without these checks.

--check-calls also compares, row by row, what VOLATILE_PATHS leaves out of the byte check on purpose: `attempts`
(HTTP attempts) together with call_status, the number of format errors and the (name, status) list of the spans.
Use it for mock runs, where attempts are deterministic; a change there means a retry count moved.

A printed diff path can be passed back as --ignore: keys that contain dots or brackets are printed quoted, e.g.
decision_metadata.spans[1].attributes["gen_ai.request.model"], and the --ignore grammar reads that form too.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("kosis.eval.parity")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
RESULTS_FILE = "results.jsonl"

# Fields that change between two runs of the same code and data. Dotted paths from the results.jsonl row;
# `[*]` matches every item of a list. An ignored path also ignores everything below it.
VOLATILE_PATHS: tuple[str, ...] = (
    "latency_seconds",  # wall-clock time of the call (eval/run_eval.py)
    "retry_wait_seconds",  # measured sleep time around retries (perf_counter), not the configured interval
    "started_at",  # run start time; kept for rows that carry it (manifest.json has it today)
    # HTTP attempts depend on transient HCX errors (429, timeouts). The number of LLM calls is still compared through
    # decision_metadata.spans (one span per call) and raw_responses, so do not exclude spans attributes wholesale.
    "attempts",
    "usage",  # token counts reported by the backend
    "decision_metadata.spans[*].start_ms",  # span offsets from the question start (kosis_chat/pipeline.py _Calls)
    "decision_metadata.spans[*].duration_ms",  # span durations
    # File mtime date of the table (kosis_chat/data/loader.py:106); differs between data copies. No row carries it
    # today (StatAnswer.source has no fetched_at); kept so a later StatOutput/source layout stays covered.
    "source.fetched_at",
    "stat_answer.source.fetched_at",
)


MANIFEST_FILE = "manifest.json"
# manifest.json keys that define what was asked and how (eval/run_eval.py). Timing, run id, source hashes and
# prompt hashes are outcomes or bookkeeping, not conditions, so they are not here.
CONDITION_KEYS: tuple[str, ...] = (
    "golden_sha256", "golden_snapshot_sha256", "data_sha256", "questions", "runs", "pipeline", "answer_format",
    "serialization", "scope_mode", "judge_mode", "form", "interval_seconds", "model",
)


class _Missing:
    """Marks a key or list item that exists on only one side."""

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()
_BARE_KEY = re.compile(r'[^.\[\]"]+')
_BRACKET = re.compile(r'\[(?:(?P<star>\*)|(?P<index>\d+)|(?P<quoted>"(?:[^"\\]|\\.)*"))\]')


class ParityError(ValueError):
    """The inputs cannot be compared (missing or unreadable file, bad JSON, bad run_index, duplicate rows)."""


def parse_path(path: str) -> tuple[str | int, ...]:
    """'a.b[*].c["d.e"]' -> ('a', 'b', '*', 'c', 'd.e'); a list index is an int, `[*]` is the wildcard '*'.

    Grammar: segments joined by '.'; a segment is an optional bare key (no '.', '[', ']' or '"') followed by any
    number of `[*]`, `[<digits>]` or `["<JSON string>"]` brackets, and is never empty. The quoted form is what
    format_path prints for keys with dots or brackets, so parse_path(format_path(p)) == p.
    """
    parts: list[str | int] = []
    pos = 0
    while True:
        start = pos
        bare = _BARE_KEY.match(path, pos)
        if bare:
            parts.append(bare.group())
            pos = bare.end()
        while (bracket := _BRACKET.match(path, pos)) is not None:
            if bracket["quoted"] is not None:
                try:
                    parts.append(json.loads(bracket["quoted"]))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad ignore path {path!r} (quoted key: {exc.msg})") from exc
            elif bracket["index"] is not None:
                parts.append(int(bracket["index"]))
            else:
                parts.append("*")
            pos = bracket.end()
        if pos == start:
            raise ValueError(f"bad ignore path {path!r}")
        if pos == len(path):
            return tuple(parts)
        if path[pos] != ".":
            raise ValueError(f"bad ignore path {path!r}")
        pos += 1


def format_path(parts: tuple[str | int, ...]) -> str:
    """('a', 0, 'b.c') -> 'a[0]["b.c"]'; keys that contain dots or brackets are quoted."""
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        elif re.search(r"[.\[\]\"]", part) or not part:
            out += f"[{json.dumps(part, ensure_ascii=False)}]"
        else:
            out += f".{part}" if out else part
    return out or "<row>"


def is_ignored(parts: tuple[str | int, ...], patterns: list[tuple[str | int, ...]]) -> bool:
    """True when some pattern matches `parts` or one of its prefixes (`'*'` matches any list index)."""
    for pattern in patterns:
        if len(pattern) <= len(parts) and all(
            (p == "*" and isinstance(q, int)) or p == q for p, q in zip(pattern, parts)
        ):
            return True
    return False


def short(value: Any, limit: int = 80) -> str:
    """Compact JSON text of a value, cut to `limit` characters."""
    text = repr(value) if value is MISSING else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def diff_values(base: Any, run: Any, patterns: list[tuple[str | int, ...]],
                path: tuple[str | int, ...] = ()) -> list[tuple[str, Any, Any]]:
    """Every (path, base value, run value) where the two JSON values differ, skipping ignored paths."""
    if is_ignored(path, patterns):
        return []
    if isinstance(base, dict) and isinstance(run, dict):
        diffs: list[tuple[str, Any, Any]] = []
        for key in [*base, *(k for k in run if k not in base)]:
            diffs += diff_values(base.get(key, MISSING), run.get(key, MISSING), patterns, (*path, key))
        shared_base = [k for k in base if k in run and not is_ignored((*path, k), patterns)]
        shared_run = [k for k in run if k in base and not is_ignored((*path, k), patterns)]
        if not diffs and shared_base != shared_run:  # same content, different bytes
            diffs.append((format_path(path) + " (key order)", shared_base, shared_run))
        return diffs
    if isinstance(base, list) and isinstance(run, list):
        diffs = []
        for index in range(max(len(base), len(run))):
            diffs += diff_values(base[index] if index < len(base) else MISSING,
                                 run[index] if index < len(run) else MISSING, patterns, (*path, index))
        return diffs
    # Same type and same JSON text: 1, 1.0 and True are equal in Python but serialize differently; NaN != NaN in
    # Python but both serialize as NaN; -0.0 == 0.0 in Python but the bytes differ. (MISSING fails the type check
    # first, so json.dumps never sees it.)
    if type(base) is type(run) and json.dumps(base) == json.dumps(run):
        return []
    return [(format_path(path), base, run)]


def load_rows(run_dir: Path) -> dict[tuple[str, int], dict]:
    """results.jsonl of a run folder (or the file itself) keyed by (question_id, run_index), in file order."""
    path = run_dir / RESULTS_FILE if run_dir.is_dir() else run_dir
    if not path.is_file():
        raise ParityError(f"{path}: no {RESULTS_FILE}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ParityError(f"{path}: cannot read ({exc})") from exc
    rows: dict[tuple[str, int], dict] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParityError(f"{path}:{line_no}: bad JSON ({exc.msg})") from exc
        if not isinstance(row, dict) or "question_id" not in row:
            raise ParityError(f"{path}:{line_no}: row has no question_id")
        try:
            key = (str(row["question_id"]), int(row.get("run_index", 0)))
        except (ValueError, TypeError) as exc:
            raise ParityError(f"{path}:{line_no}: bad run_index {row.get('run_index')!r}") from exc
        if key in rows:
            raise ParityError(f"{path}:{line_no}: duplicate row {key[0]} run {key[1]}")
        rows[key] = row
    return rows


def load_manifest(run_dir: Path) -> dict | None:
    """manifest.json of a run folder, or None when the folder has none (a bare results.jsonl was given)."""
    path = run_dir / MANIFEST_FILE if run_dir.is_dir() else run_dir.parent / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityError(f"{path}: cannot read ({exc})") from exc
    if not isinstance(data, dict):
        raise ParityError(f"{path}: not a JSON object")
    return data


def condition_diffs(base: dict, run: dict, allow: tuple[str, ...] = ()) -> list[tuple[str, Any, Any]]:
    """(key, base value, run value) for every CONDITION_KEYS entry the two manifests disagree on."""
    found = []
    for key in CONDITION_KEYS:
        if key in allow:
            continue
        old, new = base.get(key, MISSING), run.get(key, MISSING)
        if not (type(old) is type(new) and json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)):
            found.append((key, old, new))
    return found


def completeness(manifest: dict, rows: dict[tuple[str, int], dict]) -> list[str]:
    """Why a run folder is not a finished run of the golden set it names: status not "completed", or rows that do not
    match the manifest (each of `questions` question ids once per run index 0..runs-1; `question_ids`, when the
    manifest lists them, must be exactly the ids recorded). Empty when complete."""
    problems = []
    if manifest.get("status") != "completed":
        problems.append(f"manifest status is {manifest.get('status', MISSING)!r}, not 'completed'")
    questions, runs = manifest.get("questions"), manifest.get("runs")
    if not isinstance(questions, int) or not isinstance(runs, int):
        return problems + [f"manifest has no question count / runs ({questions!r}, {runs!r})"]
    ids = {qid for qid, _ in rows}
    expected_ids = manifest.get("question_ids")
    if expected_ids is not None and set(map(str, expected_ids)) != ids:
        missing, extra = sorted(set(map(str, expected_ids)) - ids), sorted(ids - set(map(str, expected_ids)))
        problems.append(f"question ids differ from the manifest: missing {missing[:5]}, extra {extra[:5]}")
    if len(ids) != questions:
        problems.append(f"{len(ids)} question ids in results, manifest says {questions}")
    expected = {(qid, index) for qid in ids for index in range(runs)}
    if set(rows) != expected:
        problems.append(f"{len(rows)} rows, expected {len(expected)} ({len(ids)} questions x {runs} runs, "
                        f"run_index 0..{runs - 1})")
    return problems


def call_record(row: dict) -> dict:
    """What --check-calls compares for one row: attempts, call status, format error count, span (name, status)."""
    spans = (row.get("decision_metadata") or {}).get("spans") or []
    return {
        "attempts": row.get("attempts", MISSING),
        "call_status": row.get("call_status", MISSING),
        "format_errors": len(row.get("format_errors") or []),
        "spans": [[s.get("name"), s.get("status")] for s in spans if isinstance(s, dict)],
    }


def compare_calls(base: dict[tuple[str, int], dict], run: dict[tuple[str, int], dict]) -> dict:
    """Rows in both runs whose call_record differs -> {key: [(field, base, run)]}."""
    out = {}
    for key in base:
        if key in run:
            old, new = call_record(base[key]), call_record(run[key])
            found = [(f, old[f], new[f]) for f in old if json.dumps(old[f], default=repr) != json.dumps(new[f], default=repr)]
            if found:
                out[key] = found
    return out


def compare(base: dict[tuple[str, int], dict], run: dict[tuple[str, int], dict],
            ignore: tuple[str, ...] = ()) -> dict:
    """Question-by-question parity of two runs -> {'identical', 'diffs', 'only_base', 'only_run', 'order_changed'}."""
    patterns = [parse_path(p) for p in (*VOLATILE_PATHS, *ignore)]
    diffs = {}
    for key in base:
        if key in run:
            found = diff_values(base[key], run[key], patterns)
            if found:
                diffs[key] = found
    only_base = [k for k in base if k not in run]
    only_run = [k for k in run if k not in base]
    order_changed = [k for k in base if k in run] != [k for k in run if k in base]
    return {"identical": not (diffs or only_base or only_run or order_changed), "diffs": diffs,
            "only_base": only_base, "only_run": only_run, "order_changed": order_changed}


def _label(key: tuple[str, int]) -> str:
    return key[0] if key[1] == 0 else f"{key[0]} (run {key[1]})"


def render(result: dict, limit: int = 80) -> str:
    """`identical`, or one block per differing question plus the questions found in only one run."""
    if result["identical"]:
        return "identical"
    lines = []
    for key, found in result["diffs"].items():
        lines.append(f"{_label(key)}: {len(found)} path(s) differ")
        lines += [f"  {path}: base={short(old, limit)} run={short(new, limit)}" for path, old, new in found]
    if result["only_base"]:
        lines.append("only in base: " + ", ".join(_label(k) for k in result["only_base"]))
    if result["only_run"]:
        lines.append("only in run: " + ", ".join(_label(k) for k in result["only_run"]))
    if result["order_changed"]:
        lines.append("row order differs between the two results.jsonl files")
    for key, found in result.get("calls", {}).items():
        lines.append(f"{_label(key)}: calls differ")
        lines += [f"  {field}: base={short(old, limit)} run={short(new, limit)}" for field, old, new in found]
    lines.append(f"different: {len(result['diffs'])} question(s), only in base {len(result['only_base'])}, "
                 f"only in run {len(result['only_run'])}"
                 + (f", calls {len(result['calls'])}" if "calls" in result else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.parity", description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="reference run_eval folder (or its results.jsonl)")
    parser.add_argument("--run", required=True, help="run_eval folder to check (or its results.jsonl)")
    parser.add_argument("--ignore", action="append", default=[], metavar="PATH",
                        help="extra dotted path to skip, `[*]` for every list item (repeatable), e.g. judge.reason")
    parser.add_argument("--width", type=int, default=80, help="characters shown per value (default 80)")
    parser.add_argument("--check-calls", action="store_true",
                        help="also compare attempts, call_status, format error count and span (name, status) per row")
    parser.add_argument("--allow-missing-manifest", action="store_true",
                        help="compare even when a side has no manifest.json (no condition or completeness check)")
    parser.add_argument("--allow-condition", action="append", default=[], metavar="KEY", choices=CONDITION_KEYS,
                        help="manifest condition key allowed to differ (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    try:
        for path in args.ignore:
            parse_path(path)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        base, run = load_rows(Path(args.base)), load_rows(Path(args.run))
        base_manifest, run_manifest = load_manifest(Path(args.base)), load_manifest(Path(args.run))
    except ParityError as exc:
        log.error("%s", exc)
        return 2
    if base_manifest is None or run_manifest is None:
        if not args.allow_missing_manifest:
            log.error("no %s on one side: the run conditions and completeness cannot be checked "
                      "(pass --allow-missing-manifest to compare the rows anyway)", MANIFEST_FILE)
            return 2
        log.warning("no %s on one side: run conditions and completeness not checked", MANIFEST_FILE)
    else:
        problems = [f"{side}: {problem}" for side, manifest, rows in (("base", base_manifest, base),
                                                                       ("run", run_manifest, run))
                    for problem in completeness(manifest, rows)]
        if problems:
            for problem in problems:
                log.error("%s", problem)
            log.error("a run that did not finish or lost rows cannot pass the gate")
            return 2
        conditions = condition_diffs(base_manifest, run_manifest, tuple(args.allow_condition))
        if conditions:
            for key, old, new in conditions:
                log.error("run condition %s differs: base=%s run=%s", key, short(old, args.width), short(new, args.width))
            log.error("the two runs did not answer the same golden set under the same settings; "
                      "pass --allow-condition KEY to compare anyway")
            return 2
    log.info("base %s: %d rows, run %s: %d rows, ignored paths %d", args.base, len(base), args.run, len(run),
             len(VOLATILE_PATHS) + len(args.ignore))
    result = compare(base, run, tuple(args.ignore))
    if args.check_calls:
        result["calls"] = compare_calls(base, run)
        result["identical"] = result["identical"] and not result["calls"]
    print(render(result, args.width))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
