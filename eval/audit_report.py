"""Offline diagnostic reports for durable and legacy eval runs; never grade semantic correctness."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from eval.recording import durable_json, durable_text


def _json_file(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value, None
    except (OSError, ValueError) as exc:
        return {}, {"category": "metadata_unavailable", "file": path.name, "error": str(exc)}


def _jsonl(path: Path):
    records, problems = [], []
    if not path.exists():
        return records, [{"category": "artifact_unavailable", "file": path.name}]
    try:
        raw_lines = path.read_bytes().splitlines()
    except OSError as exc:
        return records, [{"category": "artifact_unavailable", "file": path.name, "error": str(exc)}]
    for line_number, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(row, dict):
                raise ValueError("expected an object")
        except (ValueError, UnicodeError) as exc:
            problems.append({"category": "malformed_jsonl", "file": path.name, "line": line_number,
                             "error": str(exc), "raw_line": raw.decode("utf-8", "replace")})
        else:
            records.append((line_number, row))
    return records, problems


def _key(row: dict):
    question = row.get("question_id")
    repeat = row.get("run_index")
    return (question if isinstance(question, str) else None,
            repeat if isinstance(repeat, int) and not isinstance(repeat, bool) and repeat >= 0 else None)


def _stat(row: dict) -> dict:
    value = row.get("stat_answer")
    return value if isinstance(value, dict) else {}


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")[:180]


def _execution_metrics(results, json_mode):
    """Use result-row usage only, never add the same HTTP/generate tokens again."""
    totals, usage_rows, attempt_rows, attempts = {}, 0, 0, 0
    model_rows = []
    def merge(target, incoming):
        for key, value in incoming.items():
            if isinstance(value, dict):
                if not isinstance(target.get(key), dict):
                    target[key] = {}
                merge(target[key], value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                if not isinstance(target.get(key, 0), (int, float)):
                    target[key] = 0
                target[key] = target.get(key, 0) + value
    for _, row in results:
        usage = row.get("usage")
        if isinstance(usage, dict) and usage:
            merge(totals, usage)
            usage_rows += 1
        count = row.get("attempts")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            attempts += count
            attempt_rows += 1
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        raw = row.get("raw_responses")
        if json_mode and not judge.get("final") and isinstance(raw, list) and raw:
            model_rows.append(row)
    return {
        "usage_source": "results rows only; reported API usage, potentially partial on transport failures",
        "usage_totals": totals if usage_rows else None, "rows_with_usage": usage_rows,
        "http_attempts_from_rows": attempts if attempt_rows else None,
        "rows_with_attempt_counts": attempt_rows,
        "json_rows_with_saved_model_output": len(model_rows) if json_mode else None,
        "first_pass_success_rows": sum(row.get("call_status") == "ok" and len(row["raw_responses"]) == 1
                                      and not row.get("format_errors") for row in model_rows) if json_mode else None,
        "format_retried_rows": sum(len(row["raw_responses"]) > 1 for row in model_rows) if json_mode else None,
        "format_recovered_rows": sum(row.get("call_status") == "ok" and bool(row.get("format_errors"))
                                    for row in model_rows) if json_mode else None,
        "terminal_failure_rows": sum(row.get("call_status") not in (None, "ok") for _, row in results),
        "limits": "First-pass refers to saved JSON/model processing, not HTTP retries or semantic correctness; absent outputs are unmeasured.",
    }


def inspect_run(directory: Path) -> tuple[dict, dict]:
    manifest, manifest_problem = _json_file(directory / "manifest.json")
    inputs, _ = _json_file(directory / "input_manifest.json")
    meta = manifest or inputs
    results, result_problems = _jsonl(directory / "results.jsonl")
    events, event_problems = _jsonl(directory / "events.jsonl")
    golden, golden_problems = _jsonl(directory / "golden.jsonl")
    problems = [*result_problems, *golden_problems]
    if manifest_problem:
        problems.append(manifest_problem)
    if not events:
        problems.extend(event_problems or [{"category": "trace_unavailable", "file": "events.jsonl"}])
    else:
        problems.extend(event_problems)
    golden_ids = [row.get("id") for _, row in golden if isinstance(row.get("id"), str)]
    duplicate_golden = [identity for identity, count in collections.Counter(golden_ids).items() if count > 1]
    if duplicate_golden:
        problems.append({"category": "duplicate_golden_ids", "ids": duplicate_golden})
    if len(golden_ids) != len(golden):
        problems.append({"category": "invalid_golden_identity"})
    if golden and meta.get("questions") != len(golden):
        problems.append({"category": "golden_count_mismatch", "declared": meta.get("questions"), "observed": len(golden)})
    if golden and isinstance(meta.get("runs"), int) and "expected_rows" in meta and meta["expected_rows"] != len(golden) * meta["runs"]:
        problems.append({"category": "expected_count_mismatch"})
    expected = None
    run_count = meta.get("runs")
    if isinstance(run_count, int) and run_count > 0 and golden:
        expected = {(row["id"], index) for _, row in golden if isinstance(row.get("id"), str) for index in range(run_count)}
    for line, row in results:
        if None in _key(row):
            problems.append({"category": "invalid_result_identity", "file": "results.jsonl", "line": line, "detail": row})
    keys = [_key(row) for _, row in results]
    counted = collections.Counter(keys)
    duplicates = sorted((key for key, count in counted.items() if count > 1), key=str)
    missing = sorted(expected - set(keys), key=str) if expected is not None else None
    unexpected = sorted(set(keys) - expected, key=str) if expected is not None else []
    for key in duplicates:
        problems.append({"category": "duplicate_result", "question_id": key[0], "run_index": key[1], "count": counted[key]})
    for key in unexpected:
        problems.append({"category": "unexpected_result", "question_id": key[0], "run_index": key[1]})
    completed = {_key(row): row for _, row in results}
    event_ids = {row.get("event_id") for _, row in events if isinstance(row.get("event_id"), str)}
    errors = []
    validation_keys = set()
    for line, event in events:
        kind = event.get("type")
        if not isinstance(kind, str):
            problems.append({"category": "invalid_event_type", "file": "events.jsonl", "line": line})
            continue
        category = {"validation_error": "format_validation", "contract_validation_error": "contract_validation",
                    "generate_raised": "call_raised", "http_raised": "transport_raised"}.get(kind)
        if kind in ("http_response", "generate_response") and event.get("call_status") not in (None, "ok"):
            category = "http_attempt_failure" if kind == "http_response" else "call_failure"
        if category:
            key = _key(event)
            if kind == "validation_error":
                validation_keys.add(key)
            errors.append({"category": category, "event_id": event.get("event_id"), "event_line": line,
                           "question_id": key[0], "run_index": key[1], "generate_id": event.get("generate_id"),
                           "recovered": completed.get(key, {}).get("call_status") == "ok", "detail": event})
    for line, row in results:
        key = _key(row)
        if key not in validation_keys:
            row_errors = row.get("format_errors", [])
            if not isinstance(row_errors, list):
                problems.append({"category": "invalid_format_errors", "file": "results.jsonl", "line": line})
                row_errors = []
            for index, error in enumerate(row_errors):
                errors.append({"category": "recorded_format_error", "question_id": key[0], "run_index": key[1],
                               "reference": f"results.jsonl:{line}:format_errors:{index}", "detail": error,
                               "recovered": row.get("call_status") == "ok"})
        if row.get("call_status") not in (None, "ok"):
            errors.append({"category": "terminal_row_failure", "question_id": key[0], "run_index": key[1],
                           "reference": f"results.jsonl:{line}", "detail": row.get("call_error"), "recovered": False})
        references = row.get("event_ids", [])
        for identity in (references if isinstance(references, list) else []):
            if not isinstance(identity, str) or identity not in event_ids:
                problems.append({"category": "missing_referenced_event", "event_id": identity, "result_line": line})
    pending = sum(row.get("review_status") != "confirmed" for _, row in results)
    human_confirmed = sum(row.get("review_status") == "confirmed" and bool(row.get("reviewer"))
                          and not str(row.get("reviewer")).startswith("auto:") for _, row in results)
    auto_confirmed = sum(row.get("review_status") == "confirmed" and str(row.get("reviewer", "")).startswith("auto:") for _, row in results)
    matrix = collections.Counter(f"{row.get('kind', 'unknown')} | {_stat(row).get('outcome', 'unavailable')}" for _, row in results)
    golden_by_id = {item.get("id"): item for _, item in golden if isinstance(item.get("id"), str)}
    reason_diagnostics = []
    for _, row in results:
        refusal = _stat(row).get("refusal")
        if isinstance(refusal, dict):
            allowed = golden_by_id.get(_key(row)[0], {}).get("allowed_reasons")
            reason_diagnostics.append({"question_id": row.get("question_id"), "run_index": row.get("run_index"),
                                       "reason": refusal.get("reason"), "allowed_match": refusal.get("reason") in allowed if isinstance(allowed, list) else None})
    new_run = bool(meta.get("schema_version"))
    broken = bool(missing or duplicates or unexpected or any(problem["category"] in
                  ("malformed_jsonl", "missing_referenced_event", "invalid_result_identity", "duplicate_golden_ids", "invalid_golden_identity", "golden_count_mismatch", "expected_count_mismatch") for problem in problems))
    state = meta.get("status", "legacy_completion_unknown")
    recovered_status = None
    if manifest_problem and inputs:
        statuses = [event.get("status") for _, event in events if event.get("type") in ("run_status", "run_ended") and event.get("status") in ("running", "completed", "failed", "interrupted")]
        recovered_status = statuses[-1] if statuses else "running"
        state = recovered_status
    if broken or (new_run and (state != "completed" or not events)):
        state = "incomplete" if state not in ("failed", "interrupted") else state
    required = ("golden_sha256", "data_sha256", "model", "scope_mode", "serialization", "form", "runs", "judge_mode")
    comparable = all(name in meta for name in required) and all(meta[name] is not None for name in required if name != "form")
    comparable = comparable and isinstance(meta.get("golden_sha256"), str) and isinstance(meta.get("data_sha256"), dict) and isinstance(meta.get("model"), dict) and isinstance(meta.get("runs"), int) and not isinstance(meta.get("runs"), bool) and meta.get("runs", 0) > 0 and meta.get("judge_mode") in ("off", "rules")
    comparable = comparable and all(name in meta.get("model", {}) for name in ("backend", "model_name", "temperature", "max_tokens", "thinking_level", "max_input_tokens", "timeout_seconds", "base_url"))
    model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
    hashes = meta.get("data_sha256") if isinstance(meta.get("data_sha256"), dict) else {}
    valid_hash = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None
    model_fields = {"backend", "model_name", "temperature", "max_tokens", "thinking_level", "max_input_tokens", "timeout_seconds", "base_url"}
    comparable = (comparable and valid_hash(meta.get("golden_sha256")) and bool(hashes)
                  and all(isinstance(key, str) and key and valid_hash(value) for key, value in hashes.items())
                  and model_fields <= model.keys() and bool(model.get("backend")) and bool(model.get("model_name"))
                  and not broken and not meta.get("pretest", False))
    signature = hashlib.sha256(json.dumps({name: meta[name] for name in required}, sort_keys=True).encode()).hexdigest() if comparable else None
    report = {
        "execution_metrics": _execution_metrics(results, meta.get("answer_format") == "json"),
        "pretest": meta.get("pretest", "unknown"),
        "report_version": "eval-audit-v1", "run_id": meta.get("run_id", directory.name), "state": state,
        "declared_status": meta.get("status", "unavailable"), "recovered_status_from_events": recovered_status, "results_rows": len(results), "unique_results": len(counted),
        "expected_rows": len(expected) if expected is not None else meta.get("expected_rows"),
        "missing_results": [{"question_id": key[0], "run_index": key[1]} for key in (missing or [])] if missing is not None else None,
        "successful_result_rows": sum(row.get("call_status") == "ok" and isinstance(row.get("stat_answer"), dict) for _, row in results),
        "format_valid_rows": sum(row.get("call_status") == "ok" and isinstance(row.get("stat_answer"), dict) for _, row in results),
        "format_valid_rows_definition": "successful structured results; may include rule bypass; not model JSON compliance",
        "model_format_valid_rows": sum(row.get("call_status") == "ok" and isinstance(row.get("stat_answer"), dict)
            and not (row.get("judge") if isinstance(row.get("judge"), dict) else {}).get("final")
            and bool(row.get("raw_responses")) for _, row in results) if meta.get("answer_format") == "json" else None,
        "model_format_metric_status": "JSON outputs only, rule bypass excluded" if meta.get("answer_format") == "json" else "not_applicable_or_legacy_unknown",
        "human_confirmed_rows": human_confirmed, "auto_confirmed_rows": auto_confirmed, "pending_rows": pending,
        "semantic_correctness": "not_automatically_determined", "kind_outcome": dict(sorted(matrix.items())),
        "refusal_reason_diagnostics": reason_diagnostics,
        "default_selection_unresolved_rows": sum((row.get("decision_metadata") if isinstance(row.get("decision_metadata"), dict) else {}).get("default_selection") == "not_implemented" for _, row in results),
        "trace_availability": "events_present" if events else "unavailable",
        "observed_http_requests": sum(row.get("type") == "http_request" for _, row in events),
        "http_trace_availability": "observed" if any(row.get("type") == "http_request" for _, row in events) else "unavailable",
        "source_changed_during_run": meta.get("source_changed_during_run", "unavailable"),
        "source_snapshot_availability": "present" if (directory / "source_snapshot.json").exists() else "unavailable",
        "diagnostic_error_events": len(errors),
        "error_history_coverage": "journal_available" if events else "unavailable; zero recorded errors does not mean zero errors", "anomalies": problems,
        "comparison_signature": signature, "comparability": "known_conditions" if signature else "unknown",
        "treatment": {name: meta.get(name, "unavailable") for name in ("prompt_version", "response_schema_sha256", "answer_format")},
        "timing": meta.get("timing", {}), "causal_improvement": "not_established",
    }
    return report, {"run_id": report["run_id"], "errors": errors, "anomalies": problems,
                    "limits": "Error events may describe the same failure at multiple layers; missing traces are not zero errors."}


def save_reports(directory: Path) -> tuple[dict, dict]:
    report, errors = inspect_run(directory)
    durable_json(directory / "run_report.json", report)
    durable_json(directory / "error_report.json", errors)
    lines = [f"# Run diagnostic: {_cell(report['run_id'])}", "", f"State: {_cell(report['state'])}",
             f"Recorded rows: {report['results_rows']}; successful structured results: {report['successful_result_rows']}; human confirmed: {report['human_confirmed_rows']}; pending: {report['pending_rows']}",
             f"Model JSON success rows: {report['model_format_valid_rows']}; pretest: {report['pretest']}",
             "", "Successful structured results may include rule bypass; they are not model JSON compliance or semantic correctness. Automatic failure confirmation is not human review.",
             f"Trace: {report['trace_availability']}; HTTP: {report['http_trace_availability']}; source snapshot: {report['source_snapshot_availability']}",
             "", "| Kind and outcome | Rows |", "| --- | --- |"]
    lines.extend(f"| {_cell(key)} | {value} |" for key, value in report["kind_outcome"].items())
    lines += ["", "Execution metrics (row usage counted once):", "```json", json.dumps(report["execution_metrics"], ensure_ascii=False, indent=2), "```"]
    lines += ["", f"Missing results: {_cell(report['missing_results'])}", f"Anomalies: {len(report['anomalies'])}",
              "", "See run_report.json for full diagnostics; semantic correctness and causal improvement are not established."]
    durable_text(directory / "run_report.md", "\n".join(lines) + "\n")
    lines = [f"# Error diagnostic: {_cell(report['run_id'])}", "", errors["limits"], "",
             "| Reference | Question / repeat | Category | Recovered | Detail |", "| --- | --- | --- | --- | --- |"]
    for error in errors["errors"]:
        reference = error.get("event_id") or error.get("reference", "unknown")
        target = "events.jsonl" if error.get("event_id") else "results.jsonl"
        detail = error["detail"]
        if isinstance(detail, dict):
            detail = detail.get("error") or detail.get("call_status") or detail.get("exception_type") or detail
        lines.append(f"| [{_cell(reference)}]({target}) | {_cell(error.get('question_id'))} / {_cell(error.get('run_index'))} | {_cell(error['category'])} | {error['recovered']} | {_cell(detail)} |")
    lines += ["", "Full raw details and incomplete-artifact diagnostics: error_report.json and the original event/results journals."]
    durable_text(directory / "error_report.md", "\n".join(lines) + "\n")
    return report, errors


def save_history(root: Path) -> dict:
    reports = []
    for directory in sorted(root.iterdir()):
        if directory.is_dir() and not directory.name.startswith("_"):
            report, _ = save_reports(directory)
            reports.append(report)
    output = root / "_reports"
    output.mkdir(exist_ok=True)
    groups = collections.defaultdict(list)
    for report in reports:
        if report["comparison_signature"]:
            groups[report["comparison_signature"]].append(report["run_id"])
    history = {"runs": reports, "comparable_condition_groups": dict(sorted(groups.items())),
               "limits": "Unknown legacy metadata is not comparable. Treatment labels and times do not prove causal or quality improvement."}
    durable_json(output / "history_report.json", history)
    lines = ["# Evaluation history diagnostics", "", history["limits"], "",
             "| Run | State | Rows | Successful structured results | Pending | Comparability |", "| --- | --- | --- | --- | --- | --- |"]
    for report in reports:
        lines.append(f"| {_cell(report['run_id'])} | {_cell(report['state'])} | {report['results_rows']} | {report['format_valid_rows']} | {report['pending_rows']} | {report['comparability']} |")
    durable_text(output / "history_report.md", "\n".join(lines) + "\n")
    return history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline diagnostics; never calls an LLM or confirms semantic PASS.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path)
    target.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.run:
        save_reports(args.run)
    else:
        save_history(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
