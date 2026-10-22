"""Scoring rules, aggregation, and the offline run_eval -> review -> scorer round trip with MockClient."""

from __future__ import annotations

import csv
import json

import pytest

from eval import review, run_eval, scorer
from kosis_chat.data.loader import sha256_of
from kosis_chat.llm.model import LLMResult, MockClient


@pytest.mark.parametrize(
    ("kind", "errors", "verdict"),
    [
        ("answerable", [], "PASS"),
        ("answerable", ["no_source"], "PARTIAL"),
        ("answerable", ["no_source", "wrong_period"], "FAIL"),  # FAIL wins over PARTIAL
        ("answerable", ["false_refusal"], "FAIL"),
        ("not_answerable", [], "PASS"),
        ("not_answerable", ["insufficient_reason"], "PARTIAL"),
        ("not_answerable", ["missed_refusal"], "FAIL"),
        ("clarify", ["missing_clarification"], "PARTIAL"),
        ("clarify", ["assumed_condition"], "FAIL"),
        ("default_allowed", ["unstated_default"], "PARTIAL"),
        ("default_allowed", ["disallowed_default"], "FAIL"),
    ],
)
def test_decide_verdict_order(kind, errors, verdict):
    assert scorer.decide_verdict(kind, errors) == verdict


def test_error_type_must_fit_the_kind():
    with pytest.raises(ValueError):
        scorer.decide_verdict("not_answerable", ["wrong_value"])


def _row(kind, status="confirmed", verdict="PASS", run_index=0, call_status="ok", errors=()):
    return {
        "kind": kind,
        "review_status": status,
        "final_verdict": verdict if status == "confirmed" else None,
        "run_index": run_index,
        "call_status": call_status,
        "error_types": list(errors),
    }


def test_pending_rows_make_score_provisional_and_are_not_dropped():
    results = [_row("answerable"), _row("answerable", status="pending")]
    summary = scorer.aggregate(results)
    assert summary["provisional"] is True
    assert summary["pending"] == 1
    assert summary["by_kind"]["answerable"]["pass_rate"] == 0.5  # pending counts in the denominator


def test_run_failures_are_counted_not_excluded():
    results = [_row("answerable"), _row("answerable", verdict="FAIL", call_status="timeout", errors=["run_failure"])]
    summary = scorer.aggregate(results)
    assert summary["run_failures"] == 1
    assert summary["by_kind"]["answerable"]["total"] == 2


def test_multiple_runs_report_mean_and_spread():
    results = [_row("answerable", run_index=0), _row("answerable", verdict="FAIL", run_index=1, errors=["wrong_value"])]
    spread = scorer.aggregate(results)["overall_pass_rate_per_run"]
    assert (spread["mean"], spread["min"], spread["max"]) == (0.5, 0.0, 1.0)


@pytest.fixture
def golden(fixtures, tmp_path):
    table = fixtures / "pop_a.csv"
    items = [
        {"id": "q1", "kind": "answerable", "question": "2023년 여자 인구는?", "table": str(table),
         "table_sha256": sha256_of(table), "required_facts": [{"value": 550, "unit": "명", "period": "2023", "dims": {"성별": "여자"}}]},
        {"id": "q2", "kind": "not_answerable", "question": "2030년 인구는?", "table": str(table),
         "table_sha256": sha256_of(table), "refusal_reason": "없는 시점"},
        {"id": "q3", "kind": "clarify", "question": "인구는?", "table": str(table),
         "table_sha256": sha256_of(table), "clarify_conditions": ["시점"]},
    ]
    gdir = tmp_path / "golden" / "v1"
    gdir.mkdir(parents=True)
    (gdir / "pop_a.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return gdir


def test_golden_hash_mismatch_is_refused(golden, mock_settings, tmp_path):
    path = golden / "pop_a.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"table_sha256": "', '"table_sha256": "0'), encoding="utf-8")
    with pytest.raises(run_eval.GoldenError, match="sha256"):
        run_eval.run(golden, "B0", mock_settings, tmp_path / "results", client=MockClient("x"))


def test_round_trip_run_review_score(golden, mock_settings, tmp_path):
    replies = [
        "2023년 여자 가상 인구는 550명입니다. 출처: KOSIS / 가상 인구조사 / 가상 성별 인구",
        "2030년 자료는 표에 없습니다.",
        LLMResult(text="", status="timeout", error="slow"),
    ]
    run_dir = run_eval.run(golden, "B0", mock_settings, tmp_path / "results", client=MockClient(replies))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["golden_version"] == "v1" and manifest["stage"] == "B0"
    assert manifest["scoring_version"] == scorer.SCORING_VERSION

    results = scorer.read_jsonl(run_dir / "results.jsonl")
    assert results[0]["automatic_checks"]["expected_value_hits"] == [{"value": 550, "found_in_text": True}]
    assert results[0]["automatic_checks"]["source_mentioned"] is True
    assert results[2]["review_status"] == "confirmed" and results[2]["error_types"] == ["run_failure"]

    sheet = run_dir / "review.csv"
    assert review.export_sheet(run_dir, sheet) == 2  # the timeout row is already confirmed
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    rows[0].update(reviewer="tester")  # q1: no errors -> PASS
    rows[1].update(reviewer="tester", error_types="insufficient_reason")  # q2 -> PARTIAL
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert review.apply_sheet(run_dir, sheet) == (2, [])

    summary = scorer.aggregate(scorer.read_jsonl(run_dir / "results.jsonl"))
    assert summary["provisional"] is False
    assert summary["by_kind"]["answerable"]["pass"] == 1
    assert summary["by_kind"]["not_answerable"]["partial"] == 1
    assert summary["by_kind"]["clarify"]["fail"] == 1
    assert summary["run_failures"] == 1


def test_review_rejects_verdict_that_conflicts_with_rules(golden, mock_settings, tmp_path):
    run_dir = run_eval.run(golden, "B0", mock_settings, tmp_path / "results", client=MockClient("x"))
    sheet = run_dir / "review.csv"
    review.export_sheet(run_dir, sheet)
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    rows[0].update(reviewer="tester", error_types="wrong_value", final_verdict="PASS")
    with sheet.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    confirmed, problems = review.apply_sheet(run_dir, sheet)
    assert confirmed == 0 and "conflicts" in problems[0]
    assert all(r["review_status"] == "pending" for r in scorer.read_jsonl(run_dir / "results.jsonl"))


def test_history_row_marks_provisional(tmp_path):
    summary = scorer.aggregate([_row("answerable", status="pending")])
    path = tmp_path / "history.md"
    scorer.append_history(path, scorer.history_row(summary, "B0", "베이스라인", "run1"))
    text = path.read_text(encoding="utf-8")
    assert "| B0 | 베이스라인 | 1/0/1 | 예 |" in text


def test_pretest_runs_stay_out_of_history(golden, mock_settings, tmp_path, monkeypatch, capsys):
    run_dir = run_eval.run(golden, "pretest", mock_settings, tmp_path / "results", client=MockClient("x"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pretest"] is True
    written = []
    monkeypatch.setattr(scorer, "append_history", lambda path, row: written.append(row))
    assert scorer.main(["--run", str(run_dir), "--append-history"]) == 0
    assert written == [] and "사전 테스트" in capsys.readouterr().err


def test_interval_waits_between_calls_only(golden, mock_settings, tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(run_eval.time, "sleep", waits.append)
    client = MockClient("x")
    run_eval.run(golden, "B0", mock_settings, tmp_path / "results", client=client, interval=3.0)
    assert waits == [3.0] * (len(client.calls) - 1)
