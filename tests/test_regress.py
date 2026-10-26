"""Regression gate: self-compare is clean, injected verdict drops are caught, a different exam or grader is refused."""

from __future__ import annotations

import json

from eval import regress
from eval.scorer import write_jsonl

GRADES = [
    {"question_id": "Q1", "kind": "answerable", "type": "value", "verdict": "PASS", "error_types": [], "details": ["76,309"]},
    {"question_id": "Q2", "kind": "not_answerable", "type": None, "verdict": "PASS", "error_types": [], "details": []},
    {"question_id": "Q3", "kind": "answerable", "type": "trend", "verdict": "FAIL", "error_types": ["wrong_dims"], "details": []},
]


def _run(tmp_path, name, grades, golden="g1"):
    run = tmp_path / name
    (run / "grades" / "20261026-000000-x").mkdir(parents=True)
    manifest = {"run_id": name, "golden_sha256": golden, "scoring_version": "v1", "pipeline": "stat",
                "model": {"model_name": "HCX-DASH-002", "temperature": 0.0}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "grades" / "20261026-000000-x" / "report.json").write_text(json.dumps({"grader_sha256": "h"}), encoding="utf-8")
    write_jsonl(run / "grades" / "20261026-000000-x" / "grades.jsonl", grades)
    write_jsonl(run / "results.jsonl", [{"question_id": g["question_id"], "run_index": 0,
                                         "decision_metadata": {"route": "rules", "compose": "model"}} for g in grades])
    return run


def test_self_compare_has_no_regression(tmp_path):
    run = _run(tmp_path, "base", GRADES)
    assert regress.main(["--run", str(run), "--base", str(run)]) == 0


def test_injected_drops_are_regressions_with_stage(tmp_path):
    base = _run(tmp_path, "base", GRADES)
    worse = [dict(GRADES[0], verdict="FAIL", error_types=["wrong_value"]),
             dict(GRADES[1], verdict="FAIL", error_types=["missed_refusal"]),
             dict(GRADES[2], verdict="PASS", error_types=[])]
    new = _run(tmp_path, "new", worse)
    result = regress.compare(regress.snapshot(base), regress.snapshot(new))
    assert result["counts"] == {"kept_pass": 0, "improvement": 1, "regression": 2, "kept_not_pass": 0}
    assert {r["id"]: r["stages"] for r in result["regressions"]} == {"Q1": ["3 계산"], "Q2": ["1 판정"]}
    assert result["refusal_regressions"] == ["Q2"]
    assert regress.main(["--run", str(new), "--base", str(base)]) == 1


def test_different_golden_is_refused(tmp_path):
    base = _run(tmp_path, "base", GRADES)
    new = _run(tmp_path, "new", GRADES, golden="g2")
    assert regress.main(["--run", str(new), "--base", str(base)]) == 2


def test_accept_pins_verdicts_without_data_values(tmp_path):
    run = _run(tmp_path, "base", GRADES)
    pinned = tmp_path / "baselines" / "stat.json"
    assert regress.main(["--run", str(run), "--base", str(pinned), "--accept"]) == 0
    assert "76,309" not in pinned.read_text(encoding="utf-8")
    assert regress.main(["--run", str(run), "--base", str(pinned)]) == 0
