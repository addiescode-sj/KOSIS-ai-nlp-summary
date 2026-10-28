"""Automatic grading: each injected error maps to the RULES error type and verdict; the golden oracle grades PASS."""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from eval.golden import GoldenError, GoldenItem, load_items, read_cells
from eval.grade import evaluate, grade
from eval.harness import grade_all, oracle_answer, save_report, summarize
from kosis_ai.schemas import Fact, FactPart, Refusal, Source, StatAnswer, StatQuery, StatResult, TextPart
from tests.conftest import write_table

ROOT = Path(__file__).resolve().parents[1]
F1 = Fact(id="f1", value=500, unit="명", period="2023", source_ref="GRADE_TBL|T1|1|2023")
F2 = Fact(id="f2", value=550, unit="명", period="2023", source_ref="GRADE_TBL|T1|2|2023")
F3 = Fact(id="f3", value=1.1, derivation="f2 / f1", round=1)
SOURCE = Source(org_id="101", tbl_id="GRADE_TBL", survey_nm="가상 인구조사", tbl_nm="가상 성별 인구")
# Made-up cells with consistent value_kind (schemas.Cell rejects a symbol marked "ok").
_CELLS = {("0", "2022"): "1000", ("1", "2022"): "480", ("2", "2022"): "520",
          ("0", "2023"): "1050", ("1", "2023"): "500", ("2", "2023"): "550", ("0", "2024"): "-"}


@pytest.fixture
def table(tmp_path) -> Path:
    rows = [{"period": period, "item_id": "T1", "item_name": "가상 인구", "unit": "명", "dim1_name": "성별",
             "dim1_code": code, "dim1_label": {"0": "계", "1": "남자", "2": "여자"}[code], "value_raw": raw,
             "value": None if raw == "-" else float(raw), "value_kind": "dash" if raw == "-" else "ok"}
            for (code, period), raw in _CELLS.items()]
    return write_table(tmp_path / "data", "GRADE_TBL", rows, {"survey_name": "가상 인구조사", "organization": "가상 기관",
                       "table_name": "가상 성별 인구", "period": "년 (2022 ~ 2024)", "url": ""})


def _item(path: Path, **kw) -> GoldenItem:
    from kosis_chat.data.loader import sha256_of

    base = {"id": "q", "kind": "answerable", "type": "compare", "question": "q", "table": str(path),
            "table_sha256": sha256_of(path), "required_facts": [F1, F2, F3]}
    return GoldenItem.model_validate(base | kw)


def _answer(facts=(F1, F2, F3), text=None, **kw) -> StatAnswer:
    parts = [FactPart(fact_id=f.id) for f in facts] + ([TextPart(text=text)] if text else [])
    base = {"outcome": "answered", "result": StatResult(facts=list(facts)), "parts": parts, "source": SOURCE}
    return StatAnswer.model_validate(base | kw)


@pytest.fixture
def cells(table):
    return read_cells(table)


def test_evaluate_supports_only_arithmetic():
    assert evaluate("(f2 / f1 - 1) * 100", {"f1": 500, "f2": 550}) == pytest.approx(10.0)
    with pytest.raises(ValueError):
        evaluate("__import__('os')", {})


def test_correct_answer_passes_even_with_renamed_ids_and_rewritten_formula(table, cells):
    a = Fact(id="x", value=500, unit="명", period="2023", source_ref=F1.source_ref)
    b = Fact(id="y", value=550, unit="명", period="2023", source_ref=F2.source_ref)
    ratio = Fact(id="z", value=1.1, derivation="1 * y / x", round=2)
    g = grade(_item(table), _answer((a, b, ratio), text="2023년 기준"), cells)
    assert (g.verdict, g.error_types, g.number_errors) == ("PASS", [], 0)


@pytest.mark.parametrize(
    ("answer_kw", "errors"),
    [
        # value differs from both golden and Parquet -> wrong value and an unsupported number
        ({"facts": (F1.model_copy(update={"value": 501}), F2, F3)}, ["unsupported_claim", "wrong_value"]),
        # right cells, derived value off by more than half a unit of round
        ({"facts": (F1, F2, F3.model_copy(update={"value": 1.3}))}, ["unsupported_claim", "wrong_value"]),
        # same cell, other year (value taken from that year, so the number itself is supported)
        ({"facts": (Fact(id="f1", value=480, period="2022", source_ref="GRADE_TBL|T1|1|2022"), F2,
                    F3.model_copy(update={"value": 1.1}))}, ["wrong_period"]),
        # a number typed into connecting text
        ({"text": "약 3배"}, ["unsupported_claim"]),
        ({"outcome": "refused", "facts": (), "refusal": Refusal(reason="no_item", message="m")}, ["false_refusal"]),
    ],
)
def test_injected_errors_fail(table, cells, answer_kw, errors):
    facts = answer_kw.pop("facts", (F1, F2, F3))
    g = grade(_item(table), _answer(facts, **answer_kw), cells)
    assert g.verdict == "FAIL" and g.error_types == errors


def test_missing_aux_is_partial(table, cells):
    aux = Fact(id="a1", value=1050, period="2023", source_ref="GRADE_TBL|T1|0|2023")
    g = grade(_item(table, aux_facts=[aux]), _answer(), cells)
    assert (g.verdict, g.error_types) == ("PARTIAL", ["missing_aux"])
    g = grade(_item(table, kind="default_allowed", default_applied={"period": "latest (2023)"}, aux_facts=[aux]), _answer(default_applied={"period": "latest (2023)"}), cells)
    assert (g.verdict, g.error_types) == ("PARTIAL", ["missing_aux"])  # no grader error for default_allowed either


def test_text_numbers_from_periods_and_labels_are_not_claims(table, cells):
    labelled = [F1.model_copy(update={"dims": {"항목": "1인 (가구)", "성별": "남자"}}), F2, F3]
    item = _item(table, required_facts=labelled)
    ok = grade(item, _answer(labelled, text="2023년 1인 가구는 남자 기준입니다."), cells)
    assert ok.verdict == "PASS" and ok.number_errors == 0  # '2023' is the period, '1' is in the label
    typed = grade(item, _answer(labelled, text="2023년 1인 가구는 3명 늘었습니다."), cells)
    assert typed.number_errors == 1 and "'3'" in typed.details[0]
    from eval.grade import allowed_text_numbers
    monthly = Fact(id="f9", value=1, period="202503", dims={"항목": "혼인"}, source_ref="M|T3|00|202503")
    assert allowed_text_numbers([monthly]) == {"202503", "2025", "3", "1"}  # 2025년 3월, 1분기


def test_refusal_reason_and_review_flag(table, cells):
    item = _item(table, kind="not_answerable", required_facts=[], allowed_reasons=["no_period", "missing_value"])
    ok = grade(item, _answer((), outcome="refused", refusal=Refusal(reason="missing_value", message="m")), cells)
    assert ok.verdict == "PASS" and ok.needs_review == ["insufficient_reason"]
    wrong = grade(item, _answer((), outcome="refused", refusal=Refusal(reason="causal", message="m")), cells)
    assert wrong.error_types == ["wrong_reason"]
    assert grade(item, _answer(), cells).error_types == ["missed_refusal"]


def test_clarify_and_default(table, cells):
    clarify = _item(table, kind="clarify", required_facts=[], clarify_options=["a", "b"])
    assert grade(clarify, _answer(), cells).error_types == ["assumed_condition"]
    default = _item(table, kind="default_allowed", default_applied={"period": "latest (2023)"})
    assert grade(default, _answer(), cells).error_types == ["unstated_default"]
    assert grade(default, _answer(default_applied={"sex": "계"}), cells).error_types == ["disallowed_default"]
    assert grade(default, _answer(default_applied={"period": "2023"}), cells).verdict == "PASS"


def test_report_metrics_and_files(table, tmp_path):
    items = [_item(table, id="q1"), _item(table, id="q2", kind="not_answerable", required_facts=[], allowed_reasons=["no_period"])]
    answers = {"q1": _answer(text="3배")}  # q2 unanswered -> run_failure
    summary = summarize(grade_all(items, answers))
    assert summary["numeric_accuracy"]["rate"] == 1.0  # facts right; the text number is a number error, not a fact error
    assert summary["number_errors"] == 1 and summary["run_failures"] == 1
    assert summary["refusal_rate"]["rate"] == 0.0
    run_dir = save_report(grade_all(items, answers), {"stage": "t", "started_at": "now", "golden_version": "v",
                          "golden_sha256": "0" * 64, "questions": 2, "answers": "x"}, tmp_path)
    assert "수치 정확도" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_breaks_down_by_eda_trap(table, tmp_path):
    items = [_item(table, id="q1", eda_traps=["label_reuse", "unit_missing"]), _item(table, id="q2", eda_traps=["label_reuse"])]
    summary = summarize(grade_all(items, {"q1": oracle_answer(items[0])}))  # q2 unanswered -> FAIL under its traps
    assert summary["by_trap"] == {"label_reuse": {"PASS": 1, "FAIL": 1}, "unit_missing": {"PASS": 1}}
    run_dir = save_report(grade_all(items, {}), {"stage": "t", "started_at": "now", "golden_version": "v",
                          "golden_sha256": "0" * 64, "questions": 2, "answers": "x"}, tmp_path)
    assert "| trap=label_reuse | 0 | 0 | 2 |" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_breaks_down_complex_items_by_id_op(table, tmp_path):
    items = [_item(table, id="S-CX-AVG-01"), _item(table, id="S-CX-AVG-02"), _item(table, id="S-EDG-01")]
    summary = summarize(grade_all(items, {"S-CX-AVG-01": oracle_answer(items[0])}))
    assert summary["by_op"] == {"AVG": {"PASS": 1, "FAIL": 1}}  # S-EDG-01 has no op
    run_dir = save_report(grade_all(items, {}), {"stage": "t", "started_at": "now", "golden_version": "v",
                          "golden_sha256": "0" * 64, "questions": 3, "answers": "x"}, tmp_path)
    assert "| op=AVG | 0 | 0 | 2 |" in (run_dir / "report.md").read_text(encoding="utf-8")


# --- T1-23 slots: gold from the required cells, compared with answer.query ---

def _slots(table, cells, item_kw=None, **query):
    item = _item(table, **(item_kw or {}))
    q = StatQuery(question="q", **query) if query else None
    return grade(item, _answer(query=q), cells).slots


def test_slots_all_right_for_the_query_the_gold_implies(table, cells):
    # item axis has one label in the table ('가상 인구') -> not a slot; 성별 남자/여자 are the compare targets
    slots = _slots(table, cells, type="compare", periods=["2023"], targets=[{"성별": "남자"}, {"성별": "여자"}])
    assert slots == {"type": True, "items": True, "dims": True, "periods": True, "ranked": None}


@pytest.mark.parametrize(("wrong", "field"), [
    ({"type": "value"}, "type"),
    ({"periods": ["2022"]}, "periods"),
    ({"targets": [{"성별": "남자"}, {"성별": "계"}]}, "dims"),
])
def test_each_wrong_slot_fails_only_its_field(table, cells, wrong, field):
    query = {"type": "compare", "periods": ["2023"], "targets": [{"성별": "남자"}, {"성별": "여자"}]} | wrong
    slots = _slots(table, cells, **query)
    assert [f for f, ok in slots.items() if ok is False] == [field]


def test_empty_query_is_every_field_wrong_and_verdict_unchanged(table, cells):
    item = _item(table)
    graded = grade(item, _answer(query=StatQuery(question="q")), cells)
    assert graded.verdict == "PASS"  # slots never change the verdict
    assert graded.slots == {"type": False, "items": False, "dims": False, "periods": False, "ranked": None}


def test_totals_drop_outside_compare_and_trend_periods_use_ends(table, cells):
    # value of 남자 2022~2023 trend: the '계' filter the pipeline fills in is not a slot
    kw = {"type": "trend", "required_facts": [Fact(id="f1", value=480, period="2022", source_ref="GRADE_TBL|T1|1|2022"),
                                              Fact(id="f2", value=500, period="2023", source_ref="GRADE_TBL|T1|1|2023")]}
    slots = _slots(table, cells, kw, type="trend", periods=["2022", "2023"], filters={"성별": "남자"})
    assert slots["dims"] and slots["periods"]


def test_extreme_ranked_axis_is_checked_against_over(table, cells):
    kw = {"type": "extreme", "question": "2023년 가장 많은 성별은?",
          "required_facts": [Fact(id="f1", value=550, period="2023", source_ref="GRADE_TBL|T1|2|2023")]}
    assert _slots(table, cells, kw, type="extreme", periods=["2023"], over={"성별": []}, pick="max")["ranked"] is True
    assert _slots(table, cells, kw, type="extreme", periods=["2023"], pick="max")["ranked"] is False  # read as periods


def test_failed_run_counts_as_every_slot_wrong(table):
    items = [_item(table, id="a"), _item(table, id="b")]
    summary = summarize(grade_all(items, {"a": _answer(query=StatQuery(question="q", type="compare", periods=["2023"],
                                                                         targets=[{"성별": "남자"}, {"성별": "여자"}]))}))
    assert summary["slots"]["all"] == {"hits": 1, "total": 2, "rate": 0.5}


def test_oracle_passes_and_golden_checks_run(table, tmp_path):
    item = _item(table)
    golden = tmp_path / "g.jsonl"
    golden.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    _, items, _ = load_items(golden)
    assert grade_all(items, {"q": oracle_answer(items[0])})[0].verdict == "PASS"
    bad = json.loads(item.model_dump_json())
    bad["required_facts"][0]["value"] = 501
    bad["table_sha256"] = "0" * 64
    golden.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(GoldenError, match="sha256"):
        load_items(golden)


# --- real golden set + Parquet (gitignored; skipped when absent or not yet on Parquet) ---

def _real():
    golden = ROOT / "eval" / "golden" / "stat_golden.jsonl"
    if not golden.exists() or '"table_sha256"' not in golden.read_text(encoding="utf-8").split("\n", 1)[0] \
            or ".parquet" not in golden.read_text(encoding="utf-8").split("\n", 1)[0]:
        pytest.skip("golden set on Parquet not present (gitignored; run scripts/golden_to_parquet.py)")
    return load_items(golden)[1]


def test_golden_oracle_answers_all_pass():
    items = _real()
    grades = grade_all(items, {i.id: oracle_answer(i) for i in items})
    assert [g.question_id for g in grades if g.verdict != "PASS"] == []
    assert sum(g.number_errors for g in grades) == 0


def test_golden_injected_errors_are_all_caught():
    items = _real()
    caught = total = 0
    for item in items:
        oracle = oracle_answer(item)
        if item.kind in ("answerable", "default_allowed"):
            first = next(f for f in item.required_facts if f.source_ref and f.value is not None)
            facts = [f.model_copy(update={"value": f.value + 1}) if f.id == first.id else f for f in oracle.result.facts]
            bad = oracle.model_copy(update={"result": StatResult(facts=facts)})
            expect = "wrong_value"
        elif item.kind == "not_answerable":
            reason = next(r for r in ("causal", "out_of_scope") if r not in item.allowed_reasons)
            bad = oracle.model_copy(update={"refusal": Refusal(reason=reason, message="m")})
            expect = "wrong_reason"
        else:
            # An answer that picked a condition instead of asking (v2: an answered answer must carry parts).
            bad = StatAnswer(outcome="answered", parts=[TextPart(text="2024년 기준")], source=oracle.source)
            expect = "assumed_condition"
        g = grade(item, bad, read_cells(item.path))
        total += 1
        caught += g.verdict == "FAIL" and expect in g.error_types
    assert caught == total == len(items)
