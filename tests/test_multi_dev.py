"""T1-30: the multi-table dev set's grader (eval/multi_dev.py grade) and the dev file's shape."""

import json
from pathlib import Path

from eval.multi_dev import DEV, grade

from kosis_ai.schemas import Fact, Refusal, Source, StatAnswer, StatResult  # isort: skip

SOURCE = Source(org_id="101", tbl_id="X", survey_nm="s", tbl_nm="t")


def answered(*values):
    facts = [Fact(id=f"f{n}", value=v, source_ref=f"X|T1|00|{2020 + n}") for n, v in enumerate(values, start=1)]
    from kosis_ai.schemas import FactPart  # isort: skip

    return StatAnswer(outcome="answered", result=StatResult(facts=facts), parts=[FactPart(fact_id="f1")], source=SOURCE,
                      numbers_origin="code")


def test_answer_needs_every_cell_and_the_final_value():
    row = {"kind": "answer", "expected": {"cells": {"a": 10.0, "b": 4.0}, "value": 6, "round": 0}}
    assert grade(row, answered(10, 4, 6)) == (True, "answered")
    assert grade(row, answered(10, 4, 7))[0] is False
    assert grade(row, answered(10, 6))[0] is False


def test_refusal_checks_the_reason_when_named():
    stat = StatAnswer(outcome="refused", refusal=Refusal(reason="causal", message="m"), source=SOURCE, numbers_origin="none")
    assert grade({"kind": "refuse", "expected": {"outcome": "refused", "reason": "causal"}}, stat)[0]
    assert not grade({"kind": "refuse", "expected": {"outcome": "refused", "reason": "no_period"}}, stat)[0]
    assert grade({"kind": "refuse", "expected": {"outcome": "refused"}}, stat)[0]


def test_dev_rows_name_two_or_three_tables_and_a_gold_plan_per_answer():
    rows = [json.loads(line) for line in Path(DEV).read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 25
    for row in rows:
        assert 2 <= len(row["table_keys"]) <= 3
        if row["kind"] == "answer":
            labels = {s["table"] for s in row["gold_plan"]["steps"] if s["op"] == "select"}
            assert labels == set("ABC"[: len(row["table_keys"])]), row["qid"]
