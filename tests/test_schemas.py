"""Contract tests for src/kosis_ai/schemas.py: golden facts load losslessly, contract violations are rejected."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kosis_ai.schemas import (
    QUESTION_KINDS,
    QUESTION_TYPES,
    REFUSAL_REASONS,
    Cell,
    Fact,
    FactPart,
    Refusal,
    Source,
    StatAnswer,
    StatQuery,
    StatResult,
    TableInfo,
    TextPart,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden" / "stat_golden.jsonl"
DATA = ROOT / "data"
SOURCE = Source(org_id="101", tbl_id="DT_1B8000H", survey_nm="인구동향조사", tbl_nm="시도/인구동태건수 및 동태율")
CELL = Fact(id="f1", value=76309, unit="명", period="2025", dims={"행정구역별": "경기도"}, source_ref="DT_1B8000H|T10|31|2025")


def golden_rows():
    if not GOLDEN.exists():
        pytest.skip("golden set is not in the repo (gitignored); place it at eval/golden/")
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_facts_load_losslessly():
    facts = [f for r in golden_rows() for f in r.get("required_facts", []) + r.get("aux_facts", [])]
    assert len(facts) == 98
    for raw in facts:
        # An explicit null and a missing key mean the same thing.
        assert Fact.model_validate(raw).model_dump(exclude_none=True) == {k: v for k, v in raw.items() if v is not None}


def test_golden_labels_are_in_schema_constants():
    for r in golden_rows():
        assert r["kind"] in QUESTION_KINDS
        assert r.get("type") is None or r["type"] in QUESTION_TYPES
        for reason in [r.get("refusal_reason"), *r.get("allowed_reasons", [])]:
            assert reason is None or reason in REFUSAL_REASONS


def require_data():
    if not (DATA / "tables.csv").exists():
        pytest.skip("provided data is not in the repo (gitignored); place it at data/")


def test_tables_csv_rows_load_and_cite():
    require_data()
    with open(DATA / "tables.csv", encoding="utf-8-sig", newline="") as f:
        tables = [TableInfo.model_validate(row) for row in csv.DictReader(f)]
    assert len(tables) == 2364
    assert all((DATA / t.parquet).exists() for t in tables)
    assert tables[0].to_source().citation().startswith(f"KOSIS / {tables[0].survey_name} / ")


def test_golden_cell_facts_resolve_to_parquet_rows():
    require_data()
    pq = pytest.importorskip("pyarrow.parquet")
    facts = [f for r in golden_rows() for f in r.get("required_facts", []) + r.get("aux_facts", []) if f.get("source_ref")]
    cells = {}
    for tbl in {f["source_ref"].split("|")[0] for f in facts}:
        for row in pq.read_table(DATA / "values" / f"101_{tbl}.parquet").to_pylist():
            cell = Cell.from_row(row)
            cells[cell.ref()] = cell
    assert len(facts) == 67
    for f in facts:
        cell = cells[f["source_ref"]]
        if f.get("value") is None:
            assert cell.value is None and cell.value_raw.strip() == f["raw"]
        else:
            assert cell.value == pytest.approx(f["value"])


def test_cell_from_parquet_row_keeps_dim_order():
    row = {
        "org_id": "101", "tbl_id": "DT_101094A001", "source": "api", "prd_se": "Y", "period": "2020",
        "item_id": "T1", "item_name": "비율", "unit": "%",
        "dim1_name": "성별", "dim1_code": "S2", "dim1_label": "여자",
        "dim2_name": "년생별", "dim2_code": "A3", "dim2_label": "1985",
        "dim3_name": None, "dim3_code": None, "dim3_label": None,
        "value_raw": "69.1", "value": 69.1, "value_kind": "ok",
    }  # fmt: skip
    assert Cell.from_row(row).ref() == "DT_101094A001|T1|S2|A3|2020"


def test_answer_with_fact_parts_and_source():
    answer = StatAnswer(
        outcome="answered",
        result=StatResult(facts=[CELL]),
        parts=[TextPart(text="2025년 경기도 출생아 수는"), FactPart(fact_id="f1")],
        source=SOURCE,
    )
    assert answer.parts[1].fact_id == "f1"


def test_answer_parts_parse_from_llm_json():
    answer = StatAnswer.model_validate(
        {"outcome": "answered", "result": {"facts": [CELL.model_dump()]}, "parts": [{"kind": "fact", "fact_id": "f1"}], "source": SOURCE.model_dump()}
    )
    assert isinstance(answer.parts[0], FactPart)


def test_out_of_scope_refusal_needs_no_source():
    StatAnswer(outcome="refused", refusal=Refusal(reason="out_of_scope", message="제공 범위 밖입니다"))


def test_citation_format():
    assert SOURCE.citation() == "KOSIS / 인구동향조사 / 시도/인구동태건수 및 동태율"
    assert SOURCE.model_copy(update={"url": "https://kosis.kr/x"}).citation().endswith("(https://kosis.kr/x)")


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: StatQuery(question="q", type="causal"), id="unknown question type"),
        pytest.param(lambda: Refusal(reason="no_year", message="x"), id="unknown refusal reason"),
        pytest.param(lambda: Fact(id="f1", value="76309명", source_ref="r"), id="value is not a number"),
        pytest.param(lambda: Fact(id="f1", value=76309), id="value without source_ref or derivation"),
        pytest.param(lambda: Fact(id="f1"), id="fact with no basis at all"),
        pytest.param(lambda: StatResult(facts=[CELL, CELL]), id="duplicate fact ids"),
        pytest.param(
            lambda: StatAnswer(outcome="answered", result=StatResult(facts=[CELL]), parts=[FactPart(fact_id="f9")], source=SOURCE),
            id="V1 unknown fact reference",
        ),
        pytest.param(lambda: StatAnswer(outcome="refused", source=SOURCE), id="refused without refusal"),
        pytest.param(
            lambda: StatAnswer(outcome="answered", refusal=Refusal(reason="causal", message="x"), source=SOURCE),
            id="answered with refusal",
        ),
        pytest.param(lambda: StatAnswer(outcome="clarify", clarify_options=["출생건수"], source=SOURCE), id="clarify with one option"),
        pytest.param(lambda: StatAnswer(outcome="answered", result=StatResult(facts=[CELL])), id="answer without source"),
        pytest.param(
            lambda: StatAnswer(outcome="refused", refusal=Refusal(reason="missing_value", message="x")),
            id="in-scope refusal without source",
        ),
        pytest.param(lambda: Refusal(reason="causal", message=""), id="refusal without message"),
        pytest.param(
            lambda: Cell(org_id="101", tbl_id="T", source="api", prd_se="Y", period="2025", item_id="T1", item_name="i",
                         value_raw="-", value=0.0, value_kind="dash"),
            id="symbol cell with a number",
        ),  # fmt: skip
        pytest.param(
            lambda: Cell(org_id="101", tbl_id="T", source="api", prd_se="Y", period="2025", item_id="T1", item_name="i",
                         value_raw="12", value_kind="ok"),
            id="ok cell without a number",
        ),  # fmt: skip
        pytest.param(
            lambda: Cell(org_id="101", tbl_id="T", source="kosis", prd_se="Y", period="2025", item_id="T1", item_name="i",
                         value_raw="X", value_kind="secret"),
            id="unknown source and value kind",
        ),  # fmt: skip
    ],
)
def test_contract_violation_is_rejected(build):
    with pytest.raises(ValidationError):
        build()
