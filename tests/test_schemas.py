"""Contract tests for src/kosis_ai/schemas.py: golden facts load losslessly, contract violations are rejected."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kosis_ai.schemas import (
    CHART_KINDS,
    QUESTION_KINDS,
    QUESTION_TYPES,
    REFUSAL_REASONS,
    Cell,
    Chart,
    Dim,
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
    unit_misses = set()
    for f in facts:
        cell = cells[f["source_ref"]]
        if f.get("value") is None:
            assert cell.value is None and cell.value_raw.strip() == f["raw"]
        else:
            assert cell.value == pytest.approx(f["value"])
        fact = cell.to_fact(f["id"])
        assert (fact.source_ref, fact.period, fact.raw) == (f["source_ref"], f["period"], f.get("raw"))
        if fact.unit != f.get("unit"):
            unit_misses.add((fact.unit, cell.item_name, *(d.label for d in cell.dims)))
    # The data carries no unit for these two; the golden author filled them in by hand.
    assert {m[0] for m in unit_misses} == {None}
    assert all("합계출산율" in m or "기대수명(출생시 기대여명)" in m for m in unit_misses)


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


def cell(**kw):
    base = dict(org_id="101", tbl_id="T", source="api", prd_se="Y", period="2025", item_id="T1", item_name="i",
                value_raw="12", value=12.0, value_kind="ok")  # fmt: skip
    return Cell(**(base | kw))


@pytest.mark.parametrize(
    ("kw", "unit"),
    [
        ({"item_name": "출생건수 (명)"}, "명"),
        ({"item_name": "조출생률 (천명당)"}, "천명당"),
        ({"item_name": "x", "dims": [{"name": "기본항목별", "code": "11", "label": "출생아수(명)"}]}, "명"),
        ({"item_name": "x", "dims": [{"name": "기본항목별", "code": "41", "label": "기대수명(출생시 기대여명)"}]}, None),
        ({"item_name": "x", "dims": [{"name": "행정구역별", "code": "29", "label": "세종특별자치시(2012~)"}]}, None),
        ({"item_name": "인구밀도 (명)", "unit": "명/㎢"}, "명/㎢"),
    ],
)
def test_cell_unit_from_column_or_suffix(kw, unit):
    assert cell(**kw).unit_or_suffix() == unit


def test_symbol_cell_to_fact_keeps_raw():
    fact = cell(value_raw=" - ", value=None, value_kind="dash").to_fact("a1")
    assert (fact.value, fact.raw, fact.source_ref) == (None, "-", "T|T1|2025")


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
            lambda: StatResult(facts=[CELL, Fact(id="f3", value=1.0, derivation="f2 - f1", round=0)]),
            id="derivation references unknown fact",
        ),
        pytest.param(lambda: Fact(id="f3", value=1.0, derivation="f2 - f1"), id="derivation without round"),
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


# ---------------------------------------------------------------- T0-03 v2: stricter contract


ANSWERED = dict(outcome="answered", result=StatResult(facts=[CELL]), parts=[FactPart(fact_id="f1")], source=SOURCE)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Source(org_id="101", tbl_id="T", survey_nm="s", tbl_nm="t", extra=1), id="Source"),
        pytest.param(lambda: Fact(id="f1", value=1.0, source_ref="T|T1|1|2025", label="x"), id="Fact"),
        pytest.param(lambda: StatQuery(question="q", slot="x"), id="StatQuery"),
        pytest.param(lambda: StatResult(facts=[], warnings=[]), id="StatResult"),
        pytest.param(lambda: FactPart(fact_id="f1", text="x"), id="FactPart"),
        pytest.param(lambda: TextPart(text="x", fact_id="f1"), id="TextPart"),
        pytest.param(lambda: StatAnswer(**ANSWERED, chart_kind="bar"), id="StatAnswer"),
    ],
)
def test_unknown_field_is_rejected_on_every_model(build):
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        build()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Fact(id="1f", value=1.0, source_ref="T|T1|1|2025"), id="id is not an identifier"),
        pytest.param(lambda: Fact(id="for", value=1.0, source_ref="T|T1|1|2025"), id="id is a keyword"),
        pytest.param(lambda: Fact(id="f1", value=float("nan"), source_ref="T|T1|1|2025"), id="value is nan"),
        pytest.param(lambda: Fact(id="f1", value=float("inf"), source_ref="T|T1|1|2025"), id="value is inf"),
        pytest.param(lambda: Fact(id="f1", value=True, source_ref="T|T1|1|2025"), id="value is a bool"),
        pytest.param(lambda: Fact(id="f1", raw="-", value=0.0, source_ref="T|T1|1|2025"), id="symbol fact with a number"),
        pytest.param(lambda: Fact(id="f1", value=1.0, unit=" ", source_ref="T|T1|1|2025"), id="blank unit"),
        pytest.param(lambda: Fact(id="f1", value=1.0, source_ref="T|2025"), id="source_ref with two parts"),
        pytest.param(lambda: Fact(id="f1", value=1.0, source_ref="|T1|1|2025"), id="source_ref without table"),
        pytest.param(lambda: Fact(id="f1", value=1.0, source_ref="T|T1|1|"), id="source_ref without period"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="f2 * 2", round=0), id="derivation references itself"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="2 * 3", round=0), id="derivation without a fact"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="abs(f1)", round=0), id="derivation calls a function"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="f1 ** f2", round=0), id="derivation uses a fact as exponent"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="f1 % 2", round=0), id="derivation uses modulo"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="f1 -", round=0), id="derivation is not an expression"),
        pytest.param(lambda: Fact(id="f2", value=1.0, derivation="f1 / 2", round=11), id="round above the limit"),
        pytest.param(lambda: TextPart(text=""), id="empty text part"),
        pytest.param(lambda: StatAnswer(outcome="answered", result=StatResult(facts=[CELL]), source=SOURCE), id="answered without parts"),
        pytest.param(
            lambda: StatAnswer(outcome="refused", refusal=Refusal(reason="causal", message="x"), parts=[FactPart(fact_id="f1")],
                               result=StatResult(facts=[CELL]), source=SOURCE),
            id="refusal with a fact part",
        ),
        pytest.param(lambda: StatAnswer(**ANSWERED, default_applied={"": "2024"}), id="default with a blank key"),
        pytest.param(lambda: StatAnswer(**ANSWERED, default_applied={"period": " "}), id="default with a blank value"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin="model"), id="model origin with a cited fact"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin="none"), id="none origin with a cited fact"),
        pytest.param(
            lambda: StatAnswer(outcome="answered", parts=[TextPart(text="x")], source=SOURCE, numbers_origin="code"),
            id="code origin without a cited fact",
        ),
        pytest.param(
            lambda: StatAnswer(outcome="refused", refusal=Refusal(reason="causal", message="x"), source=SOURCE, numbers_origin="code"),
            id="code origin on a refusal",
        ),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin="model", numbers_verified=True), id="verified model numbers"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin=None, numbers_verified=True), id="verified legacy numbers"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin=None, defaults_verified=True), id="verified legacy defaults"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin="weird"), id="unknown origin"),
    ],
)
def test_v2_contract_violation_is_rejected(build):
    with pytest.raises(ValidationError):
        build()


def test_v2_valid_shapes():
    derived = Fact(id="f2", value=2.0, derivation="f1 * 2", round=0)
    code = StatAnswer(outcome="answered", result=StatResult(facts=[CELL, derived]), parts=[FactPart(fact_id="f2")], source=SOURCE,
                      numbers_origin="code", numbers_verified=True, defaults_verified=True, default_applied={"period": "2025"})
    assert code.numbers_origin == "code" and code.result.facts[1].derivation == "f1 * 2"
    model = StatAnswer(outcome="answered", parts=[TextPart(text="x")], source=SOURCE, numbers_origin="model")
    assert model.numbers_verified is False
    none = StatAnswer(outcome="refused", refusal=Refusal(reason="causal", message="x"), source=SOURCE, numbers_origin="none")
    assert none.parts == []
    assert Fact(id="a1", value=1.0, source_ref="DT_1B8000H|T10|2025").source_ref.count("|") == 2  # three parts: the minimum


def test_record_written_before_v2_still_loads():
    legacy = {"outcome": "answered", "query": {"question": "q"}, "result": {"facts": [CELL.model_dump(mode="json")]},
              "parts": [{"kind": "fact", "fact_id": "f1"}], "text": "x", "source": SOURCE.model_dump()}
    answer = StatAnswer.model_validate(legacy)
    assert answer.numbers_origin is None and answer.numbers_verified is False and answer.defaults_verified is False
    assert answer.parts == [FactPart(fact_id="f1")] and answer.result.facts == [CELL] and answer.text == "x"


def test_cell_from_row_ignores_columns_outside_the_contract():
    row = {"org_id": "101", "tbl_id": "T", "source": "api", "prd_se": "Y", "period": "2025", "item_id": "T1", "item_name": "i",
           "unit": "명", "value_raw": "12", "value": 12.0, "value_kind": "ok", "dim1_name": "성별", "dim1_code": "1",
           "dim1_label": "남자", "dim2_code": None, "build_note": "added by the Parquet build"}  # fmt: skip
    cell = Cell.from_row(row)
    assert cell.dims == [Dim(name="성별", code="1", label="남자")] and cell.value == 12.0
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        Cell(**{k: v for k, v in row.items() if not k.startswith("dim")})


# ---------------------------------------------------------------- T0-03 v2: charts in the contract


SPEC = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "data": {"values": [{"이름": "2025", "값": 1}]}, "mark": "bar"}


def test_chart_valid_shapes():
    chart = Chart(kind="bar", spec=SPEC, fact_ids=["f1"])
    answer = StatAnswer(**ANSWERED, numbers_origin="code", chart=chart)
    assert answer.chart.kind == "bar" and StatAnswer.model_validate_json(answer.model_dump_json()) == answer
    assert "timeline" in CHART_KINDS


def test_records_before_v3_load_without_their_v2_operation():
    """S5 removed StatQuery.op / period_groups; a record written before still loads, the two keys dropped."""
    old = {"question": "q", "type": "trend", "op": "pavg", "periods": ["2010", "2011", "2015", "2016"],
           "period_groups": [["2010", "2011"], ["2015", "2016"]]}
    query = StatQuery.model_validate(old)
    assert query.periods == old["periods"] and "op" not in query.model_dump() and "period_groups" not in query.model_dump()


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Chart(kind="scatter", spec=SPEC, fact_ids=["f1"]), id="unknown chart kind"),
        pytest.param(lambda: Chart(kind="bar", spec={"mark": "bar"}, fact_ids=["f1"]), id="chart without inline data"),
        pytest.param(lambda: Chart(kind="bar", spec=SPEC, fact_ids=[]), id="chart without facts"),
        pytest.param(lambda: Chart(kind="bar", spec=SPEC, fact_ids=["f1", "f1"]), id="chart with a repeated fact"),
        pytest.param(lambda: Chart(kind="bar", spec=SPEC, fact_ids=["f1"], title="x"), id="chart with an unknown field"),
        pytest.param(lambda: StatAnswer(**ANSWERED, numbers_origin="code", chart=Chart(kind="bar", spec=SPEC, fact_ids=["f9"])),
                     id="chart over an unknown fact"),
        pytest.param(lambda: StatAnswer(**ANSWERED, chart=Chart(kind="bar", spec=SPEC, fact_ids=["f1"])), id="chart without code numbers"),
        pytest.param(lambda: StatAnswer(outcome="refused", refusal=Refusal(reason="causal", message="x"), source=SOURCE,
                                        numbers_origin="none", chart=Chart(kind="bar", spec=SPEC, fact_ids=["f1"])),
                     id="chart on a refusal"),
    ],
)
def test_chart_violations_are_rejected(build):
    with pytest.raises(ValidationError):
        build()
