"""S6 answer contract: caveats, facts shown beside the answer, ask-back choices with codes, a table and grid charts,
and the general wording of a plan the model wrote. Made-up tables and questions worded unlike the golden set."""

import json

import pytest
from conftest import write_table
from pydantic import ValidationError

from kosis_chat.answer.caveats import caveats_of
from kosis_chat.answer.chart import chart_spec
from kosis_chat.answer.table import table_part
from kosis_chat.answer.template import template_parts
from kosis_chat.answer.verify import verify
from kosis_chat.compute import build, compute
from kosis_chat.compute.engine import Computed
from kosis_chat.data.loader import load_table
from kosis_chat.data.preprocessing import clean
from kosis_chat.llm.model import MockClient
from kosis_chat.pipeline import StatPipeline

from kosis_ai.schemas import Caveat, ClarifyChoice, Fact, FactPart, Source, StatAnswer, StatResult, TablePart, TableRow  # isort: skip

SOURCE = Source(org_id="101", tbl_id="FAKE", survey_nm="가상", tbl_nm="가상")


def ref(item, region, period):
    return f"FAKE_TBL_002|{item}|{region}|{period}"


def reply(answer):
    return json.dumps({"answer": answer}, ensure_ascii=False)


def answered(**kw):
    facts = [Fact(id="f1", value=10, unit="명", period="2023", dims={"연령": "65세 이상"}, source_ref="T|1|2023"),
             Fact(id="f2", value=12, unit="명", period="2024", dims={"연령": "65세 이상"}, source_ref="T|1|2024")]
    return StatAnswer(outcome="answered", result=StatResult(facts=facts), parts=[FactPart(fact_id="f1")], source=SOURCE,
                      numbers_origin="code", **kw)


# ---------------------------------------------------------------- the contract


def test_new_fields_are_checked_against_the_facts():
    ok = answered(caveats=[Caveat(kind="missing_years", text="2018년은 표에 없습니다."),
                           Caveat(kind="alternative", text="65세 이상 기준입니다.", fact_ids=["f2"])],
                  context_fact_ids=["f2"],
                  table=TablePart(columns=["2023년", "2024년"], rows=[TableRow(label="65세 이상", fact_ids=["f1", "f2"])]))
    assert ok.table.fact_ids() == ["f1", "f2"]
    with pytest.raises(ValidationError, match="caveat text names numbers"):
        answered(caveats=[Caveat(kind="alternative", text="고위 추계는 1,200명입니다.")])  # a number belongs in a fact
    with pytest.raises(ValidationError, match="context_fact_ids reference unknown facts"):
        answered(context_fact_ids=["f9"])
    with pytest.raises(ValidationError, match="one cell per column"):
        TablePart(columns=["2023년", "2024년"], rows=[TableRow(label="x", fact_ids=["f1"])])


def test_clarify_choices_carry_the_codes_of_the_options():
    choices = [ClarifyChoice(code="11020", label="서울 > 중구", axis="지역"), ClarifyChoice(code="21010", label="부산 > 중구", axis="지역")]
    StatAnswer(outcome="clarify", clarify_options=["서울 > 중구", "부산 > 중구"], clarify_choices=choices, source=SOURCE)
    with pytest.raises(ValidationError, match="clarify_choices labels"):
        StatAnswer(outcome="clarify", clarify_options=["a", "b"], clarify_choices=choices, source=SOURCE)


def test_a_record_written_before_s6_still_loads():
    old = answered().model_dump(exclude={"caveats", "context_fact_ids", "table", "clarify_choices"})
    assert StatAnswer.model_validate(old).caveats == []


# ---------------------------------------------------------------- caveats


def test_cautions_notes_and_symbol_cells_become_caveats(pop_path):
    table = clean(load_table(pop_path))
    computed = compute(table, build.value("FAKE_TBL_001|T1|1|2023", context=["FAKE_TBL_001|T1|1|2024"]))  # 2024 'x'
    computed.cautions[:] = ["missing_years: 2019, 2021", "event_source: model (가상 사건 = 2020-03)",
                            "simple_average: 가상 성비의 시점 평균은 가중치 없는 단순 평균입니다."]
    shown = [f.id for f in computed.result.facts if f.id not in computed.main]
    found = {c.kind: c for c in caveats_of(computed, table, context_ids=shown)}
    assert found["missing_years"].text == "2019년, 2021년은(는) 표에 없습니다."
    assert found["event_source"].text.startswith("'가상 사건'의 시작 시점(2020년 3월)")
    assert found["symbol_cell"].fact_ids == shown and "'x'(비공개)" in found["symbol_cell"].text
    assert "simple_average" in found


def test_a_provisional_value_gets_a_caveat(pop_path):
    table = clean(load_table(pop_path))
    computed = compute(table, build.value("FAKE_TBL_001|T1|2|2024"))  # 560p
    assert [c.kind for c in caveats_of(computed, table)] == ["provisional"]


# ---------------------------------------------------------------- the scenarios of a projection


@pytest.fixture
def projection(tmp_path):
    scenarios = (("1", "중위 추계(기본)"), ("2", "고위 추계(최대)"), ("3", "저위 추계(최소)"))
    rows = [{"period": p, "item_id": "T1", "item_name": "가상 지표", "unit": "명", "dim1_name": "가정별", "dim1_code": code,
             "dim1_label": label, "dim2_name": "지표별", "dim2_code": "10", "dim2_label": "총인구(명)",
             "value_raw": str(v), "value": float(v)}
            for p, base in (("2040", 5000), ("2050", 4800)) for code, label in scenarios
            for v in [base + {"1": 0, "2": 300, "3": -300}[code]]]
    catalog = {"survey_name": "가상 추계", "organization": "가상 기관", "table_name": "가상 장래인구", "period": "년",
               "url": "https://example.invalid/projection"}
    return load_table(write_table(tmp_path / "data", "FAKE_PROJ", rows, catalog))


def test_a_scenario_the_question_left_open_shows_the_other_scenarios(projection):
    bot = StatPipeline(projection, client=MockClient([reply("2050년 총인구는 {f1}으로 예상됩니다.")]), org_id="101")
    stat = bot.ask("2050년에 총인구는 얼마나 될까?").stat_answer
    assert stat.outcome == "answered" and stat.result.get(stat.parts[1].fact_id).value == 4800
    shown = [stat.result.get(i) for i in stat.context_fact_ids]
    assert sorted(f.value for f in shown) == [4500, 5100] and {f.period for f in shown} == {"2050"}
    (alt,) = [c for c in stat.caveats if c.kind == "alternative"]
    assert alt.fact_ids == stat.context_fact_ids and "참고: 가정별은(는) 질문에 없어 '중위 추계' 기준" in stat.text


def test_a_scenario_the_question_names_shows_no_others(projection):
    bot = StatPipeline(projection, client=MockClient([reply("{f1}입니다.")]), org_id="101")
    stat = bot.ask("고위 추계로 2040년 총인구는?").stat_answer
    assert stat.context_fact_ids == [] and not [c for c in stat.caveats if c.kind == "alternative"]


# ---------------------------------------------------------------- table and grid charts


def test_two_targets_at_two_periods_are_a_table_and_grouped_bars(household_path):
    table = clean(load_table(household_path))
    computed = compute(table, build.gap((ref("T1", "11", "2015"), ref("T1", "21", "2015")),
                                        (ref("T1", "11", "2024"), ref("T1", "21", "2024"))))
    part = table_part(computed)
    assert part.columns == ["2015년", "2024년"] and [r.label for r in part.rows] == ["서울특별시", "부산광역시"]
    chart = chart_spec("compare", computed)
    assert chart.kind == "grouped_bar" and len(chart.spec["data"]["values"]) == 4 and chart.spec["encoding"]["xOffset"]


def test_two_targets_over_many_periods_are_a_multi_line(household_path):
    table = clean(load_table(household_path))
    cells = [ref("T1", r, p) for r in ("11", "21") for p in ("2015", "2020", "2023", "2024")]
    computed = compute(table, build.written(build._plan([build.SelectStep(id="s", cells=cells)], ["s"], "trend", None)))
    assert chart_spec("trend", computed).kind == "multi_line"
    assert len(table_part(computed).columns) == 4


def test_one_series_is_no_grid(household_path):
    table = clean(load_table(household_path))
    computed = compute(table, build.trend([ref("T1", "11", p) for p in ("2015", "2020", "2023")]))
    assert table_part(computed) is None and chart_spec("trend", computed).kind == "line"


# ---------------------------------------------------------------- general template for a written plan


def test_a_written_plan_is_worded_by_what_each_number_was_computed_from(household_path):
    table = clean(load_table(household_path))
    steps = [build.SelectStep(id="a", cells=[ref("T1", "11", "2023")]), build.SelectStep(id="b", cells=[ref("T1", "21", "2023")]),
             build.CombineStep(id="d", a="a", b="b", fn="diff")]
    computed = compute(table, build.written(build._plan(steps, ["a", "b", "d"], "compare", None)))
    parts = template_parts("compare", computed)
    assert verify(parts, computed) == []
    text = "".join(p.text if hasattr(p, "text") else "{x}" for p in parts)
    assert "2023년 서울특별시·부산광역시의 차이" in text


def test_computed_holds_cautions_for_caveats():
    assert "cautions" in Computed.__dataclass_fields__


def test_a_number_inside_a_name_is_not_a_number_of_the_caveat():
    answered(caveats=[Caveat(kind="event_source", text="'코로나19 유행 직전'은(는) 사건 달력의 시작 시점 2020년 1월 기준입니다.")])
