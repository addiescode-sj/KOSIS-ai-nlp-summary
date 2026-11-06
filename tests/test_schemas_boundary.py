"""Node-boundary models (T1-28 S1b): the dataclasses moved to stage schemas.py as Pydantic models without a change.

For each model: it is a Pydantic model, its fields are the dataclass fields in the same order with the same
defaults and frozenness, a dump validates back to an equal model, and the old import path gives the same class.
The boundary base (shared/boundary.py) keeps the dataclass behaviour the pipeline relies on: positional arguments,
containers stored as passed (no copy), scalars never coerced.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel, ValidationError

from kosis_chat.analyze import schemas as analyze_schemas
from kosis_chat.compose import schemas as compose_schemas
from kosis_chat.compute import schemas as compute_schemas
from kosis_chat.judge import schemas as judge_schemas

from kosis_ai.schemas import PlanV3, Refusal, SelectStep, StatResult  # isort: skip

_REQUIRED = object()

# (new module, name, old module, fields in the dataclass order: name -> default (_REQUIRED: no default), frozen)
# Read from the dataclasses as they were before S1b (judge/answerability.py, judge/question.py, compute/analyze.py,
# compute/build.py, llm/analyze_plan.py, compute/engine.py, compute/context.py, compute/resolve.py).
MODELS = (
    (judge_schemas, "Verdict", "kosis_chat.judge.answerability",
     {"outcome": _REQUIRED, "rule": None, "refusal": None, "clarify_question": "", "clarify_options": (),
      "default_applied": {}, "final": False}, True),
    (judge_schemas, "Resolved", "kosis_chat.judge.answerability", {"codes": _REQUIRED, "ambiguous": _REQUIRED}, False),
    (judge_schemas, "Periods", "kosis_chat.judge.question",
     {"singles": (), "ranges": (), "recent_n": None, "recent": False, "spans": (), "previous": False, "before": None,
      "events": ()}, True),
    (judge_schemas, "Match", "kosis_chat.judge.question",
     {"axis": _REQUIRED, "code": _REQUIRED, "label": _REQUIRED, "form": _REQUIRED, "exact": _REQUIRED}, True),
    (judge_schemas, "Parsed", "kosis_chat.judge.question",
     {"question": _REQUIRED, "periods": _REQUIRED, "matches": _REQUIRED, "rest": _REQUIRED, "words": ()}, False),
    (analyze_schemas, "Plan", "kosis_chat.compute.analyze",
     {"question": _REQUIRED, "type": _REQUIRED, "codes": _REQUIRED, "periods": _REQUIRED, "over": None, "pick": None,
      "missing_years": [], "default_applied": {}, "unresolved": [], "pattern": None, "groups": [], "events": []}, False),
    (analyze_schemas, "TermSteps", "kosis_chat.compute.build",
     {"cells": _REQUIRED, "value": _REQUIRED, "total": None, "agg": None, "periods": ("", "")}, True),
    (analyze_schemas, "Built", "kosis_chat.compute.build",
     {"plan": _REQUIRED, "pattern": _REQUIRED, "terms": [], "context": [], "reference": None}, True),
    (analyze_schemas, "Placed", "kosis_chat.llm.analyze_plan",
     {"plan": None, "query_plan": None, "built": None, "computed": None, "problems": [], "clarify": None}, False),
    (compute_schemas, "TermResult", "kosis_chat.compute.engine",
     {"cell_ids": _REQUIRED, "value_id": _REQUIRED, "periods": _REQUIRED, "digits": _REQUIRED, "agg": None}, True),
    (compute_schemas, "Computed", "kosis_chat.compute.engine",
     {"result": _REQUIRED, "main": _REQUIRED, "note": None, "picked": {}, "ranking": [], "series": [], "pattern": None,
      "terms": [], "plan": None, "cautions": []}, True),
    (compute_schemas, "Alternatives", "kosis_chat.compute.context",
     {"axis": _REQUIRED, "chosen": _REQUIRED, "refs": _REQUIRED}, True),
    (compute_schemas, "Resolved", "kosis_chat.compute.resolve", {"cells": _REQUIRED, "cautions": []}, False),
)
IDS = [f"{module.__name__.split('.')[1]}.{name}" for module, name, *_ in MODELS]


@pytest.mark.parametrize("module,name,old,fields,frozen", MODELS, ids=IDS)
def test_model_keeps_the_dataclass_fields(module, name, old, fields, frozen):
    model = getattr(module, name)
    assert issubclass(model, BaseModel)
    assert tuple(model.model_fields) == tuple(fields)
    for field, default in fields.items():
        info = model.model_fields[field]
        if default is _REQUIRED:
            assert info.is_required(), f"{name}.{field} must stay required"
        else:
            assert not info.is_required() and info.get_default(call_default_factory=True) == default, f"{name}.{field}"
    assert bool(model.model_config.get("frozen")) is frozen


@pytest.mark.parametrize("module,name,old,fields,frozen", MODELS, ids=IDS)
def test_old_path_gives_the_same_class(module, name, old, fields, frozen):
    assert getattr(importlib.import_module(old), name) is getattr(module, name)


def test_aliases_and_compose_output_moved():
    from kosis_chat.compute import Built, Computed, build
    from kosis_chat.llm import compose

    assert build.Pattern is analyze_schemas.Pattern and build.Agg is analyze_schemas.Agg
    assert Built is analyze_schemas.Built and Computed is compute_schemas.Computed
    assert compose.ComposeOutput is compose_schemas.ComposeOutput
    assert tuple(compose_schemas.ComposeOutput.model_fields) == ("answer",)
    assert compose_schemas.ComposeOutput.model_json_schema()["title"] == "ComposeOutput"  # the schema the model sees


def _simple():
    """One instance per model whose fields dump to the same Python values (no nested model)."""
    j, a, c = judge_schemas, analyze_schemas, compute_schemas
    return [
        j.Verdict("answered", "default", default_applied={"period": "latest (2024)"}),
        j.Verdict("clarify", "same_name", clarify_question="어느 지역?", clarify_options=("가", "나"), final=True),
        j.Resolved(codes={"지역": {"11"}}, ambiguous=[("중구", ["11020", "21010"])]),
        j.Periods(singles=("2024",), ranges=(("2020", "2024"),), recent_n=5, recent=True, before=10,
                  events=(("코로나 직전", "2020"),)),
        j.Match("지역", "11", "서울특별시", "서울", True),
        a.Plan("q", "trend", {"지역": ["11"]}, ["2023", "2024"], missing_years=["2022"], pattern="avg",
               groups=[["2023", "2024"]], events=[("코로나", "2020")]),
        a.TermSteps("c1", "v1", agg="avg", periods=("2020", "2024")),
        a.Placed(problems=["문제"]),
        c.TermResult(["f1", "f2"], "f3", ("2020", "2024"), 1, "avg"),
        c.Alternatives("가정별", "중위", ["T|T1|2|2024"]),
        c.Resolved(["T|T1|11|2024"], ["default_applied: 시점 = 최신"]),
    ]


@pytest.mark.parametrize("obj", _simple(), ids=lambda o: type(o).__name__)
def test_round_trip(obj):
    again = type(obj).model_validate(obj.model_dump())
    assert again == obj and type(again) is type(obj)


def _plan() -> PlanV3:
    return PlanV3(steps=[SelectStep(id="c", cells=["T|T1|11|2024"])], show=["c"], present="value")


def test_nested_values_are_kept_as_passed():
    plan, codes, periods = _plan(), {"지역": ["11"]}, ["2024"]
    built = analyze_schemas.Built(plan, "value", context=periods)
    assert built.plan is plan and built.context is periods  # no copy: the dataclass kept the same objects
    rule = analyze_schemas.Plan("q", None, codes, periods)
    assert rule.codes is codes and rule.periods is periods
    result = StatResult(facts=[])
    computed = compute_schemas.Computed(result, ["f1"], pattern="value")
    assert computed.result is result
    refusal = Refusal(reason="causal", message="m")
    assert judge_schemas.Verdict("refused", "causal", refusal, final=True).refusal is refusal
    periods_model = judge_schemas.Periods(singles=("2024",))
    match = judge_schemas.Match("지역", "11", "서울특별시", "서울", True)
    parsed = judge_schemas.Parsed(question="q", periods=periods_model, matches=[match], rest="q")
    assert parsed.periods is periods_model and parsed.by_axis() == {"지역": [match]}


def test_scalars_are_checked_not_coerced():
    with pytest.raises(ValidationError):
        compute_schemas.TermResult(["f1"], "f1", ("2024", "2024"), 1.0)  # lax mode would store 1
    with pytest.raises(ValidationError):
        analyze_schemas.Built(_plan(), "value", reference=True)  # lax mode would store 1
    with pytest.raises(ValidationError):
        analyze_schemas.Built(_plan(), "no such pattern")
    with pytest.raises(ValidationError):
        judge_schemas.Verdict("maybe")


def test_positional_arguments_follow_the_field_order():
    j = judge_schemas
    assert j.Match("a", "b", "c", "d", False) == j.Match(axis="a", code="b", label="c", form="d", exact=False)
    with pytest.raises(TypeError):
        j.Match("a", "b", "c", "d", False, "extra")
    with pytest.raises(TypeError):
        j.Match("a", "b", "c", "d", False, axis="a")


def test_frozen_models_hash_and_copy_like_the_dataclasses():
    j = judge_schemas
    a, b = j.Match("a", "b", "c", "d", True), j.Match("a", "b", "c", "d", True)
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    with pytest.raises(ValidationError):
        a.code = "x"
    built = analyze_schemas.Built(_plan(), "value")
    moved = built.model_copy(update={"context": ["T|T1|00|2024"]})  # was dataclasses.replace(built, context=...)
    assert moved.context == ["T|T1|00|2024"] and built.context == [] and moved.plan is built.plan
    rule = analyze_schemas.Plan("q", "value", {}, [])
    rule.pattern = "sum"  # rule_plan sets fields after building the plan; not validated, as with the dataclass
    assert rule.pattern == "sum"


def test_refuse_builds_the_final_refusal():
    from kosis_chat.judge.answerability import refuse

    verdict = refuse("causal", "causal")
    assert verdict.outcome == "refused" and verdict.rule == "causal" and verdict.final
    assert verdict.refusal.reason == "causal" and verdict.refusal.message


def test_unknown_keyword_is_rejected_like_a_dataclass():
    import pydantic

    from kosis_chat.analyze.schemas import Plan

    with pytest.raises(pydantic.ValidationError):
        Plan("q", None, {}, [], typo=1)  # valid apart from the unknown keyword
