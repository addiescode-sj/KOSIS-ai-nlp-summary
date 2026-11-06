"""kosis_ai/units.py (T1-28 S2b, D-u (b) / N6): the three unit rules moved there keep their own results.

Expected values are read off the rules as they were before the move (kosis_chat/prepare/convert.infer_unit,
kosis_ai/normalize.resolve_unit, kosis_ai/schemas.Cell.unit_or_suffix); the old names must be the same objects.
"""

from __future__ import annotations

import math

import pytest

from kosis_ai import normalize, units
from kosis_ai.schemas import Cell
from kosis_chat.data import loader
from kosis_chat.prepare import convert

LONG_PER = "출생아 천명당 아주 긴 단위 이름입니다당"  # ends in 당, 22 characters: loose rule yes, normalize rule no


@pytest.mark.parametrize(
    ("unit", "item", "dims", "expected"),
    [
        pytest.param("명", "인구", [], ("명", "given"), id="given"),
        pytest.param("  가구 ", "x", [], ("가구", "given"), id="given is stripped"),
        pytest.param("월세없음", "x", [], ("월세없음", "given"), id="unregistered given unit is kept"),
        pytest.param("", "x", [], (units.UNKNOWN_UNIT, "unknown"), id="empty unit"),
        pytest.param("   ", "출생아수(명)", [], ("명", "item_name"), id="blank unit, item parentheses"),
        pytest.param(None, "출생건수 (명)", [], ("명", "item_name"), id="item parentheses"),
        pytest.param(None, "인구 ( 명 )", [], ("명", "item_name"), id="parentheses text is stripped"),
        pytest.param(None, "인구(명)(가구)", [], ("가구", "item_name"), id="last parentheses first"),
        pytest.param(None, "조사망률(천명당)", [], ("천명당", "item_name"), id="per unit"),
        pytest.param(None, f"x({LONG_PER})", [], (LONG_PER, "item_name"), id="loose per unit, any length"),
        pytest.param(None, "항목", ["인구(명)"], ("명", "dim_label"), id="dim parentheses"),
        pytest.param(None, "평균 (㎡)", [], (units.UNKNOWN_UNIT, "unknown"), id="unit not in the 16"),
        pytest.param(None, "가상 항목", ["가상(분류)"], (units.UNKNOWN_UNIT, "unknown"), id="unregistered parentheses"),
        pytest.param(None, "합계출산율", [], ("명", "override"), id="override keyword"),
        pytest.param(None, "기대수명(출생시 기대여명)", [], ("년", "override"), id="override after non-unit parentheses"),
        pytest.param(None, "항목", ["합계출산율"], ("명", "override"), id="override in a dim label"),
    ],
)
def test_infer_unit(unit, item, dims, expected):
    assert units.infer_unit(unit, item, dims) == expected


@pytest.mark.parametrize(
    ("unit", "item", "dims", "meta", "expected"),
    [
        pytest.param("명", "인구", [], None, ("명", "unit_column"), id="unit column"),
        pytest.param("  천명 ", "x", [], None, ("천명", "unit_column"), id="unit column is stripped"),
        pytest.param("가구", "인구(명)", [], None, ("가구", "unit_column"), id="unit column before parentheses"),
        pytest.param("월세없음", "x", [], None, (None, "unknown"), id="unregistered unit column"),
        pytest.param("월세없음", "출생건수 (명)", [], None, ("명", "parentheses"), id="unregistered column, parentheses"),
        pytest.param("", "x", [], None, (None, "unknown"), id="empty unit"),
        pytest.param(math.nan, "x", [], None, (None, "unknown"), id="NaN unit"),
        pytest.param(None, "평균 (㎡)", [], None, ("㎡", "parentheses"), id="unit in the 27"),
        pytest.param(None, "인구(명)(가구)", [], None, ("가구", "parentheses"), id="last parentheses first"),
        pytest.param(None, "항목(가구)", ["x(명)"], None, ("가구", "parentheses"), id="item before dims"),
        pytest.param(None, "x", ["인구(명)"], None, ("명", "parentheses"), id="dim parentheses"),
        pytest.param(None, "x(천명당)", [], None, ("천명당", "parentheses"), id="short per unit"),
        pytest.param(None, f"x({LONG_PER})", [], None, (None, "unknown"), id="per unit over 12 characters"),
        pytest.param(None, "합계출산율", [], None, ("명", "override"), id="override keyword"),
        pytest.param(None, "기대수명", [], "명", ("년", "override"), id="override before meta"),
        pytest.param(None, "인구", [], "명", ("명", "meta"), id="meta unit"),
        pytest.param(None, "인구증가율", [], "명", (None, "unknown"), id="meta unit never on a rate"),
        pytest.param(None, "조출생률", [], "명", ("명", "meta"), id="률 is not in NOT_COUNT (as before)"),
        pytest.param(None, "x", ["평균연령"], "세", (None, "unknown"), id="meta unit never on an average dim"),
    ],
)
def test_resolve_unit(unit, item, dims, meta, expected):
    assert units.resolve_unit(unit, item, dims, meta) == expected


@pytest.mark.parametrize(
    ("unit", "texts", "expected"),
    [
        pytest.param("명", ["x"], "명", id="unit column"),
        pytest.param("월세없음", [], "월세없음", id="any unit column as it is"),
        pytest.param("", ["x(가구)"], "가구", id="empty unit, suffix"),
        pytest.param(None, ["출생건수 (명)"], "명", id="item suffix"),
        pytest.param(None, ["x", "출생아수(명)"], "명", id="dim suffix"),
        pytest.param(None, ["a(명)", "b(가구)"], "명", id="first text first"),
        pytest.param(None, ["기대수명(출생시 기대여명)"], None, id="space: not a unit"),
        pytest.param(None, ["세종특별자치시(2012~)"], None, id="digits: not a unit"),
        pytest.param(None, ["x(여자백명당남자수)"], None, id="over 5 characters"),
        pytest.param(None, ["출생건수 (명) 추가"], None, id="not at the end"),
    ],
)
def test_unit_or_suffix(unit, texts, expected):
    assert units.unit_or_suffix(unit, texts) == expected


def test_cell_method_uses_the_rule():
    base = dict(org_id="101", tbl_id="T", source="api", prd_se="Y", period="2025", item_id="T1", value_raw="1",
                value=1.0, value_kind="ok")  # fmt: skip
    dims = [{"name": "기본항목별", "code": "11", "label": "출생아수(명)"}]
    assert Cell(**base, item_name="x", dims=dims).unit_or_suffix() == "명"
    assert Cell(**base, item_name="x", unit="가구", dims=dims).unit_or_suffix() == "가구"


@pytest.mark.parametrize(
    ("unit", "item", "dims", "expected"),
    [
        ("%", "x", [], "ratio"),
        ("천명당", "x", [], "ratio"),
        ("명", "출생아수", [], "count_flow"),
        ("명", "총인구", [], "count_stock"),
        ("명", "x", ["평균연령"], "average"),
        ("건", "x", [], "count_flow"),
        ("가구", "x", [], "count_stock"),
        (None, "x", [], None),
    ],
)
def test_measure_kind(unit, item, dims, expected):
    assert units.measure_kind("T", unit, item, dims) == expected


def test_old_names_are_the_same_objects():
    assert convert.infer_unit is loader.infer_unit is units.infer_unit
    assert convert._unit_in_parens is loader._unit_in_parens is units.unit_in_parens
    assert convert.KNOWN_UNITS is loader.KNOWN_UNITS is units.INFER_KNOWN_UNITS
    assert convert.UNIT_OVERRIDES is loader.UNIT_OVERRIDES is units.INFER_UNIT_OVERRIDES
    assert convert.UNKNOWN_UNIT == loader.UNKNOWN_UNIT == units.UNKNOWN_UNIT == "단위 미상"
    assert convert._PAREN is loader._PAREN is normalize._PAREN is units.PAREN
    assert normalize.resolve_unit is units.resolve_unit
    assert normalize.measure_kind is units.measure_kind
    assert normalize._known is units.known_unit
    assert normalize._paren_unit is units.paren_unit
    assert normalize._override_unit is units.override_unit
    assert normalize._keyword_kind is units.keyword_kind
    assert normalize.KNOWN_UNITS is units.RESOLVE_KNOWN_UNITS
    assert normalize.UNIT_OVERRIDES is units.RESOLVE_UNIT_OVERRIDES
    assert normalize.MEASURE_OVERRIDES is units.MEASURE_OVERRIDES
    assert normalize._UNIT_SUFFIX is units.UNIT_PAREN
    assert normalize.meta_unit is normalize._meta_unit
    # D-u (b): the two sets stay different; the loader's is the smaller one (the docs say 27 for normalize's set; the
    # literal holds 26 distinct units).
    assert len(units.INFER_KNOWN_UNITS) == 16 and len(units.RESOLVE_KNOWN_UNITS) == 26
    assert units.INFER_KNOWN_UNITS < units.RESOLVE_KNOWN_UNITS
