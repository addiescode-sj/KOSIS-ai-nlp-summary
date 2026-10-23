"""Question scopes: cutting a table down to what one question needs (oracle / auto / full)."""

from __future__ import annotations

import pytest

from kosis_chat.chatbot import Chatbot
from kosis_chat.data import scope
from kosis_chat.data.loader import load_table
from kosis_chat.data.scope import Scope
from kosis_chat.llm.model import MockClient

REGION = "행정구역별(읍면동)"
YEARS = ["2015", "2020", "2023", "2024", "2025"]


@pytest.fixture
def households(household_path):
    return load_table(household_path)


def regions(table):
    return sorted({c.dim_codes[REGION] for c in table.cells})


def test_scope_round_trips_through_dict():
    data = {"items": ["T1"], "dims": {REGION: ["11", "21"]}, "periods": ["2020", "2023"]}
    assert Scope.from_dict(data).to_dict() == data
    assert Scope.from_dict({}) == Scope() and Scope().to_dict() == {}


def test_apply_keeps_items_codes_and_period_range(households):
    cut = scope.apply(households, Scope(items=("T1",), dims={REGION: ("11", "21")}, periods=("2020", "2023")))
    assert {(c.item_code, c.dim_codes[REGION], c.period_raw) for c in cut.cells} == {
        (i, r, p) for i in ["T1"] for r in ["11", "21"] for p in ["2020", "2023"]
    }
    assert cut.meta == households.meta  # the file hash stays that of the whole table


def test_same_level_follows_kosis_code_nesting():
    codes = ["00", "03", "11", "21", "11010", "11020", "21010", "11010530", "11010540"]
    assert scope.same_level(codes, "21") == ["00", "03", "11", "21"]
    assert scope.same_level(codes, "11020") == ["11010", "11020"]  # 서울 구들만, 부산 중구는 빠짐
    assert scope.same_level(codes, "11010540") == ["11010530", "11010540"]
    assert scope.same_level(["S0", "S1", "A10"], "S1") == ["S0", "S1", "A10"]  # not numeric: one level
    assert scope.top_level(codes) == ["00", "03", "11", "21"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("2023년 제주도 1인 가구는?", ("2023", "2023")),
        ("2015년부터 2023년까지 추이", ("2015", "2023")),
        ("2019~2024년 변화", ("2019", "2024")),
        ("2020년 대비 2024년", ("2020", "2024")),
        ("2016년 이후 가장 낮은 해", ("2016", "2025")),
        ("최근 2년 값", ("2024", "2025")),
        ("최근 추이는?", ("2015", "2025")),  # five most recent periods
        ("2003년 값은?", None),  # not in the table: every period, so the model can say so
        ("1인 가구는?", None),
        ("1985년생 여성의 2020년 비율", ("2020", "2020")),  # 1985년생 is a cohort label
    ],
)
def test_question_periods(question, expected):
    assert scope.question_periods(question, YEARS) == expected


def test_auto_scope_matches_regions_items_and_years(households):
    chosen = scope.auto_scope(households, "2023년 서울 1인 가구는 몇 가구야?")
    assert chosen == Scope(items=("T1",), dims={REGION: ("11",)}, periods=("2023", "2023"))


def test_auto_scope_adds_the_total_for_share_questions(households):
    chosen = scope.auto_scope(households, "2023년 부산 일반가구 중 1인 가구 비중은?")
    assert set(chosen.items) == {"T0", "T1"} and chosen.dims[REGION] == ("21",)


def test_auto_scope_without_a_region_keeps_the_top_level(households):
    chosen = scope.auto_scope(households, "2020년 1인 가구가 가장 적은 곳은?")
    assert chosen.dims[REGION] == ("00", "03", "11", "21", "39")  # not 시군구 / 읍면동


def test_auto_scope_short_region_names_and_lower_levels(households):
    assert scope.auto_scope(households, "제주 2인 가구").dims[REGION] == ("39",)
    assert scope.auto_scope(households, "삼청동 1인 가구").dims[REGION] == ("11010540",)
    # 2-character lower-level names (중구) are not matched; they hide inside ordinary words
    assert scope.auto_scope(households, "중구 1인 가구").dims[REGION] == ("00", "03", "11", "21", "39")


def test_fit_drops_oldest_periods_until_it_fits(households):
    cut, chosen, dropped = scope.fit(households, Scope(), "long", max_chars=4000)
    assert dropped > 0 and chosen.periods[1] == "2025"
    kept = sorted({c.period_raw for c in cut.cells})
    assert kept == YEARS[dropped:]


def test_fit_widens_an_empty_period_cut(households):
    cut, chosen, _ = scope.fit(households, Scope(dims={REGION: ("11",)}, periods=("2016", "2019")), "wide")
    assert chosen.periods is None and len(cut.cells) == 15


def test_select_modes(households):
    full, record = scope.select(households, "full", "q", None, "wide")
    assert full is households and record["mode"] == "full"
    part, record = scope.select(households, "oracle", "q", {"items": ["T0"], "dims": {REGION: ["00"]}}, "wide")
    assert record["cells"] == len(part.cells) == 5
    with pytest.raises(ValueError, match="golden"):
        scope.select(households, "oracle", "q", None, "wide")
    with pytest.raises(ValueError, match="unknown scope mode"):
        scope.select(households, "best", "q", None, "wide")


def test_chatbot_sends_only_the_scoped_part(households, mock_settings):
    client = MockClient("x")
    answer = Chatbot(households, client=client, settings=mock_settings, scope_mode="auto").ask("2023년 서울 1인 가구는?")
    user = client.calls[0][1]
    assert "서울특별시" in user and "부산광역시" not in user and "종로구" not in user
    assert answer.scope["cells"] == 1 and answer.scope["table_chars"] > 0
    with pytest.raises(ValueError, match="scope mode"):
        Chatbot(households, client=client, settings=mock_settings, scope_mode="best")
