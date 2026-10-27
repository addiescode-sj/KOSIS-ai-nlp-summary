"""T1-14: answer wording (period, labels, particles) and the docs/question_types.md rules the pipeline now applies."""

from __future__ import annotations

from pathlib import Path

import pytest

from kosis_chat.answer.render import item_label, period_text
from kosis_chat.judge.answerability import EVALUATIVE
from kosis_chat.judge.messages import refusal_message, with_josa

ROOT = Path(__file__).resolve().parents[1]


def test_period_text_writes_months_as_year_and_month():
    assert period_text("202412") == "2024년 12월"
    assert period_text("202503") == "2025년 3월"
    assert period_text("2024") == "2024년"


def test_item_label_turns_table_labels_into_words():
    assert item_label("1인 (가구)") == "1인 가구"
    assert item_label("6인 이상 (가구)") == "6인 이상 가구"
    assert item_label("일반가구_계 (가구)") == "일반가구 전체"
    assert item_label("출생건수 (명)") == "출생건수"  # other unit suffixes are still dropped


def test_placeholder_particles_are_settled():
    assert with_josa("'조출생률'은(는)") == "'조출생률'은"
    assert with_josa("'이혼건수'와(과) '조이혼율'은(는)") == "'이혼건수'와 '조이혼율'은"
    assert "(는)" not in refusal_message("not_aggregatable", what="'중위 추계'·'고위 추계'")


@pytest.mark.parametrize(("question", "hit"), [
    ("1990년생 누적 혼인 비율은 높은 편이야?", True),
    ("서울 출산율은 낮은 편인가요?", True),
    ("2015년 대비 2024년 1인 가구 증가율은 서울특별시와 경기도 중 어디가 더 높아?", False),
    ("2025년 서울특별시 출생아 수는 10년 평균의 몇 % 수준이야?", False),
])
def test_evaluative_pattern_is_narrow(question, hit):
    assert bool(EVALUATIVE.search(question)) is hit


# --- real tables (gitignored data): the four rules fire on their questions and nowhere near them ---

def _pipeline(stem: str):
    from kosis_chat.data.loader import load_table
    from kosis_chat.llm.model import MockClient
    from kosis_chat.pipeline import StatPipeline

    path = ROOT / "data" / "values" / f"{stem}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not available")
    return StatPipeline(load_table(path), client=MockClient("[mock] answer"))


@pytest.mark.parametrize(("stem", "question", "outcome", "reason"), [
    ("101_DT_101094A001", "1990년생 누적 혼인 비율은 높은 편이야?", "clarify", None),
    ("101_DT_1JC1502", "2015~2024년 서울특별시 1인 가구 수를 모두 합치면 몇 가구야?", "refused", "not_aggregatable"),
    ("101_DT_1JC1502", "2010년부터 2024년까지 전국 1인 가구 수의 연평균은?", "refused", "no_period"),
    # must still be answered: flows summed over years, regions summed, a trend with a missing start year
    ("101_DT_1B8000H", "2020년부터 2024년까지 전국 혼인 건수를 모두 합하면 몇 건이야?", "answered", None),
    ("101_DT_1JC1502", "2025년 1인 가구, 지역별로 다 더하면 몇 가구야?", "answered", None),
    ("101_DT_101094A001", "1985년생 여성의 누적 혼인 비율은 2013년부터 2017년까지 어떻게 변했어?", "answered", None),
])
def test_rules_fire_only_where_the_criteria_say(stem, question, outcome, reason):
    answer = _pipeline(stem).ask(question).stat_answer
    assert answer.outcome == outcome
    assert (answer.refusal.reason if answer.refusal else None) == reason


def test_cause_part_is_declined_in_a_note():
    answer = _pipeline("101_DT_1B8000H").ask("2019~2023년 서울 혼인건수 추이와 그렇게 된 원인을 알려줘").stat_answer
    assert answer.outcome == "answered" and "원인이나 효과는 판단할 수 없습니다" in answer.text
