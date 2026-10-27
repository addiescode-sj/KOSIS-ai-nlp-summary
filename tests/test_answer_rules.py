"""T1-14: answer wording: month periods, table labels and particles written as a person would."""

from __future__ import annotations

from kosis_chat.answer.render import item_label, period_text
from kosis_chat.judge.messages import refusal_message, with_josa


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
