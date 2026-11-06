"""T1-28 S6 legacy isolation: the new code reaches kosis_chat/legacy/ only through kosis_chat/graph/legacy.py, and only
for the fallback chatbot. The Answer is kosis_chat/schemas/answer.py (the same class under every old path), and
graph/legacy.stat_answer_of gives the StatAnswer contract.to_stat_answer gives."""

import pytest

from kosis_ai.schemas import StatAnswer, StatQuery, TextPart
from kosis_chat import chatbot as old_chatbot
from kosis_chat.contract import to_source, to_stat_answer
from kosis_chat.graph import legacy as door
from kosis_chat.legacy import chatbot as legacy_chatbot
from kosis_chat.legacy import fallback as legacy_fallback
from kosis_chat.prepare.schemas import TableMeta
from kosis_chat.schemas.answer import Answer

META = TableMeta(
    tbl_id="DT_1B8000H", tbl_nm="시도/인구동태건수 및 동태율", survey_nm="인구동향조사", org_nm="통계청", period_type="Y",
    source_url="https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1B8000H", fetched_at="2026-10-22",
    input_format="parquet", file_sha256="0" * 64,
)  # fmt: skip
TEXT = "2025년 경기도 출생아 수는 76,309명입니다."


def answer(status="ok", text=TEXT, **kw):
    return Answer(text=text, status=status, table_meta=META, prompt_version="compact-v2", **kw)


def test_answer_is_one_class_under_every_path():
    assert legacy_chatbot.Answer is Answer and old_chatbot.Answer is Answer
    assert Answer.__module__ == "kosis_chat.schemas.answer"


def test_the_door_hands_on_the_fallback_names_unchanged():
    assert door.Chatbot is legacy_chatbot.Chatbot
    assert door.FALLBACK_PROMPT == legacy_fallback.FALLBACK_PROMPT == "compact-v2"
    assert door.step_fallback_chatbot is legacy_fallback.step_fallback_chatbot


@pytest.mark.parametrize("status", ["timeout", "api_error", "input_too_long"])
def test_failed_call_is_none_both_ways(status):
    failed = answer(status=status, text="")
    assert door.stat_answer_of("q", failed) is None and to_stat_answer("q", failed) is None


def test_finished_stat_answer_is_returned_as_it_is():
    stat = StatAnswer(outcome="answered", query=StatQuery(question="q"), parts=[TextPart(text="x")], text="x",
                      source=to_source(META), numbers_origin="model")
    built = answer(stat_answer=stat)
    assert door.stat_answer_of("q", built) is stat and to_stat_answer("q", built) is stat


def test_fallback_text_answer_matches_the_b0_conversion():
    for org_id in (None, "101"):
        assert door.stat_answer_of("2025년 경기도 출생아 수는?", answer(), org_id=org_id) == to_stat_answer(
            "2025년 경기도 출생아 수는?", answer(), org_id=org_id)
