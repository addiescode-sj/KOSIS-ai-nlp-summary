"""Answer -> StatAnswer (kosis_chat/contract.py); its copy in results.jsonl is checked in test_eval.py."""

import pytest
from pydantic import ValidationError

from kosis_ai.schemas import StatAnswer, TextPart
from kosis_chat.chatbot import Answer
from kosis_chat.contract import render_text, strip_model_source, to_source, to_stat_answer
from kosis_chat.data.schema import TableMeta

META = TableMeta(
    tbl_id="DT_1B8000H", tbl_nm="시도/인구동태건수 및 동태율", survey_nm="인구동향조사", org_nm="통계청", period_type="Y",
    source_url="https://kosis.kr/statHtml/statHtml.do?orgId=101&tblId=DT_1B8000H", fetched_at="2026-10-22",
    input_format="parquet", file_sha256="0" * 64,
)  # fmt: skip
TEXT = "2025년 경기도 출생아 수는 76,309명입니다."


def answer(status="ok", text=TEXT):
    return Answer(text=text, status=status, table_meta=META, prompt_version="baseline-v1")


def test_ok_answer_is_wrapped_with_a_code_built_source():
    stat = to_stat_answer("2025년 경기도 출생아 수는?", answer())
    assert stat.outcome == "answered" and stat.parts == [TextPart(text=TEXT)]
    assert stat.text == f"{TEXT}\n\n출처: {stat.source.citation()}"
    assert stat.query.question == "2025년 경기도 출생아 수는?" and stat.result.facts == []
    assert stat.source.citation() == f"KOSIS / 인구동향조사 / 시도/인구동태건수 및 동태율 ({META.source_url})"


@pytest.mark.parametrize("status", ["timeout", "api_error", "input_too_long"])
def test_failed_call_is_not_an_answer(status):
    assert to_stat_answer("q", answer(status=status, text="")) is None


def test_json_round_trip_is_lossless():
    stat = to_stat_answer("q", answer())
    assert StatAnswer.model_validate_json(stat.model_dump_json()) == stat


def test_source_without_url():
    source = to_source(TableMeta(**{**META.__dict__, "source_url": ""}), org_id="101")
    assert source.url is None and source.citation() == "KOSIS / 인구동향조사 / 시도/인구동태건수 및 동태율"


def test_contract_rules_still_apply():
    # StatAnswer itself refuses an answer with no source; the wrapper always fills one from the table meta.
    with pytest.raises(ValidationError):
        StatAnswer(outcome="answered", text=TEXT)


@pytest.mark.parametrize("label", [
    "출처: KOSIS / old", "**출처: KOSIS / old**", "**출처:** KOSIS / old",
    "**출처**: KOSIS / old", "  > - **출처**: [KOSIS](https://kosis.kr/a), 인구총조사 기반",
    "출처： KOSIS / [인구동향조사] / old",
])
def test_strip_standalone_source_variants(label):
    assert strip_model_source(f"{TEXT}\n\n{label}") == TEXT


@pytest.mark.parametrize("links", [
    "https://kosis.kr/a", "(https://kosis.kr/a)",
    "링크: [KOSIS 통계표 바로가기](https://kosis.kr/a)", "([링크](http://kosis.kr/a))",
    "(https://kosis.kr/a)\n링크: [KOSIS 통계표 바로가기](https://kosis.kr/a)",
])
def test_attached_links_and_source_separators_are_removed(links):
    text = f"{TEXT}\n\n---\n출처: old\n{links}\n---\n\n추가 설명입니다."
    assert strip_model_source(text) == f"{TEXT}\n\n추가 설명입니다."


@pytest.mark.parametrize("text", [
    "출처가 다른 자료는 합치지 않습니다.", "문장 중간 출처: 설명은 유지합니다.",
    "본 데이터는 통계표(DT_1B8000H)에서 제공된 정보입니다.",
    "\n\n\n본문\nhttps://example.org/body\n\n---\n다음 문단",
    "출처\n: 본문은 줄을 넘겨 출처 라벨로 해석하지 않습니다.",
])
def test_body_and_unrelated_links_are_preserved(text):
    assert strip_model_source(text) == text


def test_source_cleanup_stops_at_body_and_keeps_leading_spacing():
    text = "\n\n\n본문\n출처: old\nhttps://kosis.kr/a\n설명\nhttps://example.org/body"
    assert strip_model_source(text) == "\n\n\n본문\n설명\nhttps://example.org/body"


def test_render_is_idempotent_and_preserves_model_original():
    raw = f"{TEXT}\n\n출처: wrong\n(https://kosis.kr/wrong)"
    original = answer(text=raw)
    stat = to_stat_answer("q", original)
    assert original.text == raw
    assert stat.parts == [TextPart(text=TEXT)]
    assert stat.text.count("출처:") == 1
    assert stat.text.endswith(f"출처: {stat.source.citation()}")
    assert render_text(stat.text, stat.source) == stat.text
    assert render_text(raw, None) == TEXT
    assert render_text("", stat.source) == f"출처: {stat.source.citation()}"

@pytest.mark.parametrize("boundary", ["\n", "---\n"])
def test_unrelated_url_after_source_block_boundary_is_preserved(boundary):
    text = f"본문\n출처: old\n{boundary}https://example.org/body"
    expected_gap = "\n\n" if boundary == "\n" else "\n"
    assert strip_model_source(text) == f"본문{expected_gap}https://example.org/body"


def test_source_uses_url_agency_instead_of_a_global_default():
    meta = TableMeta(**{**META.__dict__, "source_url": META.source_url.replace("orgId=101", "orgId=111")})
    assert to_source(meta).org_id == "111"
    assert to_source(meta, org_id="111").org_id == "111"
    with pytest.raises(ValueError, match="conflicts"):
        to_source(meta, org_id="101")


@pytest.mark.parametrize("url", [
    "", "https://kosis.kr/table", "https://example.invalid/table?orgId=111",
    "https://kosis.kr.evil.org/table?orgId=111", "javascript:alert(1)", "https://[bad",
    "https://kosis.kr/table?orgId=abc", "https://kosis.kr/table?orgId=101&orgId=111",
    "https://kosis.kr/table?orgId=101&tblId=WRONG",
])
def test_unknown_or_conflicting_source_identity_fails_closed(url):
    with pytest.raises(ValueError):
        to_source(TableMeta(**{**META.__dict__, "source_url": url}))


def test_explicit_source_identity_without_url_query():
    meta = TableMeta(**{**META.__dict__, "source_url": "https://example.invalid/fixture"})
    assert to_source(meta, org_id="111").org_id == "111"
    with pytest.raises(ValueError):
        to_source(meta, org_id="not-an-id")


def test_filename_agency_uses_the_documented_identity_only():
    from kosis_chat.contract import org_id_from_table_path
    assert org_id_from_table_path("111_DT_X.parquet", "DT_X") == "111"
    assert org_id_from_table_path("DT_X.parquet") is None
    assert org_id_from_table_path("wrong_DT_X.parquet") is None
    with pytest.raises(ValueError):
        org_id_from_table_path("111_DT_X.parquet", "DT_Y")


@pytest.mark.parametrize("text", ["", " \n ", "출처: old", "출처: old\nhttps://kosis.kr/old"])
def test_empty_or_source_only_answer_is_not_successful(text):
    with pytest.raises(ValueError, match="blank"):
        to_stat_answer("q", answer(text=text))


@pytest.mark.parametrize("url", ["https://example.org/body", "https://kosis.kr.evil.org/body"])
def test_unrelated_adjacent_source_url_is_body_content(url):
    assert strip_model_source(f"본문\n출처: old\n{url}") == f"본문\n{url}"


@pytest.mark.parametrize("link", ["- https://kosis.kr/old", "* [표](https://kosis.kr/old)", "> - https://sub.kosis.kr/old"])
def test_bulleted_kosis_citation_links_are_removed(link):
    assert strip_model_source(f"본문\n출처: old\n{link}") == "본문"


def test_unrelated_paragraph_spacing_is_unchanged_by_source_removal():
    assert strip_model_source("첫 문단\n\n\n둘째 문단\n출처: old") == "첫 문단\n\n\n둘째 문단"


def test_plus_bulleted_source_label_is_removed():
    assert strip_model_source("본문\n+ 출처: old\n+ https://kosis.kr/old") == "본문"


def test_wrapped_answer_records_that_the_model_wrote_the_numbers():
    stat = to_stat_answer("q", answer())
    assert stat.numbers_origin == "model" and stat.numbers_verified is False and stat.defaults_verified is False
