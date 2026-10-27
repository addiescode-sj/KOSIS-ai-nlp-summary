"""T1-26 quality checks: format rules fire on real template bugs only; the judge score is capped, cached and summarized."""

from __future__ import annotations

import json

import pytest

from eval.judge import detection, format_issues, judge_rows, summary
from kosis_chat.llm.model import MockClient


@pytest.mark.parametrize(("text", "found"), [
    ("2025년 세종특별자치시의 인구밀도는 844명/㎢ 명/㎢입니다.", "단위 중복"),
    ("202412년 중구 혼인 값은 72건입니다.", "연월 코드 노출"),
    ("각각 1,110,342가구과 1,598,716가구로", "조사 '가구과'"),
    ("2023년 서울특별시 일반가구_계 가운데 1인 비중은 39.3%입니다.", "내부 라벨"),
    ("'조출생률'은(는) 비율·평균이라", "미완성 조사"),
])
def test_format_rules_catch_template_bugs(text, found):
    assert any(found in issue for issue in format_issues(text))


@pytest.mark.parametrize("text", [
    "2005년 경기도는 109,533명과 부산광역시는 25,681명이었습니다.",  # 명 has a final consonant: 과 is right
    "2008년과 2009년의 데이터는 없습니다.",
    "2023년 제주특별자치도의 1인 가구는 93,739가구입니다.",
    "2024년 대한민국의 기대수명은 83.7년입니다.",
    "2025년 3월 전국 혼인은 19,181건입니다.",
    "답변입니다.\n\n출처: KOSIS / 인구총조사 / 가구원수별 가구 - 읍면동 (일반가구_계)",  # the source line is skipped
])
def test_format_rules_leave_correct_sentences_alone(text):
    assert format_issues(text) == []


def _row(qid="q1", answer="2024년 서울특별시 출생아는 39,456명입니다.", outcome="answered"):
    return {"id": qid, "question": "질문", "answer": answer, "outcome": outcome, "facts": []}


def test_score_is_capped_by_format_issues_and_cached(tmp_path):
    cache = tmp_path / "judge.jsonl"
    client = MockClient(lambda s, u: json.dumps({"score": 2, "tags": [], "reason": "좋음"}))
    rows = [_row("ok"), _row("bad", "202412년 서울특별시 혼인은 4,401건입니다.")]
    first = judge_rows(rows, client, "HCX-007", cache=cache)
    assert [(r["score"], r["tags"]) for r in first] == [(2, []), (1, ["format_error"])]
    again = judge_rows(rows, client, "HCX-007", cache=cache)
    assert len(client.calls) == 2 and [r["judge_status"] for r in again] == ["cached", "cached"]


def test_failed_judge_leaves_no_score_and_summary_skips_it(tmp_path):
    client = MockClient("not json")
    records = judge_rows([_row()], client, "HCX-007", cache=tmp_path / "c.jsonl")
    assert records[0]["score"] is None
    assert summary(records)["judge_failed"] == ["q1"] and summary(records)["mean"] is None


def test_detection_splits_human_and_candidate_labels():
    records = [{"question_id": "a", "tags": ["ignored_part"]}, {"question_id": "b", "tags": []}]
    labels = [{"question_id": "a", "issue": "ignored_part", "source": "human"},
              {"question_id": "b", "issue": "format_error", "source": "candidate"},
              {"question_id": "zz", "issue": "other", "source": "human"}]  # not in this run: ignored
    assert detection(records, labels) == {"human": {"hits": 1, "total": 1, "missed": []},
                                          "candidate": {"hits": 0, "total": 1, "missed": ["b:format_error"]}}
