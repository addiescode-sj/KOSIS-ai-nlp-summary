"""kosis_chat/shared/{defaults,cautions,patterns,texts}.py (T1-28 S2b): the default_applied and caution strings are
made and read in one place with the same characters as before (D9 (a)); the shared patterns are written once (D16 (d))."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kosis_chat.answer import caveats, render, template, verify
from kosis_chat.compute import analyze, roles
from kosis_chat.data import scope
from kosis_chat.judge import answerability, messages, question
from kosis_chat.prepare import index
from kosis_chat.shared import cautions, defaults, patterns, texts

from kosis_ai import normalize, schemas  # isort: skip

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- default_applied


def test_default_strings():
    assert defaults.make_latest("2024") == "latest (2024)"
    assert defaults.make_previous("2024", "2023") == "latest (2024) vs previous (2023)"
    assert defaults.make_recent(["2020", "2021", "2022", "2023", "2024"]) == "recent 5 points (2020~2024)"
    assert defaults.make_recent(["202401"]) == "recent 1 points (202401~202401)"


@pytest.mark.parametrize(
    ("value", "text", "codes"),
    [
        ("latest (2024)", ("latest", ("2024",)), ("latest", ("2024",))),
        ("latest (2024) vs previous (2023)", ("previous", ("2024", "2023")), ("previous", ("2024", "2023"))),
        ("recent 5 points (2020~2024)", ("recent", ("5", "2020~2024")), ("recent", ("5", "2020", "2024"))),
        ("latest (최근)", ("latest", ("최근",)), None),  # the sentence reader takes any text, the plan reader codes only
        ("중위 추계", None, None),
    ],
)
def test_default_parsers(value, text, codes):
    assert defaults.parse_period(value) == text
    assert defaults.parse_period_codes(value) == codes


@pytest.mark.parametrize(
    ("applied", "note", "sentence"),
    [
        ({"period": "latest (2024)"}, "시점=가장 최근 시점 2024", "질문에 시점이 없어 가장 최근 시점인 2024년 기준으로 답합니다."),
        ({"period": "latest (2024) vs previous (2023)"}, "시점=가장 최근 시점 2024과 그 직전 시점 2023",
         "질문에 시점이 없어 가장 최근 시점인 2024년을 그 직전 시점 2023년과 비교합니다."),
        ({"period": "recent 3 points (2022~2024)"}, "시점=최근 3개 시점 2022~2024",
         "질문에 시점이 없어 최근 3개 시점(2022~2024년)으로 답합니다."),
        ({"가정별": "중위 추계"}, "가정별=중위 추계", "질문에 없는 조건은 가정별 중위 추계 기준으로 답합니다."),
        ({"가정별": "latest (2024)"}, "가정별=가장 최근 시점 2024", "질문에 없는 조건은 가정별 latest (2024) 기준으로 답합니다."),
    ],
)
def test_default_readers(applied, note, sentence):
    assert messages.default_note(applied) == (f"\n(질문에 없는 조건은 다음으로 정했다: {note}. "
                                              "이 조건으로 답하고 답변 첫 문장에 이 조건을 밝힌다.)")
    assert render.default_sentence(applied) == sentence


def test_default_old_names():
    assert messages._LATEST is defaults.LATEST and messages._RECENT is defaults.RECENT
    assert messages._PREVIOUS is defaults.PREVIOUS
    assert analyze._LATEST is defaults.LATEST_CODES and analyze._RECENT is defaults.RECENT_CODES
    assert analyze._PREVIOUS is defaults.PREVIOUS_CODES
    assert messages._DEFAULT_KEYS is defaults.DEFAULT_KEYS == {"period": "시점"}


# ---------------------------------------------------------------- cautions


def test_caution_strings():
    assert cautions.event_source("코로나19", "2020-01") == "event_source: model (코로나19 = 2020-01)"
    assert cautions.default_applied("가정별", "중위 추계") == "default_applied: 가정별 = 중위 추계"
    assert cautions.DEFAULT_LATEST == "default_applied: 시점 = 최신"
    assert cautions.missing_years(["2019", "2020"]) == "missing_years: 2019, 2020"
    assert cautions.recent_short(5, 3) == "recent: 5개 중 3개 시점만 숫자가 있음"
    assert cautions.simple_average("조출생률") == "simple_average: 조출생률의 시점 평균은 가중치 없는 단순 평균입니다."


def test_caution_parsers():
    m = cautions.parse_event_source("event_source: model (코로나19 = 2020-01)")
    assert (m.group("name"), m.group("start"), m.group("month")) == ("코로나19", "2020", "01")
    assert cautions.parse_event_source("event_source: model (IMF = 1997)").group("month") is None
    assert cautions.parse_caution("missing_years: 2019, 2020") == ("missing_years", "2019, 2020")
    assert cautions.parse_caution("no prefix") is None
    assert cautions.parse_default_applied("가정별 = 중위 추계") == ("가정별", "중위 추계")
    assert cautions.parse_missing_years("2019, 202001") == ["2019", "202001"]
    assert caveats._EVENT_MODEL is cautions.EVENT_MODEL and caveats._PREFIX is cautions.PREFIX


@pytest.mark.parametrize(
    ("caution", "kind", "text"),
    [
        ("event_source: model (코로나19 = 2020-01)", "event_source",
         "'코로나19'의 시작 시점(2020년 1월)은 사건 달력에 없어 모델이 알려 준 값입니다."),
        ("missing_years: 2019, 202001", "missing_years", "2019년, 2020년 1월은(는) 표에 없습니다."),
        ("recent: 5개 중 3개 시점만 숫자가 있음", "missing_years", "요청한 최근 시점 가운데 일부만 숫자가 있습니다."),
        ("default_applied: 가정별 = 중위 추계", "default_applied", "질문에 없는 가정별은(는) '중위 추계' 기준입니다."),
        ("simple_average: x의 시점 평균은 가중치 없는 단순 평균입니다.", "simple_average",
         "x의 시점 평균은 가중치 없는 단순 평균입니다."),
    ],
)
def test_caveat_from_caution(caution, kind, text):
    caveat = caveats._from_caution(caution)
    assert (caveat.kind, caveat.text) == (kind, text)


# ---------------------------------------------------------------- texts (josa)


def test_josa_moved():
    assert messages.josa is render.josa is template.josa is texts.josa
    assert messages.with_josa is texts.with_josa
    assert texts.josa("서울특별시", "이") == "서울특별시가"
    assert texts.josa("3", "은") == "3은" and texts.josa("12%", "으로") == "12%로" and texts.josa("서울", "으로") == "서울로"
    assert texts.with_josa("'조출생률'은(는) 비율") == "'조출생률'은 비율"


# ---------------------------------------------------------------- patterns


def test_patterns_are_the_shared_objects():
    assert index._UNIT_PAREN is question._UNIT_PAREN is scope._UNIT_PAREN is patterns.UNIT_PAREN
    assert answerability._UNIT_SUFFIX is render._UNIT_SUFFIX is normalize._UNIT_SUFFIX is patterns.UNIT_PAREN
    assert question._REGION_SUFFIX is scope._REGION_SUFFIX is roles._PROVINCE is patterns.REGION_SUFFIX
    assert question._YEAR is scope._YEAR is patterns.YEAR
    assert question._RANGE is scope._YEAR_RANGE is patterns.YEAR_RANGE
    assert question._SINCE is scope._YEAR_SINCE is patterns.YEAR_SINCE
    assert question._RECENT_N is scope._RECENT_N is patterns.RECENT_N
    assert question._COHORT is patterns.COHORT
    assert verify.NUMBER is schemas.NUMBER is schemas._NUMBER is patterns.NUMBER
    assert template._DERIVED_IDS is patterns.FACT_ID


@pytest.mark.parametrize(
    "label", ["출생건수 (명)", "1인(가구) ", "기대수명(출생시 기대여명) - 남", "(명)", "a (b) (c)", "없음", "x()"])
def test_unit_paren_without_group_matches_the_same(label):
    """The copies without the capture group (scope, answerability, render, normalize) only .sub it: the same."""
    assert patterns.UNIT_PAREN.sub("", label) == re.sub(r"\s*\([^()]*\)\s*$", "", label)


@pytest.mark.parametrize(
    "label", ["부산광역시", "경기도", "세종특별자치시", "전북특별자치도", "서울특별시", "동부", "전국", "도", "특별시도", "x도\n"])
def test_region_suffix_and_province_agree(label):
    old_province = re.compile(r"(?:특별시|광역시|특별자치시|특별자치도|도)$")
    assert bool(patterns.REGION_SUFFIX.search(label)) == bool(old_province.search(label))
    assert patterns.REGION_SUFFIX.sub("", label) == old_province.sub("", label)


# The pattern texts written once (aliases elsewhere); a second copy in kosis_chat/ or src/ fails this.
ONCE = {
    r"\s*\(([^()]*)\)\s*$": "src/kosis_ai/units.py",
    r"\s*\([^()]*\)\s*$": None,
    r"\(([^()]*)\)": "src/kosis_ai/units.py",
    r"(특별자치시|특별자치도|특별시|광역시|도)$": "kosis_chat/shared/patterns.py",
    r"(?:특별시|광역시|특별자치시|특별자치도|도)$": None,
    r"(?<!\d)((?:19|20)\d{2})(?!\d)": "kosis_chat/shared/patterns.py",
    r"((?:19|20)\d{2})\s*년?\s*이후": "kosis_chat/shared/patterns.py",
    r"최근\s*(\d+)\s*년": "kosis_chat/shared/patterns.py",
    r"(?:19|20)\d{2}\s*년생": "kosis_chat/shared/patterns.py",
    r"\d[\d,]*(?:\.\d+)?": "src/kosis_ai/schemas.py",
    r'r"f\d+"': "kosis_chat/shared/patterns.py",
    r"[0-9]{3}": "kosis_chat/shared/patterns.py",
    r"latest \((.+)\)": "kosis_chat/shared/defaults.py",
    r"latest[^(]*\((\d{4,6})": "kosis_chat/shared/defaults.py",
    r"event_source: model \(": "kosis_chat/shared/cautions.py",
}


@pytest.mark.parametrize(("text", "home"), ONCE.items(), ids=list(ONCE))
def test_pattern_written_once(text, home):
    files = sorted(p for d in ("kosis_chat", "src") for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    found = [p.relative_to(ROOT).as_posix() for p in files
             if text in "\n".join(line for line in p.read_text(encoding="utf-8").splitlines()
                                  if not line.lstrip().startswith("#") and "assert" not in line)]
    found = [f for f in found if f != "kosis_chat/shared/patterns.py" or home == f]  # its docstring names variants
    assert found == ([home] if home else []), f"{text!r} is written in {found}"
