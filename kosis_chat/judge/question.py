"""What a question asks of one table: periods, the labels it names (items and categories) and operation words.

A light parser for the rules in answerability.py until T1-06 builds a StatQuery with the LLM.
"""

from __future__ import annotations

import re
from functools import lru_cache

from kosis_chat.config import PROJECT_ROOT
from kosis_chat.prepare.schemas import Cell, TableInput  # noqa: F401  (Cell, TableInput: re-export, moved in T1-28 S2)
from kosis_chat.shared.events import load_events

try:
    from kosis_ai import normalize
except ImportError:  # `python -m eval.run_eval` has no src/ on the path until T0-01 makes kosis_ai an installed package
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from kosis_ai import normalize

# re-export: moved in T1-28 S1b (after the block above: judge/schemas.py imports kosis_ai)
from kosis_chat.judge.schemas import Match, Parsed, Periods, in_range  # noqa: E402
# re-export: the table index moved to prepare/index.py in T1-28 S2 (prepare comes before judge)
from kosis_chat.prepare.index import (  # noqa: E402, F401
    _REGION_AXIS,
    _UNIT_PAREN,
    ITEM_AXIS,
    SCENARIO_AXIS,
    TOTAL_LABELS,
    Axis,
    TableIndex,
    _link_parents,
    _measure,
    build_index,
    cell_measure,
    index_of,
    is_total,
    total_stem,
)
# T1-28 S2b: label helpers and the patterns two stages share moved to kosis_chat/shared/ (old names are aliases).
# LABEL_ALIASES was read from data/scope.py (the judge importing the B0 scope module).
from kosis_chat.shared.labels import LABEL_ALIASES, interval_forms, squash  # noqa: E402
from kosis_chat.shared.labels import MEMBER_TAILS, SEX_NAMES  # noqa: E402  # loop 14
from kosis_chat.shared.labels import DOING_ENDINGS, NATION_AXIS, NATION_WORD  # noqa: E402  # loop 15
from kosis_chat.shared.patterns import ORDINAL_NAME  # noqa: E402  # loop 15
from kosis_chat.shared.patterns import COHORT, PAREN, RECENT_N, REGION_SUFFIX, YEAR, YEAR_RANGE, YEAR_SINCE  # noqa: E402
from kosis_chat.shared.patterns import PREVIOUS_MONTH, PREVIOUS_SAME_MONTH, PREVIOUS_YEAR  # noqa: E402  # loop 5
from kosis_chat.shared.patterns import COUNT_STEM_SUFFIX, TOTAL_PREFIX  # noqa: E402  # loop 12

COUNT_KINDS = ("count_stock", "count_flow")

_COHORT = COHORT
_YEAR = YEAR
_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월")
# A month without its year takes the year named before it: "2025년 1월 대비 12월", "2025년 전국 혼인 중 12월".
_BARE_MONTH = re.compile(r"(?<![\d~\-])(?<![~\-]\s)(\d{1,2})\s*월(?!\s*\d)")  # not the end of a month range "1~6월"
_RANGE = YEAR_RANGE
_SINCE = YEAR_SINCE
_RECENT_N = RECENT_N
_RECENT = re.compile(r"최근|요즘|근래")
# Period expressions (v2, T1-25): a decade, a part of it with a length, a half or a quarter of a year (month
# ranges on a monthly table), last year against this year, and N years before a named year.
_DECADE = re.compile(r"((?:19|20)\d)0\s*년대(?:\s*(초반|전반|후반)\s*(\d)\s*년)?")
_HALF = re.compile(r"((?:19|20)\d{2})\s*년\s*(상|하)반기")
_QUARTER = re.compile(r"((?:19|20)\d{2})\s*년\s*([1-4])\s*분기")
_PREVIOUS_YEAR = re.compile(r"(?:작년|전년|지난\s*해)\s*(?:대비|보다|에\s*비해|과|와|하고)")
_YEARS_BEFORE = re.compile(r"(?<!\d)(\d{1,2})\s*년\s*전(?=과|보다|에|의|하고|\s|$)")  # not "2025년 전국"
# An event of the calendar (shared/events.json) followed by a time word (T0-04 S5): "코로나 직전 해", "IMF 이후",
# "코로나19 유행 직전", "외환위기 2년 전", "세월호 참사 당시". A name with no time word after it is not a period, so a
# region or a label that shares a word with an event stays a label.
_EVENT_WHEN = (r"\s*(?:유행|발생|확산|사태|시작|출범)?\s*"
               r"(?:(?P<n>\d{1,2})\s*년\s*(?P<dir>이전|이후|전|후|뒤)|(?P<word>직전|직후|이후|부터|당시|때))")
TREND_WORDS = re.compile(r"추이|변화|변했|흐름|추세")
# Operations over periods (v2): an average of several periods, a sum over them ("모두" counts with a period range).
AVG_WORDS = re.compile(r"평균")
# "누적" is not here: it names items ("누적 혼인 비율"), it does not ask for a sum over periods.
TOTAL_WORDS = re.compile(r"모두|전부")
# "얼마나 늘었나", "감소했어": a change asked without 추이/변화 (kosis_chat/shared/patterns.py since loop 2).
from kosis_chat.shared.patterns import CHANGE_WORDS  # noqa: E402
# Every form of 더하다·합하다·합치다 (더해, 합쳐, 합쳤), and the nouns and adverbs that ask for a total (합계는, 합은,
# 도합, 통틀어) (loop 1: was a few of the forms only).
SUM_WORDS = re.compile(r"더하|더한|더해|더했|합하|합한|합해|합했|합치|합친|합쳐|합쳤|합산|총합|도합|통틀어"
                       r"|합계(?:는|가|를|로|은)|합(?:은|이)(?![가-힣])")
ARITH_WORDS = re.compile(SUM_WORDS.pattern + r"|나누|나눈|몇\s*배|차이|빼면|뺀")
CAUSAL_WORDS = re.compile(r"왜|이유|원인|때문|효과|영향")
# A causal question that also asks for numbers ("추이와 원인") is answered; the LLM declines the cause part.
VALUE_ASKS = re.compile(r"추이|몇|얼마|보여")
VAGUE_WORDS = re.compile(r"통계|현황|자료|정보|수치")
OUT_OF_DOMAIN = re.compile(r"기업|기술개발|매출|수출|수입액|GDP|물가|주가|부동산|금리|환율")

_REGION_SUFFIX = REGION_SUFFIX
# A count label 'X건수' is also asked by the counted person: '출생건수' as '출생아 수' / '출생아', '사망건수' as '사망자 수'
# (KOSIS itself names the same count 출생아수 / 사망자수 in other tables; no table holds both forms, checked over the
# 2,364 tables of data/table_metadata on 10/28).
_COUNT_SUFFIX = "건수"
_COUNTED_FORMS = ("수", "아수", "자수", "아", "자")
_PERSON_ENDS = ("자", "아", "인")
_TOKEN = re.compile(r"[0-9A-Za-z가-힣~\-]+")
# A particle that joins two words of a label: '고령자의 비율', '부모와 동거', '가구주 및 배우자' (the word before it
# has two syllables or more, so '이와' or '과' alone is no particle).
_INNER_PARTICLE = re.compile(r"(?<=[가-힣]{2})(?:의|와|과|및|에서|에)\s+")
_PARTICLE = re.compile(r"(?<=[가-힣]{2})(?:은|는|이|가|을|를|의|에|도|만|와|과)$")


# squash: kosis_chat/shared/labels.py (imported above).


# ---------------------------------------------------------------- periods


def _decade_range(m: re.Match) -> tuple[str, str]:
    start = int(m.group(1)) * 10
    if m.group(2) and m.group(3):
        n = int(m.group(3))
        if m.group(2) == "후반":
            return str(start + 10 - n), str(start + 9)
        return str(start), str(start + n - 1)
    return str(start), str(start + 9)


# ponytail: S2 keeps this cache: one pattern compiled from the event calendar (shared/events.py), not per table.
@lru_cache(maxsize=1)
def _event_pattern() -> re.Pattern:
    names = sorted(load_events(), key=len, reverse=True)
    alternation = "|".join(r"\s*".join(re.escape(ch) for ch in name) for name in names)
    return re.compile(rf"(?<![0-9A-Za-z가-힣])(?P<name>{alternation}){_EVENT_WHEN}", re.IGNORECASE)


def _event_periods(text: str) -> tuple[str, list[str], list[tuple[str, str]], list[tuple[str, str]], int]:
    """(text without the event phrases, singles, ranges, (phrase, start), position of the first phrase) for the
    calendar events the question anchors on."""
    singles: list[str] = []
    ranges: list[tuple[str, str]] = []
    found: list[tuple[str, str]] = []
    first = -1
    for m in _event_pattern().finditer(text):
        start = load_events()["".join(m.group("name").split()).casefold()]
        year = int(start[:4])
        if m.group("n"):
            n = int(m.group("n"))
            singles.append(str(year - n if m.group("dir") in ("전", "이전") else year + n))
        elif m.group("word") in ("이후", "부터"):
            ranges.append((str(year), "9999"))
        else:
            singles.append(str({"직전": year - 1, "직후": year + 1}.get(m.group("word"), year)))
        found.append((m.group(0).strip(), start))
        first = m.start() if first < 0 else first
    return _event_pattern().sub(" ", text), singles, ranges, found, first


def _bare_months(text: str) -> list[str]:
    """'YYYYMM' for each month named without a year, with the last year written before it; none before it: skipped."""
    out = []
    written = [m.span() for m in _MONTH.finditer(text)]
    for m in _BARE_MONTH.finditer(text):
        if any(a <= m.start(1) < b for a, b in written):
            continue
        month = int(m.group(1))
        years = [y.group(1) for y in _YEAR.finditer(text[: m.start()])]
        if years and 1 <= month <= 12:
            out.append(f"{years[-1]}{month:02d}")
    return out


def _earlier(single: str, text: str, words: str, labels: tuple[str, ...] = ()) -> str | None:
    """The period right before a named one (a year 'YYYY' or a month 'YYYYMM') that the question compares it with:
    "N년 전" (on text), the same month a year before, the month before ("N개월 전" too), the year before (shared/patterns
    PREVIOUS_*, on words: the question without its labels). A word inside a label the question names ('전년대비
    증감률') is the label's. None when there is no such word."""
    year, month = int(single[:4]), int(single[4:]) if len(single) == 6 else None
    found = None
    if m := _YEARS_BEFORE.search(text):
        found, back = m, (int(m.group(1)), 0)
    elif m := PREVIOUS_SAME_MONTH.search(words):
        found, back = m, (1, 0)
    elif month and (m := PREVIOUS_MONTH.search(words)):
        found, back = m, (0, int(m.group(1) or 1))
    elif m := PREVIOUS_YEAR.search(words):
        found, back = m, (1, 0)
    if found is None or any(squash(found.group(0)) in label for label in labels):
        return None
    if month is None:
        return str(year - back[0]) if back[1] == 0 else None
    total = (year - back[0]) * 12 + (month - 1) - back[1]
    return f"{total // 12}{total % 12 + 1:02d}"


def asked_periods(question: str, rest: str | None = None, labels: tuple[str, ...] = ()) -> Periods:
    """The periods a question names. rest: the question without the labels it names (Parsed.rest), where the words
    for the period right before a named one are read; labels: the squashed labels it names."""
    text = _COHORT.sub(" ", question)  # "1985년생" is a cohort label, not a period
    text, event_singles, event_ranges, events, event_at = _event_periods(text)
    singles: list[str] = []
    spans: list[tuple[str, str]] = []
    for m in _HALF.finditer(text):
        spans.append((f"{m.group(1)}01", f"{m.group(1)}06") if m.group(2) == "상" else (f"{m.group(1)}07", f"{m.group(1)}12"))
    text = _HALF.sub(" ", text)
    for m in _QUARTER.finditer(text):
        q = int(m.group(2))
        spans.append((f"{m.group(1)}{3 * q - 2:02d}", f"{m.group(1)}{3 * q:02d}"))
    text = _QUARTER.sub(" ", text)
    for m in _MONTH.finditer(text):
        singles.append(f"{m.group(1)}{int(m.group(2)):02d}")
    singles += _bare_months(text)
    text = _BARE_MONTH.sub(" ", _MONTH.sub(" ", text))
    ranges = [(a, b) for a, b in _RANGE.findall(text)] + event_ranges
    text = _RANGE.sub(" ", text)
    ranges += [(a, "9999") for a in _SINCE.findall(text)]
    text = _SINCE.sub(" ", text)
    for m in _DECADE.finditer(text):
        decade = _decade_range(m)
        # "2010년대 전반(2010~2014년)과 후반(2015~2019년)": the explicit ranges say it, the decade is not a third group.
        if not any(decade[0] <= a and b <= decade[1] for a, b in ranges):
            ranges.append(decade)
    text = _DECADE.sub(" ", text)
    singles += _YEAR.findall(text)
    if event_singles:  # in question order: before the years when the event comes first ("코로나 직전 해와 2025년")
        year_at = next((m.start() for m in _YEAR.finditer(question)), len(question))
        singles = event_singles + singles if event_at < year_at else singles + event_singles
    before = _YEARS_BEFORE.search(text)
    # "10년 전과 비교해서 2025년", "2024년 3월 전월 대비", "2023년은 작년보다": the earlier period is implied (loop 5:
    # was "N년 전" of a year only)
    if len(singles) == 1 and (not ranges or before) and (earlier := _earlier(singles[0], text, question if rest is None else rest,
                                                                 labels)):
        singles.insert(0, earlier)
    recent_n = _RECENT_N.search(question)
    return Periods(
        singles=tuple(dict.fromkeys(singles)),
        ranges=tuple(ranges + spans),
        recent_n=int(recent_n.group(1)) if recent_n else None,
        recent=bool(_RECENT.search(question)),
        spans=tuple(spans),
        previous=bool(_PREVIOUS_YEAR.search(question)),
        before=int(before.group(1)) if before and len(singles) == 2 else None,
        events=tuple(events),
    )


# ---------------------------------------------------------------- table index
# Axis, TableIndex, build_index, index_of, cell_measure, is_total: kosis_chat/prepare/index.py (re-exported above).


# ---------------------------------------------------------------- label matching


def _known_unit(text: str) -> bool:
    return text in normalize.KNOWN_UNITS or (text.endswith("당") and len(text) <= 12)


def label_forms(label: str, regional_total: bool = False) -> tuple[set[str], set[str], tuple[str, ...]]:
    """(exact forms, partial forms, tokens). '출생건수 (명)' -> exact {'출생건수'}; '기대수명(출생시 기대여명)' ->
    partial {'기대수명'}; '인구(명): 65세 이상' -> tokens ('인구', '65세', '이상')."""
    text = label.strip().lstrip("-").strip()
    m = _UNIT_PAREN.search(text)
    core = text[: m.start()].strip() if m and _known_unit(m.group(1).strip()) else text
    exact = {squash(core), *(squash(a) for a in LABEL_ALIASES.get(core, ()))}
    # '부모와 동거하는 자녀', '고령자의 비율' are also asked without the particle between the words (loop 1)
    plain = _INNER_PARTICLE.sub(" ", core)
    if plain != core:
        exact.add(squash(plain))
    short = _REGION_SUFFIX.sub("", core)
    if short != core and len(short) >= 2:
        exact.add(squash(short))
    stem = squash(core)[: -len(_COUNT_SUFFIX)]
    if squash(core).endswith(_COUNT_SUFFIX) and len(stem) >= 2:
        exact.update(stem + form for form in _COUNTED_FORMS)
    # A count of people 'X자수' / 'X아수' / 'X인수' is also asked by the people: '사망자수' as '사망자', '출생아수' as
    # '출생아' (loop 1).
    person = squash(core)[:-1]
    if squash(core).endswith("수") and len(person) >= 3 and person[-1] in _PERSON_ENDS:
        exact.add(person)
    if regional_total and core in TOTAL_LABELS:
        exact.add("전국")
    # loop 6: a total marked on a stem ('임금근로자-계', '일반가구_계') is asked by its stem ('임금근로자')
    if (stem := total_stem(core)) and len(squash(stem)) >= 2:
        exact.add(squash(stem))
    # loop 6: an interval name joined another way ('20-24세' as '20~24세', '20세부터 24세까지', '20대 초반')
    exact |= interval_forms(core)
    bare = PAREN.sub(" ", core)
    head = squash(bare.split(" - ")[0])
    # '기대수명(출생시 기대여명)' is also asked as '기대수명'; not '인구(명): 0-14세', where '인구' is only a prefix.
    partial = {head} - exact if head and "(" in core and ":" not in core else set()
    tokens = tuple(t for t in (_INNER_PARTICLE.sub("", w + " ").strip() for w in _TOKEN.findall(bare))
                   if len(t) >= 2 and t != "-")
    return exact, partial, tokens


def _found(form: str, question: str, squashed: str, lower_level: bool) -> bool:
    if len(form) < 2 or form not in squashed:
        return False
    if form.isdigit():  # cohort labels such as '1985': not a year of the period ("1985년" but "1985년생" is fine)
        return re.search(rf"(?<!\d){form}(?!\d)(?!\s*년(?!생))", question) is not None
    if lower_level and len(form) == 2:  # 2-character lower-level names hide in words ("더하면" has 하면)
        return re.search(rf"(?<![가-힣]){re.escape(form)}", question) is not None
    return True


_WORD = re.compile(r"[0-9A-Za-z가-힣]+")
_NAME_WORD = 3  # the shortest word read as coming from the survey's or the table's name


def name_spans(index: TableIndex, question: str) -> list[tuple[int, int]]:
    """Loop 6: where the question (squashed) says words of the survey's name or the table's title ('가상장애인실태조사
    에서'): for each word of the question, its longest start of three characters or more that is in one of those names.
    A label form strictly inside such a span is a part of the name's word, not the label ('장애' in '장애인실태조사')."""
    meta = index.table.meta
    names = [squash(n) for n in (getattr(meta, "survey_nm", ""), getattr(meta, "tbl_nm", "")) if n]
    spans = []
    for m in _WORD.finditer(question):
        word, at = m.group(0), len(squash(question[: m.start()]))
        size = next((k for k in range(len(word), _NAME_WORD - 1, -1) if any(word[:k] in n for n in names)), 0)
        if size:
            spans.append((at, at + size))
    return spans


def _outside_names(form: str, q: str, spans: list[tuple[int, int]]) -> bool:
    """The form occurs in q at least once that is not strictly inside a word of a name (name_spans)."""
    at = q.find(form)
    while at >= 0:
        if not any(a <= at and at + len(form) <= b and b - a > len(form) for a, b in spans):
            return True
        at = q.find(form, at + 1)
    return False


_JOIN_MARK = "-_·"  # a mark that joins a group's name and an attribute ('귀농인-여자', '가구주_남자')
_HANGUL = re.compile(r"[가-힣]+")


def _core(label: str) -> str:
    text = label.strip().lstrip("-").strip()
    m = _UNIT_PAREN.search(text)
    return squash(text[: m.start()] if m and _known_unit(m.group(1).strip()) else text)


def group_parts(axis: Axis, elsewhere) -> dict[str, tuple[str, str]]:
    """Loop 7: code -> (group, attribute) for the names of a category made of the name of a group of the same category
    and an attribute, joined by a mark or written together: '귀농인-여자' and '귀농인여자' beside '귀농인' or '귀농인-계'
    (the group: another name of the category, or the stem of its total name, two characters or more). The attribute
    is a word of two Hangul characters or more that no other category of the table names (elsewhere: their exact
    forms; there that category holds the attribute). A total name and a name with no such group have none, and so has
    a regional category ('서울특별시' is no group '서울' with an attribute)."""
    if axis.regional:
        return {}
    cores = {code: _core(label) for code, label in axis.labels.items()}
    groups = {c for c in cores.values() if len(c) >= 2}
    groups |= {squash(s) for label in axis.labels.values() if (s := total_stem(label)) and len(squash(s)) >= 2}
    out: dict[str, tuple[str, str]] = {}
    for code, core in cores.items():
        if is_total(axis.labels[code]):
            continue
        group = next((core[:k] for k in range(len(core) - 1, 1, -1) if core[:k] in groups), None)  # the longest
        if group is None:
            continue
        attr = core[len(group):].lstrip(_JOIN_MARK)
        if len(attr) >= 2 and _HANGUL.fullmatch(attr):
            out[code] = (group, attr)
    if out:
        out = {code: pair for code, pair in out.items() if pair[1] not in elsewhere()}
    return out


def head_parts(axis: Axis, elsewhere) -> dict[str, tuple[str, str, str]]:
    """Loop 12: code -> (attribute, head, stem) for the names of a category made the other way round from group_parts:
    an attribute before a head noun two names or more end with ('개인사업체수', '법인사업체수' beside '총사업체수'; '맞벌이
    가구', '외벌이 가구' beside '전체 가구'). The head is the longest end (spaces left out) that two names or more share
    and whose stem (the head without a counting word, shared COUNT_STEM_SUFFIX) has two characters or more, so names
    that share only a counting word or a person ending ('전입자수', '전출자수') are not split. Every name with that end
    puts an attribute of two Hangul characters or more before it, or a word of the whole (shared TOTAL_PREFIX: 총,
    전체): that name is the whole, attribute ''. An attribute another category names (elsewhere) is that category's,
    and a head (or stem) that is a name of its own of the category is that name's: neither is split. Not a regional
    category, not a total name."""
    if axis.regional:
        return {}
    cores = {code: _core(label) for code, label in axis.labels.items() if not is_total(label)}
    if len(cores) < 2:
        return {}
    own = set(cores.values())
    # a name that is another name, or a total's stem, with an attribute after it is group_parts' shape, not this one
    own |= {squash(s) for label in axis.labels.values() if (s := total_stem(label))}
    out: dict[str, tuple[str, str, str]] = {}
    for code, core in cores.items():
        text = _UNIT_PAREN.sub("", axis.labels[code]).strip().lstrip("-").strip()
        spaced = " " in text  # a name written in words is cut only where a word starts ('맞벌이 가구')
        starts = {len(squash(text[: m.start()])) for m in re.finditer(r"\S+", text)}
        best: tuple[int, int, str, str] | None = None
        for k in range(1, len(core) - 1):
            head = core[k:]
            stem = COUNT_STEM_SUFFIX.sub("", head)
            sharing = [c for c in cores.values() if c.endswith(head) and len(c) > len(head)]
            if (len(sharing) < 2 or len(stem) < 2 or not _HANGUL.fullmatch(head) or head in own or stem in own
                    or (spaced and k not in starts)):
                continue
            attrs = [c[: -len(head)] for c in sharing]
            if all(TOTAL_PREFIX.fullmatch(a) or (len(a) >= 2 and _HANGUL.fullmatch(a) and a not in own) for a in attrs):
                # the end the most names share, then the longest ('사업체수' over '인사업체수', '가구' over '이가구')
                if best is None or (len(sharing), len(head)) > best[:2]:
                    best = (len(sharing), len(head), head, stem)
        if best is not None:
            attr = core[: -len(best[2])]
            out[code] = ("" if TOTAL_PREFIX.fullmatch(attr) else attr, best[2], best[3])
    if out:
        taken = elsewhere()
        out = {code: parts for code, parts in out.items() if parts[0] not in taken}
    if sum(1 for a, _, _ in out.values() if a) < 2:  # one attribute is no split of names
        return {}
    return out


def head_parts_of(index: TableIndex) -> dict[str, dict[str, tuple[str, str, str]]]:
    """head_parts of every category of the table, once per table (TableInput.caches)."""
    def build() -> dict[str, dict[str, tuple[str, str, str]]]:
        def other_forms(name: str) -> set[str]:
            return set().union(*(label_forms(label)[0] for n in index.axes if n != name
                                 for label in index.axes[n].labels.values()))

        return {name: head_parts(axis, lambda name=name: other_forms(name)) for name, axis in index.axes.items()}

    caches = getattr(index.table, "caches", None)
    return build() if caches is None else caches.get("judge.head_parts", build)


def group_parts_of(index: TableIndex) -> dict[str, dict[str, tuple[str, str]]]:
    """group_parts of every category of the table, once per table (TableInput.caches)."""
    def build() -> dict[str, dict[str, tuple[str, str]]]:
        exact_of: dict[str, set[str]] = {}

        def forms(name: str) -> set[str]:
            if name not in exact_of:
                exact_of[name] = set().union(*(label_forms(label)[0] for label in index.axes[name].labels.values()))
            return exact_of[name]

        return {name: group_parts(axis, lambda name=name: set().union(*(forms(n) for n in index.axes if n != name)))
                for name, axis in index.axes.items()}

    caches = getattr(index.table, "caches", None)
    return build() if caches is None else caches.get("judge.group_parts", build)


def member_forms(index: TableIndex) -> tuple[dict[tuple[str, str], tuple[str, ...]], str]:
    """Loop 14: ((axis, code) -> the words a question says a name by through the people it is about, the sex
    initials of the table).

    - A name ending with a tail of a place people attend (shared MEMBER_TAILS) by its members: '중학교' -> '중학생'.
    - A name of a sex (shared SEX_NAMES) by its first character before a noun for those members, the member word of
      a name of the table or the member tail alone: '남학생', '여중학생'.
    - Both at once, by a sex's first character, the name's first character and the member tail's last character:
      '남고생' for '남자' and '고등학교'.
    A word two names of the table give is neither's; a word holding the exact form of another name of the table is
    that name's (the shorter name keeps it as before: '대학생' stays '대학' when the table has '대학')."""
    def build():
        places: list[tuple[str, str, str, str, str]] = []  # (axis, code, core, member word, member tail)
        sexes: list[tuple[str, str, str]] = []  # (axis, code, initial)
        exact_all: dict[str, set[tuple[str, str]]] = {}
        for axis in index.axes.values():
            for code, label in axis.labels.items():
                for form in label_forms(label)[0]:
                    if len(form) >= 2:
                        exact_all.setdefault(form, set()).add((axis.name, code))
                if is_total(label):
                    continue
                core = squash(PAREN.sub(" ", _core(label)).split(" - ")[0])
                if core in SEX_NAMES:
                    sexes.append((axis.name, code, core[0]))
                    continue
                tail = next(((t, m) for t, m in MEMBER_TAILS if core.endswith(t)), None)
                if tail and _HANGUL.fullmatch(core):
                    places.append((axis.name, code, core, core[: -len(tail[0])] + tail[1], tail[1]))
        given: dict[str, set[tuple[str, str]]] = {}
        nouns = {member for *_, member, _ in places} | {tail for *_, tail in places}
        for name, code, core, member, tail in places:
            if core not in member:
                given.setdefault(member, set()).add((name, code))
        for name, code, initial in sexes:
            for noun in nouns:
                given.setdefault(initial + noun, set()).add((name, code))
        for name, code, core, member, tail in places:
            for s_name, s_code, initial in sexes:
                short = initial + core[0] + tail[-1]
                given.setdefault(short, set()).update({(name, code), (s_name, s_code)})
        out: dict[tuple[str, str], list[str]] = {}
        for form, owners in given.items():
            by_axis: dict[str, set[str]] = {}
            for a, c in owners:
                by_axis.setdefault(a, set()).add(c)
            if any(len(codes) > 1 for codes in by_axis.values()):
                continue  # two names of one category give the word
            if any(f in form and not (exact_all[f] <= owners) for f in exact_all):
                continue  # another name of the table is in the word
            for owner in owners:
                out.setdefault(owner, []).append(form)
        return {k: tuple(sorted(v, key=len, reverse=True)) for k, v in out.items()}, "".join(i for *_, i in sexes)

    caches = getattr(index.table, "caches", None)
    return build() if caches is None else caches.get("judge.member_forms", build)


def doing_forms(index: TableIndex) -> dict[tuple[str, str], tuple[str, ...]]:
    """Loop 15: (axis, code) -> the words a question says a name for people by through what they did: a name ending
    with a person suffix (_PERSON_ENDS: '취업자', '귀농인') whose stem has two Hangul characters or more, said by the
    stem and a describing ending of a 하다 verb (shared DOING_ENDINGS: '취업한', '귀농하는'). Not when the stem is a
    name of the table or a word of its title (that name or word keeps it), not a total, not a word two names of one
    category give."""
    def build():
        names = {f for axis in index.axes.values() for label in axis.labels.values() for f in label_forms(label)[0]}
        title = squash(getattr(index.table.meta, "tbl_nm", "") or "")
        words = {_PARTICLE.sub("", w) for w in _WORD.findall(getattr(index.table.meta, "tbl_nm", "") or "")}
        given: dict[str, set[tuple[str, str]]] = {}
        for axis in index.axes.values():
            for code, label in axis.labels.items():
                core = _core(label)
                stem = core[:-1]
                if (is_total(label) or core[-1:] not in _PERSON_ENDS or len(stem) < 2 or not _HANGUL.fullmatch(stem)
                        or stem in names or stem in words or (stem in title and core not in title)):
                    continue
                for ending in DOING_ENDINGS:
                    given.setdefault(stem + ending, set()).add((axis.name, code))
        out: dict[tuple[str, str], list[str]] = {}
        for form, owners in given.items():
            if len(owners) == 1:
                out.setdefault(next(iter(owners)), []).append(form)
        return {k: tuple(sorted(v, key=len, reverse=True)) for k, v in out.items()}

    caches = getattr(index.table, "caches", None)
    return build() if caches is None else caches.get("judge.doing_forms", build)


def nation_pairs(index: TableIndex) -> dict[tuple[str, str], tuple[str, ...]]:
    """Loop 15: (axis, code) -> the two-character word a question says a pair of nations by ('한중' for '중국' in a
    category '한국인 남편의 아내 국적별'): the first character of the nation fixed by the category's name or the table's
    title (shared NATION_WORD: '한국인'), then the first character of a name of the category of nations (shared
    NATION_AXIS). Only when that character starts one name of the category (codes holding the same name count as
    one), and the fixed nation is no name of the category; not for a total."""
    def build():
        out: dict[tuple[str, str], tuple[str, ...]] = {}
        title = getattr(index.table.meta, "tbl_nm", "") or ""
        for axis in index.axes.values():
            if axis.regional or not NATION_AXIS.search(axis.name):
                continue
            cores = {code: _core(label) for code, label in axis.labels.items() if not is_total(label)}
            fixed = {m.group(1) for w in (axis.name + " " + title).split() if (m := NATION_WORD.match(w))}
            fixed = {f[0] for f in fixed if f not in cores.values()}
            starts: dict[str, set[str]] = {}
            for core in cores.values():
                if _HANGUL.match(core[:1]):
                    starts.setdefault(core[0], set()).add(core)
            for code, core in cores.items():
                if core[:1] in starts and len(starts[core[0]]) == 1:
                    forms = tuple(f + core[0] for f in sorted(fixed) if f != core[0])
                    if forms:
                        out[(axis.name, code)] = forms
        return out

    caches = getattr(index.table, "caches", None)
    return build() if caches is None else caches.get("judge.nation_pairs", build)


def _said_as_pair(form: str, question: str) -> bool:
    """Loop 15: a two-character word of a pair of nations said as a word of its own: nothing Hangul before it, and a
    space, a particle or nothing Hangul after it ('한중 부부', '한중의', not '한미래')."""
    for m in re.finditer(rf"(?<![가-힣]){re.escape(form)}", question):
        after = question[m.end():]
        if not re.match(r"[가-힣]", after) or _WORD_PARTICLE.match(after):
            return True
    return False


def ordinal_form(label: str) -> str | None:
    """Loop 15: a name of an ordinal with a person suffix ('둘째아', '넷째아 이상') without the suffix ('둘째', '넷째이상'),
    squashed; None for another name. The ordinal is read with the count words (shared ORDINAL_NAME)."""
    core = _core(label)
    m = ORDINAL_NAME.match(core)
    if m is None or core[m.end():m.end() + 1] not in _PERSON_ENDS or not core[m.end():]:
        return None
    return m.group(1) + core[m.end() + 1:]


# A particle or a copula right after a word ("첫째는", "둘째인", "셋째와"), nothing Hangul after it.
_WORD_PARTICLE = re.compile(r"(?:은|는|이|가|을|를|의|도|만|와|과|랑|이랑|하고|인|이고|이야|이며)(?![가-힣])")


def _said_as_ordinal(form: str, end: str, question: str) -> bool:
    """Loop 15: the question says an ordinal name without its suffix (ordinal_form) as the name, not as a place of a
    ranking or a count of another thing: a word of its own followed by nothing Hangul, by a particle or a copula, or
    by a word holding the name's person suffix (end: '둘째 출생아', '첫째 아이'). Not "둘째로 많은" (a ranking), not
    "셋째 주" (another counting word)."""
    pattern = r"(?<![가-힣])" + r"\s*".join(re.escape(ch) for ch in form)
    for m in re.finditer(pattern, question):
        after = question[m.end():]
        if not re.match(r"[가-힣]", after.lstrip()):
            return True
        if _WORD_PARTICLE.match(after):
            return True
        word = re.match(r"\s+([가-힣]+)", after)
        if word and end in word.group(1):
            return True
    return False


def _said_as_word(form: str, question: str) -> bool:
    """Loop 15: the question says form with no Hangul right before or right after it ('아버지가 취업한 자녀', not
    '미취업한' or '취업한다')."""
    return re.search(rf"(?<![가-힣]){re.escape(form)}(?![가-힣])", question) is not None


def _said_alone(form: str, question: str, before: str) -> bool:
    """Loop 14: the question says form as a word of its own (spaces allowed inside): no Hangul right before it but a
    sex initial (before) that starts the word ('남중학생'). Not '전남학생' (전남 + 학생)."""
    pattern = r"\s*".join(re.escape(ch) for ch in form)
    for m in re.finditer(pattern, question):
        at = m.start()
        if at == 0 or not re.match(r"[가-힣]", question[at - 1]):
            return True
        if question[at - 1] in before and (at == 1 or not re.match(r"[가-힣]", question[at - 2])):
            return True
    return False


def match_labels(index: TableIndex, question: str) -> list[Match]:
    q = squash(question)
    spans = name_spans(index, question)
    found: list[Match] = []
    splits = group_parts_of(index)
    members, initials = member_forms(index)
    doings = doing_forms(index)
    pairs = nation_pairs(index)
    for axis in index.axes.values():
        split = splits.get(axis.name, {})
        for code, label in axis.labels.items():
            lower = axis.parents.get(code) is not None
            exact, partial, tokens = label_forms(label, regional_total=axis.regional and axis.parents.get(code) is None)
            # loop 6: a two-character region name is a word's start too ('대구' is not a region inside '연령대구분')
            short = lower or axis.regional

            def seen(f: str) -> bool:
                return _found(f, question, q, short) and _outside_names(f, q, spans)

            hit = next((f for f in sorted(exact, key=len, reverse=True) if seen(f)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, True))
                continue
            hit = ("".join(tokens) if len(tokens) >= 2 and all(t in q and _outside_names(t, q, spans) for t in tokens)
                   else None)
            hit = hit or next((f for f in partial if seen(f)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, False))
                continue
            # loop 14: a name said through its members ('중학생', '남학생', '남고생': member_forms)
            hit = next((f for f in members.get((axis.name, code), ())
                        if f in q and _outside_names(f, q, spans) and _said_alone(f, question, initials)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, True))
                continue
            # loop 15: a name for people said by what they did ('취업한': doing_forms)
            hit = next((f for f in doings.get((axis.name, code), ())
                        if f in q and _outside_names(f, q, spans) and _said_as_word(f, question)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, True))
                continue
            # loop 15: an ordinal name said by the ordinal alone ('둘째' for '둘째아': ordinal_form)
            if (form := ordinal_form(label)) and form in q and _outside_names(form, q, spans):
                end = _core(label)[len(ORDINAL_NAME.match(_core(label)).group(1))]
                if _said_as_ordinal(form, end, question):
                    found.append(Match(axis.name, code, label, form, True))
                    continue
            # loop 15: a nation said in a pair with the nation the category fixes ('한중': nation_pairs)
            hit = next((f for f in pairs.get((axis.name, code), ())
                        if f in q and _outside_names(f, q, spans) and _said_as_pair(f, question)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, True))
                continue
            # loop 7: a group and an attribute said apart ("귀농인 중 여자", "귀농인 여성"); an attribute is also said
            # by its other names (LABEL_ALIASES)
            if code in split:
                group, attr = split[code]
                said = next((f for f in (attr, *(squash(a) for a in LABEL_ALIASES.get(attr, ()))) if seen(f)), None)
                if seen(group) and said:
                    found.append(Match(axis.name, code, label, group + attr, False, parts=(group, said)))
                    continue
            # loop 12: an attribute before a head noun the names share (head_parts): the name is asked by its
            # attribute with the head's stem ("개인 사업체"), by the attribute alone ("법인은"), or with the stem said
            # apart; the whole by the head or its stem alone ("사업체 수")
            if code in (heads := head_parts_of(index).get(axis.name, {})):
                if (hit := _said_by_head(heads[code], question, q, spans, heads)) is not None:
                    form, parts = hit
                    found.append(Match(axis.name, code, label, form, True, parts=parts))
    return _dedupe(found, q)


def _said_by_head(parts: tuple[str, str, str], question: str, q: str, spans,
                  heads: dict[str, tuple[str, str, str]]) -> tuple[str, tuple[str, ...]] | None:
    """(form, parts) a question names a head_parts name by, or None. The whole is named by its head only when no
    attribute of that head is named (in "맞벌이 가구는 몇 가구야" the head is the attribute's and a counting word)."""
    attr, head, stem = parts

    def seen(f: str) -> bool:
        return _found(f, question, q, True) and _outside_names(f, q, spans)

    def attr_said(a: str) -> str | None:
        return next((f for f in (a, *(squash(x) for x in LABEL_ALIASES.get(a, ()))) if seen(f)), None)

    if not attr:
        if any(a and h == head and attr_said(a) for a, h, _ in heads.values()):
            return None
        return next(((f, ()) for f in (head, stem) if seen(f)), None)
    said = attr_said(attr)
    if said is None:
        return None
    joined = next((said + f for f in (head, stem) if seen(said + f)), None)
    if joined:
        return joined, ()
    apart = next((f for f in (head, stem) if seen(f)), None)
    return (said + apart, (said, apart)) if apart else (said, ())


def _inside_longer(form: str, longer: list[str] | list[Match], q: str) -> bool:
    """Every place q says form is inside a place it says one of the longer forms ('수도권' in '비수도권'). Loop 6: a
    form said once on its own too ('수도권과 비수도권') is its own label. Loop 7: a longer label matched by its parts
    said apart (Match.parts: '귀농인' and '여자' of '귀농인-여자') covers the places of its parts."""
    covered = set()
    for o in longer:
        texts = (o,) if isinstance(o, str) else (o.form,) if o.form in q or not o.parts else o.parts
        for text in texts:
            at = q.find(text)
            while at >= 0:
                covered.update(range(at, at + len(text)))
                at = q.find(text, at + 1)
    at = q.find(form)
    while at >= 0:
        if not set(range(at, at + len(form))) <= covered:
            return False
        at = q.find(form, at + 1)
    return True


def _dedupe(found: list[Match], q: str | None = None) -> list[Match]:
    # One question word, several labels: '중국' matches '중국' and '중국(한국계)'; keep the label that is the word.
    by_form: dict[tuple[str, str], list[Match]] = {}
    for m in found:
        by_form.setdefault((m.axis, m.form), []).append(m)
    kept = []
    for group in by_form.values():
        whole = [m for m in group if squash(m.label) == m.form]
        kept += whole if whole and len(whole) < len(group) else group
    # A form inside a longer matched form is not its own label: '0세' in '20세', '합계' in '합계출산율'.
    def longer(m: Match) -> list[Match]:
        return [o for o in kept if m.form != o.form and m.form in o.form]

    if q is None:
        return [m for m in kept if not longer(m)]
    return [m for m in kept if not longer(m) or not _inside_longer(m.form, longer(m), q)]


# ---------------------------------------------------------------- the parsed question


def parse(index: TableIndex, question: str) -> Parsed:
    matches = match_labels(index, question)
    rest = squash(question)
    # loop 7: a label matched by its parts said apart leaves those parts out of rest
    forms = {f for m in matches for f in ((m.form,) if m.form in rest or not m.parts else m.parts)}
    for form in sorted(forms, key=len, reverse=True):
        rest = rest.replace(form, " ")
    words = tuple(
        _PARTICLE.sub("", w) for w in re.findall(r"[가-힣]{2,}", question) if not any(f in w for f in forms)
    )
    labels = tuple(squash(m.label) for m in matches)
    return Parsed(question=question, periods=asked_periods(question, rest, labels), matches=matches, rest=rest,
                  words=words)
