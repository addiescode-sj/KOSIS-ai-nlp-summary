"""What a question asks of one table: periods, the labels it names (items and categories) and operation words.

A light parser for the rules in answerability.py until T1-06 builds a StatQuery with the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from kosis_chat.config import PROJECT_ROOT
from kosis_chat.data.events import load_events
from kosis_chat.data.schema import Cell, TableInput
from kosis_chat.data.scope import LABEL_ALIASES

try:
    from kosis_ai import normalize
except ImportError:  # `python -m eval.run_eval` has no src/ on the path until T0-01 makes kosis_ai an installed package
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from kosis_ai import normalize

ITEM_AXIS = "항목"  # the item column, next to the category names
COUNT_KINDS = ("count_stock", "count_flow")

_COHORT = re.compile(r"(?:19|20)\d{2}\s*년생")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_MONTH = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*년\s*(\d{1,2})\s*월")
_RANGE = re.compile(r"((?:19|20)\d{2})\s*년?\s*(?:~|-|부터)\s*((?:19|20)\d{2})")
_SINCE = re.compile(r"((?:19|20)\d{2})\s*년?\s*이후")
_RECENT_N = re.compile(r"최근\s*(\d+)\s*년")
_RECENT = re.compile(r"최근|요즘|근래")
# Period expressions (v2, T1-25): a decade, a part of it with a length, a half or a quarter of a year (month
# ranges on a monthly table), last year against this year, and N years before a named year.
_DECADE = re.compile(r"((?:19|20)\d)0\s*년대(?:\s*(초반|전반|후반)\s*(\d)\s*년)?")
_HALF = re.compile(r"((?:19|20)\d{2})\s*년\s*(상|하)반기")
_QUARTER = re.compile(r"((?:19|20)\d{2})\s*년\s*([1-4])\s*분기")
_PREVIOUS_YEAR = re.compile(r"(?:작년|전년|지난\s*해)\s*(?:대비|보다|에\s*비해|과|와|하고)")
_YEARS_BEFORE = re.compile(r"(?<!\d)(\d{1,2})\s*년\s*전(?=과|보다|에|의|하고|\s|$)")  # not "2025년 전국"
# An event of the calendar (data/events.json) followed by a time word (T0-04 S5): "코로나 직전 해", "IMF 이후",
# "코로나19 유행 직전", "외환위기 2년 전", "세월호 참사 당시". A name with no time word after it is not a period, so a
# region or a label that shares a word with an event stays a label.
_EVENT_WHEN = (r"\s*(?:유행|발생|확산|사태|시작|출범)?\s*"
               r"(?:(?P<n>\d{1,2})\s*년\s*(?P<dir>이전|이후|전|후|뒤)|(?P<word>직전|직후|이후|부터|당시|때))")
TREND_WORDS = re.compile(r"추이|변화|변했|흐름|추세")
# Operations over periods (v2): an average of several periods, a sum over them ("모두" counts with a period range).
AVG_WORDS = re.compile(r"평균")
# "누적" is not here: it names items ("누적 혼인 비율"), it does not ask for a sum over periods.
TOTAL_WORDS = re.compile(r"모두|전부")
# "얼마나 늘었나", "감소했어": a change asked without 추이/변화. Verb forms only, so a label such as 자연증가건수 stays out.
CHANGE_WORDS = re.compile(r"늘었|줄었|늘어|줄어|증가(?:했|한|하|세)|감소(?:했|한|하|세)")
SUM_WORDS = re.compile(r"더하|더한|합하|합한|합치|합친|합산|총합")
ARITH_WORDS = re.compile(SUM_WORDS.pattern + r"|나누|나눈|몇\s*배|차이|빼면|뺀")
CAUSAL_WORDS = re.compile(r"왜|이유|원인|때문|효과|영향")
# A causal question that also asks for numbers ("추이와 원인") is answered; the LLM declines the cause part.
VALUE_ASKS = re.compile(r"추이|몇|얼마|보여")
VAGUE_WORDS = re.compile(r"통계|현황|자료|정보|수치")
OUT_OF_DOMAIN = re.compile(r"기업|기술개발|매출|수출|수입액|GDP|물가|주가|부동산|금리|환율")

_UNIT_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
_REGION_SUFFIX = re.compile(r"(특별자치시|특별자치도|특별시|광역시|도)$")
_REGION_AXIS = re.compile(r"행정구역|시도|시군구|지역")
# Categories that are alternative scenarios (장래추계 고위/중위/저위), not parts of a whole.
SCENARIO_AXIS = re.compile(r"가정|시나리오|추계")
TOTAL_LABELS = ("합계", "계", "총계", "전체")
_TOKEN = re.compile(r"[0-9A-Za-z가-힣~\-]+")
_PARTICLE = re.compile(r"(?<=[가-힣]{2})(?:은|는|이|가|을|를|의|에|도|만|와|과)$")


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_total(label: str) -> bool:
    core = _UNIT_PAREN.sub("", label).strip()
    return core in TOTAL_LABELS or core == "전국" or core.endswith(("_계", " 계"))


# ---------------------------------------------------------------- periods


def in_range(period: str, start: str, end: str) -> bool:
    """Whether a table period ('2024', '202406') falls in a year range ('2020', '2024') or a month range
    ('202401', '202406'); a yearly period never falls in a month range."""
    if len(start) == 6:
        return len(period) == 6 and start <= period <= end
    return start <= period[:4] <= end


@dataclass(frozen=True)
class Periods:
    """Periods named in the question: single years ('2024') or months ('202401'), year or month ranges, or "최근".

    spans: ranges that name one aggregate period (a half or a quarter of a year), in question order. On a monthly
    table a value asked for such a span is the sum of its months. previous: "작년 대비" (the latest period against
    the one before it). before: "10년 전", years before the named year, already turned into a single."""

    singles: tuple[str, ...] = ()
    ranges: tuple[tuple[str, str], ...] = ()
    recent_n: int | None = None
    recent: bool = False
    spans: tuple[tuple[str, str], ...] = ()
    previous: bool = False
    before: int | None = None
    events: tuple[tuple[str, str], ...] = ()  # (event phrase, event start) a period was read from

    @property
    def explicit(self) -> bool:
        return bool(self.singles or self.ranges)

    def contains(self, period: str) -> bool:
        if not self.explicit:
            return True
        if any(period == s or (len(s) == 4 and period[:4] == s) for s in self.singles):
            return True
        return any(in_range(period, a, b) for a, b in self.ranges)


def _decade_range(m: re.Match) -> tuple[str, str]:
    start = int(m.group(1)) * 10
    if m.group(2) and m.group(3):
        n = int(m.group(3))
        if m.group(2) == "후반":
            return str(start + 10 - n), str(start + 9)
        return str(start), str(start + n - 1)
    return str(start), str(start + 9)


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


def asked_periods(question: str) -> Periods:
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
    text = _MONTH.sub(" ", text)
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
    if before and len(singles) == 1 and len(singles[0]) == 4:  # "10년 전과 비교해서 2025년": the earlier year is implied
        singles.insert(0, str(int(singles[0]) - int(before.group(1))))
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


@dataclass
class Axis:
    """The item column or one category: code -> label, code -> parent code (KOSIS codes nest by prefix)."""

    name: str
    labels: dict[str, str] = field(default_factory=dict)
    parents: dict[str, str | None] = field(default_factory=dict)

    @property
    def regional(self) -> bool:
        return bool(_REGION_AXIS.search(self.name))

    @property
    def has_total(self) -> bool:
        return any(is_total(label) for label in self.labels.values())

    def depth(self, code: str) -> int:
        n, parent = 0, self.parents.get(code)
        while parent is not None:
            n, parent = n + 1, self.parents.get(parent)
        return n

    def ancestors(self, code: str) -> list[str]:
        out, parent = [], self.parents.get(code)
        while parent is not None:
            out.append(parent)
            parent = self.parents.get(parent)
        return out


def _link_parents(axis: Axis) -> None:
    codes = set(axis.labels)
    for code in axis.labels:
        axis.parents[code] = next((code[:k] for k in range(len(code) - 1, 0, -1) if code[:k] in codes), None)


@lru_cache(maxsize=4096)
def _measure(tbl_id: str, unit: str | None, item: str, dim_labels: tuple[str, ...]) -> tuple[str | None, str | None]:
    """(unit, measure_kind) by the T1-05 rules (src/kosis_ai/normalize.py)."""
    resolved, _ = normalize.resolve_unit(unit, item, list(dim_labels))
    return resolved, normalize.measure_kind(tbl_id, resolved, item, list(dim_labels))


def cell_measure(table: TableInput, cell: Cell) -> tuple[str | None, str | None]:
    given = cell.unit if cell.unit_source == "given" else None
    # Scenario labels name every assumption ('출산율-중위 / 기대수명-중위'), which the keyword rules would read
    # as a rate in 년 for every cell; the measure is in the other labels.
    labels = tuple(label for name, label in cell.dims.items() if not SCENARIO_AXIS.search(name))
    return _measure(table.meta.tbl_id, given, cell.item, labels)


@dataclass
class TableIndex:
    table: TableInput
    axes: dict[str, Axis]
    periods: list[str]
    # (axis, code) -> every (unit, measure_kind) of its cells; a measure label has exactly one.
    measures: dict[tuple[str, str], set[tuple[str | None, str | None]]]

    def axis_of(self, cell: Cell, name: str) -> str:
        return cell.item_code if name == ITEM_AXIS else cell.dim_codes.get(name, "")

    def year_ok(self, year: str) -> bool:
        return any(p.startswith(year) for p in self.periods)

    def period_ok(self, period: str) -> bool:
        return period in self.periods if len(period) > 4 else self.year_ok(period)

    def measure_of(self, axis: str, code: str) -> tuple[str | None, str | None] | None:
        found = self.measures.get((axis, code), set())
        return next(iter(found)) if len(found) == 1 else None


def build_index(table: TableInput) -> TableIndex:
    axes: dict[str, Axis] = {ITEM_AXIS: Axis(ITEM_AXIS)}
    measures: dict[tuple[str, str], set] = {}
    periods: set[str] = set()
    for c in table.cells:
        periods.add(c.period_raw)
        axes[ITEM_AXIS].labels.setdefault(c.item_code, c.item)
        found = cell_measure(table, c)
        measures.setdefault((ITEM_AXIS, c.item_code), set()).add(found)
        for name, code in c.dim_codes.items():
            axes.setdefault(name, Axis(name)).labels.setdefault(code, c.dims.get(name, ""))
            measures.setdefault((name, code), set()).add(found)
    for axis in axes.values():
        _link_parents(axis)
    return TableIndex(table=table, axes=axes, periods=sorted(periods), measures=measures)


# ---------------------------------------------------------------- label matching


@dataclass(frozen=True)
class Match:
    axis: str
    code: str
    label: str
    form: str  # the squashed text found in the question
    exact: bool  # the whole label (minus a unit suffix) is in the question


def _known_unit(text: str) -> bool:
    return text in normalize.KNOWN_UNITS or (text.endswith("당") and len(text) <= 12)


def label_forms(label: str, regional_total: bool = False) -> tuple[set[str], set[str], tuple[str, ...]]:
    """(exact forms, partial forms, tokens). '출생건수 (명)' -> exact {'출생건수'}; '기대수명(출생시 기대여명)' ->
    partial {'기대수명'}; '인구(명): 65세 이상' -> tokens ('인구', '65세', '이상')."""
    text = label.strip().lstrip("-").strip()
    m = _UNIT_PAREN.search(text)
    core = text[: m.start()].strip() if m and _known_unit(m.group(1).strip()) else text
    exact = {squash(core), *(squash(a) for a in LABEL_ALIASES.get(core, ()))}
    short = _REGION_SUFFIX.sub("", core)
    if short != core and len(short) >= 2:
        exact.add(squash(short))
    if regional_total and core in TOTAL_LABELS:
        exact.add("전국")
    bare = re.sub(r"\([^()]*\)", " ", core)
    head = squash(bare.split(" - ")[0])
    # '기대수명(출생시 기대여명)' is also asked as '기대수명'; not '인구(명): 0-14세', where '인구' is only a prefix.
    partial = {head} - exact if head and "(" in core and ":" not in core else set()
    tokens = tuple(t for t in _TOKEN.findall(bare) if len(t) >= 2 and t != "-")
    return exact, partial, tokens


def _found(form: str, question: str, squashed: str, lower_level: bool) -> bool:
    if len(form) < 2 or form not in squashed:
        return False
    if form.isdigit():  # cohort labels such as '1985': not a year of the period ("1985년" but "1985년생" is fine)
        return re.search(rf"(?<!\d){form}(?!\d)(?!\s*년(?!생))", question) is not None
    if lower_level and len(form) == 2:  # 2-character lower-level names hide in words ("더하면" has 하면)
        return re.search(rf"(?<![가-힣]){re.escape(form)}", question) is not None
    return True


def match_labels(index: TableIndex, question: str) -> list[Match]:
    q = squash(question)
    found: list[Match] = []
    for axis in index.axes.values():
        for code, label in axis.labels.items():
            lower = axis.parents.get(code) is not None
            exact, partial, tokens = label_forms(label, regional_total=axis.regional and axis.parents.get(code) is None)
            hit = next((f for f in sorted(exact, key=len, reverse=True) if _found(f, question, q, lower)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, True))
                continue
            hit = "".join(tokens) if len(tokens) >= 2 and all(t in q for t in tokens) else None
            hit = hit or next((f for f in partial if _found(f, question, q, lower)), None)
            if hit:
                found.append(Match(axis.name, code, label, hit, False))
    return _dedupe(found)


def _dedupe(found: list[Match]) -> list[Match]:
    # One question word, several labels: '중국' matches '중국' and '중국(한국계)'; keep the label that is the word.
    by_form: dict[tuple[str, str], list[Match]] = {}
    for m in found:
        by_form.setdefault((m.axis, m.form), []).append(m)
    kept = []
    for group in by_form.values():
        whole = [m for m in group if squash(m.label) == m.form]
        kept += whole if whole and len(whole) < len(group) else group
    # A form inside a longer matched form is not its own label: '0세' in '20세', '합계' in '합계출산율'.
    return [m for m in kept if not any(m.form != o.form and m.form in o.form for o in kept)]


# ---------------------------------------------------------------- the parsed question


@dataclass
class Parsed:
    question: str
    periods: Periods
    matches: list[Match]
    rest: str  # the question with matched label text removed, for operation words
    words: tuple[str, ...] = ()  # question words that are not matched labels, particles cut off

    def by_axis(self) -> dict[str, list[Match]]:
        out: dict[str, list[Match]] = {}
        for m in self.matches:
            out.setdefault(m.axis, []).append(m)
        return out


def parse(index: TableIndex, question: str) -> Parsed:
    matches = match_labels(index, question)
    rest = squash(question)
    for form in sorted({m.form for m in matches}, key=len, reverse=True):
        rest = rest.replace(form, " ")
    forms = {m.form for m in matches}
    words = tuple(
        _PARTICLE.sub("", w) for w in re.findall(r"[가-힣]{2,}", question) if not any(f in w for f in forms)
    )
    return Parsed(question=question, periods=asked_periods(question), matches=matches, rest=rest, words=words)
