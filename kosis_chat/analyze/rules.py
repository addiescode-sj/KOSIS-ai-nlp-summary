"""Question -> what to compute (T1-06), rules first: the judge's parser finds the periods and the labels the question
names; this module adds the question type, fills the categories the question leaves out (a total, or the only
choice), picks the cells and arranges them into a v3 plan (build.py). Whatever the rules cannot decide is listed in
Plan.unresolved for the LLM step.

T1-28 S3a: moved here from compute/analyze.py (a re-export shim now); analyze/node.py step_plan_rules calls
rule_plan and build_plan. The split into rules.py / place.py is left for later.
"""

from __future__ import annotations

import re
from itertools import product

from kosis_chat.analyze import blocks as build  # T1-28 S3a: was compute/build.py
from kosis_chat.analyze.blocks import Built, Pattern
from kosis_chat.analyze.roles import is_province, pick_of, reference_index
from kosis_chat.judge.rules import resolve
from kosis_chat.prepare.schemas import Cell, TableInput
from kosis_chat.prepare.store import DimKey, store_of
from kosis_chat.judge.question import (
    ARITH_WORDS,
    AVG_WORDS,
    CHANGE_WORDS,
    ITEM_AXIS,
    SUM_WORDS,
    TOTAL_WORDS,
    TREND_WORDS,
    Match,
    Parsed,
    Periods,
    TableIndex,
    in_range,
    is_total,
    parse,
    squash,
    total_stem,
)
from kosis_chat.judge.question import group_parts_of  # isort: skip  # loop 7

from kosis_ai.schemas import PlanV3, QuestionType, SelectStep, StatQuery  # isort: skip
from kosis_chat.analyze.schemas import Plan  # isort: skip  # re-export: moved in T1-28 S1b
from kosis_chat.shared import defaults as default_strings  # isort: skip
from kosis_chat.shared.patterns import CHANGE_ASKED, PER_UNIT, RATE_OF_CHANGE, RECENT_N, REGION_SUFFIX, asked_comparison, per_scale  # isort: skip
from kosis_chat.shared.patterns import MOST_RECENT  # isort: skip  # "가장 최근 ~" names a period, not a ranking
from kosis_chat.shared.patterns import ASKS_GAP, ASKS_PERIOD, ASKS_RATIO, CHANGE_DOWN, EXTREME_MAX, RANK_WORDS, rank_places  # isort: skip
from kosis_chat.shared.patterns import EACH_PERIOD, EXCLUDE, years_step  # isort: skip  # loop 4
from kosis_chat.shared.patterns import DEST_AFTER, DEST_AXIS, ORIGIN_AFTER, ORIGIN_AXIS, REVERSE  # isort: skip  # loop 5
from kosis_chat.shared.patterns import WITHIN_AFTER  # isort: skip  # loop 6
from kosis_chat.shared.patterns import EACH_TARGET, PERCENT, WHICH_MORE  # isort: skip  # loop 7
from kosis_chat.shared.patterns import asks_level  # isort: skip  # loop 8
from kosis_chat.shared.patterns import COUNT_STEM_SUFFIX, TOTAL_AMOUNT, asks_gap_of, asks_relative  # isort: skip  # loop 9
from kosis_chat.shared.patterns import candidate_nouns  # isort: skip  # loop 12
from kosis_chat.shared.patterns import ROLE_AXES, STAY_AXIS, VISIT_AXIS, role_kind  # isort: skip  # loop 12

PERIOD_AXIS = "시점"  # Plan.over when an extreme question ranks periods ("가장 낮았던 해")
RECENT_POINTS = 5
# loop 4: also the nouns 최댓값 / 최솟값 (최댓, 최솟), as the judge's RANKING reads them
EXTREME_WORDS = re.compile(r"가장|제일|최고|최저|최대|최소|최댓|최솟|1위|꼴찌")
SHARE_WORDS = re.compile(r"비중|차지|구성비")
# "서울이 전국의 몇 퍼센트": a share when a label the question names comes right before 의 (not "몇 퍼센트 늘었어",
# not "평균의 몇 %", where 평균 is no label).
_OF_PERCENT = re.compile(r"의몇(?:퍼센트|%|프로)")
# "%p" asks for a difference of percentages: of two codes a question word matched, the ones measured in %.
_PERCENT_POINT = re.compile(r"%\s*p|퍼센트\s*포인트|%\s*포인트")
COMPARE_WORDS = re.compile(r"비교|차이|몇\s*배|보다|더\s*(?:높|낮|많|적|크|작)")
# ASKS_PERIOD (loop 3: shared/patterns.py): a period word a describing word ends in, whatever particle follows.
# A year on a monthly table that also holds the year's value is that value, unless the question asks for the months.
BY_MONTH = re.compile(r"월별|매월|매달|달마다|월간")
# The judge's default period (default_applied["period"]) read as period codes: kosis_chat/shared/defaults.py since
# T1-28 S2b (the old names are the same objects).
_LATEST = default_strings.LATEST_CODES
_RECENT = default_strings.RECENT_CODES
_PREVIOUS = default_strings.PREVIOUS_CODES
# "3인 이하", "5인 이상", "65세 이상": a range over numbered labels of one category.
_NUM_RANGE = re.compile(r"(\d+)\s*(인|세)\s*(이하|이상|미만|초과)")
_NUM_LABEL = re.compile(r"^(\d+)\s*(인|세)\s*(이상)?(?:\s*\(.*\))?$")
_BOTH_SEXES = re.compile(r"남녀|남성과\s*여성|남자와\s*여자|성별")
# Two targets at two periods (S5): how their gap changed ("격차가 벌어졌어"), or which changed more ("증가율이 더 높아").
_GAP_WORDS = re.compile(r"격차|차이")
_GAP_CHANGE = re.compile(r"변했|변화|벌어|좁혀|줄어|늘어|커졌|작아|달라|줄었|늘었")
_GROWTH_WORDS = re.compile(r"증가율|감소율|증감률|증가\s*폭|감소\s*폭|증감\s*폭|더\s*(?:많이\s*)?(?:늘|줄)")
_DECLINE = re.compile(r"감소율|감소\s*폭|더\s*(?:많이\s*)?줄")
_GROWTH_RATE = RATE_OF_CHANGE  # loop 2: the shared words of a rate of change (shared/patterns.py), as multi_rules reads them
_AMOUNT = re.compile(r"증가\s*폭|감소\s*폭|증감\s*폭|더\s*(?:많이\s*)?(?:늘|줄)")


class Unresolved(Exception):
    def __init__(self, what: list[str]) -> None:
        super().__init__(", ".join(what))
        self.what = what


# ---------------------------------------------------------------- rules: question -> plan


def _asks_share(question: str, matches: list[Match]) -> bool:
    """A share word that is not part of a label the question names ('65세 이상 인구 구성비' names a cell that is
    already a share), or "<label>의 몇 퍼센트"."""
    words = {m.group(0) for m in SHARE_WORDS.finditer(question)}
    if any(not any(w in squash(m.label) for m in matches) for w in words):
        return True
    q = squash(question)
    return any(q[: m.start()].endswith(match.form) for m in _OF_PERCENT.finditer(q) for match in matches)


def _percent_of_other(question: str, index: TableIndex, codes: dict[str, list[str]], matches: list[Match]) -> bool:
    """Loop 8: a share read only from "<label>의 몇 퍼센트" (no share word asked) of two targets of one category
    neither of which holds the other (_holds: the total, an ancestor, the group of a "group + attribute" name): one
    target as a percent of the other, a comparison (build_plan reads the reference), not a share of the total."""
    words = {m.group(0) for m in SHARE_WORDS.finditer(question)}
    if any(not any(w in squash(m.label) for m in matches) for w in words):
        return False
    several = [n for n, v in codes.items() if len(v) >= 2]
    if len(several) != 1 or len(codes[several[0]]) != 2 or several[0] not in index.axes:
        return False
    name, (a, b) = several[0], codes[several[0]]
    return not _holds(index, name, a, b) and not _holds(index, name, b, a)


def _scoped_percent(question: str, index: TableIndex, codes: dict[str, list[str]], rest: str) -> bool:
    """Loop 11: a share asked without a share word: a percent asked as a level (shared asks_level: no difference,
    more / less or change), a code named right before a word of looking inside a group (_within: "정규직 중",
    "서울 가운데") and another code named that is not that group (the part). Not of cells already in % (a rate asked
    of a group: the cell's value), not when a code the question names is measured in %, and not "<label>의 몇
    퍼센트" (one target as a percent of another)."""
    if not asks_level(rest):
        return False
    q = squash(question)
    matches = parse(index, question).matches
    if any(q[: m.start()].endswith(match.form) for m in _OF_PERCENT.finditer(q) for match in matches):
        return False  # "여자는 남자의 몇 퍼센트": one target as a percent of another (loop 8), not a share
    scopes = {(n, c) for n in codes if n in index.axes for c in _within(question, index, n) if c in codes[n]}
    if not scopes:
        return False
    if any((index.measure_of(n, c) or (None,))[0] in ("%", "퍼센트") for n, named in codes.items() if n in index.axes
           for c in named):
        return False
    items = index.axes[ITEM_AXIS].labels if ITEM_AXIS in index.axes else {}
    if not codes.get(ITEM_AXIS) and len(items) == 1 and (index.measure_of(ITEM_AXIS, next(iter(items))) or (None,))[0] in ("%", "퍼센트"):
        return False
    return any((n, c) not in scopes for n, named in codes.items() if n in index.axes for c in named)


def _percent_only(question: str, index: TableIndex, codes: dict[str, list[str]]) -> None:
    if not _PERCENT_POINT.search(question):
        return
    for name, named in codes.items():
        percent = [c for c in named if (index.measure_of(name, c) or (None,))[0] == "%"]
        if 0 < len(percent) < len(named):
            codes[name] = percent


def _question_type(question: str, periods: Periods, codes: dict[str, list[str]], matches: list[Match],
                   rest: str | None = None) -> QuestionType:
    # loop 3: a place other than the first ("세 번째로 많은", "2위", "상위 3개") ranks too. Loop 4: the ranking words are
    # read on the question without the labels it names (rest), so an ordinal in a label ('첫째아', '1순위') ranks nothing.
    ranking = question if rest is None else rest
    if EXTREME_WORDS.search(MOST_RECENT.sub(" ", ranking)) or RANK_WORDS.search(ranking):
        return "extreme"
    if _asks_share(question, matches):
        return "share"
    # loop 9: the words of a trend, a change and a comparison are read on rest too, as the ranking words are (loop 4):
    # a label that holds one ('출생성비 차이', '인구변화율') names a measure and asks for no comparison of two targets
    if periods.ranges or len(periods.singles) >= 2 or (periods.recent and TREND_WORDS.search(ranking)):
        return "trend"
    if TREND_WORDS.search(ranking) or CHANGE_WORDS.search(ranking):
        return "trend"
    if COMPARE_WORDS.search(ranking) or any(len(v) >= 2 for v in codes.values()):
        return "compare"
    return "value"


def _number_range(question: str, index: TableIndex) -> tuple[str, list[str]] | None:
    """'3인 이하' -> (axis, [1인, 2인, 3인] codes); '5인 이상' -> 5인 + the smallest 'K인 이상' above (no double count);
    '65세 이상' on a category that holds '65세이상' itself -> that code alone (loop 8)."""
    m = _NUM_RANGE.search(question)
    if not m:
        return None
    n, unit, how = int(m.group(1)), m.group(2), m.group(3)
    for name, axis in index.axes.items():
        numbered = {}
        for code, label in axis.labels.items():
            if (lm := _NUM_LABEL.match(label.strip())) and lm.group(2) == unit:
                numbered[code] = (int(lm.group(1)), bool(lm.group(3)))
        if len(numbered) < 2:
            continue
        exact = {code: k for code, (k, open_) in numbered.items() if not open_}
        opened = {code: k for code, (k, open_) in numbered.items() if open_}
        if how in ("이하", "미만"):
            top = n if how == "이하" else n - 1
            return name, sorted((c for c, k in exact.items() if k <= top), key=exact.get)
        low = n if how == "이상" else n + 1
        # loop 8: an open interval that starts at the asked bound ('65세이상' for "65세 이상") is the whole range
        if same := sorted(c for c, k in opened.items() if k == low):
            return name, same[:1]
        tail = min(((k, c) for c, k in opened.items() if k > low), default=None)
        end = tail[0] if tail else max(exact.values()) + 1
        picked = sorted((c for c, k in exact.items() if low <= k < end), key=exact.get)
        return name, picked + ([tail[1]] if tail else [])
    return None


def _both_sexes(question: str, index: TableIndex, codes: dict[str, list[str]]) -> None:
    if not _BOTH_SEXES.search(question):
        return
    for name, axis in index.axes.items():
        pair = [c for c, label in axis.labels.items() if label.strip() in ("남자", "여자")]
        if len(pair) == 2 and len(codes.get(name, [])) < 2:
            codes[name] = sorted(pair, key=lambda c: axis.labels[c] != "남자")


def _drop_children(index: TableIndex, codes: dict[str, list[str]]) -> None:
    """'기대수명' matches '기대수명(...)' and '기대수명(...) - 남' / '- 여': keep the label the others extend."""
    for name, named in codes.items():
        labels = {c: index.axes[name].labels[c].strip() for c in named}
        roots = [c for c in named if all(labels[o].startswith(labels[c]) for o in named)]
        if len(named) > 1 and len(roots) == 1:
            codes[name] = roots


def _position(question: str, index: TableIndex, name: str, code: str) -> int:
    """Where the squashed question first names a code's label (its length when it does not)."""
    q = squash(question)
    label = index.axes[name].labels[code]
    forms = [squash(label), squash(re.sub(r"\s*\(.*\)\s*$", "", label)), squash(REGION_SUFFIX.sub("", label))]
    found = [q.find(f) for f in forms if f and q.find(f) >= 0]
    return min(found) if found else len(q)


def _order_in_question(question: str, index: TableIndex, name: str, named: list[str]) -> list[str]:
    return sorted(named, key=lambda code: _position(question, index, name, code))


def _per_unit(question: str, index: TableIndex, name: str, named: list[str]) -> tuple[int, float] | None:
    """(the denominator's index in named, the scale) when the question asks one of two targets per units of the
    other (loop 2, the words multi_rules reads too: shared/patterns.py PER_UNIT, per_scale): "사망자 1천 명당 출생아
    수", "인구 1만 명당 의사 수". The denominator is the target named last before 당; a 당 inside a label the question
    names ('인구 천명당 사망률') is no such word."""
    m = PER_UNIT.search(question)
    if m is None or len(named) != 2:
        return None
    q = squash(question)
    at = len(squash(question[: m.end()])) - 1
    named_at: dict[str, int] = {}  # where the question names each target, by the form it was matched by
    for match in parse(index, question).matches:
        start = q.find(match.form)
        if match.axis == name and start >= 0:
            named_at[match.code] = min(named_at.get(match.code, start), start)
        while start >= 0:
            if start <= at < start + len(match.form):
                return None
            start = q.find(match.form, start + 1)
    before = [(p, n) for n, code in enumerate(named)
              if (p := named_at.get(code, _position(question, index, name, code))) < at]
    if not before:
        return None
    return max(before)[1], per_scale(m)


def _range_periods(index: TableIndex, start: str, end: str, months: bool = False) -> list[str]:
    """Table periods inside an asked range. A range of years is read year by year as _single_periods reads one
    year (loop 2): a year the table holds a value of is that value, not that value and its months as well; a year
    with months only is its months. Months asked (BY_MONTH) or a range of months: every period inside."""
    if end == "9999":
        end = max(x[:4] for x in index.periods)
    inside = [x for x in index.periods if in_range(x, start, end)]
    if months or len(start) != 4 or len(end) != 4:
        return inside
    yearly = {x for x in inside if len(x) == 4}
    return [x for x in inside if x in yearly or x[:4] not in yearly]


def _single_periods(index: TableIndex, single: str, months: bool = False) -> list[str]:
    """Table periods of one asked period: a year is its value when the table has one ('2024' on a monthly table
    with yearly totals), else (or when months is set) all its months."""
    if single in index.periods and not months:
        return [single]
    return [x for x in index.periods if x == single or (len(single) == 4 and x[:4] == single)]


def _every(periods: list[str], start: str, step: int | None) -> list[str]:
    """The periods of a range at an interval of years from its first year (loop 4: "2010~2020년 5년마다")."""
    if not step or len(start) != 4:
        return periods
    return [x for x in periods if (int(x[:4]) - int(start)) % step == 0]


def _periods(p: Periods, index: TableIndex, default: dict[str, str], months: bool = False,
             step: int | None = None) -> tuple[list[str], list[str]]:
    """(table periods asked, asked years the table lacks)."""
    years = sorted({x[:4] for x in index.periods})
    if p.ranges:
        found = [x for a, b in p.ranges for x in _every(_range_periods(index, a, b, months), a, step)]
        if p.singles:  # "2025년 ... 2020~2024년 평균보다": the single period belongs to the question too
            found += [x for s in p.singles for x in _single_periods(index, s, months)]
        # Only the asked ends: a census table has no 2016~2019 by design, which is not worth a note.
        ends = [y for a, b in p.ranges for y in (a, b) if b != "9999" and len(a) == 4]
        return sorted(dict.fromkeys(found)), [y for y in dict.fromkeys(ends) if y not in years]
    if p.singles:
        found = [x for s in p.singles for x in _single_periods(index, s, months)]
        return list(dict.fromkeys(found)), [s for s in p.singles if not any(x[:4] == s[:4] for x in index.periods)]
    parsed = default_strings.parse_period_codes(default.get(default_strings.PERIOD_KEY, ""))
    if parsed is None:
        return [], []
    kind, groups = parsed
    if kind == "previous":
        return [groups[1], groups[0]], []
    if kind == "recent":
        return [x for x in index.periods if groups[1] <= x <= groups[2]], []
    return [groups[0]], []


def _period_groups(p: Periods, index: TableIndex, kind: QuestionType, months: bool = False) -> list[list[str]]:
    """The period groups an op works on, in question order: one per range (a decade, a half, a quarter), and
    for a share the two single periods compared."""
    groups = [_range_periods(index, a, b, months) for a, b in p.ranges]
    if kind == "share" and not groups and len(p.singles) >= 2:
        groups = [_single_periods(index, s, months) for s in p.singles[:2]]
    return [g for g in groups if g]


def _operation(question: str, rest: str, p: Periods, index: TableIndex, kind: QuestionType,
               months: bool = False, targets: bool = False) -> tuple[Pattern | None, list[list[str]]]:
    """(pattern, period groups) when the question aggregates periods; (None, []) for the five plain types.

    rest is the question with matched labels removed, so "평균" in a label ('평균 가구원수') does not count.
    targets (loop 9): the question names two targets or more of one category to add (_adds_targets)."""
    groups = _period_groups(p, index, kind, months)
    average = bool(AVG_WORDS.search(rest))
    # "모두" / "전부" sums a range, but not a range asked period by period ("연도별로 모두 알려줘", loop 4); loop 9: and
    # the targets named at one period when it comes right before 몇 + a counting word ("서울과 부산은 모두 몇 명")
    total = bool(SUM_WORDS.search(rest) or (TOTAL_WORDS.search(rest) and p.ranges and not EACH_PERIOD.search(rest))
                 or (targets and not p.ranges and len(p.singles) <= 1 and TOTAL_AMOUNT.search(rest)))
    if kind == "share":
        if _part_of_year([x for g in groups for x in g]):  # "2025년 중 12월 비중": one share, not two compared
            return None, []
        return ("shr", groups) if len(groups) == 2 and not average and not total else (None, [])
    if average and kind != "extreme" and _GROWTH_RATE.search(rest):
        return "cagr", []  # loop 3: the average is of the rate ("연평균 증가율", "해마다 평균 몇 퍼센트씩"), not of the values
    # loop 4: the same for an amount ("한 해 평균 몇 명씩 줄었어", "연평균 얼마나 늘었어"): the average is of the change
    # between the two ends (one range, two years, or the last n years), not of the values
    ends = (len(groups) == 1 and not p.singles) or (not p.ranges and len(p.singles) == 2) or (not groups and p.recent_n)
    if average and kind != "extreme" and CHANGE_ASKED.search(rest) and ends:
        return "apc", []
    if average and groups:
        if len(groups) >= 2:
            return "pavg", groups[:2]
        single = [x for s in p.singles for x in _single_periods(index, s, months)]
        if single and single[0] not in groups[0]:
            return "vsa", [groups[0], single[:1]]
        return "avg", groups
    if average and p.recent_n:  # "최근 5년 평균": the last n numeric periods are chosen when the request is built
        return "avg", []
    if p.spans and not average:  # "2024년 상반기" on a monthly table: the months are summed
        if len(groups) >= 2 and not total:
            return "psum", groups[:2]
        if len(groups) == 1 and not (TREND_WORDS.search(rest) or CHANGE_WORDS.search(rest)):
            return "sum", groups
    if total and groups:
        return ("psum", groups[:2]) if len(groups) >= 2 else ("sum", groups)
    if total and kind != "extreme" and not p.ranges and len(p.singles) >= 2:  # "2021년과 2023년 출생아 수를 합치면": periods named one by one
        named = [x for s in p.singles for x in _single_periods(index, s, months)]
        return ("sum", [named]) if len(named) >= 2 else (None, [])
    if total and kind != "extreme":  # "서울과 부산 인구를 합치면", "출생아와 사망자를 더하면": the named targets at one period (build_plan)
        return "sum", []
    return None, []


def _excluded(question: str, index: TableIndex, codes: dict[str, list[str]], periods: list[str]) -> tuple[str, str, str] | None:
    """(axis, the category's total, the part left out) when the question asks for the rest of a category but one
    (loop 4, shared EXCLUDE): one code named of a category that has a total, before the word that leaves it out (and
    after the total when the total is named too: "전국에서 서울을 제외한", "서울 말고 나머지 지역"; not "전국 말고 서울"),
    one code of every other category, one period."""
    m = EXCLUDE.search(question)
    if m is None or len(periods) > 1:
        return None
    at = len(squash(question[: m.start()]))
    found = None
    for name, named in codes.items():
        total = _total(index, name)
        parts = [c for c in named if c != total]
        if not parts or total is None:
            continue
        if len(parts) != 1 or len(named) > 2 or found is not None:
            return None
        part = parts[0]
        where = _position(question, index, name, part)
        if where >= at or (total in named and _position(question, index, name, total) > where):
            return None
        found = (name, total, part)
    return found


def rule_plan(question: str, index: TableIndex, default_applied: dict[str, str] | None = None) -> Plan:
    parsed = parse(index, question)
    resolved = resolve(index, parsed)
    codes = {name: _order_in_question(question, index, name, sorted(cs)) for name, cs in resolved.codes.items()}
    ranged = None
    if found := _number_range(question, index):
        codes[found[0]] = found[1]
        ranged = found[0]
    _both_sexes(question, index, codes)
    _drop_children(index, codes)
    _percent_only(question, index, codes)
    default = dict(default_applied or {})
    kind = _question_type(question, parsed.periods, codes, parsed.matches, parsed.rest)
    if kind in ("value", "compare") and _scoped_percent(question, index, codes, parsed.rest):
        kind = "share"  # loop 11: "정규직 중 여자는 몇 퍼센트야": a part of the group the question looks inside
    if kind not in ("share", "extreme") and _drop_scopes(question, index, codes):
        kind = _question_type(question, parsed.periods, codes, parsed.matches, parsed.rest)
    if kind == "share" and _percent_of_other(question, index, codes, parsed.matches):
        kind = "compare"  # loop 8: "부산은 서울의 몇 퍼센트": one target over another, not a part over a whole
    pairs: list[dict[str, str]] = []
    pair_reference = None
    if found := _flows(question, index, parsed, kind):  # loop 5: names in a start and an end category
        for side, named in found[0].items():
            if named:
                codes[side] = named
            else:
                codes.pop(side, None)
        pairs, pair_reference = found[1], found[2]
        kind = _question_type(question, parsed.periods, codes, parsed.matches, parsed.rest)
        if pairs and kind not in ("share", "extreme"):
            kind = "compare"  # two flows: compared (blocks.compare reads the difference or the ratio asked)
    p = parsed.periods
    # "2024년 혼인 추이" on a monthly table: the months of the one year asked
    months = bool(BY_MONTH.search(question)) or (kind == "trend" and len(p.singles) == 1 and not p.ranges)
    periods, missing = _periods(p, index, default, months, years_step(question))
    plan = Plan(question, kind, codes, periods, missing_years=missing, default_applied=default,
                events=list(parsed.periods.events), pairs=pairs, pair_reference=pair_reference)
    plan.pattern, plan.groups = _operation(question, parsed.rest, parsed.periods, index, kind, months,
                                           _adds_targets(parsed, codes, ranged))
    if (kind in ("value", "compare") and (plan.pattern is None or (plan.pattern == "sum" and not plan.groups))
            and _excluded(question, index, codes, periods)):
        plan.pattern, plan.groups, plan.type = "excl", [], "compare"  # loop 4: the total minus the part left out
        return plan
    if plan.pattern == "sum" and parsed.periods.spans and not parsed.periods.singles:
        plan.type = "value"  # "2024년 상반기 혼인 건수": one number for one named period
    elif plan.pattern and plan.pattern != "shr" and kind not in ("value", "trend"):
        plan.type = "trend"  # an average or sum over periods is a trend question, whatever its wording suggested
    if plan.pattern is None and kind != "extreme" and (named := _named_average(parsed, index, plan)):
        plan.pattern, plan.groups, plan.type = named
        return plan
    # loop 11: a ranking among targets the question names one by one ranks them (their values or their changes),
    # not each target's change (_two_by_two)
    named_pool = _named_pool(parsed, index, plan.codes) if kind == "extreme" else None
    if plan.pattern is None and named_pool is None and (pair := _two_by_two(question, plan, parsed.rest, parsed)):
        plan.pattern, plan.groups, plan.type = pair
        return plan
    if plan.pattern is None and kind == "trend" and (ratio := _period_ratio(question, parsed.rest, plan)):
        plan.pattern, plan.groups, plan.type = "pratio", ratio, "compare"
        return plan
    if plan.pattern is None and kind in ("trend", "compare") and _values_only(parsed, kind, codes):
        plan.type = "value"  # "2021년과 2023년 서울과 부산 인구 알려줘": the cells of every named target and period
    if kind == "extreme":
        plan.pick = pick_of(question)
        if _asks_period(question, index, plan.codes):
            plan.over = PERIOD_AXIS
        elif named_pool is not None:
            plan.over, plan.candidates = named_pool
        else:
            plan.over = _scope(question, index, codes) or _ranked_axis(index, codes, question)
        if plan.over is None and len(plan.periods) >= 2:  # loop 4: nothing open but the asked periods: they are ranked
            plan.over = PERIOD_AXIS
        if plan.over is None:
            plan.unresolved.append("over")
    return plan


def _average_in_names(index: TableIndex) -> bool:
    """Loop 8: the table's title or an item name says 평균 ('평균 가구원수'): a 평균 in the question may name the
    measure, not ask for an average (loop 7 brief G), so the named-average plans below do not read it."""
    title = getattr(index.table.meta, "tbl_nm", "") or ""
    items = index.axes[ITEM_AXIS].labels.values() if ITEM_AXIS in index.axes else ()
    return any(AVG_WORDS.search(text) for text in (title, *items))


def _named_average(parsed: Parsed, index: TableIndex, plan: Plan) -> tuple[Pattern, list[list[str]], QuestionType] | None:
    """Loop 8: the average of what the question names one by one, as their sum is read (_operation's sum of periods
    named one by one, _targets_sum): a word of an average (on rest), no range, no recent periods, nothing compared or
    changed, and either several periods named one by one of one series ("2015년, 2018년, 2021년 평균": avg over
    them) or, at one period, several targets of one category each named by its own name ("서울, 부산, 대구 평균":
    tavg)."""
    p, rest = parsed.periods, parsed.rest
    if not AVG_WORDS.search(rest) or _average_in_names(index):
        return None
    if p.ranges or p.recent or p.recent_n or p.previous or p.before or p.events:
        return None
    diff, ratio = asked_comparison(rest)
    if diff or ratio or CHANGE_ASKED.search(rest) or TREND_WORDS.search(rest):
        return None
    several = [n for n, named in plan.codes.items() if len(named) >= 2]
    # loop 11: a value per units of another ("종사자 수를 사업체 1개당 평균으로"): the per-unit word already says
    # what is computed of the two targets, and the average word describes that value (as a label's 평균 does above)
    if len(several) == 1 and _per_unit(parsed.question, index, several[0], plan.codes[several[0]]) is not None:
        return None
    if len(p.singles) >= 2 and not several and len(plan.periods) >= 2:
        return "avg", [list(plan.periods)], "trend"
    if len(plan.periods) <= 1 and len(several) == 1 and _named_one_by_one(parsed, several[0], plan.codes[several[0]]):
        return "tavg", [], "value"
    return None


def _values_only(parsed: Parsed, kind: QuestionType, codes: dict[str, list[str]]) -> bool:
    """No word that asks for a change, a trend, a comparison or a formula, and the question names several periods
    one by one (a trend by count: two or more, no range, no relative period) or several codes in two categories or
    more (a compare by count: two regions and two items) (loop 2): the question lists values ("2020년, 2023년 각각
    얼마야?", "서울과 부산의 혼인건수와 조혼인율"). The type was counted (_question_type), which put a change in the
    answer nobody asked for, or failed for the second category. Read on the question without its labels."""
    p, rest = parsed.periods, parsed.rest
    if kind == "compare":
        # one pair: the compare plan lists it when no formula is asked (blocks.compare); loop 5: three targets or more
        # of one category are listed when the question names each of them (not a range, not an over-wide match)
        several = [n for n, v in codes.items() if len(v) >= 2]
        if len(several) < 2 and not (several and len(codes[several[0]]) >= 3
                                     and _named_one_by_one(parsed, several[0], codes[several[0]])):
            return False
    elif p.recent or p.recent_n or p.previous or p.before or p.events:
        return False
    elif p.ranges:  # loop 4: a range asked period by period (shared EACH_PERIOD, as the multi-table rules read it)
        if not EACH_PERIOD.search(rest):
            return False
    elif len(p.singles) < 2:
        return False
    diff, ratio = asked_comparison(rest)
    asks = (diff or ratio or TREND_WORDS.search(rest) or CHANGE_ASKED.search(rest) or ARITH_WORDS.search(rest)
            or AVG_WORDS.search(rest) or "얼마나" in rest)
    return not asks


_ONLY_ONE_SIDE = 2  # loop 12: names one category of a pair may hold that the other does not ('기타', '국외')


def _twin_axes(index: TableIndex) -> tuple[str, str] | None:
    """(the start category, the end category) of a table with two categories that hold the same names (three or
    more) in two roles, the roles read on the categories' names (shared ORIGIN_AXIS, DEST_AXIS).

    Loop 12: the same names but one or two that only one of them holds (a place outside the others: '기타 지역',
    '국외'). The roles are read in the pair (shared STAY_AXIS, VISIT_AXIS): a start that is left and an end that is
    arrived at or stayed at (전출지 × 전입지, 출신지 × 거주지), or a place stayed at and a place gone to and from each
    day (상주지 × 통근통학지: the place stayed at is the start)."""
    names = [n for n, a in index.axes.items() if n != ITEM_AXIS and len(a.labels) >= 3]
    for a, b in ((a, b) for i, a in enumerate(names) for b in names[i + 1:]):
        labels_a = {squash(x) for x in index.axes[a].labels.values()}
        labels_b = {squash(x) for x in index.axes[b].labels.values()}
        if len(labels_a & labels_b) < 3 or len(labels_a ^ labels_b) > _ONLY_ONE_SIDE:
            continue
        if ORIGIN_AXIS.search(a) and DEST_AXIS.search(b) and not DEST_AXIS.search(a):
            return a, b
        if ORIGIN_AXIS.search(b) and DEST_AXIS.search(a) and not DEST_AXIS.search(b):
            return b, a
        for x, y in ((a, b), (b, a)):
            if STAY_AXIS.search(x) and VISIT_AXIS.search(y) and not VISIT_AXIS.search(x) and not STAY_AXIS.search(y):
                return x, y
    return None


def _role_of(after: str, twins: tuple[str, str]) -> str | None:
    """Loop 12: the category of the pair a name plays (it is followed by `after`, squashed): the one whose name is of
    the kind the verb after the name's particle says (shared role_kind, ROLE_AXES: "서울에 사는" is the place stayed
    at, "서울로 통근하는" the place gone to); with no such verb, or none of the pair of that kind, the particle's role
    as before (ORIGIN_AFTER: the start, DEST_AFTER: the end)."""
    start, end = twins
    if (kind := role_kind(after)) is not None:
        of_kind = [n for n in twins if ROLE_AXES[kind].search(n)]
        if kind == "arrive":  # a place stayed at is arrived at too (DEST_AXIS holds 거주), unless the pair says apart
            of_kind = [n for n in of_kind if not STAY_AXIS.search(n)] or of_kind
        if len(of_kind) == 1:
            return of_kind[0]
    return start if ORIGIN_AFTER.match(after) else end if DEST_AFTER.match(after) else None


def _flows(question: str, index: TableIndex, parsed: Parsed, kind: QuestionType
           ) -> tuple[dict[str, list[str]], list[dict[str, str]], int | None] | None:
    """Loop 5. On a table with a start and an end category of the same names (_twin_axes), the names the question
    gives a role (a particle or word right after the name: shared ORIGIN_AFTER, DEST_AFTER) put in their category:
    ({start: [code], end: [code]}, the flows) for every name read so. A side no name was given is the category's total
    ("서울로 전입한": from everywhere). A role named twice starts a second flow ("A에서 B로 ... B에서 A로"), and a word
    that turns the flow around (REVERSE) names the second flow as the first one swapped. A share keeps the flow and
    the same flow with the side named last set to its total (the whole). None when the table has no such
    categories, or a name of them has no role in the question (the codes are left as they were)."""
    twins = _twin_axes(index)
    if twins is None:
        return None
    start, end = twins
    totals = {start: _total(index, start), end: _total(index, end)}
    codes_of: dict[str, dict[str, str]] = {}  # form -> category -> code
    for m in parsed.matches:
        if m.axis in twins:
            codes_of.setdefault(m.form, {})[m.axis] = m.code
    if not codes_of:
        return None
    q = squash(question)
    found: list[tuple[int, int, str]] = []  # (at, end, form), longer forms first where two overlap
    for form in sorted(codes_of, key=len, reverse=True):
        at = q.find(form)
        while at >= 0:
            if not any(a < at + len(form) and at < b for a, b, _ in found):
                found.append((at, at + len(form), form))
            at = q.find(form, at + 1)
    found.sort()
    flows: list[dict[str, str]] = []
    where: list[int] = []
    current: dict[str, str] = {}
    for at, stop, form in found:
        role = _role_of(q[stop:], twins)  # loop 12: the verb's kind first, then the particle
        if role is None or role not in codes_of[form]:
            return None
        if role in current:
            flows.append(current)
            current = {}
        if not current:
            where.append(at)
        current[role] = codes_of[form][role]
    flows.append(current)
    if any(totals[side] is None for flow in flows for side in twins if side not in flow):
        return None
    # the sides the first flow names; a side it does not name is left open (the total, or the axis a ranking ranks)
    codes = {side: [flows[0][side]] if side in flows[0] else [] for side in twins}
    flows = [{side: flow.get(side, totals[side]) for side in twins} for flow in flows]
    if len(flows) == 1 and (rev := REVERSE.search(q)) and rev.start() > where[0]:
        flows.append({start: flows[0][end], end: flows[0][start]})
        where.append(rev.start())
    if len(flows) > 2:
        return None
    if kind == "share" and len(flows) == 1:
        last = max(found)[2]
        part = next(side for side in twins if codes_of[last].get(side) == flows[0][side])
        flows.append({**flows[0], part: totals[part]})
        return codes, flows, None
    return codes, (flows if len(flows) == 2 else []), _pair_reference(q, where) if len(flows) == 2 else None


def _adds_targets(parsed: Parsed, codes: dict[str, list[str]], ranged: str | None) -> bool:
    """Loop 9: the question names two targets or more of one category that a word of the whole may add: each named by
    its own name (_named_one_by_one), or picked by a range of numbered labels ('3인 이하'). Not an over-wide match of
    one name (one target asked: its value, not a sum)."""
    several = [n for n, named in codes.items() if len(named) >= 2]
    if len(several) != 1:
        return False
    return several[0] == ranged or _named_one_by_one(parsed, several[0], codes[several[0]])


def _named_one_by_one(parsed: Parsed, name: str, named: list[str]) -> bool:
    """Every code is matched by a name of its own in the question (loop 5): each code has an exact match, and no
    question word matched two of them (a range such as '3인 이하' or a word matching several labels is not that)."""
    forms: dict[str, set[str]] = {}
    for m in parsed.matches:
        if m.axis == name and m.code in named and m.exact:
            forms.setdefault(m.code, set()).add(m.form)
    if set(forms) != set(named):
        return False
    used = [f for fs in forms.values() for f in fs]
    return len(used) == len(set(used))


_HOW_MANY_OF = re.compile(r"^몇\s*([가-힣]+)")


def _asks_period(question: str, index: TableIndex, codes: dict[str, list[str]]) -> bool:
    """A word that asks for a period (shared ASKS_PERIOD). Loop 11: "몇 + a counting word" is the unit of the value
    asked, not a period, when the cells the question picks are measured in that word (the table's unit of a named
    code, or of the table's only item: '평균 근속기간 (년)' asked "몇 년"); any other period word still asks one."""
    units: set[str] = set()
    for name, named in codes.items():
        units |= {squash(m[0]) for c in named if name in index.axes and (m := index.measure_of(name, c)) and m[0]}
    items = index.axes[ITEM_AXIS].labels if ITEM_AXIS in index.axes else {}
    if not codes.get(ITEM_AXIS) and len(items) == 1 and (m := index.measure_of(ITEM_AXIS, next(iter(items)))) and m[0]:
        units.add(squash(m[0]))
    for found in ASKS_PERIOD.finditer(question):
        word = _HOW_MANY_OF.match(found.group(0))
        if word is None or word.group(1) not in units:
            return True
    return False


def _named_pool(parsed: Parsed, index: TableIndex, codes: dict[str, list[str]]) -> tuple[str, list[str]] | None:
    """Loop 11: (category, candidates) when a ranking names its candidates one by one: the one category with two codes
    or more, each matched by a name of its own (_named_one_by_one, so not a range or an over-wide match), every other
    category one code at most. A named group that holds the others ("서울 종로구, 중구, 강남구 중") is where they are,
    not a candidate. Items may be the candidates too."""
    several = [n for n, named in codes.items() if len(named) >= 2]
    if len(several) != 1 or several[0] not in index.axes:
        return None
    name = several[0]
    if not _named_one_by_one(parsed, name, codes[name]):
        return None
    named = codes[name]
    pool = [c for c in named if not any(o != c and _holds(index, name, c, o) for o in named)]
    return (name, pool) if len(pool) >= 2 else None


def _pool_of(plan: Plan, index: TableIndex, name: str) -> list[str]:
    """The candidates of a ranking over a category: the ones the question names one by one (loop 11), else _pool."""
    return list(plan.candidates) if plan.candidates and plan.over == name else _pool(index, name, plan.codes.get(name))


def _listing(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """The cells of every combination of the named targets at every asked period, one select per cell (two items
    of different units are listed, not refused), in question order."""
    several = [n for n, named in plan.codes.items() if len(named) >= 2]
    problems: list[str] = []
    fixed = _fixed(plan, index, set(several), problems)
    if problems:
        raise Unresolved(problems)
    combos = [{**fixed, **dict(zip(several, combo))} for combo in product(*(plan.codes[n] for n in several))]
    refs = _refs(lookup, [(f, p) for f in combos for p in plan.periods], problems)
    if problems:
        raise Unresolved(problems)
    steps = [SelectStep(id=f"c{n}", cells=[r]) for n, r in enumerate(refs)]
    return build.written(PlanV3(steps=steps, show=[s.id for s in steps], present="value", note=note))


def _two_by_two(question: str, plan: Plan, rest: str | None = None,
                parsed: Parsed | None = None) -> tuple[Pattern, list[list[str]], QuestionType] | None:
    """(gap, grw or each, the two periods, "compare") when the question names two targets of one category and two
    years (or a range: its ends). A monthly table's yearly value stands for the year; without one the rules do not
    guess. Loop 7: a change asked the way one target's change is asked (shared CHANGE_ASKED: a verb, a rate of change,
    a percent point; read on rest, the question without its labels) is each target's change (each) unless the
    question asks which of them is more (shared WHICH_MORE); so is a rate named by its noun with a word of each
    (shared EACH_TARGET: "각각", "저마다"). Loop 8: three targets or more of one category, each named by its own name
    (_named_one_by_one), are each target's change too; the gap and which-changed-more plans stay for two."""
    rest = question if rest is None else rest
    sides = [n for n, named in plan.codes.items() if len(named) >= 2]
    years = sorted({p[:4] for p in plan.periods})
    if len(sides) != 1 or len(years) < 2:
        return None
    if len(plan.codes[sides[0]]) > 2:
        named = parsed is not None and _named_one_by_one(parsed, sides[0], plan.codes[sides[0]])
        asked = _GROWTH_WORDS.search(question) or CHANGE_ASKED.search(rest) or _PERCENT_POINT.search(rest)
        if not named or not asked or WHICH_MORE.search(rest) or years[0] not in plan.periods or years[-1] not in plan.periods:
            return None
        return "each", [[years[0]], [years[-1]]], "compare"
    if _GROWTH_WORDS.search(question):
        pattern: Pattern = "each" if EACH_TARGET.search(rest) and not WHICH_MORE.search(rest) else "grw"
    elif _GAP_WORDS.search(question) and _GAP_CHANGE.search(question):
        pattern = "gap"
    elif CHANGE_ASKED.search(rest) or _PERCENT_POINT.search(rest):
        pattern = "grw" if WHICH_MORE.search(rest) else "each"
    else:
        return None
    if years[0] not in plan.periods or years[-1] not in plan.periods:
        return None
    return pattern, [[years[0]], [years[-1]]], "compare"


def _period_ratio(question: str, rest: str, plan: Plan) -> list[list[str]] | None:
    """Loop 7: [[subject period], [reference period]] when the question asks how many times one period's value of
    one series is another's (shared asked_comparison: a ratio, no difference named): two periods of one kind, no
    category with two codes. The reference is the period named right before a comparing word ("2013년의 몇 배",
    "2015년 대비", "2014년에 비해", "2023년보다": roles.reference_index, as two targets' reference is read), else the
    earlier one (an implied period: "전년 대비 몇 배")."""
    periods = list(plan.periods)
    if len(periods) != 2 or len({len(p) for p in periods}) != 1 or any(len(v) >= 2 for v in plan.codes.values()):
        return None
    diff, ratio = asked_comparison(rest)
    # 비율 alone names a measure ("고령인구 비율 변화"), not how many times one period is the other
    if not ratio or diff or all(m.group(0) == "비율" for m in ASKS_RATIO.finditer(rest)):
        return None
    names = [f"{p[:4]}년" + (f" {int(p[4:])}월" if len(p) == 6 else "") for p in periods]
    ref = reference_index(question, names)
    if ref is None:
        ref = periods.index(min(periods))
    return [[periods[1 - ref]], [periods[ref]]]


def _period_ratio_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """Loop 7: one series at the subject period over the reference period (plan.groups); the answer keeps the ratio."""
    problems: list[str] = []
    fixed = _fixed(plan, index, set(), problems)
    if problems:
        raise Unresolved(problems)
    (subject,), (reference,) = plan.groups
    refs = _refs(lookup, [(fixed, subject), (fixed, reference)], problems)
    if problems:
        raise Unresolved(problems)
    return build.compare(refs, reference=1, note=note)


def _spaced_end(question: str, squashed_end: int) -> int:
    """The position in the question as written right after its first squashed_end characters that are not spaces."""
    seen = 0
    for at, ch in enumerate(question):
        if seen == squashed_end:
            return at
        seen += not ch.isspace()
    return len(question)


def _within(question: str, index: TableIndex, name: str) -> list[str]:
    """Loop 6: the codes of a category the question names right before a word of looking inside a group (shared
    WITHIN_AFTER: "임금근로자 중", "서울 안에서", "비임금근로자 가운데"), in question order. Other names the question
    gives may come between the name and that word ("소규모 사업체 중": the group of the small ones among the
    businesses), with a 의 after each."""
    q = squash(question)
    matches = parse(index, question).matches
    forms = sorted({m.form for m in matches}, key=len, reverse=True)

    def after_names(end: int) -> int:
        while True:
            if q.startswith("의", end):
                end += 1
            nxt = next((f for f in forms if q.startswith(f, end)), None)
            if nxt is None:
                return end
            end += len(nxt)

    found: list[tuple[int, str]] = []
    for m in matches:
        if m.axis != name:
            continue
        at = q.find(m.form)
        while at >= 0:
            if WITHIN_AFTER.match(question[_spaced_end(question, after_names(at + len(m.form))):]):
                found.append((at, m.code))
                break
            at = q.find(m.form, at + 1)
    return [c for _, c in sorted(found)]


def _holds(index: TableIndex, name: str, whole: str, code: str) -> bool:
    """Loop 7: the code `whole` of a category is a group that holds `code`: the category's total (a total name with no
    parent), an ancestor of it, or the group its name is made of (judge group_parts: '귀농인-계' holds '귀농인-여자')."""
    axis = index.axes[name]
    if whole == code:
        return False
    if whole in axis.ancestors(code) or (whole == _total(index, name) and axis.parents.get(whole) is None):
        return True
    stem = total_stem(axis.labels[whole])
    group = squash(stem) if stem else squash(axis.labels[whole])
    pair = group_parts_of(index).get(name, {}).get(code)
    return pair is not None and pair[0] == group


def _drop_scopes(question: str, index: TableIndex, codes: dict[str, list[str]]) -> bool:
    """Loop 7: a code named right before a word of looking inside a group (shared WITHIN_AFTER: "가구원 중 청년",
    "전국 가운데 대구") that holds every other code named of its category (_holds) is where the question looks, not a
    target: it is left out (a value, a trend or a comparison of the others). Not called for a share (the group is the
    whole) or a ranking (the group is the pool). True when a code was left out."""
    dropped = False
    for name, named in codes.items():
        if len(named) < 2:
            continue
        scopes = [w for w in _within(question, index, name) if w in named]
        inner = [c for c in named if c not in scopes]
        if inner and scopes and all(_holds(index, name, w, c) for w in scopes for c in inner):
            codes[name] = inner
            dropped = True
    return dropped


def _children(index: TableIndex, name: str, code: str) -> list[str]:
    axis = index.axes[name]
    return [c for c, parent in axis.parents.items() if parent == code]


def _scope(question: str, index: TableIndex, codes: dict[str, list[str]]) -> str | None:
    """Loop 6: the category a ranking looks inside a named group of (one code named right before a word of looking
    inside, WITHIN_AFTER, that has members in the table: a subtotal and its details, a region and its districts); its
    members are the candidates (_pool). None when there is no such group."""
    for name, named in codes.items():
        if len(named) == 1 and _children(index, name, named[0]) and named[0] in _within(question, index, name):
            return name
    return None


def _ranked_axis(index: TableIndex, codes: dict[str, list[str]], question: str = "") -> str | None:
    """The category an extreme question ranks: the regional one it does not pin down, else any other open one.

    Loop 12: unless the noun that names the candidates (shared candidate_nouns: the noun a ranking word's describing
    word modifies, the noun after 어느 / 어떤) is an open category's: its name's stem (the name without 별 and its
    brackets, two characters or more) starts that noun, or the noun's first word starts the stem ("가장 많은 산업은"
    for 산업별, "제일 적은 대표자 연령대" for 대표자연령별). Only one such category; a noun of a region ("시도", "지역",
    "곳") keeps the regional one, and so does a stem said elsewhere in the question (a condition)."""
    open_axes = [n for n, a in index.axes.items() if n != ITEM_AXIS and len(a.labels) > 1 and len(codes.get(n, [])) != 1]
    regional = [n for n in open_axes if index.axes[n].regional]
    nouns = candidate_nouns(question) if question else []
    named = [n for n in open_axes if not index.axes[n].regional and any(_names_axis(noun, n) for noun in nouns)]
    if len(named) == 1:
        return named[0]
    return (regional or open_axes or [None])[0]


def _names_axis(noun: str, name: str) -> bool:
    """Loop 12: a candidate noun (squashed, with what follows it) names the category `name` (_ranked_axis)."""
    stem = squash(re.sub(r"\([^)]*\)", "", name))
    stem = stem[:-1] if stem.endswith("별") else stem
    if len(stem) < 2:
        return False
    word = re.match(r"[가-힣]+?(?=은|는|이|가|을|를|의|와|과|에|도|$)", noun)
    return noun.startswith(stem) or bool(word and len(word.group(0)) >= 2 and stem.startswith(word.group(0)))


# ---------------------------------------------------------------- plan -> blocks


def _total(index: TableIndex, name: str) -> str | None:
    """The category's total: the first total name with no parent (loop 6: not a subtotal such as '임금근로자-계' under
    '취업자-계'), else the first total name."""
    axis = index.axes[name]
    totals = [c for c, label in axis.labels.items() if is_total(label)]
    if top := next((c for c in totals if axis.parents.get(c) is None), None):
        return top
    # loop 11: one name every other name nests under (an unmarked whole the index linked: '취업자' over its subtotals)
    roots = [c for c in axis.labels if axis.parents.get(c) is None]
    if len(roots) == 1 and len(axis.labels) > 1:
        return roots[0]
    return totals[0] if totals else None


class _Lookup:
    """The table's cells by `fixed` (one code per axis, ITEM_AXIS included), read through the cell store. The plural
    methods make one store request for many cells or series (a loop over regions or periods asks once)."""

    def __init__(self, table: TableInput) -> None:
        self.store = store_of(table)

    @staticmethod
    def key(fixed: dict[str, str]) -> tuple[str, DimKey]:
        return fixed.get(ITEM_AXIS, ""), tuple(sorted((k, v) for k, v in fixed.items() if k != ITEM_AXIS))

    def cell(self, fixed: dict[str, str], period: str) -> Cell | None:
        return self.store.cell((*self.key(fixed), period))

    def cells(self, wanted: list[tuple[dict[str, str], str]]) -> list[Cell | None]:
        """cell() of many (fixed, period) pairs, in order."""
        keys = [(*self.key(fixed), period) for fixed, period in wanted]
        found = self.store.cells_at(keys) if keys else {}  # nothing asked: no request
        return [found.get(key) for key in keys]

    def numeric_periods(self, fixed: dict[str, str]) -> list[str]:
        return [c.period_raw for c in self.store.series(*self.key(fixed)) if c.value_kind == "ok"]

    def numeric_periods_of(self, fixed_sets: list[dict[str, str]]) -> list[list[str]]:
        """numeric_periods() of many fixed sets, in order."""
        keys = [self.key(fixed) for fixed in fixed_sets]
        found = self.store.series_at(keys) if keys else {}  # nothing asked: no request
        return [[c.period_raw for c in found.get(key, ()) if c.value_kind == "ok"] for key in keys]

    def all_periods(self, fixed: dict[str, str]) -> list[str]:
        return [c.period_raw for c in self.store.series(*self.key(fixed))]


def _lookup(table: TableInput) -> _Lookup:
    # One per TableInput (T1-28 S2: TableInput.caches, was a module dict keyed by id(table)).
    return table.caches.get("compute.analyze.lookup", lambda: _Lookup(table))


def _fixed(plan: Plan, index: TableIndex, free: set[str], problems: list[str]) -> dict[str, str]:
    """One code per category not in `free`: the named one, else the total, else the only one."""
    out: dict[str, str] = {}
    for name, axis in index.axes.items():
        if name in free:
            continue
        named = plan.codes.get(name, [])
        if len(named) == 1:
            out[name] = named[0]
        elif len(named) > 1:
            problems.append(f"{name}: {len(named)}개 중 하나")
        elif (chosen := _defaulted(plan, axis)) is not None:  # the base scenario the judge chose (중위 추계)
            out[name] = chosen
        elif (total := _total(index, name)) is not None:
            out[name] = total
        elif len(axis.labels) == 1:
            out[name] = next(iter(axis.labels))
        else:
            problems.append(name)
    return out


def _defaulted(plan: Plan, axis) -> str | None:
    """The code of the label the judge's default_applied names for this category ('가정별': '중위 추계')."""
    label = plan.default_applied.get(axis.name)
    found = [c for c, text in axis.labels.items() if label and text.strip().startswith(label)]
    return found[0] if len(found) == 1 else None


def _one_period(plan: Plan, lookup: _Lookup, fixed_sets: list[dict[str, str]]) -> str | None:
    if plan.periods:
        return plan.periods[-1]
    shared = None
    for periods in lookup.numeric_periods_of(fixed_sets):
        found = set(periods)
        shared = found if shared is None else shared & found
    return max(shared) if shared else None


def _with(fixed: dict[str, str], name: str, code: str) -> dict[str, str]:
    return {**fixed, name: code}


def _ref_of(cell: Cell | None, fixed: dict[str, str], period: str, problems: list[str]) -> str:
    if cell is None:
        problems.append(f"칸 없음 {fixed} {period}")
        return ""
    return cell.source_ref


def _ref(lookup: _Lookup, fixed: dict[str, str], period: str, problems: list[str]) -> str:
    return _ref_of(lookup.cell(fixed, period), fixed, period, problems)


def _refs(lookup: _Lookup, wanted: list[tuple[dict[str, str], str]], problems: list[str]) -> list[str]:
    """_ref() of many (fixed, period) pairs in one store request: the same refs and problems, in the same order."""
    return [_ref_of(cell, fixed, period, problems) for (fixed, period), cell in zip(wanted, lookup.cells(wanted))]


def _note(plan: Plan) -> str | None:
    """What the answer should say about the periods: asked years the table lacks, events read from the calendar."""
    notes = [f"{'·'.join(plan.missing_years)}년은 표에 없음"] if plan.missing_years else []
    for phrase, start in plan.events:
        when = f"{start[:4]}년" + (f" {int(start[5:])}월" if len(start) > 4 else "")
        notes.append(f"'{phrase}'은(는) 사건 달력의 시작 시점 {when} 기준")
    return "; ".join(notes) or None


def _compared_each(plan: Plan, index: TableIndex) -> tuple[str, str] | None:
    """Loop 12: (the category compared, the category the comparison is repeated for) when a comparison of two codes
    of one category is asked for each of the codes of another ("서울은 부산보다 남자와 여자가 각각 몇 명 많아"): two
    categories hold two codes or more; of those holding two, exactly one has a code the question compares against
    (roles.reference_index, the word that names the reference: 보다, 대비, 의 몇 배); the one other category's codes
    are each named by a name of their own (_named_one_by_one) and a word asks for each (shared EACH_TARGET). The
    comparison is the one a single target gets (build_plan), once per code in question order (Plan.codes' order).
    None otherwise (a pair of two names, no word for each: as before)."""
    several = [n for n, named in plan.codes.items() if len(named) >= 2]
    if len(several) != 2:
        return None
    parsed = parse(index, plan.question)
    if not EACH_TARGET.search(parsed.rest):
        return None
    between = [m.form for m in parsed.matches] + _table_words(index)
    compared = [n for n in several if len(plan.codes[n]) == 2 and reference_index(
        plan.question, [index.axes[n].labels[c] for c in plan.codes[n]], between) is not None]
    if len(compared) != 1:
        return None
    repeat = next(n for n in several if n != compared[0])
    if not _named_one_by_one(parsed, repeat, plan.codes[repeat]):
        return None
    return compared[0], repeat


def build_plan(plan: Plan, index: TableIndex, table: TableInput) -> Built:
    """The blocks for the plan (build.py), or Unresolved naming what is missing (for the LLM step)."""
    if plan.unresolved:
        raise Unresolved(plan.unresolved)
    lookup = _lookup(table)
    problems: list[str] = []
    note = _note(plan)
    kind = plan.type
    if plan.pattern in ("gap", "grw", "each"):
        return _two_by_two_plan(plan, index, lookup, note)
    if plan.pattern == "pratio":
        return _period_ratio_plan(plan, index, lookup, note)
    if (plan.pattern == "sum" and not plan.groups) or plan.pattern == "tavg":
        return _targets_sum(plan, index, lookup, note, "avg" if plan.pattern == "tavg" else "sum")
    if plan.pattern in ("cagr", "apc"):
        return _cagr_plan(plan, index, lookup, note, amount=plan.pattern == "apc")
    if plan.pattern == "excl":
        return _excluded_plan(plan, index, lookup, note)
    if plan.pattern:
        return _period_plan(plan, index, lookup, note)
    if kind == "value" and (len(plan.periods) >= 2 or any(len(v) >= 2 for v in plan.codes.values())):
        return _listing(plan, index, lookup, note)
    if kind in ("value", "trend"):
        fixed = _fixed(plan, index, set(), problems)
        if problems:
            raise Unresolved(problems)
        if kind == "trend":
            # One asked period ("2023년 얼마나 늘었나"): the recent points that end there.
            numeric = [p for p in lookup.numeric_periods(fixed) if not plan.periods or p <= plan.periods[-1]]
            periods = plan.periods if len(plan.periods) >= 2 else numeric[-RECENT_POINTS:]
            refs = [r for r in _refs(lookup, [(fixed, p) for p in periods], []) if r]
            if len(refs) < 2:
                raise Unresolved([f"추이 시점 부족 {periods}"])
            return build.trend(refs, note)
        period = _one_period(plan, lookup, [fixed])
        ref = _ref(lookup, fixed, period or "", problems)
        if problems:
            raise Unresolved(problems)
        newer = [p for p in lookup.all_periods(fixed) if period and p > period] if plan.default_applied.get("period") else []
        return build.value(ref, _refs(lookup, [(fixed, p) for p in newer], []), note)
    if plan.pairs and kind in ("compare", "share"):
        return _pairs_plan(plan, index, lookup, note)
    if kind == "compare" and (each := _compared_each(plan, index)) is not None:
        side, repeat = each
        return build.repeated([build_plan(plan.model_copy(update={"codes": {**plan.codes, repeat: [code]}}), index, table)
                               for code in plan.codes[repeat]])
    if kind == "compare":
        side = next((n for n, named in plan.codes.items() if len(named) == 2), None)
        if side is None:
            raise Unresolved(["비교할 두 대상"])
        fixed = _fixed(plan, index, {side}, problems)
        if problems:
            raise Unresolved(problems)
        pair = [_with(fixed, side, c) for c in plan.codes[side]]
        if len(plan.periods) >= 2 and all(len(v) <= 1 for n, v in plan.codes.items() if n != side):
            pass  # two places and two years: compare at the later one
        period = _one_period(plan, lookup, pair)
        refs = _refs(lookup, [(f, period or "") for f in pair], problems)
        if problems:
            raise Unresolved(problems)
        if per := _per_unit(plan.question, index, side, plan.codes[side]):
            return build.compare(refs, note=note, per=per)
        labels = [index.axes[side].labels[c] for c in plan.codes[side]]
        parsed = parse(index, plan.question)
        diff, ratio = asked_comparison(parsed.rest)
        reference = reference_index(plan.question, labels, [m.form for m in parsed.matches] + _table_words(index))
        # loop 8: "부산은 서울의 몇 퍼센트". Of cells already in % a percent names their unit, unless a ratio is asked.
        in_percent = any((index.measure_of(ITEM_AXIS, f[ITEM_AXIS]) or (None,))[0] in ("%", "퍼센트")
                         for f in pair if ITEM_AXIS in f)
        if reference is not None and asks_level(parsed.rest) and (ratio or not in_percent):
            return build.level(refs, reference, note)
        # loop 9: "서울은 부산보다 몇 퍼센트 많아": the difference as a percent of the reference (not of cells in %,
        # where a percent may be a difference in points)
        if reference is not None and asks_relative(parsed.rest) and not in_percent:
            return build.relative(refs, reference, note)
        return build.compare(refs, reference, note, asked=diff or ratio, diff=diff or not ratio)
    if kind == "share":
        return _share_plan(plan, index, lookup, note)
    if kind == "extreme":
        return _extreme_plan(plan, index, lookup, note)
    raise Unresolved(["질문 유형"])


# loop 12: also "B의 몇 배" (the 의 the shared REFERENCE_AFTER reads as a reference only before 몇 배)
_REFERENCE_WORD = re.compile(r"보다|에비해|대비|와비교|과비교|을기준|를기준|의몇배")
_BY_SUFFIX = re.compile(r"(?:별|단위)$")


def _table_words(index: TableIndex) -> list[str]:
    """Loop 9: the words the table's own information gives (squashed): its title's words, its categories' names and
    its items' names without a counting word ('사업체수' -> '사업체'), each without 별 ('시도별' -> '시도'). A question
    may put such a word of the kind of its targets between a target's name and the word that makes it the reference
    ("부산 지역보다", "제조업 사업체보다"); roles.reference_index reads past them as it reads past the other names."""
    title = getattr(index.table.meta, "tbl_nm", "") or ""
    items = index.axes[ITEM_AXIS].labels.values() if ITEM_AXIS in index.axes else ()
    words: set[str] = set()
    for text in (*re.split(r"[\s,·()/]+", title), *(n for n in index.axes if n != ITEM_AXIS)):
        word = _BY_SUFFIX.sub("", squash(text))
        if len(word) >= 2 and not any(ch.isdigit() for ch in word):
            words.add(word)
    for label in items:
        stem = COUNT_STEM_SUFFIX.sub("", squash(re.sub(r"\s*\(.*\)\s*$", "", label)))
        if len(stem) >= 2:
            words.add(stem)
    return sorted(words)


def _pairs_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """Loop 5: the two flows of plan.pairs (one code in each of two categories) at one period. A share: the first
    over the second (the part, then the whole). A compare: the difference or the ratio asked; the flow named just
    before a comparing word ("보다", "에 비해", "대비") is the reference."""
    sides = set().union(*plan.pairs)
    problems: list[str] = []
    fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k not in sides}}),
                   index, sides, problems)
    if problems:
        raise Unresolved(problems)
    sets = [{**fixed, **pair} for pair in plan.pairs]
    period = _one_period(plan, lookup, sets)
    refs = _refs(lookup, [(f, period or "") for f in sets], problems)
    if problems:
        raise Unresolved(problems)
    if plan.type == "share":
        return build.share(refs[:1], refs[1], note)
    rest = parse(index, plan.question).rest
    diff, ratio = asked_comparison(rest)
    return build.compare(refs, plan.pair_reference, note, asked=diff or ratio)


def _pair_reference(q: str, where: list[int]) -> int | None:
    """The flow a comparing word follows (on the squashed question, where = each flow's first position): the later
    of the flows named before the word; None without such a word or with no flow before it."""
    word = _REFERENCE_WORD.search(q)
    if word is None:
        return None
    before = [n for n, at in enumerate(where) if at < word.start()]
    return max(before, key=lambda n: where[n]) if before else None


def _share_axis(plan: Plan, index: TableIndex) -> tuple[str, list[str], str] | None:
    """(axis, part codes, denominator code): items first ('1인 가구 비중'), then a category ('동부 가구 비중').

    The denominator is the parts' nearest common parent in the code hierarchy ('외국인근로자' -> '한국국적을 가지지 않은
    자'), else the category's total; a parent the question names with the parts is not a part."""
    order = [ITEM_AXIS] + [n for n in plan.codes if n != ITEM_AXIS]
    # loop 9: a category with two parts or more named comes before one with a single code: the single code is then
    # fixed (the region the shares are of), where taking it as the part left the other category two codes to fix
    order.sort(key=lambda n: len([c for c in plan.codes.get(n, []) if n in index.axes and c != _total(index, n)]) < 2)
    # loop 6: first the category of a group the question looks inside that holds the parts named in it
    order.sort(key=lambda n: not (n in index.axes and any(
        w in index.axes[n].ancestors(c) for w in _within(plan.question, index, n) for c in plan.codes.get(n, []))))
    for name in order:
        if name not in index.axes:
            continue
        axis = index.axes[name]
        named = plan.codes.get(name, [])
        # loop 11: a category whose only code the question names right before a word of looking inside ("정규직 중
        # 여자 비중") is where the share is taken (fixed), not the parts' category, when another category names a part
        if len(named) == 1 and named[0] in _within(plan.question, index, name) and any(
                c != _total(index, o) for o in plan.codes if o != name and o in index.axes for c in plan.codes[o]):
            continue
        parents = {axis.parents.get(c) for c in named if axis.parents.get(c) in axis.labels}
        parent = next(iter(parents)) if len(parents) == 1 else None
        total = _total(index, name)
        whole = parent if parent is not None and all(axis.parents.get(c) == parent or c == parent for c in named) else total
        # loop 6: a group the question looks inside ("취업자 중 자영업자", "임금근로자 가운데 상용근로자") is the whole
        # when it holds every part (an ancestor of each), whether it is the total, a subtotal or a parent further up
        within = _within(plan.question, index, name)
        inner = [c for c in named if c not in within]
        whole = next((w for w in within if inner and all(w in axis.ancestors(c) for c in inner)), whole)
        # loop 11: the whole the question looks inside may hold the parts as the category's total (an unmarked whole:
        # _holds), or be a marked subtotal whose details the index could not link ("비임금근로자 중 자영업자"): either
        # way it is the denominator, not a part
        # loop 11: the question looks inside a group of other categories only ("서울 남자 중 상용근로자 비중"): the
        # share is of that whole group, the category's total, not of the parts' nearest subtotal
        if not within and total is not None and any(
                _within(plan.question, index, o) for o in plan.codes if o != name and o in index.axes):
            whole = total
        if within and inner and whole not in within:
            whole = next((w for w in within if all(_holds(index, name, w, c) for c in inner)),
                         next((w for w in within if is_total(axis.labels[w])), whole))
        parts = [c for c in named if c not in (total, whole)]
        if whole and parts:
            return name, parts, whole
    return None


def _part_of_year(periods: list[str]) -> tuple[list[str], str] | None:
    """(months, year) when the periods are one year and months of it: the months are a part of the year's value."""
    years = [p for p in periods if len(p) == 4]
    months = [p for p in periods if len(p) == 6]
    if len(years) == 1 and months and all(m[:4] == years[0] for m in months):
        return months, years[0]
    return None


def _share_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    if span := _part_of_year(plan.periods):  # a year and months of it named: the months are the part, whatever else
        return _period_share(plan, index, lookup, span, note)  # the question names ("서울 출생아 중 7월")
    found = _share_axis(plan, index)
    if found is None:
        raise Unresolved(["비중의 분자·분모"])
    name, parts, total = found
    problems: list[str] = []
    fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k != name}}), index, {name}, problems)
    if problems:
        raise Unresolved(problems)
    sets = [_with(fixed, name, c) for c in [*parts, total]]
    period = _one_period(plan, lookup, sets)
    refs = _refs(lookup, [(f, period or "") for f in sets], problems)
    if problems:
        raise Unresolved(problems)
    rest = parse(index, plan.question).rest
    # loop 8: each part's share over the same whole when a word of each is asked and nothing joins the parts
    if len(parts) >= 2 and EACH_TARGET.search(rest) and not SUM_WORDS.search(rest):
        return build.shares_each(refs[:-1], refs[-1], note)
    # loop 9: two parts' shares compared (a difference, a percent point, or "A 비중은 B보다 높아"): each share over the
    # same whole and their difference; the part named right before the comparing word is the reference
    if len(parts) == 2 and asks_gap_of(squash(rest)) and not SUM_WORDS.search(rest):
        labels = [index.axes[name].labels[c] for c in parts]
        # read past the other names, the table's words and the share nouns the question says ("2인 가구 비중보다")
        forms = ([m.form for m in parse(index, plan.question).matches] + _table_words(index)
                 + [m.group(0) for m in SHARE_WORDS.finditer(squash(plan.question))])
        return build.shares_gap(refs[:-1], refs[-1], reference_index(plan.question, labels, forms), note)
    return build.share(refs[:-1], refs[-1], note)


def _period_share(plan: Plan, index: TableIndex, lookup: _Lookup, span: tuple[list[str], str], note: str | None) -> Built:
    """A part of a year ("2025년 혼인 중 12월"): the months' cells over the year's cell of one series."""
    problems: list[str] = []
    fixed = _fixed(plan, index, set(), problems)
    if problems:
        raise Unresolved(problems)
    months, year = span
    *parts, whole = _refs(lookup, [(fixed, m) for m in [*months, year]], problems)
    if problems:
        raise Unresolved(problems)
    return build.share(parts, whole, note)


def _pool(index: TableIndex, name: str, named: list[str] | None = None) -> list[str]:
    """The candidates of a ranking over a category: its top-level codes but the total; the provinces of a regional one.
    Loop 6: the members of the one group the question looks inside (_scope: named is that one code with members);
    a category whose top level is only its total (a whole, subtotals and their details): the whole's members."""
    axis = index.axes[name]
    if named and len(named) == 1 and (members := _children(index, name, named[0])):
        return members
    whole = _total(index, name)
    pool = [c for c, label in axis.labels.items() if axis.parents.get(c) is None and not is_total(label) and c != whole]
    if not pool and whole is not None:
        return _children(index, name, whole)
    if axis.regional and any(is_province(axis.labels[c]) for c in pool):
        pool = [c for c in pool if is_province(axis.labels[c])]
    return pool


def _two_ends(plan: Plan) -> tuple[str, str] | None:
    """The earliest and the latest asked period of one kind (years, or months) when two or more were asked."""
    asked = sorted(plan.periods)
    if not asked:
        return None
    same = [p for p in asked if len(p) == len(asked[-1])]
    return (same[0], same[-1]) if len(same) >= 2 and same[0] != same[-1] else None


def _extreme_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    if plan.over is None or (plan.over != PERIOD_AXIS and plan.over not in index.axes):
        raise Unresolved(["over"])  # the model fill can leave it open too
    rest = parse(index, plan.question).rest
    places = rank_places(rest)  # loop 4: not the ordinal of a label the question names
    changed = bool(CHANGE_ASKED.search(rest) or _GROWTH_WORDS.search(rest))
    if changed and plan.over != PERIOD_AXIS and not SHARE_WORDS.search(rest) and (ends := _two_ends(plan)):
        return _change_rank_plan(plan, index, lookup, ends, rest, places, note)
    # both ends of a ranking: with their difference or ratio (loop 3), or (loop 4) named each with its own ranking word
    # ("가장 많은 곳과 가장 적은 곳", "최댓값과 최솟값") for their values alone
    both_ends = not changed and places is None and EXTREME_MAX.search(rest) and pick_of(rest) == "min"
    formula = bool(ASKS_GAP.search(rest) or ASKS_RATIO.search(rest))
    if both_ends and (formula or len(EXTREME_WORDS.findall(MOST_RECENT.sub(" ", rest))) >= 2):
        return _extremes_plan(plan, index, lookup, rest, note, formula)
    share = bool(SHARE_WORDS.search(plan.question))
    problems: list[str] = []
    item_total = _total(index, ITEM_AXIS) if share else None
    codes = dict(plan.codes)
    if share and item_total:
        codes[ITEM_AXIS] = [c for c in codes.get(ITEM_AXIS, []) if c != item_total]
    if plan.over == PERIOD_AXIS:
        fixed = _fixed(plan.model_copy(update={"codes": codes}), index, set(), problems)
        if problems:
            raise Unresolved(problems)
        periods = plan.periods or lookup.all_periods(fixed)
        groups = [(fixed, p) for p in periods]
        context_sets: list[tuple[dict, str]] = []
    else:
        name = plan.over
        fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in codes.items() if k != name}}), index, {name}, problems)
        if problems:
            raise Unresolved(problems)
        pool = _pool_of(plan, index, name)
        sets = [_with(fixed, name, c) for c in pool]
        period = _one_period(plan, lookup, sets[:1])
        groups = [(f, period or "") for f in sets]
        total = _total(index, name)
        context_sets = [(_with(fixed, name, total), period or "")] if total else []
    # every candidate, denominator and context cell in one store request (a denominator is asked even when its
    # candidate has no cell: it is then not used, as before)
    with_total = bool(share and item_total)
    denominator_sets = [(_with(f, ITEM_AXIS, item_total), p) for f, p in groups] if with_total else []
    found = _refs(lookup, [*groups, *denominator_sets, *context_sets], [])
    split = len(groups) + len(denominator_sets)
    refs, denominators, context = found[:len(groups)], found[len(groups):split], [r for r in found[split:] if r]
    candidates, totals = [], []
    for n, ref in enumerate(refs):
        if not ref:
            continue
        if with_total:
            if not denominators[n]:
                continue
            totals.append(denominators[n])
        candidates.append(ref)
    if len(candidates) < 2:
        raise Unresolved(["비교 대상 부족"])
    return build.extreme(candidates, plan.pick, totals, context, note, places=places)


def _change_rank_plan(plan: Plan, index: TableIndex, lookup: _Lookup, ends: tuple[str, str], rest: str,
                      places, note: str | None) -> Built:
    """Loop 3: the candidates ranked by their change between the earliest and the latest asked period (the amount,
    or the rate when a rate of change is asked); a ranking of declines puts the most negative change first."""
    name = plan.over
    problems: list[str] = []
    fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k != name}}), index, {name}, problems)
    if problems:
        raise Unresolved(problems)
    pool = _pool_of(plan, index, name)
    found = lookup.cells([(_with(fixed, name, c), p) for c in pool for p in ends])  # both ends of every candidate at once
    codes = [c for n, c in enumerate(pool)
             if all(cell is not None and cell.value_kind == "ok" for cell in found[2 * n:2 * n + 2])]
    if len(codes) < 2:
        raise Unresolved(["비교 대상 부족"])
    pick = plan.pick or pick_of(plan.question)
    if CHANGE_DOWN.search(rest):  # "가장 많이 줄어든": the largest decline is the smallest change
        pick = "min" if pick == "max" else "max"
    rate = bool(_GROWTH_RATE.search(rest))
    return build.ranked_change(name, codes, fixed, ITEM_AXIS, ends, rate, pick, places, note)


def _extremes_plan(plan: Plan, index: TableIndex, lookup: _Lookup, rest: str, note: str | None,
                   formula: bool = True) -> Built:
    """Loop 3: the largest and the smallest of the candidates (periods of one series, or the codes of a category at
    one period) and their difference, or their ratio when one is asked."""
    problems: list[str] = []
    if plan.over == PERIOD_AXIS:
        fixed = _fixed(plan, index, set(), problems)
        if problems:
            raise Unresolved(problems)
        periods = plan.periods or lookup.numeric_periods(fixed)
        if len({len(p) for p in periods}) > 1:  # a year's value beside its months: the kind asked (months by BY_MONTH)
            want = 6 if BY_MONTH.search(plan.question) else 4
            periods = [p for p in periods if len(p) == want]
        wanted = [(fixed, p) for p in periods]
    else:
        name = plan.over
        fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k != name}}), index, {name}, problems)
        if problems:
            raise Unresolved(problems)
        sets = [_with(fixed, name, c) for c in _pool_of(plan, index, name)]
        period = _one_period(plan, lookup, sets[:1])
        wanted = [(f, period or "") for f in sets]
    cells = [c.source_ref for c in lookup.cells(wanted) if c is not None and c.value_kind == "ok"]
    if len(cells) < 2:
        raise Unresolved(["비교 대상 부족"])
    if not formula:
        return build.extremes_combined(cells, None, note)
    _, ratio = asked_comparison(rest)
    return build.extremes_combined(cells, "ratio" if ratio and not ASKS_GAP.search(rest) else "diff", note)


def _two_by_two_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """gap / grw: the two targets (question order) at the earliest and the latest asked period."""
    side = next((n for n, named in plan.codes.items() if len(named) == 2 or (plan.pattern == "each" and len(named) > 2)),
                None)
    if side is None:
        raise Unresolved(["비교할 두 대상"])
    problems: list[str] = []
    fixed = _fixed(plan, index, {side}, problems)
    if problems:
        raise Unresolved(problems)
    early, late = plan.groups[0][0], plan.groups[-1][-1]
    wanted = [(_with(fixed, side, c), p) for c in plan.codes[side] for p in (early, late)]
    found = lookup.cells(wanted)
    flat = [_ref_of(cell, f, p, problems) for (f, p), cell in zip(wanted, found)]
    cells = [flat[n:n + 2] for n in range(0, len(flat), 2)]
    if problems:
        raise Unresolved(problems)
    if plan.pattern == "each":
        return build.changes_each([tuple(c) for c in cells], _rate_asked(plan, index, found), note)
    (x1, x2), (y1, y2) = cells
    if plan.pattern == "gap":
        return build.gap((x1, y1), (x2, y2), note)
    return build.growth((x1, x2), (y1, y2), amount=bool(_AMOUNT.search(plan.question)) and not _GROWTH_RATE.search(plan.question),
                        decline=bool(_DECLINE.search(plan.question)), note=note)


def _rate_asked(plan: Plan, index: TableIndex, cells: list[Cell | None]) -> bool:
    """Loop 7: whether a change is asked as a rate (as compute/answer_facts.py reads one target's change): a rate of
    change named, or a percent asked of cells that are not in % (a percent of % cells is a change in points unless
    the rate is named by its noun, 증가율 or 변화율)."""
    rest = parse(index, plan.question).rest
    named_rate = any(w in m.group(0) for m in RATE_OF_CHANGE.finditer(rest) for w in ("률", "율"))
    if any(c is not None and (c.unit or "").strip() in ("%", "퍼센트") for c in cells):
        return named_rate
    return bool(RATE_OF_CHANGE.search(rest) or PERCENT.search(rest))


def _cagr_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None, amount: bool = False) -> Built:
    """Loop 3: the average yearly rate of change from the first to the last asked year (or the last n years of
    "최근 n년"), from those two cells only."""
    problems: list[str] = []
    fixed = _fixed(plan, index, set(), problems)
    if problems:
        raise Unresolved(problems)
    years = sorted(p for p in plan.periods if len(p) == 4)
    if not years and (m := RECENT_N.search(plan.question)):
        years = [p for p in lookup.numeric_periods(fixed) if len(p) == 4][-int(m.group(1)):]
    if len(years) < 2 or years[0] == years[-1]:
        raise Unresolved(["연평균 변화율의 처음·끝 연도"])
    early, late = years[0], years[-1]
    refs = _refs(lookup, [(fixed, early), (fixed, late)], problems)
    if problems:
        raise Unresolved(problems)
    return build.average_growth(refs[0], refs[1], int(late) - int(early), note, amount=amount)


def _excluded_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """Loop 4: the category's total minus the code the question leaves out, at one period."""
    found = _excluded(plan.question, index, plan.codes, plan.periods)
    if found is None:
        raise Unresolved(["제외할 대상"])
    name, total, part = found
    problems: list[str] = []
    fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k != name}}), index, {name}, problems)
    if problems:
        raise Unresolved(problems)
    sets = [_with(fixed, name, total), _with(fixed, name, part)]
    period = _one_period(plan, lookup, sets)
    whole, piece = _refs(lookup, [(f, period or "") for f in sets], problems)
    if problems:
        raise Unresolved(problems)
    return build.remainder(whole, piece, note)


def _targets_sum(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None, agg: str = "sum") -> Built:
    """The sum of the targets one category names (two regions, two items, three age groups) at one period; agg "avg"
    (loop 8): their average."""
    sides = [n for n, named in plan.codes.items() if len(named) >= 2]
    if len(sides) != 1:
        raise Unresolved(["더할 대상"])
    side = sides[0]
    problems: list[str] = []
    fixed = _fixed(plan, index, {side}, problems)
    if problems:
        raise Unresolved(problems)
    sets = [_with(fixed, side, c) for c in plan.codes[side]]
    period = _one_period(plan, lookup, sets)
    refs = _refs(lookup, [(f, period or "") for f in sets], problems)
    if problems:
        raise Unresolved(problems)
    return build.over_periods(refs, "avg" if agg == "avg" else "sum", "value", note)


def _period_plan(plan: Plan, index: TableIndex, lookup: _Lookup, note: str | None) -> Built:
    """Blocks for a pattern over periods from the plan's period groups; the cells of one series, fixed like a value
    question (a share pattern: the parts and the total at each period)."""
    problems: list[str] = []
    if plan.pattern == "shr":
        found = _share_axis(plan, index)
        if found is None:
            raise Unresolved(["비중의 분자·분모"])
        name, parts, total = found
        fixed = _fixed(plan.model_copy(update={"codes": {k: v for k, v in plan.codes.items() if k != name}}), index, {name}, problems)
        if problems:
            raise Unresolved(problems)
        shares = []
        # every group's parts and total in one store request, up to the first empty group; the groups are then read
        # in order as before: the first group with a problem stops it, and an empty group fails at its turn
        # (IndexError on group[-1], as the group-by-group reads did)
        empty = next((n for n, group in enumerate(plan.groups) if not group), len(plan.groups))
        cells = iter(lookup.cells([(_with(fixed, name, c), group[-1]) for group in plan.groups[:empty]
                                   for c in [*parts, total]]))
        for group in plan.groups:
            period = group[-1]
            *refs, whole = [_ref_of(next(cells), _with(fixed, name, c), period, problems) for c in [*parts, total]]
            if problems:
                raise Unresolved(problems)
            shares.append((refs, whole))
        if len(shares) != 2:
            raise Unresolved([f"shr: 비중 시점 {len(shares)}개"])
        return build.share_change(shares[0], shares[1], note)
    fixed = _fixed(plan, index, set(), problems)
    if problems:
        raise Unresolved(problems)
    groups = plan.groups
    if not groups and plan.pattern == "avg":  # "최근 n년 평균"
        numeric = lookup.numeric_periods(fixed)
        n = int(m.group(1)) if (m := RECENT_N.search(plan.question)) else RECENT_POINTS
        groups = [numeric[-n:]]
    cells = []
    found = iter(lookup.cells([(fixed, p) for group in groups for p in group]))  # every group in one store request
    for n, group in enumerate(groups):
        refs = [r for p in group if (r := _ref_of(next(found), fixed, p, []))]
        if not refs:
            raise Unresolved([f"기간 {group[0]}~{group[-1]}에 칸 없음"])
        agg = None if plan.pattern == "vsa" and n == 1 else ("sum" if plan.pattern in ("sum", "psum") else "avg")
        if agg and len(refs) < 2:
            raise Unresolved([f"{agg}에 시점 부족 {group}"])
        cells.append(refs)
    try:
        if plan.pattern in ("avg", "sum"):
            if len(cells) != 1:
                raise ValueError(f"needs 1 period group, got {len(cells)}")
            return build.over_periods(cells[0], plan.pattern, plan.type, note)
        if len(cells) != 2:
            raise ValueError(f"needs 2 period groups, got {len(cells)}")
        if plan.pattern == "vsa":
            if len(cells[1]) != 1:
                raise ValueError(f"a single period holds one cell, got {len(cells[1])}")
            return build.vs_average(cells[0], cells[1][0], note)
        return build.groups_compared(cells[0], cells[1], "avg" if plan.pattern == "pavg" else "sum", note)
    except ValueError as exc:  # the groups do not fit the pattern (one range for pavg): not a question for it
        raise Unresolved([f"{plan.pattern}: {exc}"]) from exc


# ---------------------------------------------------------------- plan -> StatQuery (the record in labels)


def to_query(plan: Plan, index: TableIndex, built: Built | None = None) -> StatQuery:
    def labels(name: str) -> list[str]:
        return [index.axes[name].labels[c].strip() for c in plan.codes.get(name, []) if c in index.axes[name].labels]

    items = labels(ITEM_AXIS) if ITEM_AXIS in index.axes else []
    side = next((n for n, named in plan.codes.items() if len(named) == 2), None) if plan.type == "compare" else None
    filters = {n: labels(n)[0] for n in plan.codes if n not in (ITEM_AXIS, side) and len(labels(n)) == 1}
    over = {}
    if plan.type == "extreme" and plan.over and plan.over != PERIOD_AXIS:
        over = {plan.over: []}
    return StatQuery(
        question=plan.question,
        type=plan.type,
        item=items[0] if items else None,
        items=items if len(items) > 1 else [],
        periods=plan.periods,
        filters=filters,
        targets=[{side: label} for label in labels(side)] if side else [],
        reference=built.reference if built is not None else None,
        over=over,
        pick=plan.pick,
        plan=built.plan if built is not None else None,
    )
