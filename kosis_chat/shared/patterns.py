"""Regular expressions that two or more stages use with the same meaning (T1-28 S2b, D16 (d)).

Only patterns whose text and flags were the same at every place (or that match exactly the same strings where they
are used) live here; each old place keeps its name as an alias of the object here. A stage that needs a different
form (word boundaries, another capture, ASCII digits only) declares it next to its use, built from the base pattern
here and named after it, so the base is written once:

- plan_engine._FACT_ID_WORD = rf"\\b{FACT_ID.pattern}\\b" (compute: renaming ids inside a derivation)
- llm/compose._PLACEHOLDER = rf"\\{{\\s*({FACT_ID.pattern})\\s*\\}}" (compose: '{f3}' in the model's sentence)
- contract._ORG_FILE = rf"({ORG_ID.pattern})_(.+)\\.parquet" (the Parquet file name '101_DT_1B8000F.parquet')

Not merged (they match different strings): kosis_ai.units.SUFFIX_UNIT (the contract's strict '(unit)'),
compute/analyze.py's greedy r"\\s*\\(.*\\)\\s*$" label variant, kosis_ai.schemas._ORG_ID (r"\\d{3}" also takes
non-ASCII digits, and nothing reads it), eval/grade.py's copies (the grader stays independent of the pipeline).
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    from kosis_ai import units
    from kosis_ai.schemas import NUMBER
except ImportError:  # same fallback as prepare/index.py: src/ is not on the path for `python -m ...` (S6 removes it)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kosis_ai import units
    from kosis_ai.schemas import NUMBER

# ---------------------------------------------------------------- labels

# A trailing "(...)" with its spaces: "출생건수 (명)" -> "출생건수" (.sub), group 1 = "명". Was 5 copies: prepare/index
# _UNIT_PAREN and kosis_ai.normalize._UNIT_SUFFIX (this text), data/scope._UNIT_PAREN, judge/answerability and
# answer/render _UNIT_SUFFIX (the same without the capture group; all of those only .sub it, which a group does not
# change).
UNIT_PAREN = units.UNIT_PAREN
# Every "(...)" in a name: kosis_ai.units.PAREN (judge/question.py's inline r"\([^()]*\)" .sub is the same match).
PAREN = units.PAREN
# The province suffix: "부산광역시" -> "부산" (.sub). Was judge/question._REGION_SUFFIX, data/scope._REGION_SUFFIX,
# compute/analyze.py inline (this text), and compute/roles._PROVINCE (the same literals in another order, without the
# group; anchored at $ with literal alternatives, so the match at each start is the same and .search finds the same).
REGION_SUFFIX = re.compile(r"(특별자치시|특별자치도|특별시|광역시|도)$")

# ---------------------------------------------------------------- periods in a question

# Was judge/question.py and data/scope.py (same texts); the scope names are aliases there.
YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
YEAR_RANGE = re.compile(r"((?:19|20)\d{2})\s*년?\s*(?:~|-|부터)\s*((?:19|20)\d{2})")
YEAR_SINCE = re.compile(r"((?:19|20)\d{2})\s*년?\s*이후")
# Also compute/analyze.py's inline re.search(r"최근\s*(\d+)\s*년", ...).
RECENT_N = re.compile(r"최근\s*(\d+)\s*년")
# "1985년생" is a cohort label, not a period (judge/question._COHORT; data/scope.py inline re.sub).
COHORT = re.compile(r"(?:19|20)\d{2}\s*년생")

# ---------------------------------------------------------------- numbers and ids

# NUMBER (imported above): a number as the grader reads it, kosis_ai.schemas.NUMBER (was its _NUMBER), the same text
# as answer/verify.NUMBER, which is now this object.
# A fact id in a derivation: answer/template._DERIVED_IDS, answer/verify.py inline re.findall(r"f\d+", ...).
FACT_ID = re.compile(r"f\d+")
# A KOSIS agency id, ASCII digits (kosis_chat/contract.py: re.fullmatch(r"[0-9]{3}", ...) twice).
ORG_ID = re.compile(r"[0-9]{3}")

# ---------------------------------------------------------------- what a question computes between two numbers
# (loop 1). The texts are read on the question with its spaces removed (labels.squash) unless said otherwise.

# "혼인 1건당", "인구 1천 명당", "10만 명당", "가구당", "1인당": a value per unit of another (on the question as
# written). Group 1 the number, group 2 its 십/백/천/만 word. Not 해당, 상당, 담당, 정당, 적당, 부당, 배당, and not a 당
# that starts a word (당시, 당해).
PER_UNIT = re.compile(r"(\d[\d,.]*)?\s*(십만|백만|천만|십|백|천|만)?\s*(?:[가-힣]{1,2}\s*)?(?<![해상담정적부배])당"
                      r"(?=$|[\s?.!,]|(?:은|는|의|으로|로)(?:$|[\s?.!,]))")
_SCALE_WORDS = {"십": 10, "백": 100, "천": 1000, "만": 10_000, "십만": 100_000, "백만": 1_000_000, "천만": 10_000_000}


def per_scale(m: re.Match) -> float:
    """The number a PER_UNIT match counts by: '1천 명당' -> 1000, '10만 명당' -> 100000, '1건당' / '가구당' -> 1."""
    number = float(m.group(1).replace(",", "")) if m.group(1) else 1.0
    return number * _SCALE_WORDS.get(m.group(2) or "", 1)


# ---------------------------------------------------------------- a change between two periods (loop 2)
# One place for the words that ask how a number changed, read by the one-table rules (analyze/rules.py), the
# multi-table rules (analyze/multi_rules.py) and the judge (judge/question.py CHANGE_WORDS is this object).

# "얼마나 늘었나", "감소했어": a change asked without 추이/변화. Verb forms only, so a label such as 자연증가건수 stays out.
CHANGE_WORDS = re.compile(r"늘었|줄었|늘어|줄어|증가(?:했|한|하|세)|감소(?:했|한|하|세)")
# The stems of the verbs and nouns of a change in either direction or none (늘다, 줄다, 오르다, 떨어지다, 변하다,
# 달라지다, 바뀌다), as they follow "몇 퍼센트".
CHANGE_VERB = r"(?:늘|줄|증가|감소|올|오르|내리|내려|떨어|변하|변했|변해|변화|달라|바뀌|바뀐)"
# A rate of change: the nouns (증가율, 감소율, 증감률, 변화율, 변동률) or "몇 퍼센트(나) + a change verb".
RATE_OF_CHANGE = re.compile(rf"증가율|감소율|증감률|증감율|변화율|변동률|몇\s*(?:퍼센트|%|프로)\s*(?:나|가|정도)?\s*{CHANGE_VERB}")
# A change asked at all (an amount unless RATE_OF_CHANGE says a rate): the verb forms above, the noun 증감 and the
# verbs of a change with no direction (변했다, 변화, 달라졌다, 바뀌었다).
CHANGE_ASKED = re.compile(CHANGE_WORDS.pattern + r"|증감(?![률율])|변했|변화(?!율)|달라졌|바뀌었|" + RATE_OF_CHANGE.pattern)

# A range of periods asked period by period (연도별, 해마다, 매년, 월별, 매달, 분기별): the periods are not aggregated.
EACH_PERIOD = re.compile(r"연도별|년도별|해마다|매년|매해|연도마다|년마다|월별|매월|매달|달마다|분기별|분기마다")

# A share of one number in another: 비중, 차지, 구성비, 몇 퍼센트 (not 몇 퍼센트 + a change verb, a rate of change).
MULTI_SHARE = re.compile(rf"비중|차지|구성비|몇(?:퍼센트|%|프로)(?!(?:나|가|정도)?{CHANGE_VERB})")
# One number over another: 몇 배, 배수, 나누다, 비율, and 배 with an ending (배나, 배야, 배인, 배로). One list for one
# table (compute/answer_facts.py, analyze/rules.py) and several (loop 2: was two lists, MULTI_RATIO and _ASKS_RATIO).
ASKS_RATIO = re.compile(r"몇배|배수|비율|나누|나눈|나눠|배(?:나|정도|가량|야|인|이|로)")
MULTI_RATIO = ASKS_RATIO
# What follows the name of the whole (the denominator): "A 중", "A 가운데", "A 대비", "A에서", "A의 몇 배".
MULTI_WHOLE_AFTER = re.compile(r"^(?:은|는|이|가)?(?:중|가운데|대비|에서|에비해|의몇|기준)")
# A difference of two numbers, and the words that ask how much fewer (the reference minus the subject). The
# one-syllable comparatives are kept out of the words they also start (loop 1 learnings): 적어 줘 (write it down),
# 작년 (last year), 크기 (size). One list for one table (ASKS_DIFF) and several (MULTI_DIFF).
_COMPARATIVE = r"많|적(?!어\s*(?:줘|주|봐|달|둬))|높|낮|크(?!기)|작(?!년)|넘|모자|덜"
ASKS_GAP = re.compile(r"차이|격차|빼|뺀")  # a difference asked by name, beside a ratio
MULTI_DIFF = re.compile(rf"{ASKS_GAP.pattern}|{_COMPARATIVE}")
ASKS_DIFF = re.compile(rf"비교|보다|대비|{MULTI_DIFF.pattern}")
MULTI_FEWER = re.compile(r"적(?!어\s*(?:줘|주|봐|달|둬))|낮|작(?!년)|모자|덜")


def asked_comparison(rest: str) -> tuple[bool, bool]:
    """(a difference asked, a ratio asked) of two values, read on the question without its labels (squashed): a
    ratio for 몇 배 / 비율 / 나누다, a difference for a comparing word, but beside a ratio only when named (차이,
    빼다: "몇 배 많아" is one ratio); neither when only the two values were asked ("서울과 부산 인구 알려줘")."""
    ratio, diff = bool(ASKS_RATIO.search(rest)), bool(ASKS_DIFF.search(rest))
    if ratio and not ASKS_GAP.search(rest):
        diff = False
    return diff, ratio
# A count item's name without its count word: '혼인건수' -> '혼인', '출생아수' -> '출생', '사망자수' -> '사망'.
COUNT_STEM_SUFFIX = re.compile(r"(?:건수|아수|자수|수)$")
# The word that makes an item name the whole of its kind: '총인구수' is asked as '인구수'.
TOTAL_PREFIX = re.compile(r"^(?:총|전체)")
