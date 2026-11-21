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


# A share of one number in another: 비중, 차지, 구성비, 몇 퍼센트 (not 몇 퍼센트 늘었/줄었, a growth rate).
MULTI_SHARE = re.compile(r"비중|차지|구성비|몇(?:퍼센트|%|프로)(?!(?:나|가|정도)?(?:늘|줄|증가|감소|올|내|올랐|떨어))")
# One number over another: 몇 배, 배수, 나누다, 비율.
MULTI_RATIO = re.compile(r"몇배|배수|나누|나눈|나눠|비율")
# What follows the name of the whole (the denominator): "A 중", "A 가운데", "A 대비", "A에서", "A의 몇 배".
MULTI_WHOLE_AFTER = re.compile(r"^(?:은|는|이|가)?(?:중|가운데|대비|에서|에비해|의몇|기준)")
# A difference of two numbers, and the words that ask how much fewer (the reference minus the subject).
MULTI_DIFF = re.compile(r"차이|격차|빼|뺀|많|적|높|낮|크|작|넘|모자|덜")
MULTI_FEWER = re.compile(r"적|낮|작|모자|덜")
# A count item's name without its count word: '혼인건수' -> '혼인', '출생아수' -> '출생', '사망자수' -> '사망'.
COUNT_STEM_SUFFIX = re.compile(r"(?:건수|아수|자수|수)$")
# The word that makes an item name the whole of its kind: '총인구수' is asked as '인구수'.
TOTAL_PREFIX = re.compile(r"^(?:총|전체)")
