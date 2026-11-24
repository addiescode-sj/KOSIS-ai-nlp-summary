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
# The period right before the one a question names (loop 5; judge/question.py asked_periods puts it in the periods,
# as it does for "N년 전"). Read on the question without the labels it names. Three kinds:
# - the same month (or the same part of the year) a year before: 전년 동월, 작년 같은 달, 지난해 같은 기간, 1년 전 같은 달;
# - the month before (a named month only): 전월, 전달, 지난달, 앞 달, 직전 달·월, 한 달 전, "N개월 전" (group 1: N);
#   not 전월세 (a rent), not 전달하다 (to pass on);
# - the year before: 전년, 전년도, 작년, 지난해, 앞 해, 직전 해·연도, 한 해 전.
PREVIOUS_SAME_MONTH = re.compile(r"(?:전년|작년|지난\s*해|한\s*해\s*전|1\s*년\s*전)\s*(?:의\s*)?(?:동월|같은\s*달|동기|같은\s*기간)")
PREVIOUS_MONTH = re.compile(r"전월(?!세)|전\s*달(?![하해했되된받])|지난\s*달|앞\s*달|직전\s*(?:달|월)|한\s*달\s*전"
                            r"|(?<!\d)(\d{1,2})\s*개월\s*전")
PREVIOUS_YEAR = re.compile(r"전년|작년|지난\s*해|앞\s*해|직전\s*(?:해|연도|년도)|한\s*해\s*전")

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
# loop 3: also "몇 퍼센트씩 늘었어" (씩: by so much each period).
RATE_OF_CHANGE = re.compile(rf"증가율|감소율|증감률|증감율|변화율|변동률|성장률|몇\s*(?:퍼센트|%|프로)\s*(?:씩)?\s*(?:나|가|정도)?\s*{CHANGE_VERB}")
# A change asked at all (an amount unless RATE_OF_CHANGE says a rate): the verb forms above, the noun 증감 and the
# verbs of a change with no direction (변했다, 변화, 달라졌다, 바뀌었다).
CHANGE_ASKED = re.compile(CHANGE_WORDS.pattern + r"|증감(?![률율])|변했|변화(?!율)|달라졌|바뀌었|" + RATE_OF_CHANGE.pattern)
# Loop 5 (compute/answer_facts.py, which of a change and a rate of change a trend answer keeps): a percent asked
# (퍼센트, %, 프로; not a percent point), and an amount asked: 몇 + a counting word ("몇 명", "몇 건", "몇 가구", "몇 세";
# not 몇 퍼센트, 몇 배, 몇 년, 몇 월, 몇 위) or a percent point (%p, 퍼센트 포인트).
PERCENT = re.compile(r"퍼센트(?!\s*포인트)|%(?!\s*(?:p|포인트))|프로(?!그|젝)")
_HOW_MANY = r"몇\s*(?!퍼센트|프로|배|번|위|등|년|개월|월|분기|시)[가-힣]"  # 몇 + a counting word (몇 명, 몇 건, 몇 가구)
ASKS_AMOUNT = re.compile(_HOW_MANY + r"|%\s*p|퍼센트\s*포인트|%\s*포인트")
# Loop 9: a word of the whole (모두, 전부, 총, 통틀어, 도합; the adverbs judge TOTAL_WORDS reads and their kin) right
# before 몇 + a counting word ("모두 몇 명", "전부 합해서 몇 건", "총 몇 가구"): the amount of the targets together. A
# word of the whole before a request ("모두 알려줘") lists the values and is not this.
TOTAL_AMOUNT = re.compile(r"(?:모두|전부|통틀어|도합|총)\s*(?:해서|합해서|합쳐서|더해서)?\s*" + _HOW_MANY)

# Two categories of one table that hold the same names in two roles (loop 5): where a flow starts and where it ends
# (전출지별 × 전입지별, 출발지 × 도착지, 출신지 × 거주지). The role of a category, read on its name:
# Loop 14: also where people were born or grew up, the place they came from (출생지, 태생지, 성장지).
ORIGIN_AXIS = re.compile(r"전출|출발|출신|유출|송출|발송|이전\s*전|출생|태생|성장")
DEST_AXIS = re.compile(r"전입|도착|목적|유입|수신|행선|거주|이전\s*후")
# The role of a name the question gives (on the squashed question, right after the name): a particle of a start
# (에서, 부터, 로부터) or of an end (으로, 로, 에, 까지), or, after a subject or object particle, a word of leaving
# (떠나다, 나가다, 빠져나가다, 전출, 출발, 출신) or of arriving (들어오다, 들어가다, 옮겨 오다, 전입, 도착, 유입).
ORIGIN_AFTER = re.compile(r"^(?:에서|으로부터|로부터|부터)"
                          r"|^(?:을|를|이|가|의|은|는)?(?:떠나|떠난|나간|나가|빠져나|유출|전출|출발|출신)")
DEST_AFTER = re.compile(r"^(?:으로|로|에|까지)(?!서)"
                        r"|^(?:을|를|이|가|의|은|는)?(?:들어오|들어온|들어간|들어가|옮겨온|옮겨간|유입|전입|도착)")
# A second flow named only as the first one turned around: "그 반대", "반대로", "거꾸로", "역방향", "역으로".
REVERSE = re.compile(r"반대|거꾸로|역방향|역으로")
# Loop 12: the roles a category of places plays, read on its name, as four kinds: where a flow leaves (ORIGIN_AXIS),
# where it arrives (DEST_AXIS), where people stay (거주지, 상주지, 주소지) and where they go to and from each day
# (통근지, 통학지, 종사지, 근무지, 직장). A pair is read in the pair: the place people stay is the end of a flow of
# moving (출신지 × 거주지) and the start of a flow of going to and from (상주지 × 통근통학지).
STAY_AXIS = re.compile(r"거주|상주|주소")
VISIT_AXIS = re.compile(r"통근|통학|종사지|근무지|직장|등교|방문지")
# Loop 12: the verb after a name and its particle that says which of those roles the name plays ("서울에 사는", "부산에
# 거주하는", "서울로 통근하는", "인천에서 출근하는" is a stay only by its particle: the verb is of going). The kind of the
# verb decides; with no such verb the particle does (ORIGIN_AFTER, DEST_AFTER).
_ROLE_PARTICLE = r"^(?:에서|에|으로|로|을|를|이|가|의|은|는)?"
ROLE_VERBS = {
    # loop 14: also being born or growing up there (the place people came from: 태어난, 출생한, 자란, 성장한)
    "leave": re.compile(_ROLE_PARTICLE + r"(?:떠나|떠난|나간|나가|빠져나|유출|전출|출발|출신|태어나|태어난|태생|출생|자란|자라|성장)"),
    "arrive": re.compile(_ROLE_PARTICLE + r"(?:들어오|들어온|들어간|들어가|옮겨온|옮겨간|유입|전입|도착)"),
    "stay": re.compile(_ROLE_PARTICLE + r"(?:사는|살고|살면서|살며|살던|산다|거주|머무|머문|머물|상주)"),
    "visit": re.compile(_ROLE_PARTICLE + r"(?:통근|출근|통학|등교|다니|다닌|다녀|오가|오간|오고가|일하|근무|방문)"),
}
ROLE_AXES = {"leave": ORIGIN_AXIS, "arrive": DEST_AXIS, "stay": STAY_AXIS, "visit": VISIT_AXIS}


def role_kind(after: str) -> str | None:
    """Loop 12: the kind of role (ROLE_VERBS) the verb right after a name and its particle says, or None."""
    return next((kind for kind, verb in ROLE_VERBS.items() if verb.match(after)), None)

# Years asked at a regular interval: "2년 간격으로", "5년마다", "3년 주기로", "5년 단위로", "격년으로" (loop 4).
# Group 1 the number of years (none: 격년, every other year).
EVERY_N_YEARS = re.compile(r"(?<!\d)(\d{1,2})\s*년\s*(?:간격|마다|주기|단위로|단위\s*로)|격년")
# A range of periods asked period by period (연도별, 해마다, 매년, 월별, 매달, 분기별, and every n years): the periods
# are not aggregated.
EACH_PERIOD = re.compile(r"연도별|년도별|해마다|매년|매해|연도마다|년마다|월별|매월|매달|달마다|분기별|분기마다|"
                         + EVERY_N_YEARS.pattern)


# Loop 7: a word that asks for the same thing of each target named ("각각", "각자", "저마다", "제각기", "따로", "각 지역"),
# on the question without its labels. 각 only as a word of its own (not 각종, 시각).
EACH_TARGET = re.compile(r"각각|각자|저마다|제각기|제각각|따로|(?<![가-힣])각(?=\s)")
# Loop 7: words that ask which of two targets is more ("어느 쪽", "어디가", "누가", "더 많이", "더 크게", "비교", "차이"),
# on the question without its labels: the two targets' changes are compared, not only given.
WHICH_MORE = re.compile(r"어느|어디가|어디의|누가|누구|비교|차이|격차|더\s*(?:많|적|크|작|높|낮|늘|줄|빠르|느리|가파르)")


def years_step(question: str) -> int | None:
    """The interval of years a question asks the periods at ("5년마다" -> 5, "격년" -> 2), None for none or 1."""
    m = EVERY_N_YEARS.search(question)
    if m is None:
        return None
    n = int(m.group(1)) if m.group(1) else 2
    return n if n >= 2 else None

# A share of one number in another: 비중, 차지, 구성비, 몇 퍼센트 (not 몇 퍼센트 + a change verb, a rate of change).
MULTI_SHARE = re.compile(rf"비중|차지|구성비|몇(?:퍼센트|%|프로)(?!(?:씩)?(?:나|가|정도)?{CHANGE_VERB})")
# One number over another: 몇 배, 배수, 나누다, 비율, and 배 with an ending (배나, 배야, 배인, 배로). One list for one
# table (compute/answer_facts.py, analyze/rules.py) and several (loop 2: was two lists, MULTI_RATIO and _ASKS_RATIO).
ASKS_RATIO = re.compile(r"몇배|배수|비율|나누|나눈|나눠|배(?:나|정도|가량|야|인|이|로)")
MULTI_RATIO = ASKS_RATIO
# What follows the name of the whole (the denominator): "A 중", "A 가운데", "A 대비", "A에서", "A의 몇 배".
MULTI_WHOLE_AFTER = re.compile(r"^(?:은|는|이|가)?(?:중|가운데|대비|에서|에비해|의몇|기준)")
# Loop 6: what follows the name of a group a question looks inside (on the question as written, right after the
# name): "A 중", "A 중에서", "A 가운데", "A 안에서", "A 내에서", "A에 속한", "A에 해당하는". 중 and 가운데 only as words of
# their own (not 중구, 중학교). The group is a share's whole and a ranking's pool (its members are the candidates).
WITHIN_AFTER = re.compile(r"^\s*(?:은|는|이|가)?\s*(?:(?:중|가운데)(?:에서|에|의)?(?![가-힣])|(?:안|내)(?:에서|의)(?![가-힣])"
                          r"|에\s*(?:속한|속하는|속해|해당하는|해당한|해당되는))")
# Loop 15: the word of looking inside of WITHIN_AFTER said as a word of its own after a space ("… 사람들 중", "…
# 집단 가운데", "… 안에서"), wherever it is in the question.
_WITHIN_WORD = re.compile(r"(?<=\s)(?:(?:중|가운데)(?:에서|에|의)?(?![가-힣])|(?:안|내)(?:에서|의)(?![가-힣]))")


def within_split(question: str) -> tuple[int, int] | None:
    """Loop 15: (start, end) on the squashed question of the one word of looking inside a group the question says as
    a word of its own (_WITHIN_WORD); None for none or two. The names before it are where the question looks, the
    names after it the parts, whatever words come between a name and the word."""
    found = list(_WITHIN_WORD.finditer(question))
    if len(found) != 1:
        return None
    m = found[0]
    start = len(re.sub(r"\s+", "", question[: m.start()]))
    return start, start + len(re.sub(r"\s+", "", m.group(0)))


# A difference of two numbers, and the words that ask how much fewer (the reference minus the subject). The
# one-syllable comparatives are kept out of the words they also start (loop 1 learnings): 적어 줘 (write it down),
# 작년 (last year), 크기 (size). One list for one table (ASKS_DIFF) and several (MULTI_DIFF).
_COMPARATIVE = r"많|적(?!어\s*(?:줘|주|봐|달|둬))|높|낮|크(?!기)|작(?!년)|넘|모자|덜"
ASKS_GAP = re.compile(r"차이|격차|빼|뺀")  # a difference asked by name, beside a ratio
# The rest of a whole but one part (loop 4), on the question as written: "서울을 제외한", "서울 빼고", "서울 말고",
# "나머지", "서울 외 지역", "서울 이외의", "그 밖의". 외 only as a word of its own (not 외국, 해외).
EXCLUDE = re.compile(r"제외|빼고|빼면|뺀|말고|나머지|그\s*밖|(?<![가-힣])(?:이)?외(?:의|에|에는)?(?![가-힣])")
MULTI_DIFF = re.compile(rf"{ASKS_GAP.pattern}|{_COMPARATIVE}")
ASKS_DIFF = re.compile(rf"비교|보다|대비|{MULTI_DIFF.pattern}")
MULTI_FEWER = re.compile(r"적(?!어\s*(?:줘|주|봐|달|둬))|낮|작(?!년)|모자|덜")

# The smaller end of a ranking (analyze/roles.py pick_of; loop 3: the adverbs 적게·작게, 최솟값, the bottom of a
# ranking: 하위, 아래·밑·뒤에서 세어). Loop 16 (moved from analyze/roles.py _MIN_WORDS): a one-syllable stem with its
# ending (_MIN_STEM_FORMS: 낮은, 적은, 작게, 짧은) also ends nouns ('면적은', '실적은', '성적은'), so it is read only where
# a word starts or right after a ranking adverb written together with it ("가장적은"), on the question as written.
_MIN_STEM_FORMS = r"낮|짧|적은|적었|적게|작은|작았|작게"
MIN_WORDS = re.compile(_MIN_STEM_FORMS + r"|최저|최소|최솟|꼴찌|하위|(?:아래|밑|뒤)에서")
_MIN_STEM_AT_START = re.compile(rf"(?:(?<![가-힣])|(?<=가장)|(?<=제일)|(?<=더)|(?<=덜))(?:{_MIN_STEM_FORMS})")
_MIN_STEM = re.compile(rf"^(?:{_MIN_STEM_FORMS})$")


def asks_min(text: str, spaced: str | None = None) -> bool:
    """Loop 16: text (the question, or its squashed rest) asks for the smaller end. spaced: the question as written,
    where a one-syllable stem form found in text is looked for at a word's start (text itself when None)."""
    spaced = text if spaced is None else spaced
    for m in MIN_WORDS.finditer(text):
        if not _MIN_STEM.match(m.group(0)):
            return True
        if any(s.group(0) == m.group(0) for s in _MIN_STEM_AT_START.finditer(spaced)):
            return True
    return False


# Loop 8. A division said with its verb: the target the particle 로 / 으로 follows ("서울 출생아 수로 나누면", "부산으로
# 나눈") is the denominator. Read on the squashed question, right after the target's name and the counting words
# that may come between (COUNT_WORD: "부산 출생아 수로", "서울 가구수로").
DIVIDE = re.compile(r"나누|나눈|나눠|나눴")
DIVISOR_AFTER = re.compile(r"^(?:으로|로)")
COUNT_WORD = re.compile(r"^(?:건수|가구수|인구수|수|명|건|가구|개|곳)")
# What follows the label a comparison is made against, on the squashed question (was analyze/roles.py
# _REFERENCE_AFTER and _OF_PERCENT; loop 12: the judge reads them too): "전국보다", "전국 대비", "전국 평균의 몇 배",
# "전국에 비해", and a 의 that is read as a reference only before 몇 배 or 몇 퍼센트 (REFERENCE_OF).
REFERENCE_AFTER = re.compile(r"^(?:평균)?(?:보다|대비|에비해|의몇배|의)")
OF_PERCENT = re.compile(r"^의몇(?:퍼센트|%|프로)")
# Loop 12: a particle that joins two names as targets side by side ("서울과 종로구", "취업자 및 제조업", "서울이랑",
# "서울하고", "서울, 종로구"), right after the first name on the squashed question.
JOIN_AFTER = re.compile(r"^(?:와|과|및|하고|이랑|랑|그리고|또는|,|·)")
# Loop 12: the subject particles after a name ("서울은", "서울이"), on the squashed question.
SUBJECT_AFTER = re.compile(r"^(?:은|는|이|가)")
# The two sexes named as one word ("남녀", "남성과 여성", "남자와 여자", "성별"; was analyze/rules.py _BOTH_SEXES).
BOTH_SEXES = re.compile(r"남녀|남성과\s*여성|남자와\s*여자|성별")
# Loop 16 (loop 13's WHOLE_AFTER, kept to the words of the whole and the words that put together): said right after
# two names named as one word (BOTH_SEXES), on the squashed question, the whole of them is meant ("남녀 전체", "남녀
# 합계", "남녀를 합친", "남녀 모두 합쳐서", "남녀 통틀어"), not each. 모두 and 전부 alone say both each ("남녀 모두
# 늘었어"), so they count only before a word that puts together.
WHOLE_AFTER = re.compile(r"^(?:의|을|를|이|가)?(?:전체|통틀어|도합|합계|합산)"
                         r"|^(?:을|를|이|가)?(?:모두|전부)?(?:합쳐|합친|합한|합해|합하|더한|더해)")


def refers_after(rest: str) -> bool:
    """Whether the squashed text right after a label makes it the reference of a comparison (REFERENCE_AFTER, a 의
    only before 몇 배 / 몇 퍼센트) or the divisor of a division said with its verb ("B로 나누면")."""
    if REFERENCE_AFTER.match(rest):
        return not rest.startswith("의") or "배" in rest[:6] or bool(OF_PERCENT.match(rest))
    return bool(DIVISOR_AFTER.match(rest)) and bool(DIVIDE.search(rest))


def asks_level(rest: str) -> bool:
    """Loop 8: one value as a percent of another ("부산은 서울의 몇 퍼센트", "서울 대비 대구 비율은 몇 %"), read on the
    question without its labels: a percent asked with no word of a difference or of more / less (not "몇 퍼센트 많아",
    a relative difference) and no change ("몇 퍼센트 늘었어")."""
    return bool(PERCENT.search(rest)) and not MULTI_DIFF.search(rest) and not CHANGE_ASKED.search(rest)


def asks_ratio_noun(rest: str) -> bool:
    """Loop 15: the noun 비율 asked of one target with nothing else that compares or divides two values (no other word
    of ASKS_RATIO: 몇 배, 나누다; no word of ASKS_DIFF: 대비, 보다, 차이, more / less) and no change, read on the question
    without its labels (a 비율 inside a name the question names is the name's). The rules read it as a share only of
    a part of a group the question looks inside (analyze/rules.py _scoped_percent), as they read asks_level."""
    if "비율" not in rest:
        return False
    other = rest.replace("비율", " ")
    return not ASKS_RATIO.search(other) and not ASKS_DIFF.search(other) and not CHANGE_ASKED.search(rest)


# Loop 9: words that compare two values by more / less (the comparatives of MULTI_DIFF: 많, 적, 높, 낮, 크, 작, 넘, 모자,
# 덜), apart from the nouns of a difference (차이, 격차, 빼다).
_MORE_LESS = re.compile(_COMPARATIVE)
# Loop 9: a word naming the reference of a comparison ("보다", "에 비해", "대비", "와 비교해"), on the squashed rest.
_AGAINST = re.compile(r"보다|에비해|대비|와비교|과비교")
AGAINST = _AGAINST  # loop 12: the multi-table rules find the reference of a relative difference by it


def asks_relative(rest: str) -> bool:
    """Loop 9: one value's difference from another as a percent of the other ("서울은 부산보다 몇 퍼센트 많아", "몇 %
    적어"): a percent and a word of more / less asked, read on the question without its labels; no difference named
    (차이, 격차: "차이가 몇 퍼센트" is left as it was), no ratio, no change. The other side of asks_level, which reads
    the same percent without a word of more / less."""
    return (bool(PERCENT.search(rest)) and bool(_MORE_LESS.search(rest)) and not ASKS_GAP.search(rest)
            and not ASKS_RATIO.search(rest) and not CHANGE_ASKED.search(rest))


def asks_gap_of(rest: str) -> bool:
    """Loop 9: two numbers of one kind compared as a difference (two parts' shares): a difference named (차이, 격차,
    빼다), a percent point, or a word naming the reference with a word of more / less ("A 비중은 B보다 얼마나 높아").
    Read on the question without its labels (squashed)."""
    return bool(ASKS_GAP.search(rest) or re.search(r"%\s*p|퍼센트\s*포인트|%\s*포인트", rest)
                or (_AGAINST.search(rest) and _MORE_LESS.search(rest)))


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

# ---------------------------------------------------------------- a ranking (loop 3)
# Read by the one-table rules (analyze/rules.py): which places of a ranking the question asks for, and whether it
# ranks periods. On the question as written.

# A count as a digit or a native Korean number word (한·두·세 ... 열), and the ordinal stems (첫째, 둘째, ...).
_COUNT_WORDS = {"하나": 1, "한": 1, "첫": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6,
                "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_COUNT = r"(\d+|" + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True)) + r")"
# One place: "3번째", "세 번째로", "둘째로", "3위", "2등" (not 3위까지 / 3위 안: several; not 1등급).
RANK_PLACE = re.compile(rf"{_COUNT}\s*번\s*째|{_COUNT}\s*째|(\d+)\s*(?:위|등)(?!\s*(?:까지|안|이내|권|내|부터)|급|분)")
# The first n places: "상위 3개", "하위 다섯 곳", "위에서 세 번째까지", "1~3위", "1위부터 3위까지", "3위까지", "3위 안",
# and a count of places after a ranking word ("가장 많은 세 곳", "높은 순으로 5개 시도"). Not "17개 시도 중" (the pool),
# not "3개월" (a period).
RANK_TOP = re.compile(
    rf"(?:상위|하위|위에서|아래에서|밑에서|뒤에서|톱|top|TOP)\s*{_COUNT}"
    r"|1\s*(?:위|등)?\s*(?:~|-|부터|에서)\s*(\d+)\s*(?:위|등)"
    rf"|{_COUNT}\s*(?:위|등|번\s*째)\s*(?:까지|안|이내|권|내)"
    rf"|{_COUNT}\s*(?:개|곳|군데)(?!월|\s*(?:시도|지역|도시|나라|국가|항목)?\s*(?:중|가운데|에서|의))")
# The periods of a series ranked: 언제, 몇 년, 어느 해, and a period word (해, 연도, 시기, 시점, 달, 분기, 때) that a
# describing word ends in ("가장 많았던 해", "높은 연도와", "증가한 달은"), whatever particle follows.
ASKS_PERIOD = re.compile(
    r"언제|몇\s*년(?!\s*(?:간|동안|째|치))|어느\s*(?:해|연도|년도|달|시기|시점|분기)|연도|(?<!\d)년도|시점은"
    r"|(?:(?<=[가-힣][던은는운된한든진])|(?<=큰))\s*(?:해|달|때|분기|시기|시점)(?=[은는이가와과랑의를을에도로]|\s|$|[?,.])")
# A word that makes a question a ranking without 가장 / 최고 ("세 번째로 많은", "2위 지역", "상위 3개", "순위").
RANK_WORDS = re.compile(r"상위|하위|순위|번\s*째|(?:첫|둘|셋|넷|다섯)\s*째|\d\s*(?:위|등)(?!급|분)")



# Loop 15: an ordinal word ('첫째', '둘째', '다섯째'), built from the count words above, at the start of a name
# ('둘째아', '넷째아 이상'): group 1 the ordinal.
ORDINAL_NAME = re.compile(r"^((?:" + "|".join(sorted(_COUNT_WORDS, key=len, reverse=True)) + r")째)")


def count_of(word: str) -> int:
    return int(word) if word.isdigit() else _COUNT_WORDS[word]


def count_words_of(n: int) -> tuple[str, ...]:
    """Loop 16: the native Korean number words for n (the count words above: 2 -> '두', '둘'), none for a number they do
    not hold. Not for 10 and up: '열' starts the numbers after it ('열세', '열두'), so '열 + a counting word' is no
    reading of a name of 10."""
    return tuple(w for w, k in _COUNT_WORDS.items() if k == n and k < 10)


def rank_places(question: str) -> tuple[str, int] | None:
    """("places", n) for the first n places of a ranking, ("place", n) for the n-th one alone, None for the first
    place only (the plain extreme question)."""
    if m := RANK_TOP.search(question):
        n = count_of(next(g for g in m.groups() if g))
        return ("places", n) if n >= 2 else None
    if m := RANK_PLACE.search(question):
        n = count_of(next(g for g in m.groups() if g))
        return ("place", n) if n >= 2 else None
    return None
# The largest end of a ranking, beside the smallest one (analyze/roles.py pick_of reads the smallest): "가장 많은 해와
# 가장 적은 해의 차이", "최댓값과 최솟값".
EXTREME_MAX = re.compile(r"많|높|최대|최댓|최고|크|큰|1위")
# A change downward (줄다, 감소, 떨어지다, 하락, 내려가다, 축소): a ranking of such changes puts the most negative first.
CHANGE_DOWN = re.compile(r"줄었|줄어|줄은|감소|떨어|하락|내려|내렸|축소")
# Any ranking word (the judge reads it with ASKS_PERIOD: a ranked period is the answer, not a period to assume).
RANKING = re.compile(r"가장|제일|최고|최저|최대|최소|최댓|최솟|꼴찌|" + RANK_WORDS.pattern)
# "가장 최근 ~" names a period, not a ranking (was analyze/rules.py MOST_RECENT).
MOST_RECENT = re.compile(r"(?:가장|제일)\s*최근")

# Loop 12: the words that pick one of a kind ("어느 산업", "어떤 연령대", "무슨 업종").
WHICH_OF = re.compile(r"(?<![가-힣])(?:어느|어떤|무슨)\s+")
_WORDS_AFTER = re.compile(r"[가-힣]+")


def _adnominal(word: str) -> bool:
    """A describing word that ends with ㄴ or 던 ("많은", "큰", "높았던", "늘어난"): it modifies the noun after it."""
    last = word[-1]
    return last == "던" or ("가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 == 4)


def candidate_nouns(question: str) -> list[str]:
    """Loop 12: the nouns a ranking question names its candidates by, squashed with what follows them in their phrase:
    the noun a ranking word's describing word modifies ("가장 많은 산업은" -> '산업은...', "제일 적은 대표자 연령대는" ->
    '대표자연령대는...'; the describing word within two words after the ranking word) and the noun after a word that
    picks one of a kind ("어느 산업의" -> '산업의...'). A noun elsewhere in the question (a condition: "산업 구분 없이")
    is not one."""
    out: list[str] = []
    for m in RANKING.finditer(question):
        if MOST_RECENT.match(question, m.start()):
            continue
        words = list(_WORDS_AFTER.finditer(question, m.end()))
        for k, w in enumerate(words[:3]):
            if _adnominal(w.group(0)) and k + 1 < len(words):
                out.append(re.sub(r"\s+", "", question[words[k + 1].start():])[:16])
                break
    for m in WHICH_OF.finditer(question):
        out.append(re.sub(r"\s+", "", question[m.end():])[:16])
    return out
