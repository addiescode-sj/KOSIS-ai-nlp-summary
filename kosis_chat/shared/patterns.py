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
