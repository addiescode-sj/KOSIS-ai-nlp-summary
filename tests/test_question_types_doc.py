"""docs/question_types.md stays in line with its sources: schema codes, golden section refs and the example numbers."""

import json
import re
from pathlib import Path

import pytest

from kosis_ai.normalize import fetch_table
from kosis_ai.schemas import REFUSAL_REASONS

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "question_types.md").read_text(encoding="utf-8")
GOLDEN = [ROOT / "eval" / "golden" / name for name in ("stat_golden.jsonl", "stat_golden_edge.jsonl")]
VALUES = ROOT / "data" / "values"

offline = pytest.mark.offline
needs_data = pytest.mark.skipif(not VALUES.exists(), reason="provided data is not in the repo (gitignored); place it at data/")


def _section(heading: str) -> str:
    start = DOC.index(heading)
    end = DOC.find("\n#", start + len(heading))
    return DOC[start:] if end == -1 else DOC[start:end]


@offline
def test_reason_code_table_matches_schema():
    rows = [line for line in _section("### 2.1").splitlines() if line.startswith("| `")]
    assert sorted(re.match(r"\| `(\w+)`", row).group(1) for row in rows) == sorted(REFUSAL_REASONS)


@offline
def test_golden_section_refs_exist_in_doc():
    present = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)?) ", DOC, flags=re.M))
    for path in GOLDEN:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            for ref in re.findall(r"question_types §([\d.]+)", line):
                assert ref in present, f"{json.loads(line)['id']} cites §{ref}, which the doc does not have"


# --- example numbers: recompute each from Parquet and check the doc prints the same text ---------------------


def cell(table: str, period: str, **filters):
    df = fetch_table(table, {"periods": [period], **filters})
    assert len(df) == 1, (table, period, filters, len(df))
    row = df.iloc[0]
    return row.value if row.value_kind == "ok" else row.value_raw


def periods(table: str, **filters) -> set[str]:
    return set(fetch_table(table, filters or None)["period"])


F, H, J, D, M = "101_DT_1B8000F", "101_DT_1B8000H", "101_DT_1JC1502", "101_DT_1B08024", "101_DT_101094A001"
TFR_F = {"기본항목별": "합계출산율(명)"}


def tfr_h(period: str, region: str) -> float:
    return cell(H, period, item="합계출산율", 행정구역별=region)


def one_person_share(period: str, region: str) -> float:
    one = cell(J, period, item="1인 (가구)", **{"행정구역별(읍면동)": region})
    total = cell(J, period, item="일반가구_계 (가구)", **{"행정구역별(읍면동)": region})
    return round(one / total * 100, 1)


def sido_tfr_2025():
    df = fetch_table(H, {"periods": ["2025"], "item": "합계출산율"})
    return df[~df["dim1_label"].isin(["전국", "국외"])].set_index("dim1_label")["value"]


def sido_one_person_share_2025():
    df = fetch_table(J, {"periods": ["2025"]})
    df = df[(df["dim1_code"].str.len() == 2) & ~df["dim1_code"].isin(["00", "03", "04", "05"])]  # 17 sido rows
    one = df[df["item_name"] == "1인 (가구)"].set_index("dim1_label")["value"]
    total = df[df["item_name"] == "일반가구_계 (가구)"].set_index("dim1_label")["value"]
    return (one / total * 100).round(1)


def births(period: str) -> int:
    return int(cell(F, period, 기본항목별="출생아수(명)"))


# (text the doc must contain, value recomputed from Parquet, expected value)
EXAMPLES = [
    ("0.748명", lambda: cell(F, "2024", **TFR_F), 0.748),
    ("2025년 0.799명", lambda: cell(F, "2025", **TFR_F), 0.799),
    ("전국 값 0.799", lambda: tfr_h("2025", "전국"), 0.799),
    ("260,562", lambda: births("2021"), 260562),
    ("230,028", lambda: births("2023"), 230028),
    ("254,341", lambda: births("2025"), 254341),
    ("−2.4%", lambda: round((births("2025") / births("2021") - 1) * 100, 1), -2.4),
    ("0.632 vs 1.094", lambda: (tfr_h("2025", "서울특별시"), tfr_h("2025", "전라남도")), (0.632, 1.094)),
    ("0.462명", lambda: round(tfr_h("2025", "전라남도") - tfr_h("2025", "서울특별시"), 3), 0.462),
    ("서울특별시 0.632명", lambda: (sido_tfr_2025().idxmin(), sido_tfr_2025().min()), ("서울특별시", 0.632)),
    ("전라남도 1.094", lambda: (sido_tfr_2025().idxmax(), sido_tfr_2025().max()), ("전라남도", 1.094)),
    ("8,244,308 / 22,495,096 = **36.6%", lambda: (
        cell(J, "2025", item="1인 (가구)", **{"행정구역별(읍면동)": "전국"}),
        cell(J, "2025", item="일반가구_계 (가구)", **{"행정구역별(읍면동)": "전국"}),
        one_person_share("2025", "전국"),
    ), (8244308, 22495096, 36.6)),
    ("서울특별시 40.5%", lambda: (sido_one_person_share_2025().idxmax(), sido_one_person_share_2025().max()), ("서울특별시", 40.5)),
    ("27.2% → 36.6%", lambda: (one_person_share("2015", "전국"), one_person_share("2025", "전국")), (27.2, 36.6)),
    ("+9.4%p", lambda: round(one_person_share("2025", "전국") - one_person_share("2015", "전국"), 1), 9.4),
    ("전체 38.8%", lambda: cell(M, "2023", 성별="전체", 년생별="1990"), 38.8),
    # symbols and gaps the refusal examples rely on
    ("세종은 2012년 신설(셀 `-`)", lambda: cell(H, "2010", item="출생건수 (명)", 행정구역별="세종특별자치시"), "-"),
    ("미공표 `-`", lambda: cell(F, "2025", 기본항목별="기대수명(출생시 기대여명)"), "-"),
    ("`X` 비밀보호", lambda: cell(J, "2024", item="6인 이상 (가구)", **{"행정구역별(읍면동)": "연동면"}), "X"),
    ("2001년부터", lambda: min(periods(F, 기본항목별="영아사망률(출생아 천명당)")), "2001"),
    ("5년 간격(1995, 2000)", lambda: sorted(p for p in periods(D) if "1994" < p < "2015"), ["1995", "2000", "2005", "2010"]),
    ("2011~2014년이 없다", lambda: {"2011", "2012", "2013", "2014"} & periods(D), set()),
]


@needs_data
@pytest.mark.parametrize(("text", "compute", "expected"), EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_doc_example_matches_parquet(text, compute, expected):
    assert text in DOC
    assert compute() == expected
