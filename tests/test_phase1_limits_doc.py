"""docs/phase1_limits.md only cites what exists: every repo path it names, and every number in a sourced table is in that source."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "phase1_limits.md"
# Paths the doc names in backticks; gitignored golden files are named but not checked.
CODE_PATH = re.compile(r"`((?:plans|eval|kosis_chat|src|docs|tests)/[\w./-]+\.\w+)`")
LINK = re.compile(r"\]\(([^)#]+)\)")
UNTRACKED = ("eval/golden/",)

offline = pytest.mark.offline


def cited_paths(doc: str) -> set[str]:
    """Repo-relative paths named in backticks or as relative links from docs/."""
    paths = {p for p in CODE_PATH.findall(doc) if not p.startswith(UNTRACKED)}
    for target in LINK.findall(doc):
        if "://" not in target:
            paths.add((Path("docs") / target).as_posix() if not target.startswith("../") else target[3:])
    return paths


def _squash(text: str) -> str:
    return re.sub(r"\s+|\*\*", "", text)


def sourced_rows(doc: str) -> list[tuple[list[str], str]]:
    """(value cells, source path) for each row of a table whose last header cell is 출처."""
    rows, in_sourced = [], False
    for line in doc.splitlines():
        if not line.startswith("|"):
            in_sourced = False
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[-1] == "출처":
            in_sourced = True
        elif in_sourced and not set(cells[0]) <= set("- "):
            link = LINK.search(cells[-1])
            rows.append((cells[1:-1], link.group(1)[3:] if link else ""))
    return rows


def unmatched_numbers(doc: str) -> list[str]:
    """Each '→'-separated piece with a digit must appear in its source file (bold and spaces ignored)."""
    problems = []
    for values, source in sourced_rows(doc):
        if not (ROOT / source).is_file():
            problems.append(f"no source file: {source!r}")
            continue
        text = _squash((ROOT / source).read_text(encoding="utf-8"))
        for cell in values:
            for piece in cell.split("→"):
                if re.search(r"\d", piece) and _squash(piece) not in text:
                    problems.append(f"{piece.strip()!r} not in {source}")
    return problems


@offline
def test_cited_paths_exist():
    missing = sorted(p for p in cited_paths(DOC_PATH.read_text(encoding="utf-8")) if not (ROOT / p).exists())
    assert missing == []


@offline
def test_sourced_numbers_are_in_their_source():
    assert unmatched_numbers(DOC_PATH.read_text(encoding="utf-8")) == []


@offline
def test_answers_all_mentor_questions():
    headings = re.findall(r"^## \d+\. (.+)$", DOC_PATH.read_text(encoding="utf-8"), re.M)
    for topic in ("아키텍처", "스키마 필드", "골든셋 분포", "메트릭", "실패 유형별 건수", "개선 우선순위"):
        assert topic in headings


@offline
@pytest.mark.parametrize(
    "old, new",
    [
        ("`kosis_chat/compute/engine.py`", "`kosis_chat/compute/engin.py`"),  # typo in a path
        ("| 17/90 → 0/90 |", "| 17/90 → 1/90 |"),  # a copied number off by one
        ("| 26/28 | [plans/T1-23.md](../plans/T1-23.md)", "| 26/28 | [plans/T1-22.md](../plans/T1-22.md)"),  # wrong plan
        ("(../eval/reports/20261026-115038-T1-25-oracle-all/", "(../eval/reports/20261026-115038-T1-25-oracle-full/"),  # no such report
    ],
)
def test_injected_errors_are_caught(old, new):
    doc = DOC_PATH.read_text(encoding="utf-8")
    assert old in doc
    broken = doc.replace(old, new, 1)
    assert [p for p in cited_paths(broken) if not (ROOT / p).exists()] or unmatched_numbers(broken)
