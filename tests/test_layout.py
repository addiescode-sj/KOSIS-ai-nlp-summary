"""Folder layout rules of the T1-28 refactor (plans/T1-28.md), checked by reading imports (ast), not by importing.

Rules that hold today are enforced now; the ones the later steps make true are noted where they tighten.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAT = ROOT / "kosis_chat"
KOSIS_AI = ROOT / "src" / "kosis_ai"

# Stage packages in data-flow order: a stage may import the ones before it, never the ones after it.
STAGES = ("prepare", "judge", "analyze", "compute", "compose", "chart", "render", "graph")
# (importer, imported) pairs allowed against the order, with the reason.
STAGE_EXCEPTIONS = {
    # D12 (a): analyze_plan.place runs the model's plan through compute to report problems back to the model
    # (kosis_chat/analyze/write.py place(), moved from llm/analyze_plan.py in S3b). Kept as is in the first refactor.
    ("analyze", "compute"),
}
# (file, old path) pairs allowed against the "no old paths in new code" rule, with the reason.
OLD_PATH_EXCEPTIONS = {
    # T1-28 S3b: graph/legacy.py builds the legacy chatbot.Answer (to_answer); Answer stays in kosis_chat/chatbot.py
    # until S6 moves the chatbot to legacy/.
    ("kosis_chat/graph/legacy.py", "kosis_chat.chatbot"),
}
NEW_PACKAGES = ("schemas", "graph", "shared", "prepare", "analyze", "compose", "chart", "render", "adapters", "legacy")
# Old module paths that new code must not import (they become re-export shims as files move).
OLD_PATHS = ("kosis_chat.data", "kosis_chat.answer", "kosis_chat.contract", "kosis_chat.chatbot", "kosis_chat.raw",
             "kosis_chat.llm.compose", "kosis_chat.llm.analyze_fill", "kosis_chat.llm.analyze_plan",
             "kosis_chat.llm.event_lookup", "kosis_chat.compute.analyze", "kosis_chat.compute.build", "kosis_chat.compute.roles",
             "kosis_chat.judge.answerability", "kosis_chat.judge.messages")
# pandas / pyarrow: kosis_ai.normalize, and where T1-28 S2 moved the loader's use (adapters, prepare, schemas/io).
FRAME_LIBS = ("pandas", "pyarrow")
FRAME_ALLOWED = ("src/kosis_ai/normalize.py", "kosis_chat/adapters/",
                 "kosis_chat/prepare/convert.py", "kosis_chat/prepare/columns.py", "kosis_chat/schemas/io.py")


def _py_files(*roots: Path) -> list[Path]:
    return sorted(p for root in roots for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _module_of(path: Path) -> str:
    rel = path.relative_to(KOSIS_AI.parent if path.is_relative_to(KOSIS_AI) else ROOT).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(path: Path) -> list[str]:
    """Absolute names of every import in the file, function-level ones included."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_of(path) if path.name == "__init__.py" else _module_of(path).rpartition(".")[0]
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            names.append(module)
            names += [f"{module}.{alias.name}" for alias in node.names]
    return names


def _hits(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _stage_of(path: Path) -> str | None:
    rel = path.relative_to(ROOT).parts
    return rel[1] if len(rel) > 2 and rel[0] == "kosis_chat" and rel[1] in STAGES else None


ALL_FILES = _py_files(CHAT, KOSIS_AI)


def test_files_are_found():
    assert len(ALL_FILES) > 50


@pytest.mark.parametrize("path", ALL_FILES, ids=_rel)
def test_import_rules(path: Path):
    rel = _rel(path)
    names = _imports(path)
    problems = []
    if not rel.startswith("kosis_chat/graph/"):
        problems += [n for n in names if _hits(n, ("langgraph",))]
    if rel != "kosis_chat/llm/model.py":  # HTTP to the model; urllib.parse (contract.py) is fine
        problems += [n for n in names if _hits(n, ("urllib.request", "urllib.error"))]
    if rel.startswith("src/kosis_ai/"):
        problems += [n for n in names if _hits(n, ("kosis_chat",))]
    if not any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in FRAME_ALLOWED):
        problems += [n for n in names if _hits(n, FRAME_LIBS)]
    top = rel.split("/")[1] if rel.startswith("kosis_chat/") and rel.count("/") >= 2 else None
    if top == "shared":
        problems += [n for n in names if _hits(n, tuple(f"kosis_chat.{s}" for s in (*STAGES, "api", "pipeline")))]
    if top in NEW_PACKAGES and top != "legacy":
        allowed = tuple(old for file, old in OLD_PATH_EXCEPTIONS if file == rel)
        problems += [n for n in names if _hits(n, OLD_PATHS) and not _hits(n, allowed)]
        if top != "graph" and top != "adapters":
            problems += [n for n in names if _hits(n, ("kosis_chat.adapters",))]
    stage = _stage_of(path)
    if stage:
        later = STAGES[STAGES.index(stage) + 1:]
        problems += [n for n in names for s in later
                     if _hits(n, (f"kosis_chat.{s}",)) and (stage, s) not in STAGE_EXCEPTIONS]
    assert not problems, f"{rel}: imports break the layout rules: {sorted(set(problems))}"


def test_langgraph_only_in_graph_and_one_stage_exception():
    """T1-28 S4: the graph (kosis_chat/graph/build.py) imports langgraph, nothing outside graph/ does; the one import
    against the stage order stays analyze -> compute (D12 a, analyze/write.py place())."""
    users = [_rel(p) for p in ALL_FILES if any(_hits(n, ("langgraph",)) for n in _imports(p))]
    assert "kosis_chat/graph/build.py" in users and all(u.startswith("kosis_chat/graph/") for u in users), users
    assert STAGE_EXCEPTIONS == {("analyze", "compute")}


def test_every_package_is_in_pyproject():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"])
    found = {_module_of(p) for p in ALL_FILES if p.name == "__init__.py"}
    assert found == listed, f"missing in pyproject: {sorted(found - listed)}, not a package: {sorted(listed - found)}"


def test_new_packages_exist():
    for name in NEW_PACKAGES:
        assert (CHAT / name / "__init__.py").is_file(), name
