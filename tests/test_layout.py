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
OLD_PATH_EXCEPTIONS: set[tuple[str, str]] = set()  # T1-28 S6: graph/legacy.py imports kosis_chat.legacy.chatbot now
# Top-level modules held to the old-path rule too (T1-28 S6), with the old paths each may still import and why.
TOP_LEVEL_NEW = {
    "kosis_chat/api.py": (),
    "kosis_chat/cli.py": (),
    # pipeline.VERSION is kosis_chat.llm.compose.VERSION, the object Phase 2 scripts/stat_baseline.py reads.
    "kosis_chat/pipeline.py": ("kosis_chat.llm.compose",),
}
# The one graph / stage file that may import kosis_chat.legacy (plans/T1-28.md S6 Done): the legacy Answer adapter,
# which also hands the fallback node (legacy/fallback.py) to graph/build.py and graph/adapters.py.
LEGACY_DOOR = "kosis_chat/graph/legacy.py"
# Packages that must not import kosis_chat.legacy (beside LEGACY_DOOR): the graph, the stages and what they build on.
NO_LEGACY = (*STAGES, "schemas", "shared", "adapters")
# B0 modules moved to kosis_chat/legacy/ in T1-28 S6: old path -> new module. The old files are re-export shims.
B0_MOVES = {
    "kosis_chat/chatbot.py": "kosis_chat.legacy.chatbot",
    "kosis_chat/raw.py": "kosis_chat.legacy.raw",
    "kosis_chat/data/scope.py": "kosis_chat.legacy.scope",
    "kosis_chat/data/serializer.py": "kosis_chat.legacy.serializer",
    "kosis_chat/llm/prompts.py": "kosis_chat.legacy.prompts",
    "kosis_chat/llm/raw_prompt.py": "kosis_chat.legacy.raw_prompt",
    "kosis_chat/llm/answer_schema.py": "kosis_chat.legacy.answer_schema",
    "kosis_chat/llm/compact_v2.py": "kosis_chat.legacy.compact_v2",
    "kosis_chat/contract.py": "kosis_chat.legacy.contract_text",
}
NEW_PACKAGES = ("schemas", "graph", "shared", "prepare", "analyze", "compose", "chart", "render", "adapters", "legacy")
# Old module paths that new code must not import (they become re-export shims as files move).
OLD_PATHS = ("kosis_chat.data", "kosis_chat.answer", "kosis_chat.contract", "kosis_chat.chatbot", "kosis_chat.raw",
             "kosis_chat.llm.compose", "kosis_chat.llm.analyze_fill", "kosis_chat.llm.analyze_plan",
             "kosis_chat.llm.event_lookup", "kosis_chat.compute.analyze", "kosis_chat.compute.build", "kosis_chat.compute.roles",
             "kosis_chat.judge.answerability", "kosis_chat.judge.messages",
             # T1-28 S6: the B0 modules moved to kosis_chat/legacy/
             "kosis_chat.llm.prompts", "kosis_chat.llm.raw_prompt", "kosis_chat.llm.answer_schema",
             "kosis_chat.llm.compact_v2")
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
    if top in NEW_PACKAGES or rel in TOP_LEVEL_NEW:  # legacy/ too since T1-28 S6 (it imports the new paths)
        allowed = tuple(old for file, old in OLD_PATH_EXCEPTIONS if file == rel) + TOP_LEVEL_NEW.get(rel, ())
        problems += [n for n in names if _hits(n, OLD_PATHS) and not _hits(n, allowed)]
    if top in NO_LEGACY and rel != LEGACY_DOOR:
        problems += [n for n in names if _hits(n, ("kosis_chat.legacy",))]
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


def test_legacy_is_reached_only_through_graph_legacy():
    """T1-28 S6 Done: no graph or stage file imports kosis_chat/legacy/ except graph/legacy.py (the Answer adapter,
    which hands the fallback node on to graph/build.py and graph/adapters.py)."""
    users = sorted(_rel(p) for p in ALL_FILES
                   if any(_hits(n, ("kosis_chat.legacy",)) for n in _imports(p)) and _rel(p).split("/")[1] in NO_LEGACY)
    assert users == [LEGACY_DOOR], users
    assert OLD_PATH_EXCEPTIONS == set()


@pytest.mark.parametrize("old,new", sorted(B0_MOVES.items()), ids=sorted(B0_MOVES))
def test_b0_old_paths_are_reexport_shims(old: str, new: str):
    """The old B0 paths only re-export (T1-28 S6): imports and the docstring, no code; the moved module is imported."""
    tree = ast.parse((ROOT / old).read_text(encoding="utf-8"))
    body = tree.body[1:] if ast.get_docstring(tree) is not None else tree.body
    code = [type(node).__name__ for node in body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not code, f"{old} holds code: {code}"
    assert new in _imports(ROOT / old), f"{old} does not import {new}"
    assert (ROOT / (new.replace(".", "/") + ".py")).is_file()
