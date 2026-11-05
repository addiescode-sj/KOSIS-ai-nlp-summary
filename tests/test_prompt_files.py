"""The prompt .md files hold the exact text sent to the model (T1-28 S3b, D15), so an editor must not change them.

Each file is read as text (universal newlines, so a CRLF checkout reads the same): it ends in exactly one newline,
which the loader drops, and no line has trailing whitespace. This stays after tests/test_prompts.py (the move check
against the old Python constants) is deleted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "kosis_chat"
FILES = sorted([*(ROOT / "analyze" / "prompts").glob("*.md"), *(ROOT / "compose" / "prompts").glob("*.md")])


def test_there_are_prompt_files():
    assert len(FILES) == 9


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_prompt_file_is_clean(path):
    text = path.read_text(encoding="utf-8")
    assert not text.startswith("﻿"), "BOM"
    assert text.endswith("\n") and not text.endswith("\n\n"), "exactly one final newline"
    assert all(line == line.rstrip() for line in text[:-1].split("\n")), "trailing whitespace"
