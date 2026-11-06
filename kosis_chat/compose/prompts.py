"""The compose stage's prompts, kept as .md files in compose/prompts/ (T1-28 S3b, D15).

A prompt's version is its file name without .md (compose-v2): the name decision records and run manifests carry.
Files are read once, through importlib.resources, so they load the same from the source folder (PYTHONPATH), an
editable install and a wheel (pyproject.toml package-data). The text is the file's text with its one final newline
dropped (the file ends with a newline; the prompt does not).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

COMPOSE = "compose-v2.md"  # compose/compose.py: the model writes the sentence around {f1} citations
COMPOSE_EXAMPLE = "compose-v2-example.md"
COMPOSE_RETRY = "compose-v2-retry.md"


def version(file: str) -> str:
    """The prompt version a file holds: its name without .md."""
    return file.removesuffix(".md")


@cache
def load(file: str) -> str:
    """The prompt text of compose/prompts/<file> (its final newline dropped)."""
    text = resources.files(__package__).joinpath("prompts", file).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text
