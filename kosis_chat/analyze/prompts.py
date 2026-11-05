"""The analyze stage's prompts, kept as .md files in analyze/prompts/ (T1-28 S3b, D15).

A prompt's version is its file name without .md (analyze-fill-v2, analyze-plan-v1, event-lookup-v1): the name
decision records and run manifests carry. Files are read once, through importlib.resources, so they load the same
from the source folder (PYTHONPATH), an editable install and a wheel (pyproject.toml package-data). The text is the
file's text with its one final newline dropped (the file ends with a newline; the prompt does not).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

FILL = "analyze-fill-v2.md"  # analyze/fill.py: the model fills what the rules left open
FILL_RETRY = "analyze-fill-v2-retry.md"
PLAN = "analyze-plan-v1.md"  # analyze/write.py: the model writes the plan as v3 blocks
PLAN_EXAMPLES = "analyze-plan-v1-examples.md"
PLAN_RETRY = "analyze-plan-v1-retry.md"
EVENT_LOOKUP = "event-lookup-v1.md"  # analyze/event_lookup.py: when an event the calendar lacks began


def version(file: str) -> str:
    """The prompt version a file holds: its name without .md."""
    return file.removesuffix(".md")


@cache
def load(file: str) -> str:
    """The prompt text of analyze/prompts/<file> (its final newline dropped)."""
    text = resources.files(__package__).joinpath("prompts", file).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text
