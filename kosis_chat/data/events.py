"""The event calendar (kosis_chat/data/events.json, T0-04 D5): events a question may anchor periods on.

Every name and alias maps to the event's start ('YYYY-MM', or 'YYYY' when the month is not one date). Names are
matched with spaces removed and case folded. The calendar is a local file for now; a database will replace it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

EVENTS_PATH = Path(__file__).resolve().parent / "events.json"


def event_key(name: str) -> str:
    return "".join(name.split()).casefold()


@lru_cache(maxsize=4)
def load_events(path: Path = EVENTS_PATH) -> dict[str, str]:
    """Every name and alias (as event_key) -> start. Do not change the returned dict: it is cached."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        for key in (name, *entry.get("aliases", [])):
            out[event_key(key)] = entry["start"]
    return out
