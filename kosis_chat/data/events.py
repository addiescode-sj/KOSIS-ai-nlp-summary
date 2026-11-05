"""Old path of the event calendar: moved to kosis_chat/shared/events.py (+ events.json) in T1-28 S2 (re-exported by name)."""

from __future__ import annotations

from kosis_chat.shared.events import EVENTS_PATH, event_key, load_events

__all__ = ["EVENTS_PATH", "event_key", "load_events"]
