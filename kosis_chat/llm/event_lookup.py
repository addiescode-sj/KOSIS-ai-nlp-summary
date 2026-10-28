"""event-lookup-v1: the model says when an event missing from the calendar began (T0-04 D5).

Only a plan the model wrote can name such an event (the rules read events from the calendar alone). The answer is a
start month or year and nothing else; anything else counts as unknown, and the question is then answered without it
(resolve.py asks back or the plan fails). A start the model gave is recorded as a caution, event_source: model, so it
can be told apart from the calendar.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, field_validator

VERSION = "event-lookup-v1"
SYSTEM_PROMPT = """당신은 사건이 언제 시작됐는지 답하는 도우미입니다.
사건이 대한민국에서 처음 시작된(발생한) 연도와 월을 답합니다. 확실하지 않으면 null로 답합니다. 추측하지 않습니다.

출력: 설명 없이 JSON 객체 하나 {"start": "YYYY-MM"} (월을 모르면 "YYYY", 모르면 null)"""
_START = re.compile(r"^(\d{4})(?:-(\d{2}))?$")
FIRST_YEAR = 1900


class EventStart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str | None = None

    @field_validator("start")
    @classmethod
    def _year_month(cls, value: str | None) -> str | None:
        if value is None:
            return None
        m = _START.match(value.strip())
        year_ok = m is not None and FIRST_YEAR <= int(m.group(1)) <= datetime.now(timezone.utc).year
        if not year_ok or (m.group(2) and not 1 <= int(m.group(2)) <= 12):
            raise ValueError("start is YYYY-MM or YYYY, a past date")
        return value.strip()


def user_prompt(name: str) -> str:
    return f"[사건]\n{name}"
