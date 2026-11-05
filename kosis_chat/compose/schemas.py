"""What the compose stage reads from the model (T1-28 S1b): the compose-v2 reply.

ComposeOutput moved here unchanged from llm/compose.py (it was Pydantic already: the structured-output schema the
model is asked for); llm/compose.py re-exports it. The class name is the schema title (model_json_schema) and the
name in validation events (llm/structured.py), so it stays ComposeOutput. Caveat is the shared contract model
(kosis_ai.schemas.Caveat) and stays there.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ComposeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    answer: str = Field(min_length=1, description="{f번호}를 넣은 답변 문장")
