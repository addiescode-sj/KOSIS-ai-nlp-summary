"""Base of the node-boundary models (T1-28 S1b): Pydantic models that behave like the dataclasses they replaced.

The stage schemas (judge/ analyze/ compute/ compose/ schemas.py) moved from dataclasses to Pydantic without changing
what any answer holds (mock results.jsonl byte for byte), so the base keeps the dataclass behaviour the code relies on:
- extra="forbid": an unknown keyword raises, as an unknown dataclass argument did.
- strict: scalar fields (str, bool, int, Literal) are checked, never coerced (no 100 -> 100.0, no True -> 1).
- containers and objects are annotated `SkipValidation[...]` (or `Any` where the type lives in a module the stage
  must not import): stored as passed, the same object and never a copy, as the dataclass stored them.
- positional arguments fill the fields in declaration order, as the dataclass __init__ did (`Match(axis, code, ...)`).
- `dataclasses.replace(x, ...)` becomes `x.model_copy(update={...})`: a shallow copy without validation, as replace
  did for these classes (none of them had a __post_init__).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class BoundaryModel(BaseModel):
    model_config = ConfigDict(strict=True, arbitrary_types_allowed=True, extra="forbid")

    def __init__(self, /, *args: Any, **data: Any) -> None:
        if args:
            names = list(type(self).model_fields)
            if len(args) > len(names):
                raise TypeError(f"{type(self).__name__}() takes {len(names)} positional arguments but {len(args)} were given")
            for name, value in zip(names, args):
                if name in data:
                    raise TypeError(f"{type(self).__name__}() got multiple values for argument {name!r}")
                data[name] = value
        super().__init__(**data)
