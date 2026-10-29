"""Where the work files live: data/, eval/golden, eval/results, logs and .env (plans/T1-27.md).

The code root is the folder that holds kosis_chat/ and src/. KOSIS_WORKSPACE moves the work root elsewhere, e.g.
stage 2 keeps data/ at its own repo root and imports this repo as vendor/phase1. It is read on every call, so a
value set after import (tests, scripts) is seen. Set it in the shell, not in .env: the .env location depends on it.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
ENV_VAR = "KOSIS_WORKSPACE"


def workspace_root() -> Path:
    """KOSIS_WORKSPACE as an absolute path, or the code root when it is unset."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value).expanduser().resolve() if value else CODE_ROOT


def data_dir() -> Path:
    return workspace_root() / "data"
