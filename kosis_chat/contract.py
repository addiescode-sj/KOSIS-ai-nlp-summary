"""Re-export shim (T1-28 S6): the B0 text handling (strip_model_source, render_text, to_stat_answer) moved to
kosis_chat/legacy/contract_text.py; the source helpers were moved to kosis_chat/render/source.py in S3a.

Every name the old module had is re-exported by name (private ones too: `import *` would drop them), the same
objects, so callers of the old path (tests, eval, Phase 2 evals/trace.py) need no change."""

import re  # noqa: F401
import sys  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import TYPE_CHECKING  # noqa: F401
from urllib.parse import parse_qs, urlsplit  # noqa: F401

from kosis_chat.config import PROJECT_ROOT  # noqa: F401
from kosis_chat.legacy.contract_text import (  # noqa: F401
    _is_kosis_link_line,
    Refusal,
    render_text,
    _SEPARATOR,
    Source,
    _SOURCE_LINE,
    _SOURCE_LINK,
    StatAnswer,
    StatQuery,
    strip_model_source,
    TextPart,
    to_stat_answer,
)
from kosis_chat.prepare.schemas import TableMeta  # noqa: F401
from kosis_chat.render.source import _ORG_FILE, _kosis_host, org_id_from_table_path, to_source  # noqa: F401
from kosis_chat.shared.patterns import ORG_ID  # noqa: F401
