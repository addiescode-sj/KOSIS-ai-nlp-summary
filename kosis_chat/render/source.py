"""The answer's source: agency and table identity from the table meta, a KOSIS URL or the Parquet file name.

T1-28 S3a: moved here from contract.py (to_source, org_id_from_table_path; contract.py re-exports them by name).
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from kosis_chat.prepare.schemas import TableMeta
from kosis_chat.shared.patterns import ORG_ID

from kosis_ai.schemas import Source  # isort: skip

# The Parquet file name '101_DT_1B8000F.parquet': the shared ORG_ID, then the table id (T1-28 S2b; same pattern text).
_ORG_FILE = re.compile(rf"({ORG_ID.pattern})_(.+)\.parquet")


def _kosis_host(host: str | None) -> bool:
    host = (host or "").lower()
    return host == "kosis.kr" or host.endswith(".kosis.kr")


def org_id_from_table_path(path: str | Path, tbl_id: str | None = None) -> str | None:
    """Read an explicit agency identifier from the documented Parquet filename convention."""
    match = _ORG_FILE.fullmatch(Path(path).name)
    if match is None:
        return None
    if tbl_id is not None and match.group(2) != tbl_id:
        raise ValueError("table filename and metadata tbl_id conflict")
    return match.group(1)


def to_source(meta: TableMeta, org_id: str | None = None) -> Source:
    """Resolve agency identity from a KOSIS URL or an explicit trusted table identifier."""
    if org_id is not None and (not isinstance(org_id, str) or ORG_ID.fullmatch(org_id) is None):
        raise ValueError("org_id must be a three-digit identifier")
    url = meta.source_url or ""
    url_org = None
    trusted_url = False
    if url:
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or re.search(r"\s", url):
                raise ValueError("source URL must be a valid HTTP(S) URL")
            if parsed.username or parsed.password:
                raise ValueError("source URL cannot contain user information")
            parsed.port  # Validate malformed port strings without opening the URL.
            query = parse_qs(parsed.query, keep_blank_values=True)
        except ValueError as exc:
            raise ValueError(f"invalid source URL: {exc}") from exc
        trusted_url = _kosis_host(parsed.hostname)
        if "orgId" in query:
            values = query["orgId"]
            if len(values) != 1 or ORG_ID.fullmatch(values[0]) is None:
                raise ValueError("source URL needs one numeric orgId")
            url_org = values[0]
        if "tblId" in query and (len(query["tblId"]) != 1 or query["tblId"][0] != meta.tbl_id):
            raise ValueError("source URL tblId conflicts with table metadata")
    if org_id is not None and url_org is not None and org_id != url_org:
        raise ValueError("explicit org_id conflicts with source URL orgId")
    resolved_org = org_id or (url_org if trusted_url else None)
    if resolved_org is None:
        raise ValueError("org_id is required when a trusted KOSIS URL has no agency identifier")
    return Source(org_id=resolved_org, tbl_id=meta.tbl_id, survey_nm=meta.survey_nm, tbl_nm=meta.tbl_nm, url=url or None)
