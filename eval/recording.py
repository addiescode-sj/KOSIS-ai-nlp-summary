"""Durable, allowlisted evaluation records. No credentials or HTTP headers are recorded."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from kosis_chat.llm.model import HCXClient, LLMClient, TraceWriteError

AUDIT_VERSION = "eval-audit-v1"


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_snapshot(root: Path) -> dict:
    paths = [*root.glob("kosis_chat/**/*.py"), *root.glob("eval/*.py"), root / "src/kosis_ai/schemas.py"]
    snapshot = {}
    for path in sorted(set(paths)):
        if path.is_file() and path.resolve().is_relative_to(root.resolve()):
            raw = path.read_bytes()
            snapshot[path.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(raw).hexdigest(), "utf8_text": raw.decode("utf-8")}
    return snapshot


def durable_text(path: Path, text: str, mode: str = "w") -> None:
    try:
        with path.open(mode, encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except (OSError, UnicodeError) as exc:
        raise TraceWriteError(f"cannot persist audit artifact {path.name}") from exc


def durable_json(path: Path, value: Any, mode: str = "w") -> None:
    durable_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n", mode)


class RunRecorder:
    def __init__(self, directory: Path):
        self.directory = directory
        self.sequence = 0
        self.context: dict[str, Any] = {}
        self.question_events: list[str] = []
        self.rows = 0
        self.redactor = None
        durable_text(directory / "events.jsonl", "", "x")
        durable_text(directory / "results.jsonl", "", "x")

    def clean(self, value):
        cleaned = self.redactor(value) if self.redactor is not None else value
        if isinstance(cleaned, dict) and cleaned != value:
            cleaned = {**cleaned, "credential_redacted": True}
        return cleaned

    def initialize(self, manifest: dict, items: list[dict], form: str | None, root: Path, system: str, schema: dict | None) -> None:
        snapshot = source_snapshot(root)
        fingerprints = {path: value["sha256"] for path, value in snapshot.items()}
        golden_text = self.clean("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
        safe_system = self.clean(system)
        safe_form = self.clean(form)
        manifest.update({"golden_snapshot_sha256": digest_text(golden_text),
                         "golden_snapshot_representation": "normalized merged JSONL; golden_sha256 is the original input bytes hash",
                         "schema_version": AUDIT_VERSION, "status": "running", "recorded_rows": 0,
                         "expected_rows": manifest["questions"] * manifest["runs"],
                         "source_fingerprints": fingerprints, "prompt_sha256": digest_text(system),
                         "prompt_snapshot_sha256": digest_text(safe_system),
                         "form_snapshot_sha256": digest_text(safe_form) if safe_form is not None else None,
                         "credential_redaction_policy": "known HCX credential values; original fingerprints may differ from redacted snapshot bytes",
                         "response_schema_sha256": digest_text(json.dumps(schema, ensure_ascii=False, sort_keys=True)) if schema else None,
                         "source_provenance": "disk snapshot before calls; not proof of historical loaded modules"})
        safe_snapshot = self.clean(snapshot)
        for value in safe_snapshot.values():
            if isinstance(value, dict) and "utf8_text" in value:
                value["snapshot_text_sha256"] = digest_text(value["utf8_text"])
        durable_json(self.directory / "source_snapshot.json", safe_snapshot, "x")
        durable_text(self.directory / "golden.jsonl", golden_text, "x")
        if form is not None:
            durable_text(self.directory / "form.txt", safe_form, "x")
        durable_json(self.directory / "input_manifest.json", self.clean({**manifest, "system_prompt": safe_system, "response_schema": schema}), "x")
        self.status(manifest)
        self.event({"type": "run_started", "expected_rows": manifest["expected_rows"]})

    def status(self, manifest: dict) -> None:
        manifest["recorded_rows"] = self.rows
        self.event({"type": "run_status", "status": manifest["status"], "recorded_rows": self.rows})
        durable_json(self.directory / "manifest.json", self.clean(manifest))

    def begin_question(self, question_id: str, run_index: int) -> None:
        self.context = {"question_id": question_id, "run_index": run_index}
        self.question_events = []
        self.event({"type": "question_started"})

    def event(self, event: dict) -> str:
        self.sequence += 1
        identity = f"E{self.sequence:07d}"
        value = {"event_id": identity, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), **self.context, **event}
        durable_text(self.directory / "events.jsonl", json.dumps(self.clean(value), ensure_ascii=False) + "\n", "a")
        if "question_id" in self.context:
            self.question_events.append(identity)
        return identity

    def append_result(self, row: dict) -> None:
        row["event_ids"] = list(self.question_events)
        durable_text(self.directory / "results.jsonl", json.dumps(self.clean(row), ensure_ascii=False) + "\n", "a")
        self.rows += 1
        self.event({"type": "question_completed", "call_status": row["call_status"], "recorded_rows": self.rows})


class RecordingClient(LLMClient):
    """Journal complete generate calls; only HCXClient supplies actual HTTP-attempt events."""

    def __init__(self, client: LLMClient, recorder: RunRecorder):
        self.client = client
        if isinstance(client, HCXClient):
            recorder.redactor = client._redact
        self.recorder = recorder
        self.name = client.name
        self.calls = 0
        self.current_generate_id: str | None = None
        self.original_observer = None
        self.attached = False

    @property
    def transport_trace(self) -> str:
        return "hcx_http" if isinstance(self.client, HCXClient) else "unavailable_for_client"

    def attach(self) -> None:
        if isinstance(self.client, HCXClient):
            self.original_observer = self.client.observer
            self.client.observer = self._http_event
            self.attached = True

    def restore(self) -> None:
        if self.attached:
            self.client.observer = self.original_observer
            self.attached = False

    def _http_event(self, event: dict) -> None:
        self.record_event(event)
        if self.original_observer is not None:
            self.original_observer(event)

    def record_event(self, event: dict) -> None:
        self.recorder.event({**event, "generate_id": self.current_generate_id})

    def generate(self, system: str, user: str):
        self.calls += 1
        self.current_generate_id = f"G{self.calls:07d}"
        self.record_event({"type": "generate_request", "system": system, "user": user,
                           "transport_trace": self.transport_trace})
        started = time.perf_counter()
        try:
            result = self.client.generate(system, user)
        except BaseException as exc:
            self.record_event({"type": "generate_raised", "exception_type": type(exc).__name__, "error": str(exc)})
            cleaned = self.recorder.clean(str(exc))
            if isinstance(exc, Exception) and cleaned != str(exc):
                if isinstance(exc, TimeoutError):
                    raise TimeoutError(cleaned) from None
                if isinstance(exc, TraceWriteError):
                    raise TraceWriteError(cleaned) from None
                raise RuntimeError(f"{type(exc).__name__}: {cleaned}") from None
            raise
        self.record_event({"type": "generate_response", "text": result.text, "call_status": result.status,
                           "error": result.error, "usage": result.usage, "attempts": result.attempts,
                           "latency_seconds": time.perf_counter() - started})
        return result
