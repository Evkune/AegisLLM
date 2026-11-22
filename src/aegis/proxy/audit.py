from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import orjson

_logger = logging.getLogger("aegis.audit")


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _escape_cef_field(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _escape_cef_ext(value: str) -> str:
    return value.replace("\\", "\\\\").replace("=", "\\=").replace("\n", " ").replace("\r", " ")


@dataclass(slots=True)
class AuditEvent:
    request_id: str
    endpoint: str
    mode: str
    decision: str
    threat_level: str
    threat_category: str
    s_mal: float
    latency_ms: float
    prompt_sha256: str
    prompt_len: int
    fast_blocked: bool
    windows_inspected: int
    backend: str
    matched_rules: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    raw_prompt: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": self.timestamp,
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "mode": self.mode,
            "decision": self.decision,
            "threat_level": self.threat_level,
            "threat_category": self.threat_category,
            "s_mal": self.s_mal,
            "latency_ms": self.latency_ms,
            "prompt_sha256": self.prompt_sha256,
            "prompt_len": self.prompt_len,
            "fast_blocked": self.fast_blocked,
            "windows_inspected": self.windows_inspected,
            "backend": self.backend,
            "matched_rules": self.matched_rules,
            "anomalies": self.anomalies,
        }
        if self.raw_prompt is not None:
            payload["raw_prompt"] = self.raw_prompt
        return payload


def format_cef(event: AuditEvent) -> str:
    severity = 8 if event.decision == "block" else (5 if event.threat_level == "suspicious" else 2)
    sig_id = _escape_cef_field(event.threat_category)
    name = _escape_cef_field(f"AegisLLM {event.decision.upper()} ({event.threat_category})")
    rules_csv = ",".join(event.matched_rules[:8])
    ext = (
        f"rt={_escape_cef_ext(event.timestamp)} "
        f"externalId={_escape_cef_ext(event.request_id)} "
        f"request={_escape_cef_ext(event.endpoint)} "
        f"act={_escape_cef_ext(event.decision)} "
        f"cs1Label=mode cs1={_escape_cef_ext(event.mode)} "
        f"cs2Label=promptSha256 cs2={event.prompt_sha256} "
        f"cs3Label=matchedRules cs3={_escape_cef_ext(rules_csv)} "
        f"cfp1Label=sMal cfp1={event.s_mal:.4f} "
        f"cfp2Label=latencyMs cfp2={event.latency_ms:.3f} "
        f"cn1Label=promptLen cn1={event.prompt_len}"
    )
    return f"CEF:0|AegisLLM|SecurityGateway|1.0|{sig_id}|{name}|{severity}|{ext}"


class AsyncAuditLogger:
    """
    Non-blocking asynchronous SIEM audit logger supporting JSON lines and CEF.
    Hashes prompt payloads with SHA-256 by default so sensitive user data is never
    persisted in cleartext logs unless explicitly enabled.
    """

    def __init__(
        self,
        fmt: Literal["json", "cef"] = "json",
        log_path: Path | None = None,
        log_raw_prompt: bool = False,
        max_queue_size: int = 2048,
    ) -> None:
        self.fmt = fmt
        self.log_path = log_path
        self.log_raw_prompt = log_raw_prompt
        self.max_queue_size = max_queue_size
        self._queue: asyncio.Queue[AuditEvent] | None = None
        self._task: asyncio.Task[None] | None = None
        self._dropped = 0
        self._written = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._task = asyncio.create_task(self._writer_loop(), name="aegis-audit-logger")

    async def stop(self) -> None:
        if self._queue is not None:
            await self._queue.join()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def emit_nowait(self, event: AuditEvent) -> None:
        if not self.log_raw_prompt:
            event.raw_prompt = None
        if self._queue is None:
            self._write_line(self._serialize(event))
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1

    def _serialize(self, event: AuditEvent) -> str:
        if self.fmt == "cef":
            return format_cef(event)
        return orjson.dumps(event.to_dict()).decode("utf-8")

    def _write_line(self, line: str) -> None:
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        else:
            _logger.info(line)
        self._written += 1

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                line = self._serialize(event)
                self._write_line(line)
            finally:
                self._queue.task_done()
