from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AegisConfig(BaseModel):
    block_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    suspicious_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_prompt_length: int = Field(default=16384, ge=1)
    latency_budget_ms: float = Field(default=15.0, gt=0.0)
    enable_normalization: bool = Field(default=True)
    engine_name: str = Field(default="aegis_pipeline")


class ProxySettings(BaseModel):
    mode: Literal["enforce", "shadow"] = Field(default="enforce")
    upstream_url: str = Field(default="http://127.0.0.1:8001")
    upstream_timeout_s: float = Field(default=60.0, gt=0.0)
    model_dir: Path = Field(default=Path("models/deberta_v3_small_aegis_v1"))
    fallback_joblib_path: Path | None = Field(
        default=Path("models/baseline_tfidf_linearsvc.joblib")
    )
    fallback_metrics_path: Path | None = Field(
        default=Path("reports/baseline_tier1_metrics.json")
    )
    allow_fallback: bool = Field(default=True)
    block_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    suspicious_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    fast_block_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_window_tokens: int = Field(default=256, ge=32, le=512)
    max_windows_per_segment: int = Field(default=4, ge=1, le=8)
    last_k_user_turns: int = Field(default=3, ge=1, le=16)
    max_batch_size: int = Field(default=4, ge=1, le=32)
    max_queue_size: int = Field(default=128, ge=1, le=4096)
    coalesce_ms: float = Field(default=1.0, ge=0.0, le=20.0)
    intra_op_threads: int | None = Field(default=None, ge=1, le=64)
    audit_format: Literal["json", "cef"] = Field(default="json")
    audit_log_path: Path | None = Field(default=None)
    log_raw_prompt: bool = Field(default=False)

    @classmethod
    def from_env(cls) -> ProxySettings:
        mode_raw = os.environ.get("AEGIS_MODE", "enforce").strip().lower()
        mode: Literal["enforce", "shadow"] = "shadow" if mode_raw == "shadow" else "enforce"

        audit_fmt_raw = os.environ.get("AEGIS_AUDIT_FORMAT", "json").strip().lower()
        audit_fmt: Literal["json", "cef"] = "cef" if audit_fmt_raw == "cef" else "json"

        block_thr_env = os.environ.get("AEGIS_BLOCK_THRESHOLD")
        block_thr = float(block_thr_env) if block_thr_env else None

        intra_env = os.environ.get("AEGIS_INTRA_OP_THREADS")
        intra_threads = int(intra_env) if intra_env and intra_env.isdigit() else None

        audit_path_env = os.environ.get("AEGIS_AUDIT_LOG_PATH")
        audit_path = Path(audit_path_env) if audit_path_env else None

        return cls(
            mode=mode,
            upstream_url=os.environ.get("AEGIS_UPSTREAM_URL", "http://127.0.0.1:8001"),
            upstream_timeout_s=float(os.environ.get("AEGIS_UPSTREAM_TIMEOUT_S", "60.0")),
            model_dir=Path(
                os.environ.get("AEGIS_MODEL_DIR", "models/deberta_v3_small_aegis_v1")
            ),
            allow_fallback=os.environ.get("AEGIS_ALLOW_FALLBACK", "true").lower()
            in ("1", "true", "yes"),
            block_threshold=block_thr,
            suspicious_threshold=float(os.environ.get("AEGIS_SUSPICIOUS_THRESHOLD", "0.35")),
            fast_block_threshold=float(os.environ.get("AEGIS_FAST_BLOCK_THRESHOLD", "0.85")),
            max_window_tokens=int(os.environ.get("AEGIS_MAX_WINDOW_TOKENS", "256")),
            max_batch_size=int(os.environ.get("AEGIS_MAX_BATCH_SIZE", "4")),
            max_queue_size=int(os.environ.get("AEGIS_MAX_QUEUE_SIZE", "128")),
            coalesce_ms=float(os.environ.get("AEGIS_COALESCE_MS", "1.0")),
            intra_op_threads=intra_threads,
            audit_format=audit_fmt,
            audit_log_path=audit_path,
            log_raw_prompt=os.environ.get("AEGIS_LOG_RAW_PROMPT", "false").lower()
            in ("1", "true", "yes"),
        )
