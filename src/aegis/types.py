from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ThreatCategory(str, Enum):
    BENIGN = "benign"
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    DATA_EXFILTRATION = "data_exfiltration"


class ThreatLevel(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class InspectionDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"


class StageTimings(BaseModel):
    canonicalize_ms: float = 0.0
    prefilter_ms: float = 0.0
    windowing_ms: float = 0.0
    queue_wait_ms: float = 0.0
    onnx_ms: float = 0.0
    total_ms: float = 0.0


class InspectionRequest(BaseModel):
    prompt: str = Field(..., description="Raw user prompt or payload to inspect")
    request_id: str | None = Field(default=None, description="Unique correlation identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Contextual telemetry metadata"
    )


class InspectionResult(BaseModel):
    decision: InspectionDecision
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    matched_rules: list[str] = Field(default_factory=list)
    request_id: str | None = None
    s_mal: float = Field(default=0.0, ge=0.0, le=1.0)
    class_probs: tuple[float, float, float] = (1.0, 0.0, 0.0)
    anomalies: list[str] = Field(default_factory=list)
    fast_blocked: bool = False
    windows_inspected: int = 0
    backend: str = "baseline"
    timings: StageTimings = Field(default_factory=StageTimings)
