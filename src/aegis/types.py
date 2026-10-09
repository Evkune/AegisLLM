from enum import Enum
from typing import Any, Dict, List, Optional
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


class InspectionRequest(BaseModel):
    prompt: str = Field(..., description="Raw user prompt or payload to inspect")
    request_id: Optional[str] = Field(default=None, description="Unique correlation identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Contextual telemetry metadata")


class InspectionResult(BaseModel):
    decision: InspectionDecision
    threat_level: ThreatLevel
    threat_category: ThreatCategory
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    matched_rules: List[str] = Field(default_factory=list)
    request_id: Optional[str] = None
