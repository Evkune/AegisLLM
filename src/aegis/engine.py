import re
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import List, Pattern, Tuple

from aegis.config import AegisConfig
from aegis.types import (
    InspectionDecision,
    InspectionRequest,
    InspectionResult,
    ThreatCategory,
    ThreatLevel,
)


class BaseInspectionEngine(ABC):
    def __init__(self, config: AegisConfig) -> None:
        self.config = config

    @abstractmethod
    def inspect(self, request: InspectionRequest) -> InspectionResult:
        pass


class BaselineEngine(BaseInspectionEngine):
    """
    Baseline heuristic and structural pattern inspection engine.
    Used for pre-filtering and initial phase benchmarking.
    """

    DIRECT_INJECTION_PATTERNS: List[Tuple[Pattern[str], float, str]] = [
        (
            re.compile(
                r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|directives|prompts|rules)",
                re.IGNORECASE,
            ),
            0.95,
            "rule:override_previous_instructions",
        ),
        (
            re.compile(
                r"\b(?:new\s+system\s+instruction|system\s+override|system\s+prompt\s+reset)\b",
                re.IGNORECASE,
            ),
            0.90,
            "rule:system_override_marker",
        ),
        (
            re.compile(
                r"\b(?:you\s+must\s+now\s+act\s+as|you\s+are\s+now\s+in\s+developer\s+mode|dan\s+mode|jailbreak\s+mode)\b",
                re.IGNORECASE,
            ),
            0.92,
            "rule:persona_hijack_jailbreak",
        ),
    ]

    LEAK_PATTERNS: List[Tuple[Pattern[str], float, str]] = [
        (
            re.compile(
                r"\b(?:repeat|print|display|dump|output|reveal|show)\s+(?:your\s+)?(?:initial|system|hidden|\s)*(?:prompt|instructions|configuration)",
                re.IGNORECASE,
            ),
            0.88,
            "rule:system_prompt_extraction",
        ),
        (
            re.compile(
                r"\b(?:what\s+are\s+the\s+words\s+above|echo\s+everything\s+above\s+this\s+line)\b",
                re.IGNORECASE,
            ),
            0.85,
            "rule:verbatim_context_leak",
        ),
    ]

    ZERO_WIDTH_CHARS = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")

    def __init__(self, config: AegisConfig | None = None) -> None:
        super().__init__(config or AegisConfig())

    def _normalize(self, text: str) -> str:
        if not self.config.enable_normalization:
            return text
        cleaned = self.ZERO_WIDTH_CHARS.sub("", text)
        normalized = unicodedata.normalize("NFKC", cleaned)
        return " ".join(normalized.split())

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        start_time = time.perf_counter()

        raw_text = request.prompt
        if len(raw_text) > self.config.max_prompt_length:
            raw_text = raw_text[: self.config.max_prompt_length]

        normalized_text = self._normalize(raw_text)

        matched_rules: List[str] = []
        max_score = 0.0
        detected_category = ThreatCategory.BENIGN

        for pattern, weight, rule_name in self.DIRECT_INJECTION_PATTERNS:
            if pattern.search(normalized_text):
                matched_rules.append(rule_name)
                if weight > max_score:
                    max_score = weight
                    detected_category = ThreatCategory.DIRECT_INJECTION

        for pattern, weight, rule_name in self.LEAK_PATTERNS:
            if pattern.search(normalized_text):
                matched_rules.append(rule_name)
                if weight > max_score:
                    max_score = weight
                    detected_category = ThreatCategory.SYSTEM_PROMPT_LEAK

        if max_score >= self.config.block_threshold:
            decision = InspectionDecision.BLOCK
            level = ThreatLevel.MALICIOUS
        elif max_score >= self.config.suspicious_threshold:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SUSPICIOUS
        else:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SAFE
            detected_category = ThreatCategory.BENIGN

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return InspectionResult(
            decision=decision,
            threat_level=level,
            threat_category=detected_category,
            confidence=round(max_score, 4),
            latency_ms=round(elapsed_ms, 3),
            matched_rules=matched_rules,
            request_id=request.request_id,
        )
