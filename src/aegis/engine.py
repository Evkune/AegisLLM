import base64
import binascii
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
    Production-grade baseline inspection engine combining deterministic
    pre-tokenization hygiene (zero-width, NFKC, anti-spacing collapse, Base64 unwrapping)
    and structural pattern inspection.
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

    ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00AD\u180E\uFFF9-\uFFFB]")
    BASE64_CANDIDATE_RE = re.compile(r"(?:^|[\s:])([A-Za-z0-9+/]{16,}={0,2})(?:$|[\s])")

    def __init__(self, config: AegisConfig | None = None) -> None:
        super().__init__(config or AegisConfig())

    def _canonicalize(self, text: str) -> Tuple[str, List[str]]:
        if not self.config.enable_normalization:
            return text, []

        anomalies: List[str] = []

        # 1. Zero-width and Unicode Format (Cf) extraction
        zw_matches = self.ZERO_WIDTH_RE.findall(text)
        if zw_matches:
            anomalies.append("anomaly:zero_width_chars")
        cleaned = self.ZERO_WIDTH_RE.sub("", text)
        cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")

        # 2. Unicode NFKC canonicalization (homoglyphs & full-width normalization)
        normalized = unicodedata.normalize("NFKC", cleaned)

        # 3. Collapse adversarial character-spacing sequences (mitigates Cisco PromptGuard bypass)
        # Check if text contains single characters separated by spaces: "I g n o r e"
        if re.search(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", normalized):
            anomalies.append("anomaly:spaced_characters")
            # Multiple spaces separate words, single space separates characters
            temp = re.sub(r"[ \t]{2,}", " _W_ ", normalized)
            temp = re.sub(r"(?<=\b[A-Za-z]) (?=[A-Za-z]\b)", "", temp)
            normalized = temp.replace(" _W_ ", " ")

        # 4. Collapse contiguous horizontal whitespace
        normalized = re.sub(r"[ \t]+", " ", normalized).strip()

        # 5. Recursive Base64 payload unwrapping
        b64_matches = self.BASE64_CANDIDATE_RE.findall(normalized)
        for cand in b64_matches:
            try:
                decoded_bytes = base64.b64decode(cand.strip(), validate=True)
                decoded_str = decoded_bytes.decode("utf-8", errors="ignore").strip()
                if len(decoded_str) >= 8 and any(ch.isalpha() for ch in decoded_str):
                    anomalies.append("anomaly:base64_payload")
                    normalized = f"{normalized} [DECODED_PAYLOAD: {decoded_str}]"
            except (binascii.Error, ValueError):
                continue

        return normalized, anomalies

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        start_time = time.perf_counter()

        raw_text = request.prompt
        if len(raw_text) > self.config.max_prompt_length:
            raw_text = raw_text[: self.config.max_prompt_length]

        canonical_text, detected_anomalies = self._canonicalize(raw_text)

        matched_rules: List[str] = list(detected_anomalies)
        max_score = 0.0
        detected_category = ThreatCategory.BENIGN

        for pattern, weight, rule_name in self.DIRECT_INJECTION_PATTERNS:
            if pattern.search(canonical_text):
                matched_rules.append(rule_name)
                if weight > max_score:
                    max_score = weight
                    detected_category = ThreatCategory.DIRECT_INJECTION

        for pattern, weight, rule_name in self.LEAK_PATTERNS:
            if pattern.search(canonical_text):
                matched_rules.append(rule_name)
                if weight > max_score:
                    max_score = weight
                    detected_category = ThreatCategory.SYSTEM_PROMPT_LEAK

        # Obfuscation anomaly elevated threat floor
        if detected_anomalies and max_score < 0.50:
            max_score = max(max_score, 0.50)
            if detected_category == ThreatCategory.BENIGN:
                detected_category = ThreatCategory.DIRECT_INJECTION

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
