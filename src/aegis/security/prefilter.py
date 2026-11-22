from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from re import Pattern

from aegis.types import ThreatCategory


class AhoCorasickMatcher:
    """
    Deterministic O(n + z) multi-pattern literal matcher with zero backtracking (ReDoS-proof).
    """

    def __init__(self, patterns: Sequence[tuple[str, float, str]]) -> None:
        self._goto: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[tuple[str, float, str, int]]] = [[]]

        for literal, weight, tag in patterns:
            key = literal.lower()
            if not key:
                continue
            state = 0
            for ch in key:
                nxt = self._goto[state].get(ch)
                if nxt is None:
                    nxt = len(self._goto)
                    self._goto[state][ch] = nxt
                    self._goto.append({})
                    self._fail.append(0)
                    self._out.append([])
                state = nxt
            self._out[state].append((key, float(weight), tag, len(key)))

        queue: deque[int] = deque()
        for nxt in self._goto[0].values():
            self._fail[nxt] = 0
            queue.append(nxt)

        while queue:
            r = queue.popleft()
            for ch, s in self._goto[r].items():
                queue.append(s)
                f = self._fail[r]
                while f != 0 and ch not in self._goto[f]:
                    f = self._fail[f]
                self._fail[s] = self._goto[f].get(ch, 0)
                self._out[s].extend(self._out[self._fail[s]])

    def find_all(self, text: str) -> list[tuple[int, int, float, str]]:
        lowered = text.lower()
        state = 0
        matches: list[tuple[int, int, float, str]] = []
        for idx, ch in enumerate(lowered):
            while state != 0 and ch not in self._goto[state]:
                state = self._fail[state]
            state = self._goto[state].get(ch, 0)
            if self._out[state]:
                for _, weight, tag, length in self._out[state]:
                    matches.append((idx - length + 1, idx + 1, weight, tag))
        return matches


@dataclass(slots=True)
class PrefilterResult:
    fast_block: bool
    score: float
    threat_category: ThreatCategory
    matched_rules: list[str] = field(default_factory=list)
    hotspot_spans: list[tuple[int, int, float]] = field(default_factory=list)
    latency_ms: float = 0.0


# Linear-time RE2-safe signature rules (strictly bounded quantifiers, zero nested repetitions)
_FAST_BLOCK_REGEX_RULES: list[tuple[Pattern[str], float, str, ThreatCategory]] = [
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|directives|prompts|rules|guidelines)\b",
            re.IGNORECASE,
        ),
        0.96,
        "rule:override_previous_instructions",
        ThreatCategory.DIRECT_INJECTION,
    ),
    (
        re.compile(
            r"\b(?:new\s+system\s+instruction|system\s+override|system\s+prompt\s+reset|override\s+system\s+prompt)\b",
            re.IGNORECASE,
        ),
        0.93,
        "rule:system_override_marker",
        ThreatCategory.DIRECT_INJECTION,
    ),
    (
        re.compile(
            r"\b(?:you\s+must\s+now\s+act\s+as|you\s+are\s+now\s+in\s+developer\s+mode|enable\s+developer\s+mode|dan\s+mode|jailbreak\s+mode|do\s+anything\s+now)\b",
            re.IGNORECASE,
        ),
        0.93,
        "rule:persona_hijack_jailbreak",
        ThreatCategory.JAILBREAK,
    ),
    (
        re.compile(
            r"\b(?:repeat|print|display|dump|output|reveal|show|leak)\s+(?:your\s+|the\s+)?(?:initial|original|hidden|full|exact|verbatim)?\s*(?:system\s+prompt|system\s+instructions|initial\s+prompt|hidden\s+prompt|hidden\s+instructions)\b",
            re.IGNORECASE,
        ),
        0.91,
        "rule:system_prompt_extraction",
        ThreatCategory.SYSTEM_PROMPT_LEAK,
    ),
    (
        re.compile(
            r"\b(?:what\s+are\s+the\s+words\s+above|echo\s+everything\s+above\s+this\s+line|print\s+everything\s+above)\b",
            re.IGNORECASE,
        ),
        0.88,
        "rule:verbatim_context_leak",
        ThreatCategory.SYSTEM_PROMPT_LEAK,
    ),
    (
        re.compile(
            r"!\[[^\]]{0,64}\]\(https?://[^\s)]{1,256}(?:secret|token|key|prompt|data|exfil|leak)=",
            re.IGNORECASE,
        ),
        0.95,
        "rule:markdown_image_exfiltration",
        ThreatCategory.DATA_EXFILTRATION,
    ),
]

# High-precision literal delimiter spoofing signatures for Aho-Corasick
_FAST_BLOCK_LITERALS: list[tuple[str, float, str]] = [
    ("<|im_start|>system", 0.97, "rule:delimiter_im_start_system"),
    ("<|system|>", 0.95, "rule:delimiter_system_tag"),
    ("[inst] <<sys>>", 0.97, "rule:delimiter_llama_sys"),
    ("<<sys>>", 0.90, "rule:delimiter_llama_sys_open"),
    ("### system override", 0.95, "rule:delimiter_md_sys_override"),
    ("begin system prompt override", 0.96, "rule:delimiter_begin_sys_override"),
    ("ignore all previous instructions", 0.97, "rule:literal_ignore_previous"),
    ("ignore previous instructions", 0.95, "rule:literal_ignore_previous"),
    ("disregard all previous instructions", 0.96, "rule:literal_disregard_previous"),
]

# Weighted anchor tokens for O(n) Hot-Spot density scoring on long/RAG contexts
_HOTSPOT_LITERALS: list[tuple[str, float, str]] = [
    ("ignore", 1.2, "hs:ignore"),
    ("disregard", 1.4, "hs:disregard"),
    ("override", 1.3, "hs:override"),
    ("previous instructions", 2.5, "hs:prev_instr"),
    ("prior instructions", 2.3, "hs:prior_instr"),
    ("system prompt", 2.4, "hs:sys_prompt"),
    ("hidden instructions", 2.4, "hs:hidden_instr"),
    ("developer mode", 2.2, "hs:dev_mode"),
    ("jailbreak", 2.0, "hs:jailbreak"),
    ("do anything now", 2.2, "hs:dan"),
    ("bypass", 1.3, "hs:bypass"),
    ("unrestricted", 1.4, "hs:unrestricted"),
    ("without restrictions", 1.8, "hs:no_restrict"),
    ("exfiltrate", 2.2, "hs:exfiltrate"),
    ("reveal your", 1.8, "hs:reveal_your"),
    ("verbatim", 1.1, "hs:verbatim"),
    ("act as", 1.0, "hs:act_as"),
    ("pretend you are", 1.5, "hs:pretend"),
    ("roleplay", 1.0, "hs:roleplay"),
    ("new instructions", 2.0, "hs:new_instr"),
    ("<|im_start|>", 2.6, "hs:im_start"),
    ("<<sys>>", 2.6, "hs:llama_sys"),
    ("important instruction", 1.9, "hs:imp_instr"),
    ("assistant:", 1.2, "hs:role_spoof"),
    ("system:", 1.3, "hs:sys_spoof"),
    ("http://", 0.7, "hs:url"),
    ("https://", 0.7, "hs:url"),
    ("secret", 0.8, "hs:secret"),
    ("api key", 1.1, "hs:api_key"),
    ("credentials", 1.0, "hs:creds"),
    ("forget everything", 2.2, "hs:forget_all"),
]

# Educational/defensive context phrases that downgrade hard fast-block to semantic ML review
_BENIGN_DEFENSIVE_PREFIX_RE = re.compile(
    r"^\s*(?:explain\s+(?:what|how)|what\s+is\s+(?:a|an)?\s*(?:prompt\s+injection|jailbreak)|how\s+(?:do|can)\s+(?:i|we)\s+(?:defend|prevent|mitigate|detect))\b",
    re.IGNORECASE,
)


class Prefilter:
    """
    Tier-0 deterministic prefilter combining Aho-Corasick literal matching,
    RE2-safe linear signatures, and an O(n) Hot-Spot scorer for long RAG prompts.
    """

    def __init__(self, fast_block_threshold: float = 0.85) -> None:
        self.fast_block_threshold = float(fast_block_threshold)
        self._literal_matcher = AhoCorasickMatcher(_FAST_BLOCK_LITERALS)
        self._hotspot_matcher = AhoCorasickMatcher(_HOTSPOT_LITERALS)

    def inspect(
        self,
        views: Sequence[str],
        hard_block_anomalies: Sequence[str] = (),
        anomalies: Sequence[str] = (),
    ) -> PrefilterResult:
        t0 = time.perf_counter()
        matched_rules: list[str] = list(anomalies)
        max_score = 0.0
        category = ThreatCategory.BENIGN

        if hard_block_anomalies:
            max_score = 0.99
            category = ThreatCategory.DIRECT_INJECTION

        for idx, view in enumerate(views):
            if not view:
                continue
            # Only defer educational questions on the raw primary view when no smuggling/obfuscation exists
            defensive_context = (
                idx == 0 and not anomalies and bool(_BENIGN_DEFENSIVE_PREFIX_RE.search(view))
            )

            for _, _, weight, rule_name in self._literal_matcher.find_all(view):
                eff_weight = weight * 0.65 if defensive_context else weight
                matched_rules.append(rule_name)
                if eff_weight > max_score:
                    max_score = eff_weight
                    category = ThreatCategory.DIRECT_INJECTION

            for pattern, weight, rule_name, rule_cat in _FAST_BLOCK_REGEX_RULES:
                if pattern.search(view):
                    eff_weight = weight * 0.65 if defensive_context else weight
                    matched_rules.append(rule_name)
                    if eff_weight > max_score:
                        max_score = eff_weight
                        category = rule_cat

        primary_text = views[0] if views else ""
        hotspots = self.score_hotspots(primary_text)
        dedup_rules = list(dict.fromkeys(matched_rules))
        fast_block = max_score >= self.fast_block_threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PrefilterResult(
            fast_block=fast_block,
            score=round(max_score, 4),
            threat_category=category,
            matched_rules=dedup_rules,
            hotspot_spans=hotspots,
            latency_ms=round(elapsed_ms, 4),
        )

    def score_hotspots(
        self,
        text: str,
        window_chars: int = 800,
        stride_chars: int = 300,
        top_k: int = 3,
    ) -> list[tuple[int, int, float]]:
        """
        Scores candidate character spans in O(n) using Aho-Corasick hits so long RAG
        documents can select Top-2 Hot-Spot windows without scanning every token with ONNX.
        """
        if not text:
            return []
        hits = self._hotspot_matcher.find_all(text)
        if not hits:
            return []

        n = len(text)
        if n <= window_chars:
            total = sum(w for _, _, w, _ in hits)
            return [(0, n, round(total, 3))]

        # Sweep sliding windows using two-pointer accumulation over sorted hits
        hits.sort(key=lambda h: h[0])
        candidates: list[tuple[int, int, float]] = []
        left = 0
        num_hits = len(hits)
        for start in range(0, n, stride_chars):
            end = min(n, start + window_chars)
            while left < num_hits and hits[left][1] <= start:
                left += 1
            score = 0.0
            k = left
            while k < num_hits and hits[k][0] < end:
                score += hits[k][2]
                k += 1
            if score > 0.0:
                candidates.append((start, end, round(score, 3)))
            if end == n:
                break

        # Non-maximum suppression (IoU > 0.4) to return disjoint top-k hotspots
        candidates.sort(key=lambda c: c[2], reverse=True)
        selected: list[tuple[int, int, float]] = []
        for cand in candidates:
            if len(selected) >= top_k:
                break
            if all(_span_iou(cand, kept) < 0.4 for kept in selected):
                selected.append(cand)
        return selected


def _span_iou(a: tuple[int, int, float], b: tuple[int, int, float]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / float(union) if union > 0 else 0.0
