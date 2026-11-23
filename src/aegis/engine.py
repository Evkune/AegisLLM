from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from aegis.config import AegisConfig
from aegis.runtime.batcher import AdaptiveMicroBatcher
from aegis.runtime.onnx_engine import BatchScorer, ModelScore
from aegis.security.canonicalizer import Canonicalizer
from aegis.security.prefilter import Prefilter
from aegis.security.windowing import (
    ExtractedSegment,
    TextWindow,
    WindowBuilder,
    extract_openai_segments,
)
from aegis.types import (
    InspectionDecision,
    InspectionRequest,
    InspectionResult,
    StageTimings,
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
    Deterministic Tier-0 inspection engine backed by the shared `Canonicalizer`
    and Aho-Corasick / RE2-safe `Prefilter`.
    """

    def __init__(self, config: AegisConfig | None = None) -> None:
        super().__init__(config or AegisConfig())
        self.canonicalizer = Canonicalizer(max_input_chars=self.config.max_prompt_length)
        self.prefilter = Prefilter(fast_block_threshold=self.config.block_threshold)

    def inspect(self, request: InspectionRequest) -> InspectionResult:
        t0 = time.perf_counter()
        raw_text = request.prompt
        if not raw_text:
            return InspectionResult(
                decision=InspectionDecision.ALLOW,
                threat_level=ThreatLevel.SAFE,
                threat_category=ThreatCategory.BENIGN,
                confidence=0.0,
                s_mal=0.0,
                latency_ms=0.0,
                matched_rules=[],
                request_id=request.request_id,
                backend="tier0:deterministic",
            )

        if len(raw_text) > self.config.max_prompt_length:
            raw_text = raw_text[: self.config.max_prompt_length]

        if self.config.enable_normalization:
            canon = self.canonicalizer.canonicalize(raw_text)
            views = canon.views
            anomalies = canon.anomalies
            hard_anomalies = canon.hard_block_anomalies
            canon_ms = canon.latency_ms
        else:
            views = [raw_text]
            anomalies = []
            hard_anomalies = []
            canon_ms = 0.0

        pre = self.prefilter.inspect(
            views=views,
            hard_block_anomalies=hard_anomalies,
            anomalies=anomalies,
        )

        score = pre.score
        category = pre.threat_category

        if anomalies and score < 0.50:
            score = 0.50
            if category == ThreatCategory.BENIGN:
                category = ThreatCategory.DIRECT_INJECTION

        if score >= self.config.block_threshold:
            decision = InspectionDecision.BLOCK
            level = ThreatLevel.MALICIOUS
        elif score >= self.config.suspicious_threshold:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SUSPICIOUS
        else:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SAFE
            category = ThreatCategory.BENIGN

        total_ms = (time.perf_counter() - t0) * 1000.0
        return InspectionResult(
            decision=decision,
            threat_level=level,
            threat_category=category,
            confidence=round(score, 4),
            s_mal=round(score, 4),
            latency_ms=round(total_ms, 3),
            matched_rules=pre.matched_rules,
            anomalies=anomalies,
            fast_blocked=decision == InspectionDecision.BLOCK,
            windows_inspected=len(views),
            backend="tier0:deterministic",
            request_id=request.request_id,
            timings=StageTimings(
                canonicalize_ms=canon_ms,
                prefilter_ms=pre.latency_ms,
                total_ms=round(total_ms, 4),
            ),
        )


class AegisPipeline:
    """
    Full Defense-in-Depth inspection pipeline:
    Tier-0 Canonicalizer -> Tier-0 Aho-Corasick/RE2 Prefilter (Fast-Block) ->
    Role-Aware Windowing (`L <= 256`, Head+Tail+Top-2 Hot-Spots) ->
    Adaptive Micro-Batched ONNX INT8 Inference -> Window Max-Pooling Policy.
    """

    def __init__(
        self,
        scorer: BatchScorer,
        batcher: AdaptiveMicroBatcher | None = None,
        block_threshold: float | None = None,
        suspicious_threshold: float = 0.35,
        fast_block_threshold: float = 0.85,
        max_window_tokens: int = 256,
        max_windows_per_segment: int = 4,
        last_k_user_turns: int = 3,
    ) -> None:
        self.scorer = scorer
        self.batcher = batcher
        self.canonicalizer = Canonicalizer()
        self.prefilter = Prefilter(fast_block_threshold=fast_block_threshold)
        self.window_builder = WindowBuilder(
            max_window_tokens=max_window_tokens,
            max_windows_per_segment=max_windows_per_segment,
            token_counter=scorer.count_tokens,
        )
        self.last_k_user_turns = int(last_k_user_turns)
        self.block_threshold = (
            float(block_threshold)
            if block_threshold is not None
            else float(scorer.metadata.tau_95)
        )
        self.suspicious_threshold = min(float(suspicious_threshold), self.block_threshold)

    def inspect_sync(
        self,
        text: str,
        request_id: str | None = None,
        role: str = "user",
    ) -> InspectionResult:
        segments = [ExtractedSegment(role=role, text=text, turn_index=0)] if text else []
        return self._inspect_segments_sync(segments, request_id=request_id)

    async def inspect_prompt_async(
        self,
        text: str,
        request_id: str | None = None,
        role: str = "user",
    ) -> InspectionResult:
        segments = [ExtractedSegment(role=role, text=text, turn_index=0)] if text else []
        return await self._inspect_segments_async(segments, request_id=request_id)

    async def inspect_openai_payload_async(
        self,
        payload: Mapping[str, Any],
        request_id: str | None = None,
    ) -> InspectionResult:
        segments = extract_openai_segments(payload, last_k_user_turns=self.last_k_user_turns)
        return await self._inspect_segments_async(segments, request_id=request_id)

    def _prepare_tier0(
        self,
        segments: Sequence[ExtractedSegment],
    ) -> tuple[
        InspectionResult | None,
        list[TextWindow],
        list[str],
        list[str],
        float,
        float,
        float,
        float,
        ThreatCategory,
    ]:
        total_canon_ms = 0.0
        total_pre_ms = 0.0
        total_win_ms = 0.0
        all_windows: list[TextWindow] = []
        all_anomalies: list[str] = []
        all_rules: list[str] = []
        best_tier0_score = 0.0
        best_tier0_cat = ThreatCategory.BENIGN

        for seg in segments:
            if not seg.text:
                continue
            canon = self.canonicalizer.canonicalize(seg.text)
            total_canon_ms += canon.latency_ms
            all_anomalies.extend(canon.anomalies)

            pre = self.prefilter.inspect(
                views=canon.views,
                hard_block_anomalies=canon.hard_block_anomalies,
                anomalies=canon.anomalies,
            )
            total_pre_ms += pre.latency_ms
            all_rules.extend(pre.matched_rules)

            if pre.score > best_tier0_score:
                best_tier0_score = pre.score
                best_tier0_cat = (
                    ThreatCategory.INDIRECT_INJECTION
                    if seg.role in ("tool", "function")
                    and pre.threat_category == ThreatCategory.DIRECT_INJECTION
                    else pre.threat_category
                )

            if pre.fast_block:
                dedup_rules = list(dict.fromkeys(all_rules))
                dedup_anoms = list(dict.fromkeys(all_anomalies))
                total_ms = total_canon_ms + total_pre_ms
                fast_res = InspectionResult(
                    decision=InspectionDecision.BLOCK,
                    threat_level=ThreatLevel.MALICIOUS,
                    threat_category=best_tier0_cat,
                    confidence=round(best_tier0_score, 4),
                    s_mal=round(best_tier0_score, 4),
                    class_probs=(
                        round(max(0.0, 1.0 - best_tier0_score), 6),
                        round(best_tier0_score, 6)
                        if best_tier0_cat != ThreatCategory.JAILBREAK
                        else 0.0,
                        round(best_tier0_score, 6)
                        if best_tier0_cat == ThreatCategory.JAILBREAK
                        else 0.0,
                    ),
                    latency_ms=round(total_ms, 3),
                    matched_rules=dedup_rules,
                    anomalies=dedup_anoms,
                    fast_blocked=True,
                    windows_inspected=0,
                    backend=f"tier0+fast_block/{self.scorer.backend_name}",
                    timings=StageTimings(
                        canonicalize_ms=round(total_canon_ms, 4),
                        prefilter_ms=round(total_pre_ms, 4),
                        total_ms=round(total_ms, 4),
                    ),
                )
                return (
                    fast_res,
                    [],
                    dedup_rules,
                    dedup_anoms,
                    total_canon_ms,
                    total_pre_ms,
                    total_win_ms,
                    best_tier0_score,
                    best_tier0_cat,
                )

            plan = self.window_builder.build(
                views=canon.neural_views or canon.views,
                hotspot_spans=pre.hotspot_spans,
                source_role=seg.role,
            )
            total_win_ms += plan.latency_ms
            all_windows.extend(plan.windows)

        return (
            None,
            all_windows,
            list(dict.fromkeys(all_rules)),
            list(dict.fromkeys(all_anomalies)),
            total_canon_ms,
            total_pre_ms,
            total_win_ms,
            best_tier0_score,
            best_tier0_cat,
        )

    def _finalize_from_scores(
        self,
        t0: float,
        windows: Sequence[TextWindow],
        scores: Sequence[ModelScore],
        rules: list[str],
        anomalies: list[str],
        canon_ms: float,
        pre_ms: float,
        win_ms: float,
        queue_wait_ms: float,
        onnx_wall_ms: float,
        tier0_score: float,
        tier0_cat: ThreatCategory,
        request_id: str | None,
    ) -> InspectionResult:
        if not scores:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return InspectionResult(
                decision=InspectionDecision.ALLOW,
                threat_level=ThreatLevel.SAFE,
                threat_category=ThreatCategory.BENIGN,
                confidence=0.0,
                s_mal=0.0,
                latency_ms=round(total_ms, 3),
                matched_rules=rules,
                anomalies=anomalies,
                windows_inspected=0,
                backend=self.scorer.backend_name,
                request_id=request_id,
                timings=StageTimings(
                    canonicalize_ms=round(canon_ms, 4),
                    prefilter_ms=round(pre_ms, 4),
                    windowing_ms=round(win_ms, 4),
                    total_ms=round(total_ms, 4),
                ),
            )

        # Max-Pooling across all inspected windows and decoded views
        best_idx = max(range(len(scores)), key=lambda i: scores[i].s_mal)
        best_score = scores[best_idx]
        best_window = windows[best_idx]

        # Defense-in-depth score fusion: combine neural probability with Tier-0 non-blocking floor
        fused_s_mal = max(best_score.s_mal, tier0_score * 0.75)
        category = best_score.predicted_category
        if best_score.s_mal >= self.block_threshold and category == ThreatCategory.BENIGN:
            category = (
                ThreatCategory.JAILBREAK
                if best_score.probs[2] > best_score.probs[1]
                else ThreatCategory.DIRECT_INJECTION
            )
        if best_window.source_role in ("tool", "function") and category == ThreatCategory.DIRECT_INJECTION:
            category = ThreatCategory.INDIRECT_INJECTION
        if tier0_score * 0.75 > best_score.s_mal and tier0_cat != ThreatCategory.BENIGN:
            category = tier0_cat

        if fused_s_mal >= self.block_threshold:
            decision = InspectionDecision.BLOCK
            level = ThreatLevel.MALICIOUS
        elif fused_s_mal >= self.suspicious_threshold:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SUSPICIOUS
        else:
            decision = InspectionDecision.ALLOW
            level = ThreatLevel.SAFE
            category = ThreatCategory.BENIGN

        total_ms = (time.perf_counter() - t0) * 1000.0
        return InspectionResult(
            decision=decision,
            threat_level=level,
            threat_category=category,
            confidence=round(fused_s_mal, 4),
            s_mal=round(fused_s_mal, 6),
            class_probs=best_score.probs,
            latency_ms=round(total_ms, 3),
            matched_rules=rules,
            anomalies=anomalies,
            fast_blocked=False,
            windows_inspected=len(windows),
            backend=self.scorer.backend_name,
            request_id=request_id,
            timings=StageTimings(
                canonicalize_ms=round(canon_ms, 4),
                prefilter_ms=round(pre_ms, 4),
                windowing_ms=round(win_ms, 4),
                queue_wait_ms=round(queue_wait_ms, 4),
                onnx_ms=round(onnx_wall_ms, 4),
                total_ms=round(total_ms, 4),
            ),
        )

    def _inspect_segments_sync(
        self,
        segments: Sequence[ExtractedSegment],
        request_id: str | None,
    ) -> InspectionResult:
        t0 = time.perf_counter()
        if not segments:
            return self._finalize_from_scores(
                t0, [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ThreatCategory.BENIGN, request_id
            )

        (
            fast_res,
            windows,
            rules,
            anomalies,
            canon_ms,
            pre_ms,
            win_ms,
            tier0_score,
            tier0_cat,
        ) = self._prepare_tier0(segments)

        if fast_res is not None:
            fast_res.request_id = request_id
            fast_res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            fast_res.timings.total_ms = fast_res.latency_ms
            return fast_res

        t_onnx = time.perf_counter()
        scores = self.scorer.score_batch([w.text for w in windows]) if windows else []
        onnx_wall_ms = (time.perf_counter() - t_onnx) * 1000.0

        return self._finalize_from_scores(
            t0,
            windows,
            scores,
            rules,
            anomalies,
            canon_ms,
            pre_ms,
            win_ms,
            0.0,
            onnx_wall_ms,
            tier0_score,
            tier0_cat,
            request_id,
        )

    async def _inspect_segments_async(
        self,
        segments: Sequence[ExtractedSegment],
        request_id: str | None,
    ) -> InspectionResult:
        t0 = time.perf_counter()
        if not segments:
            return self._finalize_from_scores(
                t0, [], [], [], [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ThreatCategory.BENIGN, request_id
            )

        (
            fast_res,
            windows,
            rules,
            anomalies,
            canon_ms,
            pre_ms,
            win_ms,
            tier0_score,
            tier0_cat,
        ) = self._prepare_tier0(segments)

        if fast_res is not None:
            fast_res.request_id = request_id
            fast_res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            fast_res.timings.total_ms = fast_res.latency_ms
            return fast_res

        t_onnx = time.perf_counter()
        if windows:
            window_texts = [w.text for w in windows]
            if self.batcher is not None:
                scores, queue_wait_ms = await self.batcher.score_many(window_texts)
            else:
                scores = self.scorer.score_batch(window_texts)
                queue_wait_ms = 0.0
        else:
            scores, queue_wait_ms = [], 0.0
        onnx_wall_ms = max(0.0, (time.perf_counter() - t_onnx) * 1000.0 - queue_wait_ms)

        return self._finalize_from_scores(
            t0,
            windows,
            scores,
            rules,
            anomalies,
            canon_ms,
            pre_ms,
            win_ms,
            queue_wait_ms,
            onnx_wall_ms,
            tier0_score,
            tier0_cat,
            request_id,
        )
