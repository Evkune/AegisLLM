from __future__ import annotations

import os

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_STAGE_BUCKETS_MS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0, 250.0, 500.0)


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="utf-8") as fh:
            fields = fh.read().split()
        rss_pages = int(fields[1])
        return rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


class ProxyMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests_total = Counter(
            "aegis_requests_total",
            "Total inspected requests by endpoint, mode, decision, and category.",
            ["endpoint", "mode", "decision", "category"],
            registry=self.registry,
        )
        self.fast_blocks_total = Counter(
            "aegis_fast_blocks_total",
            "Requests blocked at Tier-0 prefilter before ONNX inference.",
            registry=self.registry,
        )
        self.backpressure_rejects_total = Counter(
            "aegis_backpressure_rejects_total",
            "Requests rejected with HTTP 429 due to queue saturation.",
            registry=self.registry,
        )
        self.stage_latency_ms = Histogram(
            "aegis_stage_latency_ms",
            "Per-stage processing latency in milliseconds.",
            ["stage"],
            buckets=_STAGE_BUCKETS_MS,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "aegis_inference_queue_depth",
            "Current number of windows queued for ONNX inference.",
            registry=self.registry,
        )
        self.process_rss_bytes = Gauge(
            "aegis_process_rss_bytes",
            "Current resident set size (RSS) of the gateway process in bytes.",
            registry=self.registry,
        )

    def observe_inspection(
        self,
        endpoint: str,
        mode: str,
        decision: str,
        category: str,
        fast_blocked: bool,
        canonicalize_ms: float,
        prefilter_ms: float,
        windowing_ms: float,
        queue_wait_ms: float,
        onnx_ms: float,
        total_ms: float,
        queue_depth: int,
    ) -> None:
        self.requests_total.labels(
            endpoint=endpoint,
            mode=mode,
            decision=decision,
            category=category,
        ).inc()
        if fast_blocked:
            self.fast_blocks_total.inc()

        self.stage_latency_ms.labels(stage="canonicalize").observe(canonicalize_ms)
        self.stage_latency_ms.labels(stage="prefilter").observe(prefilter_ms)
        self.stage_latency_ms.labels(stage="windowing").observe(windowing_ms)
        self.stage_latency_ms.labels(stage="queue_wait").observe(queue_wait_ms)
        self.stage_latency_ms.labels(stage="onnx").observe(onnx_ms)
        self.stage_latency_ms.labels(stage="e2e").observe(total_ms)
        self.queue_depth.set(queue_depth)
        rss = _read_rss_bytes()
        if rss > 0:
            self.process_rss_bytes.set(rss)

    def render_prometheus(self) -> bytes:
        rss = _read_rss_bytes()
        if rss > 0:
            self.process_rss_bytes.set(rss)
        return generate_latest(self.registry)
