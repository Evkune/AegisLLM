from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from aegis.runtime.onnx_engine import BatchScorer, ModelScore


class BackpressureError(RuntimeError):
    """Raised when the inference queue is saturated (maps to HTTP 429)."""


@dataclass(slots=True)
class _QueuedItem:
    text: str
    bucket: int
    enqueued_at: float
    future: asyncio.Future[tuple[ModelScore, float]]


@dataclass(slots=True)
class BatcherStats:
    queue_depth: int
    max_queue_size: int
    total_dispatched: int
    total_batches: int
    rejected_backpressure: int


def _length_bucket(text: str) -> int:
    # Approximate token length bucket (<=64, <=128, <=256) to avoid padding short prompts to 256
    approx_tokens = len(text) // 4
    if approx_tokens <= 56:
        return 64
    if approx_tokens <= 116:
        return 128
    return 256


class AdaptiveMicroBatcher:
    """
    Zero-idle adaptive micro-batcher for CPU ONNX Runtime inference:
    - Idle path (C=1): dispatches immediately with 0 ms wait.
    - Loaded path (C>1): drains queued windows up to `max_batch_size` with optional
      sub-millisecond coalesce (`coalesce_ms <= 1.5 ms`) and groups by sequence-length
      bucket so short prompts are never padded to 256 tokens.
    - Backpressure: rejects immediately with `BackpressureError` when queue reaches
      `max_queue_size` (proxy returns HTTP 429).
    """

    def __init__(
        self,
        scorer: BatchScorer,
        max_batch_size: int = 4,
        max_queue_size: int = 128,
        coalesce_ms: float = 1.0,
    ) -> None:
        self.scorer = scorer
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_queue_size = max(1, int(max_queue_size))
        self.coalesce_s = max(0.0, float(coalesce_ms)) / 1000.0

        self._queue: asyncio.Queue[_QueuedItem] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-ort")
        self._worker_task: asyncio.Task[None] | None = None
        self._started = False
        self._total_dispatched = 0
        self._total_batches = 0
        self._rejected = 0

    async def start(self) -> None:
        if self._started:
            return
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._started = True
        self._worker_task = asyncio.create_task(self._worker_loop(), name="aegis-micro-batcher")

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._queue is not None:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(RuntimeError("Batcher stopped before dispatch"))
        self._executor.shutdown(wait=False)

    def stats(self) -> BatcherStats:
        depth = self._queue.qsize() if self._queue is not None else 0
        return BatcherStats(
            queue_depth=depth,
            max_queue_size=self.max_queue_size,
            total_dispatched=self._total_dispatched,
            total_batches=self._total_batches,
            rejected_backpressure=self._rejected,
        )

    async def score_many(self, texts: Sequence[str]) -> tuple[list[ModelScore], float]:
        if not texts:
            return [], 0.0
        if not self._started or self._queue is None:
            await self.start()
        assert self._queue is not None

        if self._queue.qsize() + len(texts) > self.max_queue_size:
            self._rejected += 1
            raise BackpressureError(
                f"Inference queue saturated ({self._queue.qsize()}/{self.max_queue_size})"
            )

        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        futures: list[asyncio.Future[tuple[ModelScore, float]]] = []
        for text in texts:
            fut: asyncio.Future[tuple[ModelScore, float]] = loop.create_future()
            item = _QueuedItem(
                text=text,
                bucket=_length_bucket(text),
                enqueued_at=now,
                future=fut,
            )
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull as exc:
                self._rejected += 1
                raise BackpressureError("Inference queue saturated") from exc
            futures.append(fut)

        outcomes = await asyncio.gather(*futures)
        scores = [pair[0] for pair in outcomes]
        max_queue_wait_ms = max(pair[1] for pair in outcomes)
        return scores, round(max_queue_wait_ms, 4)

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        loop = asyncio.get_running_loop()

        while True:
            first = await self._queue.get()
            batch: list[_QueuedItem] = [first]

            # Zero-idle path: only coalesce if additional items are already arriving
            if self._queue.empty() and self.coalesce_s > 0.0:
                # Immediately dispatch single item on idle worker (0 ms wait!)
                pass
            else:
                deadline = time.perf_counter() + self.coalesce_s
                while len(batch) < self.max_batch_size:
                    try:
                        nxt = self._queue.get_nowait()
                        batch.append(nxt)
                    except asyncio.QueueEmpty:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0.0:
                            break
                        try:
                            nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                            batch.append(nxt)
                        except asyncio.TimeoutError:
                            break

            # Bucket by approximate token length to keep short prompts fast
            buckets: dict[int, list[_QueuedItem]] = {}
            for item in batch:
                buckets.setdefault(item.bucket, []).append(item)

            for bucket_items in buckets.values():
                dispatch_ts = time.perf_counter()
                texts = [it.text for it in bucket_items]
                try:
                    scores = await loop.run_in_executor(
                        self._executor,
                        self.scorer.score_batch,
                        texts,
                    )
                    self._total_batches += 1
                    self._total_dispatched += len(bucket_items)
                    for it, sc in zip(bucket_items, scores):
                        wait_ms = (dispatch_ts - it.enqueued_at) * 1000.0
                        if not it.future.done():
                            it.future.set_result((sc, wait_ms))
                except Exception as exc:  # noqa: BLE001
                    for it in bucket_items:
                        if not it.future.done():
                            it.future.set_exception(exc)
