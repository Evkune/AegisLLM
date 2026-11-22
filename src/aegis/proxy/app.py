from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import orjson
from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from aegis.config import ProxySettings
from aegis.engine import AegisPipeline
from aegis.proxy.audit import AsyncAuditLogger, AuditEvent, hash_prompt
from aegis.proxy.metrics import ProxyMetrics
from aegis.runtime.batcher import AdaptiveMicroBatcher, BackpressureError
from aegis.runtime.onnx_engine import BatchScorer, create_runtime_scorer
from aegis.security.windowing import extract_openai_segments
from aegis.types import InspectionDecision, InspectionResult

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _forward_headers(headers: Request) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.headers.items()
        if k.lower() not in _HOP_BY_HOP_HEADERS
    }


def _security_headers(result: InspectionResult, mode: str) -> dict[str, str]:
    effective = (
        f"shadow-{result.decision.value}"
        if mode == "shadow" and result.decision == InspectionDecision.BLOCK
        else result.decision.value
    )
    return {
        "X-Aegis-Decision": effective,
        "X-Aegis-Category": result.threat_category.value,
        "X-Aegis-Score": f"{result.s_mal:.4f}",
        "X-Aegis-Latency-Ms": f"{result.latency_ms:.3f}",
        "X-Aegis-Request-Id": result.request_id or "",
    }


def _blocked_openai_payload(result: InspectionResult) -> dict[str, Any]:
    return {
        "error": {
            "message": (
                f"Request blocked by AegisLLM security gateway "
                f"(category={result.threat_category.value}, score={result.s_mal:.4f})."
            ),
            "type": "aegis_security_policy_violation",
            "code": "prompt_injection_blocked",
            "param": None,
            "aegis": {
                "request_id": result.request_id,
                "threat_category": result.threat_category.value,
                "threat_level": result.threat_level.value,
                "s_mal": result.s_mal,
                "matched_rules": result.matched_rules,
                "anomalies": result.anomalies,
                "fast_blocked": result.fast_blocked,
                "latency_ms": result.latency_ms,
            },
        }
    }


def create_app(
    settings: ProxySettings | None = None,
    scorer: BatchScorer | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = settings or ProxySettings.from_env()
    metrics = ProxyMetrics()

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        active_scorer = scorer or create_runtime_scorer(
            model_dir=cfg.model_dir,
            allow_fallback=cfg.allow_fallback,
            fallback_joblib_path=cfg.fallback_joblib_path,
            fallback_metrics_path=cfg.fallback_metrics_path,
            intra_op_threads=cfg.intra_op_threads,
        )
        active_scorer.warmup()

        batcher = AdaptiveMicroBatcher(
            scorer=active_scorer,
            max_batch_size=cfg.max_batch_size,
            max_queue_size=cfg.max_queue_size,
            coalesce_ms=cfg.coalesce_ms,
        )
        await batcher.start()

        pipeline = AegisPipeline(
            scorer=active_scorer,
            batcher=batcher,
            block_threshold=cfg.block_threshold,
            suspicious_threshold=cfg.suspicious_threshold,
            fast_block_threshold=cfg.fast_block_threshold,
            max_window_tokens=cfg.max_window_tokens,
            max_windows_per_segment=cfg.max_windows_per_segment,
            last_k_user_turns=cfg.last_k_user_turns,
        )

        audit_logger = AsyncAuditLogger(
            fmt=cfg.audit_format,
            log_path=cfg.audit_log_path,
            log_raw_prompt=cfg.log_raw_prompt,
        )
        await audit_logger.start()

        limits = httpx.Limits(max_connections=100, max_keepalive_connections=32)
        client_kwargs: dict[str, Any] = {
            "base_url": cfg.upstream_url.rstrip("/"),
            "timeout": httpx.Timeout(cfg.upstream_timeout_s),
            "limits": limits,
        }
        if upstream_transport is not None:
            client_kwargs["transport"] = upstream_transport
        http_client = httpx.AsyncClient(**client_kwargs)

        fastapi_app.state.settings = cfg
        fastapi_app.state.scorer = active_scorer
        fastapi_app.state.batcher = batcher
        fastapi_app.state.pipeline = pipeline
        fastapi_app.state.audit_logger = audit_logger
        fastapi_app.state.metrics = metrics
        fastapi_app.state.http_client = http_client
        fastapi_app.state.ready = True

        try:
            yield
        finally:
            fastapi_app.state.ready = False
            await http_client.aclose()
            await batcher.stop()
            await audit_logger.stop()

    fastapi_app = FastAPI(
        title="AegisLLM Security Gateway",
        version="0.2.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    def _record_telemetry(
        endpoint: str,
        result: InspectionResult,
        inspected_text: str,
    ) -> None:
        st = fastapi_app.state
        stats = st.batcher.stats()
        st.metrics.observe_inspection(
            endpoint=endpoint,
            mode=st.settings.mode,
            decision=result.decision.value,
            category=result.threat_category.value,
            fast_blocked=result.fast_blocked,
            canonicalize_ms=result.timings.canonicalize_ms,
            prefilter_ms=result.timings.prefilter_ms,
            windowing_ms=result.timings.windowing_ms,
            queue_wait_ms=result.timings.queue_wait_ms,
            onnx_ms=result.timings.onnx_ms,
            total_ms=result.timings.total_ms,
            queue_depth=stats.queue_depth,
        )
        st.audit_logger.emit_nowait(
            AuditEvent(
                request_id=result.request_id or "",
                endpoint=endpoint,
                mode=st.settings.mode,
                decision=result.decision.value,
                threat_level=result.threat_level.value,
                threat_category=result.threat_category.value,
                s_mal=result.s_mal,
                latency_ms=result.latency_ms,
                prompt_sha256=hash_prompt(inspected_text),
                prompt_len=len(inspected_text),
                fast_blocked=result.fast_blocked,
                windows_inspected=result.windows_inspected,
                backend=result.backend,
                matched_rules=result.matched_rules,
                anomalies=result.anomalies,
                raw_prompt=inspected_text if st.settings.log_raw_prompt else None,
            )
        )

    async def _proxy_openai_request(request: Request, upstream_path: str) -> Response:
        req_id = request.headers.get("x-request-id") or f"aegis-{uuid.uuid4().hex[:12]}"
        raw_body = await request.body()
        try:
            payload = orjson.loads(raw_body) if raw_body else {}
        except orjson.JSONDecodeError:
            return ORJSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
            )
        if not isinstance(payload, dict):
            return ORJSONResponse(
                status_code=400,
                content={"error": {"message": "JSON body must be an object", "type": "invalid_request_error"}},
            )

        st = fastapi_app.state
        segments = extract_openai_segments(payload, last_k_user_turns=st.settings.last_k_user_turns)
        combined_text = "\n".join(s.text for s in segments)

        try:
            result = await st.pipeline.inspect_openai_payload_async(payload, request_id=req_id)
        except BackpressureError as exc:
            st.metrics.backpressure_rejects_total.inc()
            return ORJSONResponse(
                status_code=429,
                headers={"Retry-After": "1"},
                content={
                    "error": {
                        "message": str(exc),
                        "type": "rate_limit_error",
                        "code": "aegis_backpressure",
                    }
                },
            )

        _record_telemetry(upstream_path, result, combined_text)
        sec_headers = _security_headers(result, st.settings.mode)

        if result.decision == InspectionDecision.BLOCK and st.settings.mode == "enforce":
            return ORJSONResponse(
                status_code=403,
                headers=sec_headers,
                content=_blocked_openai_payload(result),
            )

        forward_hdrs = _forward_headers(request)
        is_stream = bool(payload.get("stream", False))

        if is_stream:
            upstream_req = st.http_client.build_request(
                method="POST",
                url=upstream_path,
                content=raw_body,
                headers=forward_hdrs,
            )
            try:
                upstream_resp = await st.http_client.send(upstream_req, stream=True)
            except httpx.HTTPError as exc:
                return ORJSONResponse(
                    status_code=502,
                    headers=sec_headers,
                    content={"error": {"message": f"Upstream LLM unavailable: {exc}", "type": "upstream_error"}},
                )

            resp_headers = {
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in _HOP_BY_HOP_HEADERS
            }
            resp_headers.update(sec_headers)

            async def _stream_bytes() -> AsyncIterator[bytes]:
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(
                _stream_bytes(),
                status_code=upstream_resp.status_code,
                headers=resp_headers,
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            )

        try:
            upstream_resp = await st.http_client.post(
                upstream_path,
                content=raw_body,
                headers=forward_hdrs,
            )
        except httpx.HTTPError as exc:
            return ORJSONResponse(
                status_code=502,
                headers=sec_headers,
                content={"error": {"message": f"Upstream LLM unavailable: {exc}", "type": "upstream_error"}},
            )

        resp_headers = {
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in _HOP_BY_HOP_HEADERS
        }
        resp_headers.update(sec_headers)
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            media_type=upstream_resp.headers.get("content-type", "application/json"),
        )

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy_openai_request(request, "/v1/chat/completions")

    @fastapi_app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy_openai_request(request, "/v1/completions")

    @fastapi_app.post("/v1/inspect")
    async def inspect_endpoint(request: Request) -> Response:
        req_id = request.headers.get("x-request-id") or f"aegis-{uuid.uuid4().hex[:12]}"
        raw_body = await request.body()
        try:
            payload = orjson.loads(raw_body) if raw_body else {}
        except orjson.JSONDecodeError:
            return ORJSONResponse(status_code=400, content={"error": "Invalid JSON body"})
        if not isinstance(payload, dict):
            return ORJSONResponse(status_code=400, content={"error": "JSON body must be an object"})

        st = fastapi_app.state
        try:
            if "messages" in payload:
                segments = extract_openai_segments(payload, last_k_user_turns=st.settings.last_k_user_turns)
                inspected_text = "\n".join(s.text for s in segments)
                result = await st.pipeline.inspect_openai_payload_async(payload, request_id=req_id)
            else:
                prompt = str(payload.get("prompt", ""))
                inspected_text = prompt
                role = str(payload.get("role", "user"))
                result = await st.pipeline.inspect_prompt_async(prompt, request_id=req_id, role=role)
        except BackpressureError as exc:
            st.metrics.backpressure_rejects_total.inc()
            return ORJSONResponse(
                status_code=429,
                headers={"Retry-After": "1"},
                content={"error": str(exc)},
            )

        _record_telemetry("/v1/inspect", result, inspected_text)
        return ORJSONResponse(
            status_code=200,
            headers=_security_headers(result, st.settings.mode),
            content=result.model_dump(),
        )

    @fastapi_app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    @fastapi_app.get("/readyz")
    async def readyz() -> Response:
        st = fastapi_app.state
        is_ready = bool(getattr(st, "ready", False))
        scorer_obj = getattr(st, "scorer", None)
        batcher_obj = getattr(st, "batcher", None)
        stats = batcher_obj.stats() if batcher_obj is not None else None
        payload = {
            "ready": is_ready,
            "mode": st.settings.mode if hasattr(st, "settings") else "unknown",
            "backend": scorer_obj.backend_name if scorer_obj is not None else "none",
            "onnx_loaded": bool(scorer_obj.is_onnx) if scorer_obj is not None else False,
            "block_threshold": st.pipeline.block_threshold if hasattr(st, "pipeline") else None,
            "queue_depth": stats.queue_depth if stats is not None else 0,
        }
        return ORJSONResponse(status_code=200 if is_ready else 503, content=payload)

    @fastapi_app.get("/metrics")
    async def prometheus_metrics() -> Response:
        data = fastapi_app.state.metrics.render_prometheus()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return fastapi_app


app = create_app()
