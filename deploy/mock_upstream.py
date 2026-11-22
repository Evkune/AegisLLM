from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse, StreamingResponse

app = FastAPI(title="AegisLLM Mock Upstream LLM", default_response_class=ORJSONResponse)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> object:
    body = await request.json()
    model = str(body.get("model", "mock-llm"))
    if body.get("stream"):
        async def _sse() -> AsyncIterator[bytes]:
            for token in ("Aegis ", "upstream ", "OK."):
                chunk = {
                    "id": "chatcmpl-mock-stream",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                await asyncio.sleep(0.001)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-mock-001",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Mock upstream response OK."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 16, "completion_tokens": 6, "total_tokens": 22},
    }


@app.post("/v1/completions")
async def completions(request: Request) -> dict[str, object]:
    body = await request.json()
    return {
        "id": "cmpl-mock-001",
        "object": "text_completion",
        "model": str(body.get("model", "mock-llm")),
        "choices": [{"index": 0, "text": "Mock completion OK.", "finish_reason": "stop"}],
    }


if __name__ == "__main__":
    port = int(os.environ.get("MOCK_UPSTREAM_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
