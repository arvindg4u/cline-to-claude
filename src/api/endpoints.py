"""HTTP routes.

Cline (and any OpenAI-compatible client) talks to ``/v1/chat/completions``;
the proxy translates to Anthropic's ``/v1/messages``. The Anthropic-wire
routes are exposed as passthrough so the same process can also serve Claude
Code-style clients without translation.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.api.dashboard import dashboard_response
from src.conversion.request_converter import convert_openai_to_claude
from src.conversion.response_converter import (
    OpenAIStreamTranslator,
    convert_claude_stream_to_openai,
    convert_claude_to_openai_response,
)
from src.core.client import AnthropicClient
from src.core.config import config
from src.core.failures import load_failure_state
from src.core.logging import logger
from src.core.model_manager import model_manager
from src.core.stats import stats
from src.models.openai import OpenAIChatRequest

router = APIRouter()

anthropic_client = AnthropicClient(
    config.anthropic_api_key,
    config.anthropic_base_url,
    config.request_timeout,
    config.max_retries,
    custom_headers=config.get_upstream_headers(),
)

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "X-Accel-Buffering": "no",
}

CHAT_ENDPOINT = "/v1/chat/completions"


async def validate_client_api_key(
    authorization: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)
) -> None:
    """Enforce CLIENT_API_KEY when one is configured (open by default)."""
    if not config.client_api_key:
        return

    client_api_key = x_api_key
    if not client_api_key and authorization:
        client_api_key = (
            authorization[len("Bearer "):]
            if authorization.lower().startswith("bearer ")
            else authorization
        )

    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning("Rejected request: invalid client API key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Provide the key configured as CLIENT_API_KEY.",
        )


async def _prepend(first: str, rest: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Re-inject the line consumed while priming the upstream stream."""
    yield first
    async for line in rest:
        yield line


async def _finalize_stream(
    frames: AsyncGenerator[str, None], translator: OpenAIStreamTranslator
) -> AsyncGenerator[str, None]:
    """Forward translated frames and bank streaming token usage at the end."""
    try:
        async for frame in frames:
            yield frame
    finally:
        stats.add_tokens(translator.last_usage)


@router.post(CHAT_ENDPOINT)
@router.post("/chat/completions")
async def create_chat_completion(
    request: OpenAIChatRequest,
    http_request: Request,
    _: None = Depends(validate_client_api_key),
):
    """OpenAI Chat Completions -> Anthropic Messages (streaming supported)."""
    t0 = time.monotonic()
    request_id = str(uuid.uuid4())

    try:
        try:
            claude_payload = convert_openai_to_claude(request, model_manager)
        except ValueError as e:
            stats.record(CHAT_ENDPOINT, status=400, latency=time.monotonic() - t0)
            raise HTTPException(status_code=400, detail=str(e))

        upstream_model = claude_payload["model"]
        stats.note_model(upstream_model)
        logger.info(
            f"[{request_id}] {CHAT_ENDPOINT} model={upstream_model} "
            f"stream={bool(request.stream)} tools={len(claude_payload.get('tools') or [])} "
            f"messages={len(claude_payload['messages'])} max_tokens={claude_payload['max_tokens']}"
        )

        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            include_usage = request.wants_usage_in_stream()
            if include_usage is None:
                include_usage = config.stream_usage_default
            translator = OpenAIStreamTranslator(request.model, request_id, include_usage)

            # Prime the upstream stream so failures surface as real HTTP
            # status codes instead of an error frame inside a 200 response.
            upstream = anthropic_client.stream_message(claude_payload, request_id)
            first_line: Optional[str] = None
            try:
                first_line = await upstream.__anext__()
            except StopAsyncIteration:
                first_line = None

            lines: AsyncGenerator[str, None] = (
                _prepend(first_line, upstream) if first_line is not None else upstream
            )
            stats.record(CHAT_ENDPOINT, status=200, latency=time.monotonic() - t0)
            converted = convert_claude_stream_to_openai(
                lines,
                request.model,
                request_id,
                include_usage,
                config.stream_keepalive_secs,
                translator,
            )
            return StreamingResponse(
                _finalize_stream(converted, translator),
                media_type="text/event-stream",
                headers=STREAM_HEADERS,
            )

        claude_response = await anthropic_client.create_message(claude_payload, request_id)
        openai_response = convert_claude_to_openai_response(claude_response, request.model)
        stats.record(
            CHAT_ENDPOINT,
            model=upstream_model,
            status=200,
            latency=time.monotonic() - t0,
            usage=claude_response.get("usage"),
        )
        return JSONResponse(openai_response)

    except HTTPException as e:
        stats.record(CHAT_ENDPOINT, status=e.status_code, latency=time.monotonic() - t0)
        raise
    except Exception as e:  # pragma: no cover - defensive
        import traceback

        logger.error(f"[{request_id}] unexpected error: {e}")
        logger.error(traceback.format_exc())
        stats.record(CHAT_ENDPOINT, status=500, latency=time.monotonic() - t0)
        raise HTTPException(status_code=500, detail=f"Proxy error: {e}")


# ── Model discovery ───────────────────────────────────────────────────────
@router.get("/v1/models")
@router.get("/models")
async def list_models():
    """Advertise the models Cline may request."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "created": created, "owned_by": "anthropic"}
            for model in model_manager.list_models()
        ],
    }


# ── Anthropic-wire passthrough (Claude Code and friends) ──────────────────
async def _passthrough_anthropic(http_request: Request, path: str) -> Response:
    raw = await http_request.body()
    try:
        payload: Dict[str, Any] = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    url = f"{anthropic_client.base_url}{path}"
    headers = dict(anthropic_client.headers)

    if payload.get("stream"):
        async def stream_gen() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=anthropic_client.timeout_config) as client:
                async with client.stream(
                    "POST", url, content=raw, headers=headers
                ) as upstream:
                    if upstream.status_code >= 400:
                        yield await upstream.aread()
                        return
                    async for chunk in upstream.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_gen(), media_type="text/event-stream", headers=STREAM_HEADERS)

    async with httpx.AsyncClient(timeout=anthropic_client.timeout_config) as client:
        upstream = await client.post(url, content=raw, headers=headers)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@router.post("/v1/messages")
async def passthrough_messages(http_request: Request):
    """Anthropic Messages passthrough: point ANTHROPIC_BASE_URL here if you like."""
    return await _passthrough_anthropic(http_request, "/v1/messages")


@router.post("/v1/messages/count_tokens")
async def passthrough_count_tokens(http_request: Request):
    """Token counting passthrough (Anthropic wire)."""
    raw = await http_request.body()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    return await anthropic_client.count_tokens(payload)


# ── Ops endpoints ─────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "upstream": config.anthropic_base_url,
        "upstream_configured": bool(config.anthropic_api_key),
        "client_api_key_validation": bool(config.client_api_key),
        "wire": "openai-chat -> anthropic-messages",
    }


@router.get("/test-connection")
async def test_connection():
    """One tiny upstream round trip; always 200 so scripts can read `ok`."""
    payload = {
        "model": config.default_model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
    }
    try:
        response = await anthropic_client.create_message(payload, "test-connection")
    except HTTPException as e:
        return JSONResponse(
            status_code=200,
            content={
                "ok": False,
                "status": e.status_code,
                "error": e.detail,
                "upstream": config.anthropic_base_url,
                "model": config.default_model,
            },
        )

    text = "".join(
        block.get("text", "")
        for block in response.get("content") or []
        if block.get("type") == "text"
    )
    return {
        "ok": True,
        "upstream": config.anthropic_base_url,
        "model": response.get("model") or config.default_model,
        "reply": text.strip()[:200],
        "usage": response.get("usage") or {},
    }


@router.get("/api/status")
async def api_status():
    """Machine-readable status for the dashboard (also used by scripts)."""
    return {
        "proxy": config.as_status(),
        "stats": stats.snapshot(),
        **load_failure_state(),
    }


@router.get("/")
async def root():
    """Native status dashboard (HTML). JSON lives at /api/status."""
    return dashboard_response()