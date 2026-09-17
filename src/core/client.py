"""Async upstream client for the Anthropic Messages API.

Uses httpx directly (not the openai SDK) because the upstream wire format is
Anthropic's. Retries cover transient upstream failures; the streaming path
retries only until the first byte is forwarded, after which a partial stream
is passed through as-is.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from fastapi import HTTPException

from src.core.failures import record_failure
from src.core.logging import logger

RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
MAX_BACKOFF_SECS = 8.0


class AnthropicClient:
    """Thin, observable wrapper around POST {base_url}/v1/messages."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 180,
        max_retries: int = 2,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.headers = dict(custom_headers or {})
        # Streaming upstreams can go silent for minutes (thinking), so only the
        # connect/write phases get short timeouts; reads use the full budget.
        self.timeout_config = httpx.Timeout(timeout, connect=15.0, write=30.0, pool=15.0)

    # ── Public API ────────────────────────────────────────────────────────
    async def create_message(
        self, payload: Dict[str, Any], request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """POST /v1/messages and return the parsed (non-streaming) response."""
        url = f"{self.base_url}/v1/messages"
        attempt = 0
        last_error: Optional[HTTPException] = None

        while attempt <= self.max_retries:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=self.timeout_config) as client:
                    response = await client.post(url, json=payload, headers=self.headers)
            except httpx.HTTPError as e:
                last_error = HTTPException(
                    status_code=502, detail=f"Upstream connection failed: {e}"
                )
                logger.warning(f"[{request_id}] upstream transport error: {e}")
                if attempt <= self.max_retries:
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                record_failure(502, str(e), payload.get("model"), request_id)
                raise last_error

            if response.status_code >= 400:
                detail = self._error_detail(response.status_code, response.text)
                if response.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                    logger.warning(
                        f"[{request_id}] upstream {response.status_code}, retry "
                        f"{attempt}/{self.max_retries}: {detail[:200]}"
                    )
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                record_failure(response.status_code, response.text, payload.get("model"), request_id)
                raise HTTPException(status_code=response.status_code, detail=detail)

            try:
                return response.json()
            except json.JSONDecodeError:
                record_failure(502, response.text, payload.get("model"), request_id)
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream returned non-JSON body: {response.text[:300]}",
                )

        raise last_error or HTTPException(status_code=502, detail="Upstream request failed")

    async def stream_message(
        self, payload: Dict[str, Any], request_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Yield raw SSE lines from POST /v1/messages (stream=true).

        Retries are only attempted while opening the response; once the first
        line has been yielded the stream is terminal.
        """
        url = f"{self.base_url}/v1/messages"
        attempt = 0

        while True:
            attempt += 1
            client = httpx.AsyncClient(timeout=self.timeout_config)
            stream_cm = client.stream("POST", url, json=payload, headers=self.headers)
            try:
                response = await stream_cm.__aenter__()
            except httpx.HTTPError as e:
                await client.aclose()
                if attempt <= self.max_retries:
                    logger.warning(f"[{request_id}] stream open failed ({e}), retrying")
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                record_failure(502, str(e), payload.get("model"), request_id)
                raise HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")

            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")
                await stream_cm.__aexit__(None, None, None)
                await client.aclose()
                detail = self._error_detail(response.status_code, body)
                if response.status_code in RETRYABLE_STATUS and attempt <= self.max_retries:
                    logger.warning(
                        f"[{request_id}] stream open got {response.status_code}, retry "
                        f"{attempt}/{self.max_retries}: {detail[:200]}"
                    )
                    await asyncio.sleep(self._backoff(attempt))
                    continue
                record_failure(response.status_code, body, payload.get("model"), request_id)
                raise HTTPException(status_code=response.status_code, detail=detail)

            logger.debug(f"[{request_id}] upstream stream open {response.status_code}")
            break

        try:
            async for line in response.aiter_lines():
                # Blank lines only separate events; the converter re-emits them.
                if line:
                    yield line
        finally:
            await stream_cm.__aexit__(None, None, None)
            await client.aclose()

    async def count_tokens(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v1/messages/count_tokens passthrough."""
        url = f"{self.base_url}/v1/messages/count_tokens"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=self.headers)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=self._error_detail(response.status_code, response.text),
            )
        return response.json()

    # ── Helpers ───────────────────────────────────────────────────────────
    def _backoff(self, attempt: int) -> float:
        return min(MAX_BACKOFF_SECS, 0.5 * (2 ** (attempt - 1)))

    def _error_detail(self, status_code: int, body: str) -> str:
        """Human-readable error text, kept short enough for a chat UI."""
        text = (body or "").strip()
        message = text
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                error = parsed.get("error")
                if isinstance(error, dict) and error.get("message"):
                    message = f"{error.get('type', 'error')}: {error['message']}"
                elif isinstance(error, str):
                    message = error
                elif parsed.get("detail"):
                    message = str(parsed["detail"])
        except (json.JSONDecodeError, TypeError):
            pass
        message = message[:600] if message else "no response body"

        if status_code == 401:
            return f"Upstream rejected the credential (401). {message}"
        if status_code == 403:
            return f"Upstream denied access (403). {message}"
        if status_code == 404:
            return (
                "Upstream could not find the endpoint or model (404). "
                f"Check ANTHROPIC_BASE_URL and the mapped model id. {message}"
            )
        if status_code == 413:
            return f"Request too large for upstream (413). {message}"
        if status_code == 429:
            return f"Upstream rate limit or quota exhausted (429). {message}"
        if status_code == 529:
            return f"Upstream overloaded (529). {message}"
        if status_code >= 500:
            return f"Upstream error ({status_code}). {message}"
        return f"Upstream rejected the request ({status_code}). {message}"