"""A minimal Anthropic Messages API server used by the end-to-end tests.

Speaks just enough of the wire format: non-streaming message objects, the SSE
event sequence for streaming (thinking, text and tool_use blocks), error
responses, and /v1/messages/count_tokens.
"""

import json
import threading
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

TEXT_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_fake_text",
            "type": "message",
            "role": "assistant",
            "model": "claude-fake-1",
            "content": [],
            "usage": {"input_tokens": 11, "cache_read_input_tokens": 4, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
]

TOOL_EVENTS: List[Dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_fake_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-fake-1",
            "content": [],
            "usage": {"input_tokens": 42, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Running."}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "execute_command", "input": {}},
    },
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"comm'}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": 'and": "ls"}'}},
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 19},
    },
    {"type": "message_stop"},
]

TEXT_RESPONSE: Dict[str, Any] = {
    "id": "msg_fake_sync",
    "type": "message",
    "role": "assistant",
    "model": "claude-fake-1",
    "content": [
        {"type": "thinking", "thinking": "let me think"},
        {"type": "text", "text": "All done."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 21, "output_tokens": 8, "cache_read_input_tokens": 5},
}

TOOL_RESPONSE: Dict[str, Any] = {
    "id": "msg_fake_sync_tool",
    "type": "message",
    "role": "assistant",
    "model": "claude-fake-1",
    "content": [
        {"type": "text", "text": "Listing files."},
        {
            "type": "tool_use",
            "id": "toolu_9",
            "name": "execute_command",
            "input": {"command": "ls -la"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 30, "output_tokens": 12},
}


def sse_frames(events: List[Dict[str, Any]]) -> str:
    """Serialise events exactly the way api.anthropic.com does."""
    out: List[str] = []
    for event in events:
        out.append(f"event: {event['type']}")
        out.append(f"data: {json.dumps(event)}")
        out.append("")
    return "\n".join(out) + "\n"


class FakeAnthropic:
    """uvicorn-hosted stand-in for the upstream Messages API."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.headers_seen: List[Dict[str, str]] = []
        self.mode = "text"
        self.error_status: Optional[int] = None
        self.error_body: Dict[str, Any] = {}
        self.stream_delay = 0.0
        self.last_payload: Dict[str, Any] = {}
        self.app = self._build_app()
        self.server: Optional[uvicorn.Server] = None
        self.thread: Optional[threading.Thread] = None
        self.base_url = ""

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/messages")
        async def messages(request: Request):
            payload = await request.json()
            self.requests.append(payload)
            self.headers_seen.append(dict(request.headers))
            self.last_payload = payload

            if self.error_status:
                return JSONResponse(
                    status_code=self.error_status,
                    content=self.error_body
                    or {
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "slow down"},
                    },
                )

            if payload.get("stream"):
                return StreamingResponse(
                    self._iter_events(self._stream_events()), media_type="text/event-stream"
                )

            return JSONResponse(content=self._message_response())

        @app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request):
            payload = await request.json()
            self.requests.append(payload)
            return JSONResponse(content={"input_tokens": 12})

        return app

    def _stream_events(self) -> List[Dict[str, Any]]:
        return TOOL_EVENTS if self._responds_with_tool() else TEXT_EVENTS

    def _message_response(self) -> Dict[str, Any]:
        return TOOL_RESPONSE if self._responds_with_tool() else TEXT_RESPONSE

    def _responds_with_tool(self) -> bool:
        """'tool' always; 'tool_then_text' only until a tool result comes back."""
        if self.mode == "tool":
            return True
        if self.mode == "tool_then_text":
            return not self._has_tool_result()
        return False

    def _has_tool_result(self) -> bool:
        for message in self.last_payload.get("messages") or []:
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                return True
        return False

    async def _iter_events(self, events: List[Dict[str, Any]]):
        import asyncio

        for event in events:
            if self.stream_delay:
                await asyncio.sleep(self.stream_delay)
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def start(self, port: int = 0) -> str:
        server_config = uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(server_config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()

        deadline = time.time() + 10
        while time.time() < deadline:
            if self.server.started and self.server.servers:
                break
            time.sleep(0.02)
        else:  # pragma: no cover - environment failure
            raise RuntimeError("fake Anthropic server did not start")

        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        return self.base_url

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)