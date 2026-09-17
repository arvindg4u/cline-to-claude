"""Convert Anthropic Messages responses into the OpenAI shape Cline expects.

Non-streaming responses are a single function; streaming uses a small stateful
translator because Anthropic's content-block events must be re-projected onto
OpenAI's flat `delta` chunks (notably: tool arguments arrive as
`input_json_delta` fragments that accumulate per tool index).
"""

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.core.constants import STOP_REASON_TO_FINISH_REASON, Constants
from src.core.logging import logger

KEEPALIVE_COMMENT = ": keepalive\n\n"


def sse(payload: Dict[str, Any]) -> str:
    """Serialise one OpenAI SSE `data:` frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Anthropic usage -> OpenAI usage (cache reads count as prompt tokens)."""
    usage = usage or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    prompt_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_creation
    completion_tokens = int(usage.get("output_tokens") or 0)
    result: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cache_read:
        result["prompt_tokens_details"] = {"cached_tokens": cache_read}
    if cache_creation:
        result["cache_creation_input_tokens"] = cache_creation
    return result


def map_finish_reason(stop_reason: Optional[str], has_tool_calls: bool = False) -> str:
    """Anthropic stop_reason -> OpenAI finish_reason."""
    if has_tool_calls and stop_reason in (None, Constants.STOP_TOOL_USE):
        return Constants.FINISH_TOOL_CALLS
    if not stop_reason:
        return Constants.FINISH_STOP
    return STOP_REASON_TO_FINISH_REASON.get(stop_reason, Constants.FINISH_STOP)


def convert_claude_to_openai_response(
    claude_response: Dict[str, Any], requested_model: Optional[str] = None
) -> Dict[str, Any]:
    """Anthropic message object -> OpenAI ChatCompletion object."""
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in claude_response.get("content") or []:
        block_type = block.get("type")
        if block_type == Constants.CONTENT_TEXT:
            text_parts.append(block.get("text", ""))
        elif block_type == Constants.CONTENT_THINKING:
            reasoning_parts.append(block.get("thinking", ""))
        elif block_type == Constants.CONTENT_TOOL_USE:
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": Constants.TOOL_FUNCTION,
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )

    message: Dict[str, Any] = {
        "role": Constants.ROLE_ASSISTANT,
        "content": "".join(text_parts) if text_parts else None,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    message_id = claude_response.get("id") or f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": message_id if message_id.startswith("chatcmpl-") else f"chatcmpl-{message_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model or claude_response.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": map_finish_reason(
                    claude_response.get("stop_reason"), bool(tool_calls)
                ),
            }
        ],
        "usage": openai_usage(claude_response.get("usage")),
    }


class OpenAIStreamTranslator:
    """Statefully re-project Anthropic SSE events onto OpenAI chat chunks."""

    def __init__(
        self,
        requested_model: Optional[str] = None,
        request_id: str = "",
        include_usage: bool = True,
    ) -> None:
        self.requested_model = requested_model or ""
        self.request_id = request_id
        self.include_usage = include_usage
        self.created = int(time.time())
        self.message_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.model = requested_model or ""
        self.started = False
        self.finish_reason: Optional[str] = None
        self.usage: Dict[str, Any] = {}
        self.last_usage: Optional[Dict[str, Any]] = None
        self.tool_calls_seen = 0
        self.block_tool_index: Dict[int, int] = {}
        self.error: Optional[Dict[str, Any]] = None

    # ── Frame builders ────────────────────────────────────────────────────
    def chunk(
        self,
        delta: Optional[Dict[str, Any]],
        finish_reason: Optional[str] = None,
        choices: bool = True,
    ) -> str:
        payload: Dict[str, Any] = {
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        if choices and delta is not None:
            payload["choices"] = [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ]
        else:
            payload["choices"] = []
        return sse(payload)

    def ensure_started(self) -> List[str]:
        """Emit the assistant role chunk exactly once, before any content."""
        if self.started:
            return []
        self.started = True
        return [self.chunk({"role": Constants.ROLE_ASSISTANT, "content": ""})]

    # ── Event handling ───────────────────────────────────────────────────
    def handle(self, event: Dict[str, Any]) -> List[str]:
        """Translate one Anthropic SSE event into zero or more SSE frames."""
        event_type = event.get("type") or event.get("event")

        if event_type == Constants.EVENT_PING:
            return []
        if event_type == Constants.EVENT_ERROR:
            error = event.get("error") or {}
            self.error = {
                "message": error.get("message") or "upstream stream error",
                "type": error.get("type") or "api_error",
            }
            logger.warning(f"[{self.request_id}] upstream SSE error: {self.error['message']}")
            return []

        frames = self.ensure_started()

        if event_type == Constants.EVENT_MESSAGE_START:
            message = event.get("message") or {}
            if message.get("id"):
                self.message_id = f"chatcmpl-{message['id']}"
            self.model = self.model or message.get("model") or ""
            self.usage.update(message.get("usage") or {})
        elif event_type == Constants.EVENT_CONTENT_BLOCK_START:
            frames.extend(self._on_block_start(event))
        elif event_type == Constants.EVENT_CONTENT_BLOCK_DELTA:
            frames.extend(self._on_block_delta(event))
        elif event_type == Constants.EVENT_MESSAGE_DELTA:
            delta = event.get("delta") or {}
            self.finish_reason = map_finish_reason(
                delta.get("stop_reason"), self.tool_calls_seen > 0
            )
            self.usage.update(event.get("usage") or {})
        return frames

    def _on_block_start(self, event: Dict[str, Any]) -> List[str]:
        block = event.get("content_block") or {}
        if block.get("type") != Constants.CONTENT_TOOL_USE:
            return []
        tool_index = self.tool_calls_seen
        self.tool_calls_seen += 1
        self.block_tool_index[int(event.get("index") or 0)] = tool_index
        return [
            self.chunk(
                {
                    "tool_calls": [
                        {
                            "index": tool_index,
                            "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "type": Constants.TOOL_FUNCTION,
                            "function": {"name": block.get("name") or "", "arguments": ""},
                        }
                    ]
                }
            )
        ]

    def _on_block_delta(self, event: Dict[str, Any]) -> List[str]:
        delta = event.get("delta") or {}
        delta_type = delta.get("type")

        if delta_type == Constants.DELTA_TEXT:
            text = delta.get("text") or ""
            return [self.chunk({"content": text})] if text else []
        if delta_type == Constants.DELTA_THINKING:
            thinking = delta.get("thinking") or ""
            return [self.chunk({"reasoning_content": thinking})] if thinking else []
        if delta_type == Constants.DELTA_INPUT_JSON:
            tool_index = self.block_tool_index.get(int(event.get("index") or 0))
            if tool_index is None:
                return []
            return [
                self.chunk(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "function": {"arguments": delta.get("partial_json") or ""},
                            }
                        ]
                    }
                )
            ]
        # signature_delta and anything else carry no client-visible content.
        return []

    # ── Termination ───────────────────────────────────────────────────────
    def finalize(self) -> List[str]:
        """Finishing chunk, optional usage chunk, then [DONE]."""
        frames = self.ensure_started()

        if self.error:
            frames.append(
                sse(
                    {
                        "error": {
                            "message": self.error["message"],
                            "type": self.error["type"],
                            "code": "upstream_error",
                        }
                    }
                )
            )
            frames.append("data: [DONE]\n\n")
            return frames

        finish_reason = self.finish_reason or map_finish_reason(
            None, self.tool_calls_seen > 0
        )
        frames.append(self.chunk({}, finish_reason=finish_reason))

        if self.include_usage:
            self.last_usage = openai_usage(self.usage)
            frames.append(
                sse(
                    {
                        "id": self.message_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": self.last_usage,
                    }
                )
            )
        frames.append("data: [DONE]\n\n")
        return frames


def _parse_sse_line(line: str, current_event: Optional[str]):
    """Return (event_dict_or_None, new_current_event_name)."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None, current_event
    if stripped.startswith("event:"):
        return None, stripped[len("event:"):].strip()
    if not stripped.startswith("data:"):
        return None, current_event

    data = stripped[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None, current_event
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        logger.debug(f"Skipping unparsable SSE data: {data[:120]}")
        return None, current_event
    if not isinstance(event, dict):
        return None, current_event
    if not event.get("type") and current_event:
        event["type"] = current_event
    return event, current_event


async def convert_claude_stream_to_openai(
    upstream_lines: AsyncGenerator[str, None],
    requested_model: Optional[str] = None,
    request_id: str = "",
    include_usage: bool = True,
    keepalive_secs: float = 15.0,
    translator: Optional[OpenAIStreamTranslator] = None,
) -> AsyncGenerator[str, None]:
    """Re-emit an Anthropic SSE stream as OpenAI chat completion chunks.

    A pre-built ``translator`` may be supplied so the caller can read
    ``translator.last_usage`` once the stream has been drained.
    """
    translator = translator or OpenAIStreamTranslator(
        requested_model, request_id, include_usage
    )

    # Send the role chunk immediately so the client starts rendering.
    for frame in translator.ensure_started():
        yield frame

    iterator = upstream_lines.__aiter__()
    pending: Optional[asyncio.Task] = None
    current_event: Optional[str] = None

    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())

            if keepalive_secs and keepalive_secs > 0:
                # Wait without cancelling the pending read: a timeout just
                # emits a comment frame and resumes waiting on the same task,
                # so an idle upstream never ends the stream.
                done, _ = await asyncio.wait({pending}, timeout=keepalive_secs)
                if not done:
                    yield KEEPALIVE_COMMENT
                    continue
            else:
                await asyncio.wait({pending})

            try:
                line = pending.result()
            except StopAsyncIteration:
                pending = None
                break
            except Exception as e:
                pending = None
                logger.warning(f"[{request_id}] upstream stream aborted: {e}")
                translator.error = {"message": str(e), "type": "api_error"}
                break
            pending = None

            event, current_event = _parse_sse_line(line, current_event)
            if event is None:
                continue
            for frame in translator.handle(event):
                yield frame
            if translator.error:
                break
    finally:
        if pending is not None:
            pending.cancel()
            try:
                await pending
            except BaseException:
                pass
        try:
            await upstream_lines.aclose()
        except Exception:
            pass

    for frame in translator.finalize():
        yield frame