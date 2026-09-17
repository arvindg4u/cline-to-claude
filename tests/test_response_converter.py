"""Response translation: Anthropic Messages -> OpenAI Chat Completions."""

import asyncio
import json

import pytest

from src.conversion.response_converter import (
    KEEPALIVE_COMMENT,
    OpenAIStreamTranslator,
    convert_claude_stream_to_openai,
    convert_claude_to_openai_response,
    map_finish_reason,
    openai_usage,
)
from tests.fake_upstream import TEXT_EVENTS, TEXT_RESPONSE, TOOL_EVENTS, TOOL_RESPONSE


def parse_frames(frames):
    """Turn raw SSE strings into parsed payloads (or the '[DONE]' marker)."""
    parsed = []
    for frame in frames:
        for line in frame.strip().splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            parsed.append(payload if payload == "[DONE]" else json.loads(payload))
    return parsed


def deltas(chunks):
    """Extract the delta dicts from a parsed chunk list."""
    out = []
    for chunk in chunks:
        if chunk == "[DONE]":
            continue
        for choice in chunk.get("choices") or []:
            out.append(choice["delta"])
    return out


def stream_chunks(events):
    translator = OpenAIStreamTranslator("cline-model", "req-1", True)
    frames = []
    for event in events:
        frames.extend(translator.handle(event))
    frames.extend(translator.finalize())
    return translator, parse_frames(frames)


# ── Non-streaming ─────────────────────────────────────────────────────────
def test_non_streaming_text_response():
    response = convert_claude_to_openai_response(TEXT_RESPONSE, "cline-model")

    assert response["object"] == "chat.completion"
    assert response["model"] == "cline-model"
    assert response["id"] == "chatcmpl-msg_fake_sync"
    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "All done."
    assert choice["message"]["reasoning_content"] == "let me think"
    assert "tool_calls" not in choice["message"]

    # cache reads count as prompt tokens; cached count is reported separately.
    assert response["usage"] == {
        "prompt_tokens": 26,
        "completion_tokens": 8,
        "total_tokens": 34,
        "prompt_tokens_details": {"cached_tokens": 5},
    }


def test_non_streaming_tool_response():
    response = convert_claude_to_openai_response(TOOL_RESPONSE, "cline-model")

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Listing files."
    assert choice["message"]["tool_calls"] == [
        {
            "id": "toolu_9",
            "type": "function",
            "function": {"name": "execute_command", "arguments": '{"command": "ls -la"}'},
        }
    ]


def test_tool_only_response_has_null_content():
    response = convert_claude_to_openai_response(
        {
            "id": "msg_x",
            "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}],
            "stop_reason": "tool_use",
        }
    )

    assert response["choices"][0]["message"]["content"] is None
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "{}"


@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        (None, "stop"),
        ("something_new", "stop"),
    ],
)
def test_finish_reason_mapping(stop_reason, expected):
    assert map_finish_reason(stop_reason) == expected


def test_usage_defaults_to_zeroes():
    assert openai_usage(None) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


# ── Streaming ─────────────────────────────────────────────────────────────
def test_streaming_text_translation():
    translator, chunks = stream_chunks(TEXT_EVENTS)

    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(d.get("content", "") for d in deltas(chunks)) == "Hello"

    finish = chunks[-3]
    assert finish["choices"][0]["finish_reason"] == "stop"
    assert finish["id"] == "chatcmpl-msg_fake_text"

    usage_chunk = chunks[-2]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 7,
        "total_tokens": 22,
        "prompt_tokens_details": {"cached_tokens": 4},
    }

    assert chunks[-1] == "[DONE]"
    assert translator.last_usage["total_tokens"] == 22
    assert translator.finish_reason == "stop"


def test_streaming_tool_use_and_thinking():
    translator, chunks = stream_chunks(TOOL_EVENTS)
    streamed = deltas(chunks)

    assert "".join(d.get("reasoning_content", "") for d in streamed) == "hmm"
    assert "".join(d.get("content", "") for d in streamed) == "Running."

    tool_start = [d for d in streamed if d.get("tool_calls") and d["tool_calls"][0].get("id")]
    assert len(tool_start) == 1
    assert tool_start[0]["tool_calls"][0]["index"] == 0
    assert tool_start[0]["tool_calls"][0]["id"] == "toolu_1"
    assert tool_start[0]["tool_calls"][0]["function"] == {
        "name": "execute_command",
        "arguments": "",
    }

    fragments = "".join(
        d["tool_calls"][0]["function"]["arguments"]
        for d in streamed
        if d.get("tool_calls") and not d["tool_calls"][0].get("id")
    )
    assert json.loads(fragments) == {"command": "ls"}

    assert chunks[-3]["choices"][0]["finish_reason"] == "tool_calls"
    assert translator.tool_calls_seen == 1
    assert translator.last_usage == {
        "prompt_tokens": 42,
        "completion_tokens": 19,
        "total_tokens": 61,
    }


def test_streaming_without_usage_chunk():
    translator = OpenAIStreamTranslator("m", "req", include_usage=False)
    frames = []
    for event in TEXT_EVENTS:
        frames.extend(translator.handle(event))
    frames.extend(translator.finalize())

    chunks = parse_frames(frames)
    assert chunks[-1] == "[DONE]"
    assert all("usage" not in chunk for chunk in chunks[:-1])
    assert translator.last_usage is None


def test_streaming_error_event_terminates_stream():
    translator = OpenAIStreamTranslator("m", "req", True)
    frames = translator.handle(
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    )
    frames.extend(translator.finalize())

    chunks = parse_frames(frames)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["error"]["message"] == "busy"
    assert chunks[1]["error"]["code"] == "upstream_error"
    assert chunks[-1] == "[DONE]"


def test_unknown_and_ping_events_are_ignored():
    translator = OpenAIStreamTranslator("m", "req", True)
    assert translator.handle({"type": "ping"}) == []
    assert translator.handle({"type": "content_block_stop", "index": 0})[0].startswith("data: ")


def test_keepalive_comments_during_upstream_silence():
    async def upstream():
        yield f"data: {json.dumps(TEXT_EVENTS[0])}"
        await asyncio.sleep(0.15)  # longer than the keepalive interval
        for event in TEXT_EVENTS[1:]:
            yield f"data: {json.dumps(event)}"

    async def collect():
        frames = []
        async for frame in convert_claude_stream_to_openai(
            upstream(), "cline-model", "req-keepalive", True, 0.05
        ):
            frames.append(frame)
        return frames

    frames = asyncio.run(collect())
    assert KEEPALIVE_COMMENT in frames

    chunks = parse_frames(frames)
    assert chunks[-1] == "[DONE]"
    assert "".join(d.get("content", "") for d in deltas(chunks)) == "Hello"


def test_stream_without_keepalive_still_completes():
    async def upstream():
        for event in TEXT_EVENTS:
            yield f"data: {json.dumps(event)}"

    async def collect():
        return [
            frame
            async for frame in convert_claude_stream_to_openai(
                upstream(), "m", "req", True, 0
            )
        ]

    chunks = parse_frames(asyncio.run(collect()))
    assert chunks[-1] == "[DONE]"
    assert "".join(d.get("content", "") for d in deltas(chunks)) == "Hello"


def test_upstream_failure_mid_stream_surfaces_as_error_frame():
    async def upstream():
        yield f"data: {json.dumps(TEXT_EVENTS[0])}"
        raise RuntimeError("connection reset")

    async def collect():
        return [
            frame
            async for frame in convert_claude_stream_to_openai(
                upstream(), "m", "req-boom", True, 0
            )
        ]

    chunks = parse_frames(asyncio.run(collect()))
    assert chunks[1]["error"]["message"] == "connection reset"
    assert chunks[-1] == "[DONE]"