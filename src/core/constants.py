"""Shared string constants for both wire formats.

Keeping the Anthropic and OpenAI vocabularies side by side makes the
translation code self-documenting and avoids typos in the converters.
"""

from typing import Dict


class Constants:
    # ── OpenAI roles (client wire) ──
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_DEVELOPER = "developer"
    ROLE_TOOL = "tool"

    # ── Anthropic content block types ──
    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"
    CONTENT_THINKING = "thinking"
    CONTENT_REDACTED_THINKING = "redacted_thinking"

    # ── Anthropic streaming delta types ──
    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"
    DELTA_THINKING = "thinking_delta"
    DELTA_SIGNATURE = "signature_delta"

    # ─ Anthropic stop reasons ──
    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_STOP_SEQUENCE = "stop_sequence"
    STOP_TOOL_USE = "tool_use"
    STOP_REFUSAL = "refusal"
    STOP_PAUSE_TURN = "pause_turn"
    STOP_ERROR = "error"

    # ── OpenAI finish reasons ──
    FINISH_STOP = "stop"
    FINISH_LENGTH = "length"
    FINISH_TOOL_CALLS = "tool_calls"
    FINISH_CONTENT_FILTER = "content_filter"

    # ── Structured outputs ──
    OUTPUT_JSON_SCHEMA = "json_schema"

    # ── OpenAI tool plumbing ──
    TOOL_FUNCTION = "function"
    TOOL_CHOICE_AUTO = "auto"
    TOOL_CHOICE_NONE = "none"
    TOOL_CHOICE_REQUIRED = "required"

    # ── Anthropic SSE event names ──
    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_PING = "ping"
    EVENT_ERROR = "error"

    # ── Image source types ──
    IMAGE_SOURCE_BASE64 = "base64"
    IMAGE_SOURCE_URL = "url"


# Anthropic stop_reason -> OpenAI finish_reason
STOP_REASON_TO_FINISH_REASON: Dict[str, str] = {
    Constants.STOP_END_TURN: Constants.FINISH_STOP,
    Constants.STOP_STOP_SEQUENCE: Constants.FINISH_STOP,
    Constants.STOP_PAUSE_TURN: Constants.FINISH_STOP,
    Constants.STOP_MAX_TOKENS: Constants.FINISH_LENGTH,
    Constants.STOP_TOOL_USE: Constants.FINISH_TOOL_CALLS,
    Constants.STOP_REFUSAL: Constants.FINISH_CONTENT_FILTER,
}