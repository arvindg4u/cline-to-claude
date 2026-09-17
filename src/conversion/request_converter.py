"""Convert an OpenAI Chat Completions request into an Anthropic Messages request.

The tricky parts are (a) Anthropic keeps the system prompt out of `messages`,
(b) tool results must live in `user` turns as `tool_result` blocks, and
(c) turns must not repeat the same role back to back, so consecutive user /
tool messages are merged into a single user turn.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

from src.core.config import config
from src.core.constants import Constants
from src.core.logging import logger
from src.models.openai import OpenAIChatRequest, OpenAIMessage

TEXT_PART_TYPES = {"text", "input_text", "output_text"}
IMAGE_PART_TYPES = {"image_url", "input_image", "image"}
TOOL_RESULT_PART_TYPES = {"tool_result"}


def convert_openai_to_claude(request: OpenAIChatRequest, model_manager) -> Dict[str, Any]:
    """Translate the whole request. Raises ValueError on unusable input."""
    system_parts: List[str] = []
    claude_messages: List[Dict[str, Any]] = []

    for message in request.messages:
        role = (message.role or Constants.ROLE_USER).strip().lower()

        if role in (Constants.ROLE_SYSTEM, Constants.ROLE_DEVELOPER):
            text = _flatten_text(message.content)
            if text.strip():
                system_parts.append(text.strip())
            continue

        if role == Constants.ROLE_TOOL:
            _append_user_blocks(claude_messages, [_tool_result_block(message)])
            continue

        if role == Constants.ROLE_ASSISTANT:
            blocks = _assistant_blocks(message)
            if blocks:
                claude_messages.append({"role": Constants.ROLE_ASSISTANT, "content": blocks})
            continue

        blocks = _user_blocks(message)
        if blocks:
            _append_user_blocks(claude_messages, blocks)

    if not claude_messages:
        raise ValueError("No convertible messages were provided in the request.")

    model = model_manager.resolve(request.model or "")
    tools = _convert_tools(request.tools)
    tool_choice = _convert_tool_choice(request.tool_choice, tools, request.parallel_tool_calls)
    if tool_choice is None and _tool_choice_is_none(request.tool_choice):
        tools = []

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": config.clamp_max_tokens(request.requested_max_tokens()),
        "messages": claude_messages,
    }

    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if request.stream:
        payload["stream"] = True

    if request.temperature is not None:
        payload["temperature"] = _clamp_unit(request.temperature)
    if request.top_p is not None:
        payload["top_p"] = _clamp_unit(request.top_p)

    stop_sequences = _normalise_stop(request.stop)
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences

    metadata = _normalise_metadata(request)
    if metadata:
        payload["metadata"] = metadata

    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    _apply_thinking(payload, request)
    return payload


# ── Text and content-block helpers ────────────────────────────────────────
def _flatten_text(content: Any) -> str:
    """Collect every text fragment of an OpenAI content value into one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                block_type = item.get("type")
                if block_type in TEXT_PART_TYPES or "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _order_user_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Anthropic requires tool_result blocks first inside a user turn."""
    tool_results = [b for b in blocks if b.get("type") == Constants.CONTENT_TOOL_RESULT]
    others = [b for b in blocks if b.get("type") != Constants.CONTENT_TOOL_RESULT]
    return tool_results + others


def _append_user_blocks(claude_messages: List[Dict[str, Any]], blocks: List[Dict[str, Any]]) -> None:
    """Append blocks to the trailing user turn, merging when one already exists."""
    if not blocks:
        return
    if claude_messages and claude_messages[-1]["role"] == Constants.ROLE_USER:
        existing = claude_messages[-1]["content"]
        merged: List[Dict[str, Any]] = []
        if isinstance(existing, str):
            if existing.strip():
                merged.append({"type": Constants.CONTENT_TEXT, "text": existing})
        elif isinstance(existing, list):
            merged.extend(existing)
        merged.extend(blocks)
        claude_messages[-1]["content"] = _order_user_blocks(merged)
        return
    claude_messages.append(
        {
            "role": Constants.ROLE_USER,
            "content": _order_user_blocks(blocks),
        }
    )


def _user_blocks(message: OpenAIMessage) -> List[Dict[str, Any]]:
    """Build user content blocks from a string or multimodal part list."""
    content = message.content
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": Constants.CONTENT_TEXT, "text": content}] if content.strip() else []

    if not isinstance(content, list):
        return [{"type": Constants.CONTENT_TEXT, "text": str(content)}]

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                blocks.append({"type": Constants.CONTENT_TEXT, "text": item})
            continue
        if not isinstance(item, dict):
            continue
        part_type = item.get("type")

        if part_type in TEXT_PART_TYPES:
            text = str(item.get("text", ""))
            if text.strip():
                blocks.append({"type": Constants.CONTENT_TEXT, "text": text})
        elif part_type in IMAGE_PART_TYPES:
            block = _image_block(item)
            if block:
                blocks.append(block)
        elif part_type in TOOL_RESULT_PART_TYPES:
            tool_use_id = item.get("tool_use_id") or item.get("tool_call_id") or ""
            blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_RESULT,
                    "tool_use_id": tool_use_id,
                    "content": _tool_result_content(item.get("content")),
                }
            )
        elif part_type == Constants.CONTENT_THINKING:
            # Reasoning from a previous turn is never replayed upstream.
            continue
        elif item.get("text"):
            blocks.append({"type": Constants.CONTENT_TEXT, "text": str(item["text"])})
        else:
            logger.debug(f"Dropping unsupported user content part: {part_type}")
    return blocks


def _image_block(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """data: URLs become base64 sources; http(s) URLs become url sources."""
    source = item.get("source")
    if isinstance(source, dict):
        # Already an Anthropic-shaped image block.
        return {"type": Constants.CONTENT_IMAGE, "source": source}

    url = None
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    elif isinstance(image_url, str):
        url = image_url
    if not url:
        return None
    if not url.startswith("data:"):
        return {
            "type": Constants.CONTENT_IMAGE,
            "source": {"type": Constants.IMAGE_SOURCE_URL, "url": url},
        }

    header, _, data = url.partition(",")
    media_type = "image/png"
    if ";" in header:
        media_type = header[5:].split(";")[0] or media_type
    return {
        "type": Constants.CONTENT_IMAGE,
        "source": {
            "type": Constants.IMAGE_SOURCE_BASE64,
            "media_type": media_type,
            "data": data,
        },
    }


def _assistant_blocks(message: OpenAIMessage) -> List[Dict[str, Any]]:
    """Assistant text plus tool_use blocks (thinking is intentionally dropped)."""
    blocks: List[Dict[str, Any]] = []
    text = _flatten_text(message.content)
    if text.strip():
        blocks.append({"type": Constants.CONTENT_TEXT, "text": text})

    for tool_call in message.tool_calls or []:
        name = tool_call.function.name if tool_call.function else None
        if not name:
            logger.debug("Skipping assistant tool_call without a function name")
            continue
        blocks.append(
            {
                "type": Constants.CONTENT_TOOL_USE,
                "id": tool_call.id or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "input": _parse_arguments(tool_call.function.arguments),
            }
        )
    return blocks


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    """Cline sends JSON strings; tolerate dicts and malformed strings."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("tool_call arguments were not valid JSON; wrapping raw text")
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": arguments}


# ── Tool result helpers ───────────────────────────────────────────────────
def _tool_result_block(message: OpenAIMessage) -> Dict[str, Any]:
    """Build an Anthropic tool_result block from an OpenAI `tool` message."""
    tool_use_id = message.tool_call_id or ""
    if not tool_use_id:
        logger.debug("tool message without tool_call_id; upstream may reject it")
    return {
        "type": Constants.CONTENT_TOOL_RESULT,
        "tool_use_id": tool_use_id,
        "content": _tool_result_content(message.content),
    }


def _tool_result_content(content: Any):
    """Anthropic accepts a string or a list of text/image blocks."""
    if content is None:
        return "(no output)"
    if isinstance(content, str):
        return content if content.strip() else "(no output)"
    if isinstance(content, dict):
        if content.get("type") in TEXT_PART_TYPES and "text" in content:
            text = str(content.get("text", ""))
            return text if text.strip() else "(no output)"
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        strings: List[str] = []
        blocks: List[Dict[str, Any]] = []
        has_rich = False
        for item in content:
            if isinstance(item, str):
                strings.append(item)
                continue
            if not isinstance(item, dict):
                strings.append(str(item))
                continue
            part_type = item.get("type")
            if part_type in IMAGE_PART_TYPES:
                image = _image_block(item)
                if image:
                    has_rich = True
                    blocks.append(image)
                continue
            if part_type in TEXT_PART_TYPES or "text" in item:
                strings.append(str(item.get("text", "")))
            else:
                strings.append(json.dumps(item, ensure_ascii=False))

        text = "\n".join(part for part in strings if part)
        if not has_rich:
            return text if text.strip() else "(no output)"
        ordered: List[Dict[str, Any]] = []
        if text.strip():
            ordered.append({"type": Constants.CONTENT_TEXT, "text": text})
        ordered.extend(blocks)
        return ordered
    return str(content)


# ── Tools and tool choice ─────────────────────────────────────────────────
def _normalise_schema(schema: Any) -> Dict[str, Any]:
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    if "type" not in schema:
        schema = {"type": "object", **schema}
    return schema


def _convert_tools(tools: Any) -> List[Dict[str, Any]]:
    """OpenAI function tools -> Anthropic tool definitions."""
    result: List[Dict[str, Any]] = []
    for tool in tools or []:
        function = tool.function if hasattr(tool, "function") else (tool or {}).get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not name:
            logger.debug("Skipping tool definition without a name")
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "input_schema": _normalise_schema(function.get("parameters") or function.get("input_schema")),
        }
        description = function.get("description")
        if description:
            entry["description"] = str(description)
        result.append(entry)
    return result


def _tool_choice_is_none(tool_choice: Any) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == Constants.TOOL_CHOICE_NONE


def _convert_tool_choice(
    tool_choice: Any, tools: List[Dict[str, Any]], parallel_tool_calls: Optional[bool]
) -> Optional[Dict[str, Any]]:
    """OpenAI tool_choice -> Anthropic tool_choice (None when tools are absent)."""
    if not tools or _tool_choice_is_none(tool_choice):
        return None

    choice: Dict[str, Any] = {"type": Constants.TOOL_CHOICE_AUTO}
    if isinstance(tool_choice, str):
        mode = tool_choice.strip().lower()
        if mode in ("required", "any"):
            choice = {"type": "any"}
        elif mode in ("auto", "", "default"):
            choice = {"type": Constants.TOOL_CHOICE_AUTO}
    elif isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type", "")).strip().lower()
        function = tool_choice.get("function")
        if choice_type == Constants.TOOL_FUNCTION and isinstance(function, dict) and function.get("name"):
            choice = {"type": "tool", "name": function["name"]}
        elif choice_type == "tool" and tool_choice.get("name"):
            choice = {"type": "tool", "name": tool_choice["name"]}
        elif choice_type in ("required", "any"):
            choice = {"type": "any"}
        elif choice_type == "none":
            return None
        else:
            choice = {"type": Constants.TOOL_CHOICE_AUTO}

    if parallel_tool_calls is False:
        choice["disable_parallel_tool_use"] = True
    return choice


# ─ Misc request normalisation ────────────────────────────────────────────
def _normalise_stop(stop: Any) -> List[str]:
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, (list, tuple)):
        return [str(item) for item in stop if isinstance(item, str) and item]
    return []


def _normalise_metadata(request: OpenAIChatRequest) -> Dict[str, Any]:
    """Anthropic metadata only supports user_id; everything else is dropped."""
    metadata = request.metadata if isinstance(request.metadata, dict) else {}
    user_id = metadata.get("user_id") or request.user
    return {"user_id": str(user_id)} if user_id else {}


def _clamp_unit(value: Any) -> float:
    """Anthropic temperature/top_p range is 0..1."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    return max(0.0, min(1.0, number))


def _requested_thinking_budget(request: OpenAIChatRequest) -> int:
    """Resolve how many thinking tokens the client asked for (0 = disabled)."""
    if config.thinking_budget_tokens <= 0:
        return 0

    thinking = request.thinking if isinstance(request.thinking, dict) else None
    if thinking:
        explicit = thinking.get("budget_tokens")
        if isinstance(explicit, (int, float)) and explicit > 0:
            return int(explicit)
        if str(thinking.get("type", "")).lower() == "disabled":
            return 0

    effort = request.reasoning_effort
    extra = getattr(request, "model_extra", None) or {}
    reasoning = extra.get("reasoning") if isinstance(extra, dict) else None
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return 0
        effort = effort or reasoning.get("effort")

    if effort is None and thinking is None and not isinstance(reasoning, dict):
        # The client never asked for reasoning; do not force thinking on.
        return 0
    return config.thinking_budget(effort)


def _apply_thinking(payload: Dict[str, Any], request: OpenAIChatRequest) -> None:
    """Enable extended thinking and satisfy Anthropic's constraints for it."""
    budget = _requested_thinking_budget(request)
    if budget <= 0:
        return

    # Anthropic requires budget_tokens < max_tokens.
    if budget >= payload["max_tokens"]:
        raised = config.clamp_max_tokens(budget + 1024)
        if raised <= budget:
            logger.warning(
                "Skipping extended thinking: THINKING_BUDGET_TOKENS exceeds MAX_TOKENS_LIMIT"
            )
            return
        payload["max_tokens"] = raised

    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    # Thinking is incompatible with a custom temperature upstream.
    payload["temperature"] = 1.0