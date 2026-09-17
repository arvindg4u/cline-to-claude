"""Request translation: OpenAI Chat Completions -> Anthropic Messages."""

import pytest

from src.conversion.request_converter import convert_openai_to_claude
from src.core.model_manager import model_manager
from src.models.openai import OpenAIChatRequest


def build(**payload) -> OpenAIChatRequest:
    return OpenAIChatRequest.model_validate(payload)


def test_system_and_user_text():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "system", "content": "You are Cline."},
                {"role": "developer", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ],
        ),
        model_manager,
    )

    assert payload["system"] == "You are Cline.\n\nBe terse."
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert "stream" not in payload


def test_assistant_tool_call_and_tool_result_round_trip():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": "sure",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "execute_command",
                                "arguments": '{"command": "ls"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
                {"role": "tool", "tool_call_id": "call_1", "content": ""},
            ],
        ),
        model_manager,
    )

    assistant = payload["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {"type": "text", "text": "sure"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "execute_command",
            "input": {"command": "ls"},
        },
    ]

    # Both tool results land in one user turn; empty output is never empty text.
    tool_turn = payload["messages"][2]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "a.txt"},
        {"type": "tool_result", "tool_use_id": "call_1", "content": "(no output)"},
    ]


def test_consecutive_user_messages_merge_with_tool_results_first():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "user", "content": "now what?"},
            ],
        ),
        model_manager,
    )

    assert len(payload["messages"]) == 2
    blocks = payload["messages"][1]["content"]
    assert [block["type"] for block in blocks] == ["tool_result", "text"]
    assert blocks[1]["text"] == "now what?"


def test_images_become_base64_or_url_sources():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,QUJD"},
                        },
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    ],
                }
            ],
        ),
        model_manager,
    )

    blocks = payload["messages"][0]["content"]
    assert blocks[1]["source"] == {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "QUJD",
    }
    assert blocks[2]["source"] == {"type": "url", "url": "https://x/y.png"}


def test_malformed_tool_arguments_are_wrapped_not_dropped():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{not json"},
                        }
                    ],
                }
            ],
        ),
        model_manager,
    )

    assert payload["messages"][0]["content"][0]["input"] == {"_raw": "{not json"}


def test_missing_tool_call_id_gets_a_generated_one():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [{"type": "function", "function": {"name": "read", "arguments": "{}"}}],
                }
            ],
        ),
        model_manager,
    )

    assert payload["messages"][0]["content"][0]["id"].startswith("toolu_")


def test_empty_messages_rejected():
    with pytest.raises(ValueError):
        convert_openai_to_claude(build(model="claude-sonnet-4-5", messages=[]), model_manager)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "run a shell command",
            "parameters": {"properties": {"command": {"type": "string"}}},
        },
    },
    {"type": "function", "function": {"name": "no_schema"}},
    {"type": "function", "function": {"description": "nameless"}},
]


def test_tools_get_object_schemas_and_nameless_entries_are_dropped():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            tools=TOOLS,
        ),
        model_manager,
    )

    tools = payload["tools"]
    assert [tool["name"] for tool in tools] == ["execute_command", "no_schema"]
    assert tools[0]["description"] == "run a shell command"
    # A schema without "type" still has to be a valid object schema upstream.
    assert tools[0]["input_schema"] == {
        "type": "object",
        "properties": {"command": {"type": "string"}},
    }
    assert tools[1]["input_schema"] == {"type": "object", "properties": {}}


@pytest.mark.parametrize(
    "tool_choice,expected,keeps_tools",
    [
        ("auto", {"type": "auto"}, True),
        (None, {"type": "auto"}, True),
        ("required", {"type": "any"}, True),
        (
            {"type": "function", "function": {"name": "execute_command"}},
            {"type": "tool", "name": "execute_command"},
            True,
        ),
        ("none", None, False),
    ],
)
def test_tool_choice_mapping(tool_choice, expected, keeps_tools):
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            tools=[TOOLS[0]],
            tool_choice=tool_choice,
        ),
        model_manager,
    )

    assert payload.get("tool_choice") == expected
    assert ("tools" in payload) is keeps_tools


def test_parallel_tool_calls_false_disables_parallel_use():
    payload = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            tools=[TOOLS[0]],
            parallel_tool_calls=False,
        ),
        model_manager,
    )

    assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_token_limits_stop_sequences_and_metadata(config_overrides):
    config_overrides(default_max_tokens=8192, min_tokens_limit=100, max_tokens_limit=5000)

    defaulted = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            stop=["END"],
            user="cline-user",
        ),
        model_manager,
    )
    assert defaulted["max_tokens"] == 5000  # default 8192 clamped to the limit
    assert defaulted["stop_sequences"] == ["END"]
    assert defaulted["metadata"] == {"user_id": "cline-user"}

    clamped = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            max_completion_tokens=999999,
            temperature=3.5,
            top_p=-1,
        ),
        model_manager,
    )
    assert clamped["max_tokens"] == 5000
    assert clamped["temperature"] == 1.0
    assert clamped["top_p"] == 0.0

    floored = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=5,
        ),
        model_manager,
    )
    assert floored["max_tokens"] == 100


def test_thinking_is_opt_in_and_scales_with_effort(config_overrides):
    config_overrides(thinking_budget_tokens=0)
    plain = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            reasoning_effort="xhigh",
        ),
        model_manager,
    )
    assert "thinking" not in plain

    config_overrides(
        thinking_budget_tokens=8000, thinking_min_budget=1024, max_tokens_limit=128000
    )
    enabled = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=4000,
            temperature=0.2,
            reasoning_effort="high",
        ),
        model_manager,
    )
    assert enabled["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    # Anthropic needs max_tokens > budget_tokens and temperature == 1.
    assert enabled["max_tokens"] > 8000
    assert enabled["temperature"] == 1.0

    xhigh = convert_openai_to_claude(
        build(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "go"}],
            reasoning={"enabled": True, "effort": "xhigh"},
        ),
        model_manager,
    )
    assert xhigh["thinking"]["budget_tokens"] == 16000

    silent = convert_openai_to_claude(
        build(model="claude-sonnet-4-5", messages=[{"role": "user", "content": "go"}]),
        model_manager,
    )
    assert "thinking" not in silent


@pytest.mark.parametrize(
    "requested,expected",
    [
        ("claude-opus-4-1", "claude-opus-4-1"),
        ("sonnet", "claude-sonnet-4-5"),
        ("claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"),
        ("my-haiku", "claude-haiku-4-5"),
        ("gpt-4o", "claude-sonnet-4-5"),
        ("", "claude-sonnet-4-5"),
    ],
)
def test_model_resolution(requested, expected):
    assert model_manager.resolve(requested) == expected


def test_model_map_and_passthrough(config_overrides):
    config_overrides(model_map={"fast": "claude-haiku-4-5"}, passthrough_unknown_models=False)
    assert model_manager.resolve("fast") == "claude-haiku-4-5"
    assert model_manager.resolve("FAST") == "claude-haiku-4-5"
    assert model_manager.resolve("mystery-model") == "claude-sonnet-4-5"

    config_overrides(passthrough_unknown_models=True)
    assert model_manager.resolve("mystery-model") == "mystery-model"