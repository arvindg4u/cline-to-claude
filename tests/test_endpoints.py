"""End-to-end tests: Cline-shaped HTTP in, fake Anthropic upstream out."""

import json

import pytest
from fastapi.testclient import TestClient

from src.main import app
from tests.fake_upstream import FakeAnthropic


@pytest.fixture(scope="module")
def upstream():
    fake = FakeAnthropic()
    fake.start()
    yield fake
    fake.stop()


@pytest.fixture
def client(upstream, point_upstream):
    upstream.mode = "text"
    upstream.error_status = None
    upstream.error_body = {}
    upstream.requests.clear()
    upstream.headers_seen.clear()
    point_upstream(upstream.base_url)
    with TestClient(app) as test_client:
        yield test_client


def sse_payloads(body: str):
    payloads = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def streamed_text(payloads) -> str:
    text = ""
    for payload in payloads:
        if payload == "[DONE]":
            continue
        for choice in payload.get("choices") or []:
            text += choice["delta"].get("content", "")
    return text


CHAT_REQUEST = {
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 128,
}


def test_non_streaming_chat_completion(client, upstream):
    response = client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "All done."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 26

    # The upstream must have received Anthropic wire format with auth headers.
    assert len(upstream.requests) == 1
    sent = upstream.requests[0]
    assert sent["model"] == "claude-sonnet-4-5"
    assert sent["max_tokens"] == 128
    assert "stream" not in sent
    headers = {key.lower(): value for key, value in upstream.headers_seen[0].items()}
    assert headers["x-api-key"] == "test-upstream-key"
    assert headers["anthropic-version"] == "2023-06-01"


def test_streaming_chat_completion(client, upstream):
    response = client.post(
        "/v1/chat/completions",
        json={**CHAT_REQUEST, "stream": True, "stream_options": {"include_usage": True}},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = sse_payloads(response.text)

    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert streamed_text(payloads) == "Hello"
    assert payloads[-2]["usage"]["total_tokens"] == 22
    assert payloads[-1] == "[DONE]"
    assert upstream.requests[0]["stream"] is True


def test_streaming_tool_call_is_reassembled(client, upstream):
    upstream.mode = "tool"
    response = client.post("/v1/chat/completions", json={**CHAT_REQUEST, "stream": True})

    payloads = sse_payloads(response.text)
    deltas = [
        choice["delta"]
        for payload in payloads
        if payload != "[DONE]"
        for choice in payload.get("choices") or []
    ]

    assert streamed_text(payloads) == "Running."
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "hmm"

    arguments = "".join(
        d["tool_calls"][0]["function"]["arguments"]
        for d in deltas
        if d.get("tool_calls") and not d["tool_calls"][0].get("id")
    )
    assert json.loads(arguments) == {"command": "ls"}

    finishes = [
        choice["finish_reason"]
        for payload in payloads
        if payload != "[DONE]"
        for choice in payload.get("choices") or []
        if choice["finish_reason"]
    ]
    assert finishes == ["tool_calls"]


def test_non_streaming_tool_call_is_reassembled(client, upstream):
    upstream.mode = "tool"
    body = client.post("/v1/chat/completions", json=CHAT_REQUEST).json()

    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert message["tool_calls"][0]["function"]["name"] == "execute_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"command": "ls -la"}


def test_upstream_error_status_is_forwarded(client, upstream):
    upstream.error_status = 429

    response = client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert response.status_code == 429
    assert "429" in response.json()["detail"]

    streamed = client.post("/v1/chat/completions", json={**CHAT_REQUEST, "stream": True})
    assert streamed.status_code == 429
    assert "rate_limit_error" in streamed.json()["detail"]


def test_upstream_400_is_forwarded_with_message(client, upstream):
    upstream.error_status = 400
    upstream.error_body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "max_tokens too large"},
    }

    response = client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert response.status_code == 400
    assert "max_tokens too large" in response.json()["detail"]


def test_empty_messages_returns_400(client):
    response = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert response.status_code == 400
    assert "No convertible messages" in response.json()["detail"]


def test_models_listing(client):
    body = client.get("/v1/models").json()

    assert body["object"] == "list"
    ids = [entry["id"] for entry in body["data"]]
    assert "claude-sonnet-4-5" in ids
    assert len(ids) == len(set(ids))


def test_health_and_status(client):
    health = client.get("/health").json()
    assert health["status"] == "healthy"
    assert health["wire"] == "openai-chat -> anthropic-messages"

    client.post("/v1/chat/completions", json=CHAT_REQUEST)
    status = client.get("/api/status").json()
    assert status["proxy"]["models"]["middle"] == "claude-sonnet-4-5"
    assert status["stats"]["total_requests"] >= 1
    assert status["stats"]["by_endpoint"]["/v1/chat/completions"] >= 1
    assert status["stats"]["tokens_in"] > 0


def test_test_connection_reports_upstream_reply(client):
    body = client.get("/test-connection").json()

    assert body["ok"] is True
    assert body["reply"] == "All done."


def test_test_connection_reports_failure(client, upstream):
    upstream.error_status = 401

    body = client.get("/test-connection").json()
    assert body["ok"] is False
    assert body["status"] == 401
    assert "401" in body["error"]


def test_anthropic_passthrough_routes(client, upstream):
    message = client.post(
        "/v1/messages",
        json={"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert message.status_code == 200
    assert message.json()["content"][1]["text"] == "All done."
    assert upstream.requests[-1]["messages"] == [{"role": "user", "content": "hi"}]

    counted = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert counted.status_code == 200
    assert counted.json() == {"input_tokens": 12}


def test_dashboard_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cline" in response.text


def test_client_api_key_enforcement(client, config_overrides):
    config_overrides(client_api_key="shared-secret")

    unauthorized = client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert unauthorized.status_code == 401

    wrong = client.post(
        "/v1/chat/completions", json=CHAT_REQUEST, headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401

    allowed = client.post(
        "/v1/chat/completions",
        json=CHAT_REQUEST,
        headers={"Authorization": "Bearer shared-secret"},
    )
    assert allowed.status_code == 200


def test_chat_completions_alias_without_v1_prefix(client):
    response = client.post("/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "All done."


CLINE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def test_agentic_tool_loop_round_trip(client, upstream):
    """Simulate Cline's loop: tool_call out, tool_result back in."""
    upstream.mode = "tool_then_text"

    first = client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "tools": CLINE_TOOLS}
    ).json()
    assert first["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = first["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "execute_command"

    # The upstream must have seen Anthropic tool definitions.
    assert upstream.requests[0]["tools"][0]["name"] == "execute_command"

    second = client.post(
        "/v1/chat/completions",
        json={
            **CHAT_REQUEST,
            "tools": CLINE_TOOLS,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": tool_call["id"], "content": "total 0\n"},
            ],
        },
    ).json()

    assert second["choices"][0]["message"]["content"] == "All done."
    sent = upstream.requests[-1]["messages"]
    assert sent[1]["content"][0]["type"] == "tool_use"
    assert sent[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": "total 0\n",
                "is_error": None,
            }
        ],
    }