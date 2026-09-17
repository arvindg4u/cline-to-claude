"""Pydantic models for the OpenAI Chat Completions wire format Cline speaks.

Everything is deliberately lenient (``extra="allow"``) because Cline sends
provider-specific extras (`reasoning_effort`, `thinking`, cache hints, ...)
that the proxy should ignore rather than reject.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class OpenAIFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    # Cline normally sends a JSON string, but some providers echo a dict.
    arguments: Optional[Union[str, Dict[str, Any]]] = None


class OpenAIToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: Optional[str] = "function"
    index: Optional[int] = None
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_call_id: Optional[str] = None
    # Reasoning traces echoed back by Cline; dropped on the Anthropic side.
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None


class OpenAIToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Optional[str] = "function"
    function: Dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[OpenAIMessage] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[OpenAIToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    thinking: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    user: Optional[str] = None
    n: Optional[int] = None

    def requested_max_tokens(self) -> Optional[int]:
        """Newer OpenAI clients send max_completion_tokens; Cline sends both."""
        return self.max_completion_tokens or self.max_tokens

    def wants_usage_in_stream(self) -> Optional[bool]:
        """`stream_options.include_usage` as sent, or None when unspecified."""
        options = self.stream_options or {}
        if "include_usage" in options:
            return bool(options.get("include_usage"))
        return None