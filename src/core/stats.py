"""In-memory request statistics for the status dashboard.

Single-process counters only (no persistence). Tokens are tracked for
non-streaming responses where the upstream returns a usage block; streaming
responses contribute request counts through add_tokens() when the final
message_delta carries usage.
"""

import threading
import time
from typing import Any, Dict, Optional


class ProxyStats:
    """Thread-safe counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.total_requests = 0
        self.ok_requests = 0
        self.errors = 0
        self.by_endpoint: Dict[str, int] = {}
        self.by_model: Dict[str, int] = {}
        self.by_status: Dict[str, int] = {}
        self.tokens_in = 0
        self.tokens_out = 0
        self.tokens_cached = 0
        self.latency_sum = 0.0
        self.latency_count = 0
        self.last_error_at: Optional[float] = None

    def record(
        self,
        endpoint: str,
        model: Optional[str] = None,
        status: int = 200,
        latency: Optional[float] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one completed request."""
        with self._lock:
            self.total_requests += 1
            if 200 <= status < 400:
                self.ok_requests += 1
            else:
                self.errors += 1
                self.last_error_at = time.time()
            self.by_endpoint[endpoint] = self.by_endpoint.get(endpoint, 0) + 1
            self.by_status[str(status)] = self.by_status.get(str(status), 0) + 1
            if model:
                self.by_model[model] = self.by_model.get(model, 0) + 1
            if latency is not None:
                self.latency_sum += latency
                self.latency_count += 1
            if usage:
                self._add_usage_locked(usage)

    def note_model(self, model: Optional[str]) -> None:
        """Count a request against a model name (endpoint counting happens elsewhere)."""
        if not model:
            return
        with self._lock:
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def add_tokens(self, usage: Optional[Dict[str, Any]]) -> None:
        """Add token usage without counting a new request."""
        if not usage:
            return
        with self._lock:
            self._add_usage_locked(usage)

    def _add_usage_locked(self, usage: Dict[str, Any]) -> None:
        # Anthropic shapes (input/output/cache_read_input_tokens) and OpenAI
        # shapes (prompt/completion_tokens) are both accepted.
        self.tokens_in += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        self.tokens_out += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        self.tokens_cached += int(
            usage.get("cache_read_input_tokens", 0)
            or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            or (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy of all counters."""
        with self._lock:
            avg_latency = self.latency_sum / self.latency_count if self.latency_count else 0.0
            return {
                "uptime_secs": time.time() - self.started_at,
                "started_at": self.started_at,
                "total_requests": self.total_requests,
                "ok_requests": self.ok_requests,
                "errors": self.errors,
                "error_rate": (self.errors / self.total_requests) if self.total_requests else 0.0,
                "last_error_at": self.last_error_at,
                "avg_latency_ms": round(avg_latency * 1000, 1),
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "tokens_cached": self.tokens_cached,
                "by_endpoint": dict(self.by_endpoint),
                "by_model": dict(self.by_model),
                "by_status": dict(self.by_status),
            }


stats = ProxyStats()