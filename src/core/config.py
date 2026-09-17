"""Configuration for the Cline-to-Claude proxy.

Cline (VS Code extension or CLI, `openai-compatible` provider) speaks the
OpenAI Chat Completions wire format. This proxy translates that into the
Anthropic Messages API, so the upstream is anything that speaks Anthropic
wire: api.anthropic.com, an Anthropic-compatible gateway, or another local
proxy (for example claude-code-proxy on 127.0.0.1:4013).
"""

import os
import sys
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

# Load .env from the project root before any variable is read. Real
# environment variables win over the file so systemd/docker overrides work.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        print(f"Warning: {name}='{raw}' is not a number, using {default}.")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        print(f"Warning: {name}='{raw}' is not a number, using {default}.")
        return default


def parse_model_map(raw: Optional[str]) -> Dict[str, str]:
    """Parse MODEL_MAP, e.g. ``gpt-4o=claude-opus-4-1,my-alias:claude-haiku-4-5``.

    Both ``=`` and ``:`` separators are accepted (Anthropic model ids contain
    ``-`` not ``:``, so a colon is unambiguous). Malformed entries are skipped
    with a warning instead of failing startup.
    """
    mapping: Dict[str, str] = {}
    if not raw:
        return mapping
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        sep = "=" if "=" in part else (":" if ":" in part else None)
        if sep is None:
            print(f"Warning: MODEL_MAP entry '{part}' is not 'from=to', skipped.")
            continue
        src, dst = part.split(sep, 1)
        src, dst = src.strip(), dst.strip()
        if src and dst:
            mapping[src] = dst
        else:
            print(f"Warning: MODEL_MAP entry '{part}' has an empty side, skipped.")
    return mapping


# Reasoning effort reported by Cline -> multiplier on THINKING_BUDGET_TOKENS.
EFFORT_MULTIPLIERS: Dict[str, float] = {
    "minimal": 0.1,
    "low": 0.25,
    "medium": 0.5,
    "high": 1.0,
    "xhigh": 2.0,
}


class Config:
    def __init__(self) -> None:
        # ── Upstream: Anthropic Messages API ──
        self.anthropic_base_url = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        ).rstrip("/")
        # A bearer-style token may be supplied instead of an x-api-key.
        self.anthropic_auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or self.anthropic_auth_token
        # "x-api-key" (Anthropic default) or "bearer" (most Anthropic-compatible
        # gateways / OAuth-style tokens).
        self.anthropic_auth_mode = os.environ.get("ANTHROPIC_AUTH_MODE", "x-api-key").strip().lower()
        if self.anthropic_auth_mode not in ("x-api-key", "bearer"):
            print(
                f"Warning: unknown ANTHROPIC_AUTH_MODE '{self.anthropic_auth_mode}', "
                "falling back to 'x-api-key'."
            )
            self.anthropic_auth_mode = "x-api-key"
        self.anthropic_version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
        # Optional comma-separated beta flags (e.g. "prompt-caching-2024-07-31").
        self.anthropic_beta = (os.environ.get("ANTHROPIC_BETA") or "").strip()

        # ── Server ──
        self.host = os.environ.get("HOST", "127.0.0.1")
        self.port = _env_int("PORT", 4014)
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.proxy_log_dir = os.environ.get("PROXY_LOG_DIR", os.path.join(_PROJECT_ROOT, "logs"))

        # ── Token limits ──
        self.max_tokens_limit = _env_int("MAX_TOKENS_LIMIT", 128000)
        self.min_tokens_limit = _env_int("MIN_TOKENS_LIMIT", 100)
        # Anthropic requires max_tokens; use this when the client omits it.
        self.default_max_tokens = _env_int("DEFAULT_MAX_TOKENS", 8192)

        # ── Connection settings ──
        self.request_timeout = _env_int("REQUEST_TIMEOUT", 180)
        self.max_retries = _env_int("MAX_RETRIES", 2)

        # ── Model mapping ──
        # Upstream model ids. `BIG` serves opus-class requests, `MIDDLE`
        # sonnet-class, `SMALL` haiku-class, DEFAULT_MODEL everything unknown.
        self.big_model = os.environ.get("BIG_MODEL", "claude-opus-4-1")
        self.middle_model = os.environ.get("MIDDLE_MODEL", "claude-sonnet-4-5")
        self.small_model = os.environ.get("SMALL_MODEL", "claude-haiku-4-5")
        self.default_model = os.environ.get("DEFAULT_MODEL", self.middle_model)
        self.model_map = parse_model_map(os.environ.get("MODEL_MAP"))
        # When true, model ids the proxy does not recognise are forwarded
        # unchanged (useful for Anthropic-compatible gateways that front many
        # vendors). When false they fall back to DEFAULT_MODEL.
        self.passthrough_unknown_models = _env_bool("PASSTHROUGH_UNKNOWN_MODELS", False)

        # ── Extended thinking ──
        # 0 disables thinking translation entirely, even if Cline asks for it.
        self.thinking_budget_tokens = _env_int("THINKING_BUDGET_TOKENS", 0)
        self.thinking_min_budget = _env_int("THINKING_MIN_BUDGET_TOKENS", 1024)

        # ── Client auth ──
        # Optional shared secret Cline must send. Empty disables validation,
        # which is the sane default for a loopback-only proxy.
        self.client_api_key = (
            os.environ.get("CLIENT_API_KEY") or os.environ.get("PROXY_API_KEY") or ""
        )

        # ── Streaming ──
        # SSE comment keepalive during silent upstream phases (reasoning burns
        # tokens before any text arrives and some clients abort idle streams).
        self.stream_keepalive_secs = _env_float("STREAM_KEEPALIVE_SECS", 15.0)
        # Emit a usage-only final chunk when the client asked for it via
        # stream_options.include_usage (Cline uses this to show context usage).
        self.stream_usage_default = _env_bool("STREAM_INCLUDE_USAGE", True)

    # ── Auth helpers ──────────────────────────────────────────────────────
    def validate_api_key(self) -> bool:
        """True when an upstream credential is configured."""
        return bool(self.anthropic_api_key)

    def validate_client_api_key(self, client_api_key: Optional[str]) -> bool:
        """Validate the key Cline sent. No CLIENT_API_KEY means 'open'."""
        if not self.client_api_key:
            return True
        return client_api_key == self.client_api_key

    # ── Header helpers ────────────────────────────────────────────────────
    def get_custom_headers(self) -> Dict[str, str]:
        """Collect CUSTOM_HEADER_* environment variables into HTTP headers."""
        custom_headers: Dict[str, str] = {}
        for env_key, env_value in dict(os.environ).items():
            if env_key.startswith("CUSTOM_HEADER_"):
                header_name = env_key[len("CUSTOM_HEADER_"):].replace("_", "-")
                if header_name:
                    custom_headers[header_name] = env_value
        return custom_headers

    def get_upstream_headers(self) -> Dict[str, str]:
        """Headers for every upstream Anthropic call."""
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
            "User-Agent": "cline-to-claude/1.0.0",
        }
        if self.anthropic_beta:
            headers["anthropic-beta"] = self.anthropic_beta
        if self.anthropic_api_key:
            if self.anthropic_auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.anthropic_api_key}"
            else:
                headers["x-api-key"] = self.anthropic_api_key
        headers.update(self.get_custom_headers())
        return headers

    # ── Token / thinking helpers ──────────────────────────────────────────
    def clamp_max_tokens(self, requested: Optional[int]) -> int:
        """Anthropic requires max_tokens; keep it inside configured bounds."""
        value = int(requested) if requested else self.default_max_tokens
        if value < self.min_tokens_limit:
            return self.min_tokens_limit
        if value > self.max_tokens_limit:
            return self.max_tokens_limit
        return value

    def thinking_budget(self, effort: Optional[str]) -> int:
        """Translate a Cline reasoning effort into an Anthropic think budget."""
        if self.thinking_budget_tokens <= 0:
            return 0
        multiplier = EFFORT_MULTIPLIERS.get((effort or "medium").strip().lower(), 0.5)
        budget = int(self.thinking_budget_tokens * multiplier)
        # Anthropic requires budget_tokens >= 1024 and < max_tokens.
        return max(self.thinking_min_budget, budget)

    # ── Introspection ─────────────────────────────────────────────────────
    def expose_models(self) -> List[str]:
        """Model ids advertised to Cline at GET /v1/models."""
        seen: List[str] = []
        for model in (
            [self.default_model, self.big_model, self.middle_model, self.small_model]
            + sorted(set(self.model_map.values()))
        ):
            if model and model not in seen:
                seen.append(model)
        return seen

    def as_status(self) -> Dict[str, Any]:
        """JSON-serializable view of the config for /api/status."""
        return {
            "anthropic_base_url": self.anthropic_base_url,
            "anthropic_auth_mode": self.anthropic_auth_mode,
            "anthropic_version": self.anthropic_version,
            "wire": "openai-chat -> anthropic-messages",
            "max_tokens_limit": self.max_tokens_limit,
            "default_max_tokens": self.default_max_tokens,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "keepalive_secs": self.stream_keepalive_secs,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "client_key_validation": bool(self.client_api_key),
            "passthrough_unknown_models": self.passthrough_unknown_models,
            "models": {
                "default": self.default_model,
                "big": self.big_model,
                "middle": self.middle_model,
                "small": self.small_model,
            },
            "model_map": dict(self.model_map),
            "exposed_models": self.expose_models(),
        }


try:
    config = Config()
    print(
        f"[config] loaded: upstream='{config.anthropic_base_url}' "
        f"auth='{config.anthropic_auth_mode}' "
        f"key={'set' if config.anthropic_api_key else 'missing'} "
        f"model='{config.default_model}'"
    )
except Exception as e:  # pragma: no cover - startup guard
    print(f"[config] ERROR: {e}")
    sys.exit(1)