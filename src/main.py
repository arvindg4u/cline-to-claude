from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router as api_router
import uvicorn
import sys
from src.core.config import config

app = FastAPI(title="Cline-to-Claude API Proxy", version="1.0.0")

# Cline desktop/extension clients call from a browser context.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(api_router)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Cline-to-Claude API Proxy v1.0.0")
        print("")
        print("Usage: python start_proxy.py")
        print("")
        print("Exposes an OpenAI-compatible API for Cline and forwards it to")
        print("an Anthropic-compatible Messages API upstream.")
        print("")
        print("Required environment variables:")
        print("  ANTHROPIC_API_KEY - Upstream Anthropic (or gateway) API key")
        print("")
        print("Optional environment variables:")
        print("  ANTHROPIC_BASE_URL - Upstream base URL (default: https://api.anthropic.com)")
        print("  ANTHROPIC_AUTH_TOKEN - Bearer token used instead of ANTHROPIC_API_KEY")
        print("  ANTHROPIC_AUTH_MODE - 'x-api-key' (default) or 'bearer'")
        print("  ANTHROPIC_VERSION - anthropic-version header (default: 2023-06-01)")
        print("  ANTHROPIC_BETA - Optional anthropic-beta header value")
        print("  BIG_MODEL - Upstream model for opus-class requests")
        print("  MIDDLE_MODEL - Upstream model for sonnet-class requests")
        print("  SMALL_MODEL - Upstream model for haiku-class requests")
        print("  DEFAULT_MODEL - Upstream model for anything else")
        print("  MODEL_MAP - Extra aliases, e.g. 'fast=claude-haiku-4-5,smart=claude-opus-4-1'")
        print("  PASSTHROUGH_UNKNOWN_MODELS - true forwards unknown ids unchanged")
        print("  CLIENT_API_KEY - Shared secret Cline must send (unset = open)")
        print("  THINKING_BUDGET_TOKENS - Base extended-thinking budget (0 = disabled)")
        print("  DEFAULT_MAX_TOKENS - max_tokens used when the client omits it")
        print("  MAX_TOKENS_LIMIT - Upper clamp for max_tokens")
        print("  HOST - Server host (default: 127.0.0.1)")
        print("  PORT - Server port (default: 4014)")
        print("  LOG_LEVEL - Logging level (default: INFO)")
        print("  REQUEST_TIMEOUT - Upstream timeout in seconds (default: 180)")
        print("  MAX_RETRIES - Retries for transient upstream failures (default: 2)")
        print("  STREAM_KEEPALIVE_SECS - SSE keepalive interval, 0 disables (default: 15)")
        print("")
        print("Cline setup:")
        print(f"  Provider: OpenAI Compatible")
        print(f"  Base URL: http://{config.host}:{config.port}/v1")
        print(f"  API key:  {config.client_api_key or '<anything, validation is off>'}")
        print(f"  Model:    any alias, e.g. {config.default_model}")
        print("")
        sys.exit(0)

    # Configuration summary
    print(" Cline-to-Claude API Proxy v1.0.0")
    print("✅ Configuration loaded successfully")
    print(f"   Upstream Base URL: {config.anthropic_base_url}")
    print(f"   Upstream Auth: {config.anthropic_auth_mode}")
    print(f"   Wire: OpenAI chat/completions -> Anthropic messages")
    print(f"   Default Model: {config.default_model}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Server: {config.host}:{config.port}")
    print(
        "   Client API Key Validation: "
        f"{'Enabled' if config.client_api_key else 'Disabled'}"
    )
    print("")

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()

    # Validate and set default if invalid
    valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
    if log_level not in valid_levels:
        log_level = 'info'

    # Start server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()