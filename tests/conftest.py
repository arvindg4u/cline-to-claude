"""Shared pytest fixtures.

Environment variables are pinned here *before* any ``src.*`` module is
imported, because ``src.core.config`` builds its singleton at import time and
``load_dotenv`` deliberately lets real environment variables win.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "test-upstream-key")
os.environ.setdefault("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
os.environ.setdefault("PORT", "4099")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("THINKING_BUDGET_TOKENS", "0")
os.environ.setdefault("STREAM_KEEPALIVE_SECS", "0")
os.environ.setdefault("PROXY_LOG_DIR", str(PROJECT_ROOT / "logs-test"))
os.environ.pop("CLIENT_API_KEY", None)

import pytest  # noqa: E402

from src.core.config import config  # noqa: E402


@pytest.fixture
def point_upstream():
    """Repoint the proxy's upstream client at a fake server for one test."""
    from src.api import endpoints

    original = (config.anthropic_base_url, endpoints.anthropic_client.base_url)

    def _point(base_url: str, headers: dict = None):
        config.anthropic_base_url = base_url.rstrip("/")
        endpoints.anthropic_client.base_url = base_url.rstrip("/")
        if headers:
            endpoints.anthropic_client.headers.update(headers)

    yield _point

    config.anthropic_base_url, endpoints.anthropic_client.base_url = original


@pytest.fixture
def config_overrides():
    """Temporarily override config attributes, restoring them afterwards."""
    original: dict = {}

    def _set(**kwargs):
        for key, value in kwargs.items():
            if key not in original:
                original[key] = getattr(config, key)
            setattr(config, key, value)

    yield _set

    for key, value in original.items():
        setattr(config, key, value)