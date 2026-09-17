"""Durable record of the last upstream failure, surfaced by /api/status.

Mirrors the behaviour of the companion claude-code-proxy: the newest failure
is written to ``logs/last_upstream_failure.json`` (overwritten) and archived
to ``logs/failures/failure-YYYYmmdd-HHMMSS.json`` so long outages stay
auditable and short ones do not spam.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.config import config

_ARCHIVE_LIMIT = 200


def _log_dir() -> str:
    return config.proxy_log_dir


def record_failure(
    status: int,
    error: str,
    model: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """Persist the newest upstream failure. Never raises."""
    payload = {
        "status": status,
        "error": str(error)[:2000],
        "model": model,
        "request_id": request_id,
        "at": time.time(),
    }
    try:
        log_dir = _log_dir()
        os.makedirs(os.path.join(log_dir, "failures"), exist_ok=True)
        with open(os.path.join(log_dir, "last_upstream_failure.json"), "w") as f:
            json.dump(payload, f)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(os.path.join(log_dir, "failures", f"failure-{stamp}.json"), "w") as f:
            json.dump(payload, f)
        _prune(os.path.join(log_dir, "failures"))
    except Exception:
        # Telemetry must never break a request.
        pass


def _prune(failures_dir: str) -> None:
    names = sorted(f for f in os.listdir(failures_dir) if f.startswith("failure-"))
    for name in names[:-_ARCHIVE_LIMIT]:
        try:
            os.remove(os.path.join(failures_dir, name))
        except OSError:
            pass


def load_failure_state() -> Dict[str, Any]:
    """Return {last_upstream_failure, failure_history} for /api/status."""
    log_dir = _log_dir()
    last_failure: Optional[Dict[str, Any]] = None
    failure_history: List[Dict[str, Any]] = []

    try:
        path = os.path.join(log_dir, "last_upstream_failure.json")
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "at" not in raw:
                # Legacy dump (status + payload only): keep the status so old
                # watchers still see e.g. a 429, but mark it as stale.
                last_failure = {
                    "status": raw.get("status"),
                    "error": "(legacy dump: error text not recorded)",
                    "model": None,
                    "request_id": None,
                    "at": 0,
                }
            else:
                last_failure = raw
    except Exception:
        last_failure = None

    try:
        failures_dir = os.path.join(log_dir, "failures")
        if os.path.isdir(failures_dir):
            paths = sorted(
                (os.path.join(failures_dir, n) for n in os.listdir(failures_dir)
                 if n.startswith("failure-")),
                reverse=True,
            )[:20]
            for path in paths:
                try:
                    with open(path) as f:
                        entry = json.load(f)
                    if isinstance(entry, dict):
                        failure_history.append(
                            {
                                "status": entry.get("status"),
                                "error": str(entry.get("error", ""))[:300],
                                "model": entry.get("model"),
                                "at": entry.get("at", 0),
                            }
                        )
                except Exception:
                    continue
    except Exception:
        failure_history = []

    return {"last_upstream_failure": last_failure, "failure_history": failure_history}