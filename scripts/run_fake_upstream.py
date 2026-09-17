#!/usr/bin/env python3
"""Dev helper: run the fake Anthropic upstream on a fixed port.

Usage: python3 scripts/run_fake_upstream.py [port] [text|tool]
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tests.fake_upstream import FakeAnthropic  # noqa: E402


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4020
    mode = sys.argv[2] if len(sys.argv) > 2 else "text"
    fake = FakeAnthropic()
    fake.mode = mode
    print(f"fake Anthropic upstream on {fake.start(port=port)} (mode={mode})", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fake.stop()


if __name__ == "__main__":
    main()
