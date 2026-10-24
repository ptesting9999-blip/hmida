#!/usr/bin/env python3
"""Start ULTRON. Open the printed URL in Chrome, click the core, and talk.

    python run.py
"""

from __future__ import annotations

import os
import sys
import webbrowser

import uvicorn

from ultron.server import create_app, load_config


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("! ANTHROPIC_API_KEY is not set. ULTRON needs it to think.")
        print("  Set it, e.g.  export ANTHROPIC_API_KEY=sk-ant-...")
        # not fatal — an `ant auth login` profile also works — so we continue

    config = load_config()
    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", 8787))
    url = f"http://{host}:{port}"

    print("=" * 52)
    print("  ULTRON is online.")
    print(f"  Open {url} in Chrome, then click the core.")
    print("=" * 52)

    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
