#!/usr/bin/env bash
# ULTRON one-click launcher (macOS / Linux).
#   ./start.sh            -> the orb HUD in your browser
#   ./start.sh terminal   -> the terminal version
set -e
cd "$(dirname "$0")"

command -v python3 >/dev/null 2>&1 || { echo "Install Python 3.10+ first."; exit 1; }
[ -d venv ] || python3 -m venv venv
./venv/bin/python -m pip install -q --upgrade pip >/dev/null 2>&1 || true
./venv/bin/python -m pip install -q -r requirements.txt

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  if [ -f apikey.txt ]; then
    export ANTHROPIC_API_KEY="$(cat apikey.txt)"
  else
    read -rp "Paste your Anthropic API key (sk-ant-...): " ANTHROPIC_API_KEY
    export ANTHROPIC_API_KEY
    echo "$ANTHROPIC_API_KEY" > apikey.txt
    echo "(saved to apikey.txt - it won't ask again)"
  fi
fi

if [ "${1:-}" = "terminal" ]; then
  ./venv/bin/python cli.py
else
  ./venv/bin/python run.py
fi
