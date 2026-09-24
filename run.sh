#!/usr/bin/env bash
# Retrocausal Echo — Video: local dev server.  Needs Python 3.10+ and ffmpeg (brew install ffmpeg).
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg not found — install it first (macOS: brew install ffmpeg)"; exit 1; }
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
PORT="${PORT:-8000}"
echo "→ http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn serve:app --app-dir src --host 127.0.0.1 --port "$PORT" "$@"
