"""Vercel Python function: the RC–VFX API, served under /api (see vercel.json rewrites)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI  # noqa: E402

import serve  # noqa: E402

app = FastAPI(title="RC–VFX API")
app.mount("/api", serve.app)
