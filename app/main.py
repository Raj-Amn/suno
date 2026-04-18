"""
main.py — FastAPI application factory.

Mounts:
  GET  /          → serves static/index.html
  GET  /health    → JSON health check + model status
  WS   /ws/audio  → real-time audio ingestion + transcription
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.transcriber import preload
from app.ws_handler import handle_ws

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="Real-time STT",
    description="Multilingual speech-to-text via faster-whisper + FastAPI WebSocket",
    version="0.1.0",
)

# Mount static files (index.html etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Warm up the Whisper model so the first request has no cold-start delay."""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, preload)
    logger.info(
        "Server ready — model=%s language=%s",
        settings.model_size,
        settings.language,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "model": settings.model_size,
        "language": settings.language,
        "sample_rate": settings.sample_rate,
        "chunk_samples": settings.chunk_samples,
        "buffer_window_seconds": settings.buffer_window_seconds,
    })


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket) -> None:
    await handle_ws(websocket)
