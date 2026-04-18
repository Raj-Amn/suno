"""
ws_handler.py — WebSocket endpoint.

Orchestrates the full pipeline per connection:
  binary audio frame → AudioBuffer → VADProcessor → transcribe() → push text

Message protocol (JSON):
  Server → Client:
    { "type": "partial", "text": "..." }   — during speech
    { "type": "final",   "text": "..." }   — after end-of-utterance
    { "type": "error",   "text": "..." }   — on failure
    { "type": "ready"                   }   — after model warm-up
"""

from __future__ import annotations

import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.audio import AudioBuffer
from app.config import settings
from app.transcriber import transcribe
from app.vad import VADProcessor

logger = logging.getLogger(__name__)


async def handle_ws(websocket: WebSocket) -> None:
    """Main WebSocket handler — one coroutine per browser connection."""
    await websocket.accept()
    logger.info("WebSocket connected: %s", websocket.client)

    audio_buf = AudioBuffer()
    vad = VADProcessor()
    frame_counter = 0

    # Notify client the backend is ready
    await _send(websocket, {"type": "ready"})

    try:
        while True:
            raw = await websocket.receive_bytes()

            # Push raw bytes into rolling buffer; get decoded chunk back
            chunk = audio_buf.push(raw)
            if len(chunk) == 0:
                continue

            # Run VAD on the new chunk
            end_of_utterance = vad.process(chunk)
            frame_counter += 1

            # ── Force-flush if buffer is full ──────────────────────────────
            force_flush = audio_buf.sample_count >= settings.buffer_max_samples

            if end_of_utterance or force_flush:
                audio = audio_buf.get_audio()
                audio_buf.clear()
                vad.reset()

                if len(audio) < settings.sample_rate * 0.3:
                    # < 300ms — too short to transcribe meaningfully
                    continue

                logger.debug(
                    "Transcribing %.2fs of audio (force=%s)", audio_buf.duration_seconds, force_flush
                )

                text = await transcribe(audio)

                if text:
                    msg_type = "final" if end_of_utterance else "partial"
                    await _send(websocket, {"type": msg_type, "text": text})
                    logger.info("[%s] %s", msg_type.upper(), text)

            # ── Periodic partial during long speech ────────────────────────
            elif vad.is_speaking and frame_counter % 30 == 0:
                audio = audio_buf.get_audio()
                if len(audio) >= settings.sample_rate:
                    text = await transcribe(audio)
                    if text:
                        await _send(websocket, {"type": "partial", "text": text})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: %s", websocket.client)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await _send(websocket, {"type": "error", "text": str(exc)})
        except Exception:
            pass


async def _send(ws: WebSocket, payload: dict) -> None:
    """Safe JSON send; swallows errors if connection is already gone."""
    try:
        await ws.send_text(json.dumps(payload))
    except Exception:
        pass
