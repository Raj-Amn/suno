"""
audio.py — Audio buffer management and format helpers.

Responsibilities:
- Accept raw float32 PCM chunks from the WebSocket
- Maintain a rolling buffer up to `buffer_max_samples`
- Provide slices ready for Whisper (float32, mono, 16kHz)
- Handle browser→backend format conversion (Int16 or Float32)
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class AudioBuffer:
    """Thread-safe(ish) rolling audio buffer for a single WebSocket session."""

    def __init__(self) -> None:
        self._buf: list[np.ndarray] = []
        self._total_samples: int = 0

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def push(self, raw_bytes: bytes) -> np.ndarray:
        """
        Decode a raw binary WebSocket frame into float32 samples and
        append them to the rolling buffer.

        Browser sends Int16 PCM (16-bit signed little-endian).
        We normalise to [-1.0, 1.0] float32 for Whisper.
        """
        chunk = _decode_chunk(raw_bytes)
        self._buf.append(chunk)
        self._total_samples += len(chunk)
        self._trim()
        return chunk

    # ── Access ────────────────────────────────────────────────────────────────

    def get_audio(self) -> np.ndarray:
        """Return the complete buffered audio as a contiguous float32 array."""
        if not self._buf:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buf, dtype=np.float32)

    def clear(self) -> None:
        """Flush buffer after a successful transcription."""
        self._buf.clear()
        self._total_samples = 0

    @property
    def duration_seconds(self) -> float:
        return self._total_samples / settings.sample_rate

    @property
    def sample_count(self) -> int:
        return self._total_samples

    # ── Internals ─────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """Drop oldest chunks when buffer exceeds the configured window."""
        max_s = settings.buffer_max_samples
        while self._total_samples > max_s and self._buf:
            dropped = self._buf.pop(0)
            self._total_samples -= len(dropped)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_chunk(raw: bytes) -> np.ndarray:
    """
    Convert browser-sent Int16 PCM bytes → float32 numpy array in [-1, 1].
    Browser AudioWorklet sends Int16 (2 bytes per sample).
    """
    try:
        pcm_int16 = np.frombuffer(raw, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0
    except Exception:
        logger.warning("Failed to decode audio chunk (%d bytes), skipping.", len(raw))
        return np.zeros(0, dtype=np.float32)


def chunk_for_vad(audio: np.ndarray) -> np.ndarray:
    """
    Convert float32 audio to int16 PCM bytes for webrtcvad.
    webrtcvad expects: 16kHz, mono, 16-bit PCM.
    Frame sizes accepted: 10ms, 20ms, or 30ms → 160, 320, 480 samples at 16kHz.
    """
    return (audio * 32768).astype(np.int16)
