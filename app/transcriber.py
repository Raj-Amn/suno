"""
transcriber.py — faster-whisper singleton.

Loads the model once at startup and exposes a single async-safe
`transcribe(audio_array)` method that offloads CPU-bound work to a
ThreadPoolExecutor so the asyncio event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Generator

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from app.config import settings

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    """Load the Whisper model exactly once. lru_cache guarantees singleton."""
    logger.info(
        "Loading faster-whisper model '%s' (compute_type=%s)…",
        settings.model_size,
        settings.compute_type,
    )
    model = WhisperModel(
        settings.model_size,
        device="cpu",
        compute_type="int8",  # int8 is fastest on CPU / Apple Silicon
    )
    logger.info("Model loaded successfully.")
    return model


def _run_transcription(audio: np.ndarray) -> list[str]:
    """
    Synchronous transcription — runs in the thread pool.
    Returns a list of text segments (usually one or a few strings).
    """
    model = _load_model()
    language = settings.whisper_language  # None → auto-detect

    segments: Generator[Segment, None, None]
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=3,          # lower beam → faster on short clips
        vad_filter=False,     # we do VAD ourselves
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    return [seg.text.strip() for seg in segments if seg.text.strip()]


async def transcribe(audio: np.ndarray) -> str:
    """
    Async entry point. Dispatches CPU work to thread pool and awaits result.
    Returns a single string (joined segments) or empty string if nothing heard.
    """
    if audio is None or len(audio) == 0:
        return ""

    loop = asyncio.get_running_loop()
    texts = await loop.run_in_executor(_executor, _run_transcription, audio)
    return " ".join(texts)


def preload() -> None:
    """Call at server startup to warm up the model before first request."""
    _load_model()
