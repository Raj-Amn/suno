"""
vad.py — Energy-based Voice Activity Detection (no external VAD library).
"""
from __future__ import annotations
import numpy as np
from app.config import settings

ENERGY_THRESHOLD = 0.002
MIN_SPEECH_FRAMES = 3


class SpeechState:
    IDLE = "idle"
    SPEAKING = "speaking"


class VADProcessor:
    def __init__(self) -> None:
        self._state = SpeechState.IDLE
        self._silence_frames = 0
        self._speech_frames = 0

    def process(self, audio_chunk: np.ndarray) -> bool:
        energy = float(np.mean(audio_chunk ** 2))
        is_speech = energy > ENERGY_THRESHOLD

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            self._state = SpeechState.SPEAKING
        else:
            if self._state == SpeechState.SPEAKING:
                self._silence_frames += 1
                if self._silence_frames >= settings.vad_silence_frames:
                    self._state = SpeechState.IDLE
                    self._speech_frames = 0
                    return True
        return False

    def reset(self) -> None:
        self._state = SpeechState.IDLE
        self._silence_frames = 0
        self._speech_frames = 0

    @property
    def is_speaking(self) -> bool:
        return self._state == SpeechState.SPEAKING
