"""
config.py — Centralised application configuration via pydantic-settings.
All values are overridable through environment variables or a .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Whisper model
    model_size: str = "medium"
    language: str = "auto"
    compute_type: str = "auto"

    # Audio pipeline
    sample_rate: int = 16000
    chunk_samples: int = 1024
    buffer_window_seconds: float = 5.0

    # VAD
    vad_aggressiveness: int = 2
    vad_silence_frames: int = 15

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def whisper_language(self) -> str | None:
        """Return None for auto-detect (Whisper interprets None as multilingual)."""
        return None if self.language.lower() == "auto" else self.language

    @property
    def bytes_per_chunk(self) -> int:
        """Each sample is float32 = 4 bytes."""
        return self.chunk_samples * 4

    @property
    def buffer_max_samples(self) -> int:
        return int(self.sample_rate * self.buffer_window_seconds)


# Module-level singleton — import this everywhere
settings = Settings()
