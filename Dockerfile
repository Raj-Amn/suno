FROM python:3.11-slim

WORKDIR /app

# System deps (ffmpeg needed by faster-whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . .

# Install Python deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    faster-whisper \
    pydantic-settings \
    python-dotenv \
    numpy

# Railway will provide PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
