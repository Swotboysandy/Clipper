# syntax=docker/dockerfile:1
# Small, fast base
FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf-cache \
    JOBS_DIR=/tmp/clipper_jobs \
    DEFAULT_MODEL=small \
    DEFAULT_DEVICE=cpu \
    DEFAULT_COMPUTE_CPU=int8 \
    RES_OPTIONS="use-vc attempts:2 timeout:3" \
    PORT=7860

# Runtime deps (ffmpeg for media, curl for healthcheck)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# For local dev; Koyeb overrides $PORT at runtime
EXPOSE 7860

# Healthcheck hits your /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/health" || exit 1

# Use shell so ${PORT} expands correctly on Koyeb
CMD ["sh","-c","exec gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --timeout 180 --workers 1 --threads 8"]