# Small base
FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf-cache \
    JOBS_DIR=/tmp/clipper_jobs \
    DEFAULT_MODEL=small \
    DEFAULT_DEVICE=cpu \
    DEFAULT_COMPUTE_CPU=int8 \
    PORT=7860   # default for local runs; Koyeb overrides it

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Use shell so $PORT expands; fall back to 7860 if not set
CMD ["sh","-c","exec gunicorn app:app --bind 0.0.0.0:${PORT:-7860} --timeout 180 --workers 1 --threads 8"]
