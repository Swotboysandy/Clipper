# Small base
FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf-cache \
    JOBS_DIR=/tmp/clipper_jobs \
    DEFAULT_MODEL=small \
    DEFAULT_DEVICE=cpu \
    DEFAULT_COMPUTE_CPU=int8

# Only the runtime libs we need (ffmpeg)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Koyeb exposes $PORT
CMD ["gunicorn","app:app","--bind","0.0.0.0:${PORT}","--timeout","180","--workers","1","--threads","8"]
