# --- Dockerfile (Render free) ---
FROM python:3.11-slim

# System deps (ffmpeg + Noto fonts for Devanagari)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-noto-core wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Runtime env (use tmp since there’s no persistent disk on free plan)
ENV HF_HOME=/tmp/hf \
    JOBS_DIR=/tmp/jobs \
    PYTHONUNBUFFERED=1 \
    DEFAULT_MODEL=small

RUN mkdir -p /tmp/hf /tmp/jobs

# Render injects $PORT at runtime
CMD gunicorn -k gthread -w 1 -t 0 -b 0.0.0.0:$PORT app:app
