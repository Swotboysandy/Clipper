FROM python:3.10-slim

# System deps (ffmpeg is required by your app)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Env defaults (safe for small instances)
ENV DEFAULT_MODEL=small \
    DEFAULT_DEVICE=cpu \
    DEFAULT_COMPUTE_CPU=int8 \
    HF_HOME=/tmp/hf-cache \
    JOBS_DIR=/tmp/clipper_jobs

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Koyeb sets $PORT at runtime. We bind to it via CMD.
CMD ["bash", "-lc", "gunicorn app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1 --threads 8"]