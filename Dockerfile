FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# env for Render free plan (ephemeral)
ENV PORT=10000
ENV HF_HOME=/tmp/hf
ENV JOBS_DIR=/tmp/jobs
ENV PYTHONUNBUFFERED=1
ENV DEFAULT_MODEL=small

RUN mkdir -p /tmp/hf /tmp/jobs

CMD gunicorn -k gthread -w 1 -t 0 -b 0.0.0.0:$PORT app:app
