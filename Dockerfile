FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# ffmpeg + Noto fonts for Hindi/Gurmukhi
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \    fonts-noto \    fonts-noto-core \    fonts-noto-extra \    fonts-noto-color-emoji \    ca-certificates \ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Render persistent disk will be mounted at /data
ENV PORT=7860 \
    HF_HOME=/data/hf-cache \    JOBS_DIR=/data/jobs \    DEFAULT_MODEL=small

EXPOSE 7860

CMD ["gunicorn", "app:app", "-k", "gthread", "--threads", "8", "--workers", "1", "--timeout", "900", "--bind", "0.0.0.0:7860"]
