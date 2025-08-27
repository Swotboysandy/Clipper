# Dockerfile
FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl dnsutils ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# yt-dlp (latest) + python pkgs
RUN pip install --no-cache-dir --upgrade pip yt-dlp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
WORKDIR /app
COPY . .

# Persist model cache and jobs on Render disk
ENV HF_HOME=/data/hf-cache \
    JOBS_DIR=/data/jobs \
    RES_OPTIONS="use-vc attempts:2 timeout:3"

EXPOSE 7860
CMD ["bash","-lc","echo -e 'nameserver 1.1.1.1\nnameserver 8.8.8.8' >/etc/resolv.conf; python app.py"]
