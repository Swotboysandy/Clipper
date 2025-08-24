# YouTube → Auto Subtitles (Hindi / Hinglish / English)

A single-container app that downloads a YouTube clip, optionally transcribes + burns captions (Hindi/Hinglish/English), and renders a 9:16 output. Designed for Render.

## Deploy (Render)

1. Push this folder to a GitHub repo.
2. On Render → **New → Blueprint** → connect the repo (uses `render.yaml`)  
   or **New → Web Service** (env: Docker), then add a Disk mounted at `/data`.
3. Open the URL. The UI is served by Flask and uses same-origin API.

### Environment (already set in Dockerfile/render.yaml)
- `HF_HOME=/data/hf-cache`      – model cache persists across deploys
- `JOBS_DIR=/data/jobs`         – output folder for results
- `DEFAULT_MODEL=small`         – safer for small instances (change to medium/large-v3 if you have RAM)

## Usage
- Paste a YouTube URL, scrub the preview, click **Set Start/End**.
- Toggle **Burn subtitles** if you want captions in the final.
- Click **Download & Generate** and watch logs stream.
- When done, click **Download result**.
- If you disabled burn, the file ends with `_nosubs_9x16.mp4`.

## Local dev
```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:7860
```
