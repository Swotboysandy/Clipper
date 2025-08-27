# Clipper

**Clipper** is a single-container app that turns any YouTube link into a clean, share-ready video clip — with optional **Hindi / Hinglish / English** subtitles automatically burned in. Perfect for shorts, reels, and quick highlights.

🔗 **Live demo:** [https://clipper.koyeb.app/](https://clipper.koyeb.app/)

---

## Highlights

* 🎬 **Trim by time** — pick start/end and clip just that part
* 💬 **Auto captions** — Hindi / Hinglish (romanized) / English
* 🧠 **Fast, local ASR** — powered by `faster-whisper`
* 🧱 **Burned-in or clean** — choose with/without subtitles
* 📱 **Layouts** — 16:9 original or 9:16 vertical (Shorts)
* 🛰️ **Robust YouTube fetching** — multiple clients/UAs; cookies supported
* 🟢 **Live progress logs** — streamed from the server
* ⬇️ **One-click download** — get the final file immediately
* 🧩 **Single container** — Flask + Gunicorn; no external services needed

---

## Quick Start (Local)

```bash
git clone <your-fork-or-repo>
cd clipper
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:7860
```

> Optional: put a Netscape-format cookies file at `./cookies/cookies.txt` or set `COOKIES_B64` (see **Cookies** below) for more reliable YouTube access.

---

## Deploy

### Koyeb (recommended)

1. **Create a Service** → “Deploy from GitHub” (this repo).
2. Use the provided **Dockerfile**.
3. (Optional) Add a **file** with your cookies at path `/tmp/cookies.txt`, **or** add an env var `COOKIES_B64` containing base64 of that cookies file.
4. Hit **Deploy**. The app binds to `$PORT` automatically.

### Render

1. Push this folder to GitHub.
2. On Render → **New → Web Service** (Environment: **Docker**).
3. Add a **Disk** mounted at `/data` if you want models/jobs to persist.
4. (Optional) Add `COOKIES_B64` or mount a cookies file.
5. Deploy, then open your service URL.

---

## How to Use

1. Paste a **YouTube URL**.
2. Scrub the preview, click **Set Start/End**.
3. Toggle **Burn subtitles** if you want captions.
4. Choose **layout** (16:9 or 9:16), **quality**, **subtitle mode** (Hindi/Hinglish/English).
5. Click **Download & Generate** and watch the logs.
6. When you see **DONE**, click **Download result**.

* If you disable “burn”, the output ends with `_nosubs_<layout>`.
* 9:16 mode crops and scales to a vertical frame for Shorts/Reels.

---

## Cookies (recommended)

YouTube may occasionally ask for verification. Using cookies improves reliability.

**Option A: Env var (works on Koyeb/Render)**

* Create a Netscape cookies file (e.g., via browser exporter).
* Base64-encode it and set env var `COOKIES_B64` to that value.
* On boot, Clipper writes it to `/tmp/cookies.txt` and uses it.

**Option B: Bundle a file**

* Put `cookies/cookies.txt` in the repo **or**
* Upload a file to the platform at path `/tmp/cookies.txt`.

> If cookies rotate or expire, re-export fresh ones.

---

## Configuration

These are auto-sane for most users. Override via env vars.

| Variable              | Default                                     | What it does                                       |
| --------------------- | ------------------------------------------- | -------------------------------------------------- |
| `HF_HOME`             | `/data/hf-cache` (Render) / `/tmp/hf-cache` | Hugging Face cache directory                       |
| `JOBS_DIR`            | `/data/jobs` (Render) / `/tmp/clipper_jobs` | Where outputs go                                   |
| `DEFAULT_MODEL`       | `small` (prod) / `large-v3` (code default)  | Faster-whisper model id                            |
| `DEFAULT_DEVICE`      | `cpu`                                       | `cpu` or `cuda`                                    |
| `DEFAULT_COMPUTE_CPU` | `int8`                                      | CPU compute type (`int8`/`int8_float16`/`float32`) |
| `DEFAULT_COMPUTE_GPU` | `float16`                                   | GPU compute type (`float16`/`float32`)             |
| `FONTS_DIR`           | `./fonts`                                   | Optional TTFs (e.g., Noto Sans Devanagari)         |
| `COOKIES_PATH`        | (auto)                                      | Path to cookies file if not using `COOKIES_B64`    |
| `COOKIES_B64`         | —                                           | Base64 of Netscape cookies file                    |
| `PROXY`               | —                                           | e.g., `http://user:pass@host:port` for yt-dlp      |
| `HARD_TIMEOUT_SEC`    | `900`                                       | Kill long-running jobs after this many seconds     |
| `STALL_TIMEOUT_SEC`   | `300`                                       | Kill if no log output for this many seconds        |

---

## API (optional)

Clipper exposes a tiny HTTP API the UI uses:

* `POST /start` → start a job
  Form fields include: `url`, `start`, `end`, `burn`, `layout`, `output_mode`, `quality`, `container`, etc.
  Returns: `{"job_id": "<id>"}`

* `GET /logs/<job_id>` → **SSE** stream of logs and progress
  Emits `DONE:<filename>` when complete.

* `GET /download/<job_id>/<filename>` → download the output

* `GET /health` → `{ok:true}`

* `GET /diag` → quick DNS/egress sanity check

---

## Notes & Tips

* If downloads fail with “Sign in to confirm you’re not a bot”, use fresh cookies.
* 9:16 mode uses a smart crop + scale; keep important content near center.
* For better Hindi rendering, add Devanagari/Gurmukhi TTFs to `./fonts` and set `FONTS_DIR`.
* GPU boxes: set `DEFAULT_DEVICE=cuda` and `DEFAULT_COMPUTE_GPU=float16`.

---

## License

MIT — do what you like; attribution appreciated.
