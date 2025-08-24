import os
import re
import json
import uuid
import threading
import queue
import subprocess
import shlex
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# ---- Speed up HF downloads / avoid timeouts ----
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # if hf-transfer installed

# Optional proxy / extra flags for yt-dlp (configure via Render env vars if needed)
PROXY = os.getenv("PROXY", "").strip()             # e.g. http://user:pass@host:port
YTDLP_EXTRA = os.getenv("YTDLP_EXTRA", "").strip()  # any extra flags

# =================== OUTPUT / STYLE ===================
OUTPUT_W, OUTPUT_H = 720, 1280  # 9:16 target
CROP_VF = f"crop=in_h*9/16:in_h,scale={OUTPUT_W}:{OUTPUT_H}"

# Fonts
FONT_LATIN      = "Montserrat SemiBold"
FONT_DEVANAGARI = "Noto Sans Devanagari"
FONT_GURMUKHI   = "Noto Sans Gurmukhi"

# Optional Windows font dir (ignored on Linux/Render if missing)
FONT_DEVANAGARI_PATH_DIR = os.getenv("FONT_DEVANAGARI_PATH_DIR", r"C:\Users\sunny\Downloads\Noto_Sans_Devanagari")

FONT_SIZE = 52
BORDER = 3
SHADOW = 1
LINE_SPACING = 2
MARGIN_V = 190
PRIMARY = "&H00FFFFFF"  # white
OUTLINE = "&H00000000"  # black

# Kinetic animation (CapCut-ish)
POP_MS = 140
FADE_MS = 160
IN_BOUNCE_PIXELS = 28

# Wrapping: short, tidy lines
MAX_WORDS_PER_LINE = 4
MAX_CHARS_PER_LINE = 22

# Defaults
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "small")
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_CPU = "int8"
DEFAULT_COMPUTE_GPU = "float16"

# ======= Flask app & job store =======
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = Path(__file__).parent.resolve()
JOBS_DIR = Path(os.getenv("JOBS_DIR", BASE_DIR / "jobs")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)
JOBS: Dict[str, Dict] = {}  # job_id -> {queue, done, out_path, workdir, cookies}

# ===== Transliteration helpers (for Hinglish) =====
_has_aksh = False
_has_indic = False
try:
    from aksharamukha import transliterate as aksh_trans
    _has_aksh = True
except Exception:
    pass

try:
    from indic_transliteration.sanscript import transliterate as itr_trans, DEVANAGARI, ITRANS
    _has_indic = True
except Exception:
    pass

def devanagari_to_hinglish(txt: str) -> str:
    if not txt:
        return ""
    try:
        if _has_aksh:
            roman = aksh_trans.process('Devanagari', 'HK', txt)
            roman = roman.replace("~n", "n").replace("aa", "a").replace("\.n", "n").replace(".n", "n")
            return roman
        if _has_indic:
            roman = itr_trans(txt, DEVANAGARI, ITRANS)
            roman = roman.replace("aa", "a").replace("ee", "i").replace("oo", "u")
            return roman
    except Exception:
        pass
    # Fallback naive map
    repl = [
        ("ख", "kh"), ("घ", "gh"), ("छ", "chh"), ("झ", "jh"), ("ठ", "th"), ("ढ", "dh"),
        ("थ", "th"), ("ध", "dh"), ("श", "sh"), ("ष", "sh"), ("च", "ch"), ("ज", "j"),
        ("ट", "t"), ("ड", "d"), ("क", "k"), ("ग", "g"), ("त", "t"), ("न", "n"), ("प", "p"),
        ("ब", "b"), ("म", "m"), ("य", "y"), ("र", "r"), ("ल", "l"), ("व", "v"), ("स", "s"),
        ("ह", "h"), ("ञ", "ny"), ("ङ", "ng"),
        ("ा", "a"), ("ि", "i"), ("ी", "i"), ("ु", "u"), ("ू", "u"), ("े", "e"), ("ै", "ai"),
        ("ो", "o"), ("ौ", "au"), ("ृ", "ri"), ("ं", "n"), ("ँ", "n"), ("ः", "h"), ("्", ""),
        ("अ", "a"), ("आ", "a"), ("इ", "i"), ("ई", "i"), ("उ", "u"), ("ऊ", "u"), ("ए", "e"),
        ("ऐ", "ai"), ("ओ", "o"), ("औ", "au"), ("ऑ", "o"), ("ऋ", "ri"), ("ॠ", "ri"),
        ("क़", "q"), ("फ़", "f"), ("ज़", "z"), ("ड़", "r"), ("ढ़", "rh")
    ]
    out = txt
    for a, b in repl:
        out = out.replace(a, b)
    return out

# =================== Utility / Core ===================
def parse_hhmmss(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) == 1:
        ss = int(parts[0]); return f"00:00:{ss:02d}"
    if len(parts) == 2:
        mm, ss = map(int, parts); return f"00:{mm:02d}:{ss:02d}"
    if len(parts) == 3:
        h, m, s2 = map(int, parts); return f"{h:02d}:{m:02d}:{s2:02d}"
    return None

def detect_script(text: str) -> str:
    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:  # Devanagari
            return "devanagari"
        if 0x0A00 <= code <= 0x0A7F:  # Gurmukhi
            return "gurmukhi"
    return "latin"

def font_for_text(text: str, output_mode: str) -> str:
    if output_mode in ("hinglish", "english"):
        return FONT_LATIN
    s = detect_script(text)
    if s == "devanagari": return FONT_DEVANAGARI
    if s == "gurmukhi":   return FONT_GURMUKHI
    return FONT_LATIN

def sanitize_ass(text: str) -> str:
    text = (text or "")
    text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\u0900-\u097F\u0A00-\u0A7F’'&\-]+|[.,!?;:।]", text or "")
    merged = []
    for t in toks:
        if t in [".", ",", "!", "?", ";", ":", "।"] and merged:
            merged[-1] += t
        else:
            merged.append(t)
    return merged

def wrap_lines_short(text: str, max_words=4, max_chars=22) -> List[str]:
    w = tokenize(text)
    if not w: return [""]
    lines, cur, cur_len = [], [], 0
    for tok in w:
        add_len = len(tok) + (1 if cur else 0)
        if (len(cur) >= max_words) or (cur_len + add_len > max_chars):
            lines.append(" ".join(cur)); cur, cur_len = [tok], len(tok)
        else:
            cur.append(tok); cur_len += add_len
    if cur: lines.append(" ".join(cur))
    if len(lines) > 2:
        merged = " ".join(lines[:-1])
        return [merged, lines[-1]] if len(merged) <= max_chars * 2 else [merged + " " + lines[-1]]
    return lines

def sec_to_ass(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); cs = int((t - int(t)) * 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def build_ass_dialogue(start: float, end: float, lines: List[str], fontname: str, anim_enabled: bool) -> str:
    tags = "{"
    tags += f"\\fn{fontname}"
    if anim_enabled:
        cx = OUTPUT_W // 2
        y1 = OUTPUT_H - MARGIN_V + IN_BOUNCE_PIXELS
        y2 = OUTPUT_H - MARGIN_V
        pop_cs = max(int(POP_MS / 10), 8)
        tags += f"\\fad({POP_MS},{FADE_MS})\\move({cx},{y1},{cx},{y2})"
        tags += f"\\t(0,{pop_cs},\\fscx110\\fscy110)\\t({pop_cs},{pop_cs+1},\\fscx100\\fscy100)"
    tags += "}"
    body = r"\N".join(sanitize_ass(l) for l in lines if l)
    return f"Dialogue: 0,{sec_to_ass(start)},{sec_to_ass(end)},Sub,,0,0,0,,{tags}{body}\n"

def build_ass_file(events: List[str], ass_path: Path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUTPUT_W}
PlayResY: {OUTPUT_H}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{FONT_LATIN},{FONT_SIZE},{PRIMARY},&H00FFFFFF,{OUTLINE},&H66000000,-1,0,0,0,100,100,{LINE_SPACING},0,1,{BORDER},{SHADOW},2,40,40,{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass = header + "".join(events)
    ass_path.write_text(ass, encoding="utf-8")

def run_streamed(cmd: list, qlog: queue.Queue, title=""):
    try:
        if title: qlog.put(f"→ {title}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, universal_newlines=True)
        for line in proc.stdout:
            qlog.put(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except Exception as e:
        qlog.put(f"[ERROR] {e}")
        raise

def cut_video(src: str, dst: str, start: Optional[str], end: Optional[str], qlog: queue.Queue):
    if not (start and end):
        if src != dst:
            Path(dst).write_bytes(Path(src).read_bytes())
        return
    qlog.put(f"Trimming with FFmpeg → {start}..{end}")
    cmd = ["ffmpeg","-y","-ss", start,"-to", end,"-i", src,"-c:v","copy","-c:a","copy", dst]
    run_streamed(cmd, qlog, "ffmpeg (trim)")

def _probe_formats(url: str, cookies: Optional[Path], qlog: queue.Queue) -> dict:
    cmd = ["yt-dlp","-J","--no-warnings", url]
    if cookies and cookies.exists():
        cmd[1:1] = ["--cookies", str(cookies)]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        qlog.put(f"[probe] {e.output.strip()}")
        return {}
    except Exception as e:
        qlog.put(f"[probe error] {e}")
        return {}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "com.google.android.youtube/19.09.37 (Linux; U; Android 13) gzip",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
]
YTDLP_PLAYER_CLIENTS = ["android", "ios", "web", "web_creator", "tv"]

def _make_ytdlp_cmd(url: str, out_path: str, cookies: Optional[Path],
                    use_sections: bool, start: Optional[str], end: Optional[str],
                    fmt: str, player_client: Optional[str]=None, ua: Optional[str]=None) -> list:
    cmd = [
        "yt-dlp",
        "--force-ipv4",
        "-N", "1",
        "--hls-prefer-native",
        "--downloader", "ffmpeg",
        "--downloader-args", "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_at_eof 1",
        "--retries", "10",
        "--fragment-retries", "10",
        "--retry-sleep", "2,4,8,16",
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--no-part",
        "-o", out_path,
        url,
    ]
    if cookies and cookies.exists():
        cmd[1:1] = ["--cookies", str(cookies)]
    if use_sections and start and end:
        cmd[cmd.index(url):cmd.index(url)] = ["--download-sections", f"*{start}-{end}"]
    if player_client:
        cmd += ["--extractor-args", f"youtube:player_client={player_client}"]
    if ua:
        cmd += ["--user-agent", ua,
                "--add-header", "Referer: https://www.youtube.com",
                "--add-header", "Accept-Language: en-US,en;q=0.9"]
    if PROXY:
        cmd += ["--proxy", PROXY]
    if YTDLP_EXTRA:
        cmd += shlex.split(YTDLP_EXTRA)
    return cmd

def download_youtube(url: str, out_path: str, start: Optional[str], end: Optional[str],
                     cookies_file: Optional[Path], qlog: queue.Queue):
    meta = _probe_formats(url, cookies_file, qlog)
    if meta:
        formats = meta.get("formats") or []
        has_video = any(f.get("vcodec", "none") != "none" for f in formats)
        if not has_video:
            raise RuntimeError("This ID appears to be images-only / not a normal video. Try another link.")
    else:
        qlog.put("[probe] empty — will try downloads anyway")

    tmp_full = str(Path(out_path).with_suffix(".full.mp4"))

    fmts = [
        "bv*+ba[ext=m4a]/b[ext=mp4]/best",
        "bv*+ba/b/best",
        "bv*[height<=1080]+ba/b[height<=1080]/best",
        "bv*[height<=1080][vcodec*=avc1]+ba/b[height<=1080][vcodec*=avc1]/best",
    ]
    attempts = []
    for fmt in fmts:
        for client in YTDLP_PLAYER_CLIENTS:
            for ua in USER_AGENTS:
                attempts += [
                    dict(fmt=fmt, sections=True,  cookies=True,  client=client, ua=ua),
                    dict(fmt=fmt, sections=True,  cookies=False, client=client, ua=ua),
                    dict(fmt=fmt, sections=False, cookies=True,  client=client, ua=ua),
                    dict(fmt=fmt, sections=False, cookies=False, client=client, ua=ua),
                ]

    last_err = None
    for i, opt in enumerate(attempts, 1):
        try:
            use_sections = opt["sections"]
            fmt          = opt["fmt"]
            with_cookies = opt["cookies"]
            client       = opt["client"]
            ua           = opt["ua"]
            cookies      = cookies_file if with_cookies else None

            qlog.put(f"Attempt {i}: fmt='{fmt}' client={client} ua={'android' if 'android' in ua.lower() else 'web/ios'} sections={use_sections} cookies={'yes' if cookies else 'no'}")

            target = out_path if use_sections else tmp_full
            cmd = _make_ytdlp_cmd(url, target, cookies, use_sections, start, end, fmt, player_client=client, ua=ua)
            run_streamed(cmd, qlog, "yt-dlp")

            if use_sections:
                qlog.put("✅ Downloaded clip successfully.")
                return
            else:
                cut_video(tmp_full, out_path, start, end, qlog)
                qlog.put("✅ Downloaded full video and trimmed successfully.")
                try: Path(tmp_full).unlink(missing_ok=True)
                except Exception: pass
                return
        except Exception as e:
            last_err = e
            qlog.put(f"[attempt failed] {e}")

    raise last_err or RuntimeError("yt-dlp failed with all strategies")

def extract_clean_audio(in_video: str, out_wav: str, qlog: queue.Queue):
    cmd = [
        "ffmpeg","-y",
        "-i", in_video,
        "-ac", "1", "-ar", "16000",
        "-af", "highpass=f=100,lowpass=f=7000,afftdn=nr=18,loudnorm=I=-18:TP=-2:LRA=11",
        out_wav
    ]
    run_streamed(cmd, qlog, "ffmpeg (pre-clean audio)")

def asr_segments_with_words(
    in_audio: str,
    model_id: str,
    device: str,
    compute_type: str,
    qlog: queue.Queue,
    lang_code: Optional[str],
    output_mode: str
):
    from faster_whisper import WhisperModel
    import unicodedata, regex as r

    def _norm(txt: str) -> str:
        if not txt: return ""
        txt = unicodedata.normalize("NFC", txt)
        out = []
        prev_comb = None
        for ch in txt:
            if unicodedata.category(ch) == "Mn":
                if prev_comb == ch:
                    continue
                prev_comb = ch
                out.append(ch)
            else:
                prev_comb = None
                out.append(ch)
        txt = "".join(out)
        txt = r.sub(r"(.)\1{2,}", r"\1\1", txt)
        return txt.strip()

    qlog.put(f"Loading faster-whisper model: {model_id}")
    model = WhisperModel(model_id, device=device, compute_type=compute_type)

    if output_mode == "english":
        language = None
        task = "translate"
    else:
        language = "hi" if (lang_code in (None, "", "auto")) else lang_code
        task = "transcribe"

    qlog.put(f"Transcribing (task={task}, language={language or 'auto'}; word timestamps on)…")

    segments, info = model.transcribe(
        in_audio,
        vad_filter=False,
        word_timestamps=True,
        beam_size=5,
        language=language,
        temperature=0.2,
        no_speech_threshold=0.6,
        task=task,
    )

    segs = []
    for seg in segments:
        text = _norm((seg.text or ""))
        if not text:
            continue

        avg_lp = getattr(seg, "avg_logprob", None)
        nsp = getattr(seg, "no_speech_prob", None)
        if nsp is not None and nsp > 0.85:
            continue
        if avg_lp is not None and avg_lp < -1.2:
            continue

        if output_mode == "hinglish":
            text_out = devanagari_to_hinglish(text)
        else:
            text_out = text

        words_out = []
        for w in (getattr(seg, "words", None) or []):
            wtxt = _norm((w.word or ""))
            if not wtxt:
                continue
            if output_mode == "hinglish":
                wtxt = devanagari_to_hinglish(wtxt)
            words_out.append({"start": w.start, "end": w.end, "word": wtxt})

        segs.append({"start": seg.start, "end": seg.end, "text": text_out, "words": words_out})
    return segs, info

def chunk_by_words(words: List[Dict], max_words=4) -> List[Tuple[float, float, str]]:
    if not words: return []
    chunks, i = [], 0
    while i < len(words):
        j = min(i + max_words, len(words))
        group = words[i:j]
        start = group[0]["start"]; end = group[-1]["end"]
        text = " ".join([w["word"] for w in group]).strip()
        chunks.append((start, end, re.sub(r"\s+", " ", text)))
        i = j
    return chunks

def make_events_from_asr(asr_segs: List[Dict], anim_enabled: bool, output_mode: str) -> List[str]:
    events = []
    for seg in asr_segs:
        words = seg.get("words") or []
        def has_letters(s: str) -> bool:
            return re.search(r"[A-Za-z\u0900-\u097F\u0A00-\u0A7F]", s or "") is not None
        if not words:
            text = seg.get("text", "")
            if not has_letters(text):
                continue
            lines = wrap_lines_short(text, MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
            events.append(build_ass_dialogue(seg["start"], seg["end"], lines, font_for_text(text, output_mode), anim_enabled))
            continue
        for (start, end, text) in chunk_by_words(words, MAX_WORDS_PER_LINE):
            if not has_letters(text):
                continue
            lines = wrap_lines_short(text, MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
            events.append(build_ass_dialogue(start, end, lines, font_for_text(text, output_mode), anim_enabled))
    return events

# --- Windows path escaping for ffmpeg filter args ---
def _escape_ffmpeg_filter_path(p: str) -> str:
    """
    Make a Windows-ish path safe inside ffmpeg -vf:
      - use forward slashes
      - escape drive colon as '\:'
      - escape single quotes
    """
    s = p.replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = s[0] + "\\:" + s[2:]  # e.g. C:/... -> C\:/
    s = s.replace("'", r"\'")
    return s

def burn_subs(input_video: str, ass_file: str, output_video: str, qlog: queue.Queue):
    ass_escaped = _escape_ffmpeg_filter_path(ass_file)
    fontsdir_clause = ""
    try:
        if FONT_DEVANAGARI_PATH_DIR:
            d = Path(FONT_DEVANAGARI_PATH_DIR)
            if d.exists() and d.is_dir():
                fontsdir = _escape_ffmpeg_filter_path(str(d))
                fontsdir_clause = f":fontsdir='{fontsdir}'"
    except Exception:
        pass
    vf = f"{CROP_VF},ass='{ass_escaped}'{fontsdir_clause}"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", vf,
        "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ]
    run_streamed(cmd, qlog, "ffmpeg (burn)")

def render_no_subs(input_video: str, output_video: str, qlog: queue.Queue):
    vf = f"{CROP_VF}"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", vf,
        "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ]
    run_streamed(cmd, qlog, "ffmpeg (render no-subs)")

def job_worker(job_id: str, form: Dict):
    job = JOBS[job_id]
    qlog: queue.Queue = job["queue"]
    workdir: Path = job["workdir"]

    try:
        url = form.get("url", "").strip()
        start = parse_hhmmss(form.get("start", ""))
        end   = parse_hhmmss(form.get("end", ""))
        anim_enabled = form.get("anim", "true") == "true"
        burn = (form.get("burn", "true") == "true")
        model = form.get("model", DEFAULT_MODEL).strip()
        device = form.get("device", DEFAULT_DEVICE).strip()
        compute = form.get("compute", DEFAULT_COMPUTE_CPU).strip()
        lang = form.get("lang", "auto").strip()
        local_model_dir = form.get("local_model", "").strip()
        outbase = (form.get("outbase", "youtube_clip").strip() or "youtube_clip")
        output_mode = (form.get("output_mode", "hindi") or "hindi").lower()

        model_id = local_model_dir if (local_model_dir and Path(local_model_dir).exists()) else model

        dl_mp4 = str((workdir / f"{outbase}.mp4").resolve())
        ass_path = (workdir / f"{outbase}.ass").resolve()
        out_video = str((workdir / f"{outbase}_{'final' if burn else 'nosubs'}_9x16.mp4").resolve())

        cookies_file = job.get("cookies")

        # 1) Download
        download_youtube(url, dl_mp4, start, end, cookies_file, qlog)

        # No subtitles path
        if not burn:
            qlog.put("Subtitles disabled — rendering 9:16 without subs…")
            render_no_subs(dl_mp4, out_video, qlog)
            qlog.put(f"✅ Done: {out_video}")
            job["out_path"] = out_video
            return

        # 2) Clean audio
        clean_wav = str((workdir / f"{outbase}_clean16k.wav").resolve())
        extract_clean_audio(dl_mp4, clean_wav, qlog)

        # 3) ASR
        qlog.put("Loading ASR and transcribing…")
        segs, info = asr_segments_with_words(clean_wav, model_id, device, compute, qlog, lang, output_mode)
        lang_det = getattr(info, "language", "unknown")
        prob = getattr(info, "language_probability", 0.0)
        qlog.put(f"Detected language: {lang_det} | prob={prob:.2f}")

        # 4) Build ASS
        qlog.put("Building subtitle events…")
        events = make_events_from_asr(segs, anim_enabled, output_mode)
        if not events:
            qlog.put("No word events generated — falling back to line-level.")
            for s in segs:
                if s.get("text"):
                    lines = wrap_lines_short(s["text"], MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
                    events.append(build_ass_dialogue(s["start"], s["end"], lines, font_for_text(s["text"], output_mode), anim_enabled))

        qlog.put(f"Writing ASS → {ass_path.name}")
        build_ass_file(events, ass_path)

        # 5) Burn
        qlog.put(f"Burning & rendering 9:16 → {Path(out_video).name}")
        burn_subs(dl_mp4, str(ass_path), out_video, qlog)

        qlog.put(f"✅ Done: {out_video}")
        job["out_path"] = out_video
    except Exception as e:
        qlog.put(f"❌ Failed: {e}")
    finally:
        job["done"] = True

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start")
def start_job():
    form = request.form.to_dict()
    files = request.files
    job_id = uuid.uuid4().hex
    workdir = JOBS_DIR / job_id
    workdir.mkdir(parents=True, exist_ok=True)

    qlog = queue.Queue()
    JOBS[job_id] = {"queue": qlog, "done": False, "out_path": "", "workdir": workdir, "cookies": None}

    # ---- FIXED COOKIES HANDLING ----
    if "cookies" in files:
        f = files["cookies"]
        if f and getattr(f, "filename", ""):
            dst = workdir / "cookies.txt"   # ✅ correct
            f.save(dst)
            JOBS[job_id]["cookies"] = dst
    # ---------------------------------

    t = threading.Thread(target=job_worker, args=(job_id, form), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.get("/logs/<job_id>")
def stream_logs(job_id: str):
    if job_id not in JOBS:
        return "no job", 404
    qlog = JOBS[job_id]["queue"]

    def gen():
        while True:
            try:
                line = qlog.get(timeout=0.5)
                yield f"data: {line}\n\n"
            except queue.Empty:
                pass
            if JOBS[job_id]["done"]:
                out_path = JOBS[job_id]["out_path"]
                if out_path and Path(out_path).exists():
                    yield f"data: DONE:{Path(out_path).name}\n\n"
                else:
                    yield "data: FAIL: job ended with error\n\n"
                break
    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    })

@app.get("/download/<job_id>/<fname>")
def download(job_id: str, fname: str):
    if job_id not in JOBS:
        return "no job", 404
    workdir: Path = JOBS[job_id]["workdir"]
    return send_from_directory(workdir, fname, as_attachment=True)

@app.get("/")
def root():
    return send_from_directory(Path(__file__).parent.resolve(), "index.html")

@app.get("/favicon.ico")
def favicon():
    return "", 204

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")), debug=False)
