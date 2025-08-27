import os, re, json, uuid, threading, time, queue, subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# ---------------- Env & constants ----------------
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

PROXY = os.getenv("PROXY", "").strip()
COOKIES_PATH_ENV = os.getenv("COOKIES_PATH", "").strip()

LAYOUTS = {
    "original_16x9": {"w": 1920, "h": 1080, "vf": "scale=-2:1080", "margin_v": 90},
    "shorts_9x16":   {"w":  720, "h": 1280, "vf": "crop=in_h*9/16:in_h,scale=720:1280", "margin_v": 190},
}

FONT_LATIN      = "Montserrat SemiBold"
FONT_DEVANAGARI = "Noto Sans Devanagari"
FONT_GURMUKHI   = "Noto Sans Gurmukhi"
FONT_DEVANAGARI_PATH_DIR = r"C:\Users\sunny\Downloads\Noto_Sans_Devanagari"

FONT_SIZE = 52
BORDER = 3
SHADOW = 1
LINE_SPACING = 2
PRIMARY = "&H00FFFFFF"
OUTLINE = "&H00000000"

POP_MS = 140
FADE_MS = 160
IN_BOUNCE_PIXELS = 28

MAX_WORDS_PER_LINE = 4
MAX_CHARS_PER_LINE = 22

DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_CPU = "int8"
DEFAULT_COMPUTE_GPU = "float16"

HARD_TIMEOUT_SEC  = int(os.getenv("HARD_TIMEOUT_SEC", "900"))
STALL_TIMEOUT_SEC = int(os.getenv("STALL_TIMEOUT_SEC", "300"))

# --------------- App scaffolding ---------------
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})
BASE_DIR = Path(__file__).parent.resolve()
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(BASE_DIR / "jobs")))
JOBS_DIR.mkdir(exist_ok=True)
JOBS: Dict[str, Dict] = {}

# -------------- (Optional) transliteration --------------
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
    if not txt: return ""
    try:
        if _has_aksh:
            roman = aksh_trans.process('Devanagari', 'HK', txt)
            return roman.replace("~n","n").replace("aa","a").replace("\\.n","n").replace(".n","n")
        if _has_indic:
            roman = itr_trans(txt, DEVANAGARI, ITRANS)
            return roman.replace("aa","a").replace("ee","i").replace("oo","u")
    except Exception:
        pass
    repl = [("ख","kh"),("घ","gh"),("छ","chh"),("झ","jh"),("ठ","th"),("ढ","dh"),("थ","th"),("ध","dh"),
            ("श","sh"),("ष","sh"),("च","ch"),("ज","j"),("ट","t"),("ड","d"),("क","k"),("ग","g"),
            ("त","t"),("न","n"),("प","p"),("ब","b"),("म","m"),("य","y"),("र","r"),("ल","l"),
            ("व","v"),("स","s"),("ह","h"),("ञ","ny"),("ङ","ng"),
            ("ा","a"),("ि","i"),("ी","i"),("ु","u"),("ू","u"),("े","e"),("ै","ai"),("ो","o"),("ौ","au"),
            ("ृ","ri"),("ं","n"),("ँ","n"),("ः","h"),("्",""),("अ","a"),("आ","a"),("इ","i"),("ई","i"),
            ("उ","u"),("ऊ","u"),("ए","e"),("ऐ","ai"),("ओ","o"),("औ","au"),("ऑ","o"),("ऋ","ri"),("ॠ","ri"),
            ("क़","q"),("फ़","f"),("ज़","z"),("ड़","r"),("ढ़","rh")]
    out = txt
    for a,b in repl: out = out.replace(a,b)
    return out

# -------------- Helpers --------------
def parse_hhmmss(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s: return None
    p = s.split(":")
    if   len(p)==1: return f"00:00:{int(p[0]):02d}"
    elif len(p)==2: return f"00:{int(p[0]):02d}:{int(p[1]):02d}"
    elif len(p)==3: return f"{int(p[0]):02d}:{int(p[1]):02d}:{int(p[2]):02d}"
    return None

def _to_seconds(hhmmss: Optional[str]) -> Optional[int]:
    if not hhmmss: return None
    h,m,s = [int(x) for x in hhmmss.split(":")]
    return h*3600 + m*60 + s

def detect_script(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "devanagari"
        if 0x0A00 <= o <= 0x0A7F: return "gurmukhi"
    return "latin"

def font_for_text(text: str, output_mode: str) -> str:
    if output_mode in ("hinglish","english"): return FONT_LATIN
    s = detect_script(text)
    if s=="devanagari": return FONT_DEVANAGARI
    if s=="gurmukhi":   return FONT_GURMUKHI
    return FONT_LATIN

def sanitize_ass(text: str) -> str:
    text = (text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return re.sub(r"\s+"," ",text).strip()

def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\u0900-\u097F\u0A00-\u0A7F’'&\-]+|[.,!?;:।]", text or "")
    merged = []
    for t in toks:
        if t in [".",",","!","?",";",":","।"] and merged: merged[-1] += t
        else: merged.append(t)
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
        return [merged, lines[-1]] if len(merged) <= max_chars*2 else [merged + " " + lines[-1]]
    return lines

def sec_to_ass(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); cs = int((t - int(t)) * 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def build_ass_dialogue(start: float, end: float, lines: List[str], fontname: str,
                       anim_enabled: bool, play_w: int, play_h: int, margin_v: int) -> str:
    tags = "{"
    tags += f"\\fn{fontname}"
    if anim_enabled:
        cx = play_w // 2
        y1 = play_h - margin_v + IN_BOUNCE_PIXELS
        y2 = play_h - margin_v
        pop_cs = max(int(POP_MS/10), 8)
        tags += f"\\fad({POP_MS},{FADE_MS})\\move({cx},{y1},{cx},{y2})"
        tags += f"\\t(0,{pop_cs},\\fscx110\\fscy110)\\t({pop_cs},{pop_cs+1},\\fscx100\\fscy100)"
    tags += "}"
    body = r"\N".join(sanitize_ass(l) for l in lines if l)
    return f"Dialogue: 0,{sec_to_ass(start)},{sec_to_ass(end)},Sub,,0,0,0,,{tags}{body}\n"

def build_ass_file(events: List[str], ass_path: Path, play_w: int, play_h: int, margin_v: int):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub,{FONT_LATIN},{FONT_SIZE},{PRIMARY},&H00FFFFFF,{OUTLINE},&H66000000,-1,0,0,0,100,100,{LINE_SPACING},0,1,{BORDER},{SHADOW},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_path.write_text(header + "".join(events), encoding="utf-8")

# ----------- Streamed runner -----------
def run_streamed(cmd: list, qlog: queue.Queue, title: str = "", hard_timeout=HARD_TIMEOUT_SEC, stall_timeout=STALL_TIMEOUT_SEC):
    if title: qlog.put(f"→ {title}")
    qlog.put(" ".join([sh if " " not in sh else f'"{sh}"' for sh in cmd]))
    start_ts = time.time(); last_line_ts = start_ts
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    def reader():
        nonlocal last_line_ts
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    last_line_ts = time.time()
                    qlog.put(line)
        except Exception as e:
            qlog.put(f"[stream error] {e}")
    threading.Thread(target=reader, daemon=True).start()
    while proc.poll() is None:
        time.sleep(1)
        now = time.time()
        if hard_timeout and (now - start_ts) > hard_timeout:
            qlog.put(f"[timeout] Exceeded hard timeout ({hard_timeout}s). Killing process…"); proc.kill(); break
        if stall_timeout and (now - last_line_ts) > stall_timeout:
            qlog.put(f"[stall] No output for {stall_timeout}s. Killing process…"); proc.kill(); break
    code = proc.wait()
    if code != 0: raise subprocess.CalledProcessError(code, cmd)

# ----------------- Downloading -----------------
def cut_video(src: str, dst: str, start: Optional[str], end: Optional[str], qlog: queue.Queue):
    if not (start and end):
        if src != dst: Path(dst).write_bytes(Path(src).read_bytes())
        return
    qlog.put(f"Trimming with FFmpeg → {start}..{end}")
    cmd = ["ffmpeg","-y","-ss", start,"-to", end,"-i", src,"-c:v","copy","-c:a","copy", dst]
    run_streamed(cmd, qlog, "ffmpeg (trim)")

def _probe_formats(url: str, cookies: Optional[Path], qlog: queue.Queue) -> dict:
    cmd = ["yt-dlp","-J","--no-warnings"]
    if cookies and cookies.exists(): cmd += ["--cookies", str(cookies)]
    if PROXY: cmd += ["--proxy", PROXY]
    cmd += [url]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        qlog.put(f"[probe] {e.output.strip()}"); return {}
    except Exception as e:
        qlog.put(f"[probe error] {e}"); return {}

USER_AGENTS = {
    "tv": "Mozilla/5.0 (Chromium) YTTV/1.0 (Linux; Android 9) Cobalt/Version",
    "web_safari": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15",
    "web": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "android": "com.google.android.youtube/19.09.37 (Linux; U; Android 13) gzip",
    "ios": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15",
}
CLIENTS_ORDER = ["tv","web_safari","web","ios","android"]

def _make_ytdlp_cmd(url: str, out_path: str, cookies: Optional[Path],
                    use_sections: bool, start: Optional[str], end: Optional[str],
                    fmt: str, client: str, merge_ext: str, extra_down_args: Optional[str]=None,
                    live_from_start: bool=False) -> list:
    ua = USER_AGENTS.get(client, USER_AGENTS["web"])
    dl_args = "ffmpeg_i:-reconnect 1 -reconnect_streamed 1 -reconnect_at_eof 1"
    if extra_down_args: dl_args = f"{dl_args} {extra_down_args}"
    cmd = [
        "yt-dlp","--force-ipv4","--no-warnings","--geo-bypass","-N","2",
        "--hls-prefer-native","--downloader","ffmpeg","--downloader-args", dl_args,
        "--retries","15","--fragment-retries","15","--concurrent-fragments","1","--socket-timeout","15",
        "--user-agent", ua,
        "--add-header","Referer: https://www.youtube.com",
        "--add-header","Origin: https://www.youtube.com",
        "--add-header","Accept-Language: en-US,en;q=0.9",
        "-f", fmt, "--merge-output-format", merge_ext, "--no-part",
        "-o", out_path, "--extractor-args", f"youtube:player_client={client}",
    ]
    if live_from_start: cmd.append("--live-from-start")
    if PROXY: cmd += ["--proxy", PROXY]
    if cookies and cookies.exists(): cmd += ["--cookies", str(cookies)]
    if use_sections and start and end: cmd += ["--download-sections", f"*{start}-{end}"]
    cmd += [url]
    return cmd

def build_ytdlp_format(quality: str, container: str) -> str:
    qmax = {"best": None, "1080p":1080, "1440p":1440, "2160p":2160}.get(quality, None)
    v_sel = "bv*" if qmax is None else f"bv*[height<={qmax}]"
    if container == "mp4":
        v_pref = f"{v_sel}[ext=mp4]/{v_sel}"; a_pref = "ba[ext=m4a]/ba"; b_pref = "b[ext=mp4]" if qmax is None else f"b[ext=mp4][height<={qmax}]"
    else:
        v_pref = v_sel; a_pref = "ba"; b_pref = "b" if qmax is None else f"b[height<={qmax}]"
    return f"{v_pref}+{a_pref}/{b_pref}/best"

def is_live_from_meta_or_url(meta: dict, url: str) -> bool:
    # Strong hints from meta
    if meta.get("is_live"): return True
    ls = (meta.get("live_status") or "").lower()
    if ls in {"is_live","is_upcoming"}: return True
    # Fallback by URL pattern
    return "/live/" in url

def _yt_dlp_version(qlog: queue.Queue):
    try:
        v = subprocess.check_output(["yt-dlp","--version"], text=True).strip()
        qlog.put(f"[yt-dlp] version {v}")
    except Exception:
        pass

def download_vod(url: str, out_path: str, start: Optional[str], end: Optional[str],
                 cookies_file: Optional[Path], qlog: queue.Queue, fmt_force: str, container: str):
    tmp_full = str(Path(out_path).with_suffix(f".full.{container}"))
    attempts = []
    for client in CLIENTS_ORDER:
        attempts += [dict(use_sections=True, client=client), dict(use_sections=False, client=client)]
    last_err = None
    for i, opt in enumerate(attempts, 1):
        try:
            use_sections = opt["use_sections"]; client = opt["client"]
            qlog.put(f"Attempt {i}: fmt='{fmt_force}' client={client} sections={use_sections}")
            target = out_path if use_sections else tmp_full
            cmd = _make_ytdlp_cmd(url, target, cookies_file, use_sections, start, end, fmt_force, client, container)
            run_streamed(cmd, qlog, "yt-dlp")
            if use_sections:
                qlog.put("✅ Downloaded the selected clip."); return out_path
            else:
                cut_video(tmp_full, out_path, start, end, qlog)
                qlog.put("✅ Downloaded full video and trimmed.")
                try: Path(tmp_full).unlink(missing_ok=True)
                except Exception: pass
                return out_path
        except Exception as e:
            last_err = e; qlog.put(f"[attempt failed] {e}")
        finally:
            try: Path(tmp_full).unlink(missing_ok=True)
            except Exception: pass
    raise last_err or RuntimeError("yt-dlp failed with all strategies")

def download_live(url: str, out_path: str,
                  cookies_file: Optional[Path], qlog: queue.Queue,
                  fmt_force: str, container: str,
                  live_from_start: bool, record_seconds: Optional[int]):
    extra_down_args = None
    if record_seconds and record_seconds > 0:
        extra_down_args = f"-t {int(record_seconds)}"
        qlog.put(f"[live] Will record for {record_seconds} seconds then stop.")
    attempts = [dict(client=c) for c in CLIENTS_ORDER]
    last_err = None
    for i, opt in enumerate(attempts, 1):
        try:
            client = opt["client"]
            qlog.put(f"Attempt {i} (LIVE): fmt='{fmt_force}' client={client} live_from_start={live_from_start}")
            cmd = _make_ytdlp_cmd(url, out_path, cookies_file, False, None, None, fmt_force, client, container, extra_down_args, live_from_start)
            run_streamed(cmd, qlog, "yt-dlp (live)")
            qlog.put("✅ Live capture saved."); return out_path
        except Exception as e:
            last_err = e; qlog.put(f"[live attempt failed] {e}")
    raise last_err or RuntimeError("yt-dlp live failed with all strategies")

# ---------------- Audio / ASR / Render ----------------
def extract_clean_audio(in_video: str, out_wav: str, qlog: queue.Queue):
    cmd = ["ffmpeg","-y","-i", in_video,"-ac","1","-ar","16000",
           "-af","highpass=f=100,lowpass=f=7000,afftdn=nr=18,loudnorm=I=-18:TP=-2:LRA=11", out_wav]
    run_streamed(cmd, qlog, "ffmpeg (pre-clean audio)")

def asr_segments_with_words(in_audio: str, model_id: str, device: str, compute_type: str,
                            qlog: queue.Queue, lang_code: Optional[str], output_mode: str):
    from faster_whisper import WhisperModel
    import unicodedata
    def _norm(txt: str) -> str:
        if not txt: return ""
        txt = unicodedata.normalize("NFC", txt)
        out, prev_comb = [], None
        for ch in txt:
            if unicodedata.category(ch) == "Mn":
                if prev_comb == ch: continue
                prev_comb = ch; out.append(ch)
            else:
                prev_comb = None; out.append(ch)
        txt = "".join(out)
        txt = re.sub(r"(.)\1{2,}", r"\1\1", txt)
        return txt.strip()

    qlog.put(f"Loading faster-whisper model: {model_id}")
    model = WhisperModel(model_id, device=device, compute_type=compute_type)

    language, task = (None, "translate") if output_mode=="english" \
                     else (("hi" if (lang_code in (None,"","auto")) else lang_code), "transcribe")

    qlog.put(f"Transcribing (task={task}, language={language or 'auto'}; word timestamps on)…")
    segments, info = model.transcribe(in_audio, vad_filter=False, word_timestamps=True, beam_size=5,
                                      language=language, temperature=0.2, no_speech_threshold=0.6, task=task)
    segs = []
    for seg in segments:
        text = _norm((seg.text or ""))
        if not text:
            continue

        if getattr(seg, "no_speech_prob", 0) > 0.85: continue
        if getattr(seg, "avg_logprob", 0) < -1.2:    continue
        text_out = devanagari_to_hinglish(text) if output_mode=="hinglish" else text
        words_out = []
        for w in (getattr(seg, "words", None) or []):
            wtxt = _norm((w.word or ""))
            if not wtxt:
                continue
            if output_mode=="hinglish": wtxt = devanagari_to_hinglish(wtxt)
            words_out.append({"start": w.start, "end": w.end, "word": wtxt})
        segs.append({"start": seg.start, "end": seg.end, "text": text_out, "words": words_out})
    return segs, info

def chunk_by_words(words: List[Dict], max_words=4) -> List[Tuple[float, float, str]]:
    if not words: return []
    chunks, i = [], 0
    while i < len(words):
        j = min(i+max_words, len(words)); group = words[i:j]
        start = group[0]["start"]; end = group[-1]["end"]
        text = re.sub(r"\s+"," ", " ".join([w["word"] for w in group]).strip())
        chunks.append((start, end, text)); i = j
    return chunks

def make_events_from_asr(asr_segs: List[Dict], anim_enabled: bool, output_mode: str,
                         play_w: int, play_h: int, margin_v: int) -> List[str]:
    events = []
    def has_letters(s: str) -> bool: return re.search(r"[A-Za-z\u0900-\u097F\u0A00-\u0A7F]", s or "") is not None
    for seg in asr_segs:
        words = seg.get("words") or []
        if not words:
            text = seg.get("text","")
            if not has_letters(text):
                continue
            lines = wrap_lines_short(text, MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
            events.append(build_ass_dialogue(seg["start"], seg["end"], lines, font_for_text(text, output_mode),
                                             anim_enabled, play_w, play_h, margin_v))
            continue
        for (start, end, text) in chunk_by_words(words, MAX_WORDS_PER_LINE):
            if not has_letters(text): continue
            lines = wrap_lines_short(text, MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
            events.append(build_ass_dialogue(start, end, lines, font_for_text(text, output_mode),
                                             anim_enabled, play_w, play_h, margin_v))
    return events

def _escape_ffmpeg_filter_path(p: str) -> str:
    s = p.replace("\\","/")
    if len(s) >= 2 and s[1] == ":": s = s[0] + "\\:" + s[2:]
    return s.replace("'","\\'")

def burn_subs(input_video: str, ass_file: str, output_video: str, qlog: queue.Queue, vf_clause: str):
    ass_escaped = _escape_ffmpeg_filter_path(ass_file)
    fontsdir_clause = ""
    try:
        d = Path(FONT_DEVANAGARI_PATH_DIR)
        if d.exists() and d.is_dir() and any(d.glob("*.ttf")):
            fontsdir_clause = f":fontsdir='{_escape_ffmpeg_filter_path(str(d))}'"
        else:
            qlog.put(f"[fonts] Skipping fontsdir (not found or no .ttf): {d}")
    except Exception as e:
        qlog.put(f"[fonts] Warning: {e}")
    vf = f"{vf_clause},ass=filename='{ass_escaped}'{fontsdir_clause}"
    cmd = ["ffmpeg","-y","-i", input_video,"-vf", vf,
           "-filter:a","loudnorm=I=-16:TP=-1.5:LRA=11",
           "-c:v","libx264","-crf","18","-preset","medium",
           "-c:a","aac","-b:a","192k", output_video]
    run_streamed(cmd, qlog, "ffmpeg (burn)")

def render_no_subs(input_video: str, output_video: str, qlog: queue.Queue, vf_clause: str):
    cmd = ["ffmpeg","-y","-i", input_video,"-vf", vf_clause,
           "-filter:a","loudnorm=I=-16:TP=-1.5:LRA=11",
           "-c:v","libx264","-crf","18","-preset","medium",
           "-c:a","aac","-b:a","192k", output_video]
    run_streamed(cmd, qlog, "ffmpeg (render no-subs)")

# ---------------- Job worker ----------------
def _select_cookies_file(job_cookies: Optional[Path]) -> Optional[Path]:
    if job_cookies and job_cookies.exists(): return job_cookies
    if COOKIES_PATH_ENV:
        p = Path(COOKIES_PATH_ENV)
        if p.exists(): return p
    local = BASE_DIR / "cookies.txt"
    return local if local.exists() else None

def job_worker(job_id: str, form: Dict):
    job = JOBS[job_id]; qlog: queue.Queue = job["queue"]; workdir: Path = job["workdir"]
    try:
        _yt_dlp_version(qlog)

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
        layout = form.get("layout", "original_16x9").strip().lower()
        raw_download = (form.get("raw", "false") == "true")
        quality = (form.get("quality", "best") or "best").lower()
        container = (form.get("container", "mkv") or "mkv").lower()

        # LIVE options
        is_live_forced   = (form.get("is_live", "false") == "true")
        live_from_start  = (form.get("live_from_start", "false") == "true")
        live_duration_in = (form.get("live_duration", "") or "").strip()
        live_seconds = None
        if live_duration_in:
            if ":" in live_duration_in:
                parts = [int(x) for x in live_duration_in.split(":")]
                live_seconds = parts[0]*3600 + parts[1]*60 + parts[2] if len(parts)==3 else parts[0]*60 + parts[1]
            else:
                try: live_seconds = int(live_duration_in)
                except: live_seconds = None

        layout_cfg = LAYOUTS.get(layout, LAYOUTS["original_16x9"])
        play_w, play_h, margin_v, vf_clause = layout_cfg["w"], layout_cfg["h"], layout_cfg["margin_v"], layout_cfg["vf"]
        model_id = local_model_dir if (local_model_dir and Path(local_model_dir).exists()) else model

        out_ext  = ".mkv" if container == "mkv" else ".mp4"
        dl_path  = str((workdir / f"{outbase}{out_ext}").resolve())
        ass_path = (workdir / f"{outbase}.ass").resolve()
        suffix   = "raw" if raw_download else ("final" if burn else "nosubs")
        out_path = str((workdir / f"{outbase}_{suffix}_{layout}{out_ext}").resolve())

        cookies_from_request = job.get("cookies")
        cookies_file = _select_cookies_file(cookies_from_request)
        if cookies_file: qlog.put(f"[cookies] Using cookies file: {cookies_file}")

        # Probe & detect live
        meta = _probe_formats(url, cookies_file, qlog)
        detected_live = is_live_from_meta_or_url(meta, url)
        is_live = is_live_forced or detected_live
        if detected_live: qlog.put("[probe] Stream appears to be LIVE.")
        if is_live_forced and not detected_live: qlog.put("[note] Forced LIVE mode.")

        fmt_force = build_ytdlp_format(quality, container)

        # ---------- LIVE ----------
        if is_live:
            qlog.put("[live] Starting live capture…")
            # If user supplied Start/End in live, we will post-trim after recording.
            # Ensure we at least record long enough; UI tries to set it, but if not, we don't block.
            download_live(url, dl_path, cookies_file, qlog, fmt_force, container, live_from_start, live_seconds)

            # Optional post-trim for live if Start/End present
            trimmed_path = dl_path
            if start and end:
                live_trim_path = str((workdir / f"{outbase}.trim{out_ext}").resolve())
                cut_video(dl_path, live_trim_path, start, end, qlog)
                trimmed_path = live_trim_path

            if raw_download:
                if Path(trimmed_path) != Path(out_path): Path(out_path).write_bytes(Path(trimmed_path).read_bytes())
                job["out_path"] = out_path; qlog.put(f"✅ Done (live raw): {out_path}"); return

            if not burn:
                qlog.put("[live] Rendering without subtitles…")
                render_no_subs(trimmed_path, out_path, qlog, vf_clause)
                job["out_path"] = out_path; qlog.put(f"✅ Done: {out_path}"); return

            qlog.put("[live] Running ASR on captured segment…")
            clean_wav = str((workdir / f"{outbase}_clean16k.wav").resolve())
            extract_clean_audio(trimmed_path, clean_wav, qlog)
            segs, info = asr_segments_with_words(clean_wav, model_id, device, compute, qlog, lang, output_mode)
            events = make_events_from_asr(segs, anim_enabled, output_mode, play_w, play_h, margin_v)
            if not events:
                for s in segs:
                    if s.get("text"):
                        lines = wrap_lines_short(s["text"], MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
                        events.append(build_ass_dialogue(s["start"], s["end"], lines, font_for_text(s["text"], output_mode),
                                                         anim_enabled, play_w, play_h, margin_v))
            build_ass_file(events, ass_path, play_w, play_h, margin_v)
            burn_subs(trimmed_path, str(ass_path), out_path, qlog, vf_clause)
            job["out_path"] = out_path; qlog.put(f"✅ Done (live + subs): {out_path}"); return

        # ---------- VOD ----------
        qlog.put("[vod] Downloading VOD…")
        download_vod(url, dl_path, start, end, cookies_file, qlog, fmt_force, container)

        if raw_download:
            if Path(dl_path) != Path(out_path): Path(out_path).write_bytes(Path(dl_path).read_bytes())
            job["out_path"] = out_path; qlog.put(f"✅ Done (raw): {out_path}"); return

        if not burn:
            qlog.put(f"Subtitles disabled — rendering with layout={layout}…")
            render_no_subs(dl_path, out_path, qlog, vf_clause)
            job["out_path"] = out_path; qlog.put(f"✅ Done: {out_path}"); return

        clean_wav = str((workdir / f"{outbase}_clean16k.wav").resolve())
        extract_clean_audio(dl_path, clean_wav, qlog)
        qlog.put("Loading ASR and transcribing…")
        segs, info = asr_segments_with_words(clean_wav, model_id, device, compute, qlog, lang, output_mode)
        qlog.put(f"Detected language: {getattr(info,'language','unknown')} | prob={getattr(info,'language_probability',0.0):.2f}")
        qlog.put("Building subtitle events…")
        events = make_events_from_asr(segs, anim_enabled, output_mode, play_w, play_h, margin_v)
        if not events:
            for s in segs:
                if s.get("text"):
                    lines = wrap_lines_short(s["text"], MAX_WORDS_PER_LINE, MAX_CHARS_PER_LINE)
                    events.append(build_ass_dialogue(s["start"], s["end"], lines, font_for_text(s["text"], output_mode),
                                                     anim_enabled, play_w, play_h, margin_v))
        qlog.put(f"Writing ASS → {Path(ass_path).name}")
        build_ass_file(events, ass_path, play_w, play_h, margin_v)
        qlog.put(f"Burning & rendering ({layout}) → {Path(out_path).name}")
        burn_subs(dl_path, str(ass_path), out_path, qlog, vf_clause)
        qlog.put(f"✅ Done: {out_path}")
        job["out_path"] = out_path

    except Exception as e:
        qlog.put(f"❌ Failed: {e}")
    finally:
        job["done"] = True

# ---------------- HTTP ----------------
@app.get("/health")
def health(): return {"ok": True}

@app.post("/start")
def start_job():
    form = request.form.to_dict()
    files = request.files
    job_id = uuid.uuid4().hex
    workdir = JOBS_DIR / job_id; workdir.mkdir(parents=True, exist_ok=True)
    qlog = queue.Queue()
    JOBS[job_id] = {"queue": qlog, "done": False, "out_path": "", "workdir": workdir, "cookies": None}

    if "cookies" in files and files["cookies"]:
        f = files["cookies"]
        if getattr(f, "filename", ""):
            dst = workdir / "cookies.txt"; f.save(dst); JOBS[job_id]["cookies"] = dst

    t = threading.Thread(target=job_worker, args=(job_id, form), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.get("/logs/<job_id>")
def stream_logs(job_id: str):
    if job_id not in JOBS: return "no job", 404
    qlog = JOBS[job_id]["queue"]
    def gen():
        while True:
            try: yield f"data: {qlog.get(timeout=0.5)}\n\n"
            except queue.Empty: pass
            if JOBS[job_id]["done"]:
                out_path = JOBS[job_id]["out_path"]
                if out_path and Path(out_path).exists():
                    yield f"data: DONE:{Path(out_path).name}\n\n"
                else:
                    yield "data: FAIL: job ended with error\n\n"
                break
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/download/<job_id>/<fname>")
def download(job_id: str, fname: str):
    if job_id not in JOBS: return "no job", 404
    workdir: Path = JOBS[job_id]["workdir"]
    return send_from_directory(workdir, fname, as_attachment=True)

@app.get("/")
def root():
    return send_from_directory(Path(__file__).parent.resolve(), "index.html")

@app.get("/favicon.ico")
def favicon(): return "", 204

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
