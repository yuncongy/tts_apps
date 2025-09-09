

"""
Prefill and run
export ELEVEN_API_KEY="xi-..."                         # required if not typing in UI
export ELEVEN_VOICE_ID="YOUR_VOICE_ID"                 # required if not typing in UI
export ELEVEN_MODEL_ID="eleven_monolingual_v1"         # optional
export ELEVEN_VOICE_SETTINGS_JSON='{"stability":0.4}'  # optional
export OUTPUT_DIR="./outputs"                          # optional
streamlit run app.py

"""
# app.py - Streamlit UI for batch TTS with ElevenLabs API (text only, no ref audio)
# Run: streamlit run app.py
import os
import io
import time
import json
import queue
import hashlib
import pathlib
import requests
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Dict, Any

# ----------------------------- Helpers & Config -----------------------------

def slugify(s: str, max_len: int = 40) -> str:
    """Very light slug for filenames."""
    keep = []
    for ch in s.strip():
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in [' ', '-', '_']:
            keep.append('-')
        else:
            keep.append('')
    slug = ''.join(keep).strip('-')
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug[:max_len] or "sample"

def infer_sep(name: str, head: bytes) -> str:
    """Infer delimiter: TSV if tabs are common, else comma."""
    try:
        sample = head.decode('utf-8', errors='ignore')
        if sample.count('\t') > sample.count(','):
            return '\t'
    except Exception:
        pass
    if name.lower().endswith('.tsv'):
        return '\t'
    return ','

def ensure_dir(p: str) -> str:
    path = pathlib.Path(p).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

# ----------------------------- API Client -----------------------------------
# --- NEW: Sound Effects config & client ---
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json, requests

@dataclass
class ElevenSFXConfig:
    api_key: str
    model_id: str = "eleven_text_to_sound_v2"
    output_format: str = "mp3_44100_128"  # enum expected by API
    base_url: str = "https://api.elevenlabs.io"
    loop: bool = False
    duration_seconds: Optional[float] = None   # 0.5–30, or None to auto
    prompt_influence: Optional[float] = 0.3    # 0–1

class ElevenSFXClient:
    def __init__(self, cfg: ElevenSFXConfig):
        self.cfg = cfg

    def synth(self, text: str) -> bytes:
        if not text or not text.strip():
            raise ValueError("Empty text provided.")
        url = f"{self.cfg.base_url}/v1/sound-generation"
        headers = {
            "xi-api-key": self.cfg.api_key,
            # choose Accept by container format
            "accept": "audio/mpeg" if self.cfg.output_format.startswith("mp3") else "audio/wav",
            "content-type": "application/json",
        }
        params = {"output_format": self.cfg.output_format}
        body = {"text": text}
        if self.cfg.model_id: body["model_id"] = self.cfg.model_id
        if self.cfg.loop: body["loop"] = True
        if self.cfg.duration_seconds: body["duration_seconds"] = self.cfg.duration_seconds
        if self.cfg.prompt_influence is not None:
            body["prompt_influence"] = float(self.cfg.prompt_influence)

        r = requests.post(url, headers=headers, params=params, data=json.dumps(body), timeout=120)
        if r.status_code == 200:
            return r.content
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        err = requests.HTTPError(f"HTTP {r.status_code}: {detail}")
        err.response = r
        raise err


# ----------------------------- Streamlit UI ---------------------------------

st.set_page_config(page_title="ElevenLabs Batch TTS", layout="wide")

st.title("🎧 ElevenLabs Batch TTS (Text → Audio)")
st.caption("Upload a CSV/TSV of texts. No reference audio required. The app will call the ElevenLabs API sequentially.")

with st.expander("🔑 API & Sound Effects Settings", expanded=True):
    api_key = st.text_input("API Key (XI-API-KEY)", value=os.getenv("ELEVEN_API_KEY",""), type="password")

    # SFX-specific options
    model_id = st.text_input("Model ID (default: eleven_text_to_sound_v2)",
                             value=os.getenv("ELEVEN_MODEL_ID","eleven_text_to_sound_v2"))
    output_format = st.selectbox("Output format (enum)", 
                                 ["mp3_44100_128", "mp3_22050_32", "wav_44100", "mulaw_8000"], index=0)
    loop = st.checkbox("Loop (smooth looping if supported)", value=False)
    duration_val = st.number_input("Duration (seconds, 0.5–30; 0 = auto)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
    prompt_influence = st.slider("Prompt influence (0–1)", 0.0, 1.0, 0.3, 0.05)

with st.expander("📄 Dataset", expanded=True):
    up = st.file_uploader("Upload CSV/TSV", type=["csv","tsv","txt"], accept_multiple_files=False)
    text_col_name = None
    df = None
    if up is not None:
        # Peek at head to infer delimiter
        head = up.getvalue()[:4096]
        sep = infer_sep(up.name, head)
        df = pd.read_csv(io.BytesIO(up.getvalue()), sep=sep)
        sep_display = "\\t" if sep == "\t" else sep
        st.success(f"Parsed {len(df)} rows (sep='{sep_display}').")

        if len(df.columns) == 1:
            text_col_name = df.columns[0]
        else:
            text_col_name = st.selectbox("Which column contains the text?", list(df.columns))
        st.dataframe(df.head(10))

with st.expander("📦 Output & Batch Controls", expanded=True):
    out_dir_default = os.getenv("OUTPUT_DIR", "./outputs")
    out_dir = st.text_input("Output directory", value=out_dir_default)
    fn_pattern = st.text_input("Filename pattern", value="{row:05d}_{slug}.{ext}", help="Available keys: row, slug, hash8, ext")
    start_row = st.number_input("Start row (0-based index)", min_value=0, value=0, step=1)
    limit_rows = st.number_input("Max rows to process (0 = all)", min_value=0, value=0, step=1)
    rpm = st.number_input("Requests per minute (throttle)", min_value=0, value=0, step=1, help="0 = no throttling; otherwise adds delay to respect rate limits.")
    backoff_base = st.number_input("Backoff base (seconds) for 429/5xx", min_value=1.0, value=2.0, step=0.5)
    backoff_max = st.number_input("Backoff cap (seconds)", min_value=1.0, value=30.0, step=1.0)
    skip_existing = st.checkbox("Skip if output file already exists", value=True)

    colA, colB = st.columns(2)
    run_btn = colA.button("▶️ Start Batch", type="primary", use_container_width=True)
    stop_btn = colB.button("⏹️ Stop", use_container_width=True)

# Session state to manage running flag
if "running" not in st.session_state:
    st.session_state.running = False
if stop_btn:
    st.session_state.running = False
if run_btn:
    st.session_state.running = True

# ----------------------------- Batch Logic ----------------------------------

log_rows = []
log_placeholder = st.empty()
progress = st.progress(0.0, text="Idle")

def format_filename(row_idx: int, text: str, ext: str) -> str:
    slug = slugify(text, max_len=60)
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return fn_pattern.format(row=row_idx, slug=slug, hash8=h, ext=ext)

def save_log(out_path: str, rows: list):
    if not rows:
        return
    import csv
    ensure_dir(out_path)
    log_file = pathlib.Path(out_path) / "generation_log.csv"
    write_header = not log_file.exists()
    with log_file.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row","text","status","filepath","error","latency_s"])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    rows.clear()

def throttle_sleep(rpm: int):
    if rpm and rpm > 0:
        time.sleep(60.0 / float(rpm))

def parse_voice_settings(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        st.warning("Could not parse voice settings JSON. Ignoring.")
        return None

if st.session_state.running and df is not None and text_col_name is not None:
    # Validate config (SFX: no voice_id required)
    if not api_key:
        st.error("Please provide your API Key.")
        st.session_state.running = False
    else:
        cfg = ElevenSFXConfig(
            api_key=api_key,
            model_id=(model_id.strip() if model_id else "eleven_text_to_sound_v2"),
            output_format=output_format,          # e.g., mp3_44100_128 / wav_44100 / mulaw_8000
            loop=loop,
            duration_seconds=(None if duration_val == 0 else float(duration_val)),
            prompt_influence=float(prompt_influence),
        )
        client = ElevenSFXClient(cfg)

        def ext_from_format(fmt: str) -> str:
            # mp3_* maps to .mp3; wav_* and mulaw_* (WAV container) map to .wav
            return "mp3" if str(fmt).startswith("mp3") else "wav"

        total = len(df)
        if limit_rows and limit_rows > 0:
            end_row = min(total, start_row + limit_rows)
        else:
            end_row = total

        out_dir_abs = ensure_dir(out_dir)
        st.info(f"Processing rows [{start_row} … {end_row-1}] into: {out_dir_abs}")

        for i in range(start_row, end_row):
            if not st.session_state.running:
                st.warning("Stopped by user.")
                break

            text = str(df.iloc[i][text_col_name])
            ext = "mp3" if output_format == "mp3" else "wav"
            fname = format_filename(i, text, ext)
            fpath = str(pathlib.Path(out_dir_abs) / fname)

            if skip_existing and pathlib.Path(fpath).exists():
                log_rows.append({"row": i, "text": text[:1000], "status":"skipped_exists", "filepath": fpath, "error":"", "latency_s": 0.0})
                progress.progress((i - start_row + 1) / max(1, (end_row - start_row)), text=f"Skipped row {i} (exists).")
                continue

            # Backoff loop for 429/5xx
            attempt = 0
            t0 = time.time()
            while True:
                try:
                    audio = client.synth(text)
                    pathlib.Path(fpath).write_bytes(audio)
                    latency = time.time() - t0
                    log_rows.append({"row": i, "text": text[:1000], "status":"ok", "filepath": fpath, "error":"", "latency_s": round(latency,3)})
                    break
                except requests.HTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    # 429 / 5xx -> backoff
                    if code == 429 or (code is not None and 500 <= code < 600):
                        attempt += 1
                        delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
                        progress.progress((i - start_row) / max(1, (end_row - start_row)), text=f"Rate-limited or server error ({code}). Backing off {delay:.1f}s… (attempt {attempt})")
                        time.sleep(delay)
                        continue
                    else:
                        # Non-retryable error
                        log_rows.append({"row": i, "text": text[:1000], "status":"http_error", "filepath":"", "error":str(e), "latency_s": round(time.time()-t0,3)})
                        break
                except Exception as e:
                    log_rows.append({"row": i, "text": text[:1000], "status":"error", "filepath":"", "error":str(e), "latency_s": round(time.time()-t0,3)})
                    break

            # Periodic log flush
            if len(log_rows) >= 20:
                save_log(out_dir_abs, log_rows)

            # Throttle between successes
            throttle_sleep(int(rpm))

            # Progress indicator
            done = (i - start_row + 1)
            progress.progress(done / max(1, (end_row - start_row)), text=f"Processed row {i} / {end_row-1}")

        save_log(out_dir_abs, log_rows)
        st.success("Batch complete or stopped. Log written to generation_log.csv")
else:
    st.info("Configure settings and click Start Batch.")

with st.expander("❓Notes"):
    st.markdown("""
- **Input file**: Provide a CSV or TSV with exactly one column of text, or select the text column.
- **Output files**: Named with your pattern (e.g. `{row:05d}_{slug}.{ext}`) in the chosen output directory.
- **Throttle**: Set Requests/Minute to respect your ElevenLabs plan's rate limits.
- **Backoff**: 429 and 5xx responses trigger exponential backoff (cap = Backoff cap).
- **Resume**: Re-run with the same output folder and `Skip existing` enabled.
- **Logging**: A `generation_log.csv` accumulates statuses.
- **Security**: API keys entered here are not stored on disk by this app.
    """)
