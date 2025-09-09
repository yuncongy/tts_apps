
# tts_apps — Text & Sound Generation Utilities

This repository collects small, practical tools I built for working with **text‑to‑speech (TTS)** and **sound‑effects** workflows. It includes:
- **ElevenLabs / Downloader** — automation tools to bulk‑download SFX you see in the Explore page, plus a best‑effort scraper for public categories.
- **TTS_Generation** — starter scripts for running different TTS models (e.g., Index‑TTS, DIA, Chatterbox) via SSH on a remote GPU box.

---

## Quick Start (Repo‑wide)

Requirements: **Python 3.9+** 

```bash
# 1) Clone & enter
git clone https://github.com/yuncongy/tts_apps.git
cd tts_apps

# 2) Create and activate a virtual environment (recommended)
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Install per folder
Each subfolder has its own lightweight requirements.

```bash
# ElevenLabs downloader tools
cd Elevenlabs/Downloader (path might be changed)
pip install -r requirements.txt  # (or see README in this folder)

# Go back to repo root when needed
cd ../../
```

---

## What’s Included

```
tts_apps/
├─ Elevenlabs_tools/
|  └─ API/
│     ├─ app.py
│     ├─ README.md                     # Detailed instructions for this folder
│     ├─ requirements.txt
│     └─ outputs/ (created at runtime)
│  └─ Downloader/
│     ├─ sfx_explore_downloader.py     # Playwright-based Explore page automation
│     ├─ eleven_sfx_downloader.py      # "library" mode: public-category static links
│     ├─ README.md                     # Detailed instructions for this folder
└─ TTS_Generation/
   ├─ create_index-tts_ssh.py
   ├─ create_dia_tts_ssh.py
   └─ create_chatterbox_tts_ssh.py
```

---

## Quick Usage — ElevenLabs Downloader (2 scripts)

> Full details are in **`Elevenlabs/Downloader/README.md`**. 

### A) Explore-page automation (recommended)
Downloads the exact SFX you see in the Explore UI.

```bash
cd Elevenlabs/Downloader
pip install playwright && python -m playwright install chromium

# First run opens a browser so you can log in; saves session state.json
python sfx_explore_downloader.py --out ./public_explore --max-items 100

# Subsequent headless runs
python sfx_explore_downloader.py --out ./public_explore --max-items 200 --headless
```

**Outputs:** audio files in `--out` and `metadata.csv` with columns  
`id, filename, text, source_url, file_size_bytes, sha256, downloaded_at_iso`.

### B) Public category “library” mode
Scrapes static audio links (if present) from public category pages.

```bash
cd Elevenlabs/Downloader
python eleven_sfx_downloader.py --mode library --categories booms,whooshes,bass --max-per-cat 150 --out ./public_library
```

**Outputs:** files under `public_library/<category>/…` and `metadata.csv` with the same columns as above.

---

## About the `TTS_Generation` Scripts

This folder contains three model‑specific runners intended for remote GPU execution via SSH:

- `create_index-tts_ssh.py`
- `create_dia_tts_ssh.py`
- `create_chatterbox_tts_ssh.py`

**What they do (high level):**
- Read a list of prompts (CSV/TSV) and generate speech with the chosen model.
- Stream logs, manage output naming, and support long‑running batches over SSH.

> **Clarification Requested:** To document exact usage, please confirm for each script:
> - CLI flags (e.g., `--input`, `--out`, `--device`, any model path/env vars).
> - Expected input file schema (column names, examples).
> - Output format (sample rate, extension, any metadata).
> - Any required environment modules (PyTorch/CUDA version, HF cache directory).

Once confirmed, I’ll add runnable examples here and a tiny `requirements.txt` for this folder.

---

## Why this repo?

- **Pragmatic automation**: The ElevenLabs tooling mirrors how I ship quick, reliable utilities around third‑party platforms.
- **Scalable generation**: The TTS scripts reflect my experience running large batch jobs, logging, and reproducible outputs.

---

## License / ToS

- Respect ElevenLabs Terms of Service and rate limits when using the downloaders.

