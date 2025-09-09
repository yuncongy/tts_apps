
# ElevenLabs — Downloader Tools

This folder contains two Python tools to save ElevenLabs **sound effects (SFX)** to your local machine with clean metadata.

> **TL;DR**
> - Use `sfx_explore_downloader.py` to download the **exact** SFX you see in the Explore page.
> - Use `eleven_sfx_downloader.py --mode library` only if the category pages expose **static** audio links (often they don't).

---

## Requirements

- Python **3.9+**
- `requests`, `beautifulsoup4` (installed via `requirements.txt`)
- For Explore automation: **Playwright** + **Chromium**

### Install

```bash
# From repo root
cd Elevenlabs/Downloader

# Option A: use the provided requirements
pip install -r requirements.txt

# Option B: install manually
pip install requests beautifulsoup4
pip install playwright && python -m playwright install chromium
```

> If you use a virtual environment, activate it before installing:
> - macOS/Linux: `source .venv/bin/activate`
> - Windows PowerShell: `.venv\Scripts\Activate.ps1`

---

## Script 1 — `sfx_explore_downloader.py` (Recommended)

Automates `https://elevenlabs.io/app/sound-effects` with Playwright:
- Scrolls to load items
- Clicks each **Download** button
- Saves the file
- Writes `metadata.csv`

### Usage

```bash
# First run: visible browser to let you log in (if needed)
python sfx_explore_downloader.py --out ./public_explore --max-items 100

# Subsequent runs: headless, reusing saved session (state.json)
python sfx_explore_downloader.py --out ./public_explore --max-items 200 --headless
```

**Options**
- `--out` (default: `public_explore`) — output directory
- `--max-items` (default: `60`) — maximum downloads per run
- `--headless` — run without a visible browser window
- `--state` (default: `state.json`) — file to persist your login session

**Outputs**
- Saved audio files in `--out`
- `--out/metadata.csv` with columns:

| column              | description                                      |
|---------------------|--------------------------------------------------|
| `id`                | Stable hash of the download URL                  |
| `filename`          | Saved filename                                   |
| `text`              | Nearby card/title text captured from the UI      |
| `source_url`        | Explore page URL                                 |
| `file_size_bytes`   | Size of the saved file                           |
| `sha256`            | SHA‑256 of the saved file (useful for dedup)     |
| `downloaded_at_iso` | ISO‑8601 timestamp when the file was saved       |

---

## Script 2 — `eleven_sfx_downloader.py` (Public **library** mode)

Attempts to download audio directly linked in **public category** pages (no JavaScript rendering). If the page is JS‑driven, you may see empty folders — prefer Script 1 for Explore.

### Usage

```bash
# Auto-discovery (best-effort; may find few)
python eleven_sfx_downloader.py --mode library --out ./public_library

# Explicit categories (recommended)
python eleven_sfx_downloader.py --mode library   --categories booms,whooshes,bass,braams --max-per-cat 150   --out ./public_library
```

**Outputs**
- Saved files in `./public_library/<category>/…`
- `./public_library/metadata.csv` with the same columns as Script 1.

---

## Tips & Troubleshooting

- **Nothing downloads in Explore** → Run without `--headless` once, log in, and retry. Ensure `python -m playwright install chromium` was run.
- **Empty folders in library mode** → The site likely uses JS to inject items. Use `sfx_explore_downloader.py` instead.
- **Be gentle** → These scripts include short sleeps, but if you see throttling, lower `--max-items` and retry later.

---

## Notes for Recruiters

This folder demonstrates:
- Practical automation around a real product UI (Playwright).
- Robust metadata capture (size, hash, timestamp) to support dataset building and deduplication.
- Clear, reproducible instructions and outputs.
