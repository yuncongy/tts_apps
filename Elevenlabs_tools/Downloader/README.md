
# ElevenLabs SFX Bulk Downloader

This repo contains two Python utilities for downloading sound effects (SFX) from ElevenLabs:

- **`sfx_explore_downloader.py`** — Automates the **Explore** page (`/app/sound-effects`) using Playwright to click the visible **Download** buttons and save files locally. Recommended if you want the exact clips you see in Explore.
- **`eleven_sfx_downloader.py`** — Provides a **`library`** mode that scrapes public SFX category pages and downloads directly linked audio (works only if links are present in static HTML; many pages are JS-rendered). *(History/API mode intentionally omitted in this README per request.)*

> ⚠️ Please respect ElevenLabs’ Terms of Service and rate limits. Automated downloading may break if the website changes or if access requires authentication.

---

## Requirements

- Python **3.9+** (macOS / Linux / Windows)
- Ability to run a headless browser (Playwright installs Chromium automatically)

### Quick Setup (copy‑paste)

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Base dependencies for both scripts
pip install requests beautifulsoup4

# For the Explore downloader (required)
pip install playwright
python -m playwright install chromium
```

---

## Script A — `sfx_explore_downloader.py` (Explore page automation)

This script opens `https://elevenlabs.io/app/sound-effects`, scrolls to load items, clicks each **Download** button, and saves files to disk. It also writes a `metadata.csv` describing each file.

### Usage

```bash
# First run (visible browser): lets you sign in if needed and saves session to state.json
python sfx_explore_downloader.py --out ./public_explore --max-items 100

# Subsequent runs (headless, reusing saved state)
python sfx_explore_downloader.py --out ./public_explore --max-items 200 --headless
```

**Common flags**

- `--out` — Output folder (default: `public_explore`)
- `--max-items` — Maximum clips to download per run (default: 60)
- `--headless` — Run without opening a browser window
- `--state` — Path to reuse login session (default: `state.json`)

**Outputs**

- Audio files saved under `--out`
- `--out/metadata.csv` with columns:

| column              | description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `id`                | Stable hash of the download URL                                             |
| `filename`          | Saved filename                                                              |
| `text`              | Nearby card/title text captured from the Explore UI                         |
| `source_url`        | The Explore page URL                                                        |
| `file_size_bytes`   | Size of the saved file                                                      |
| `sha256`            | SHA‑256 hash of the saved file (useful for dedup)                           |
| `downloaded_at_iso` | Timestamp in ISO 8601 when the file was saved                               |

> Notes:
> - First time you may see the window asking you to sign in. The script waits briefly, then stores the session at `state.json` so later runs can be headless.
> - If nothing downloads, re-run **without** `--headless` and confirm you can click **Download** manually. If the site UI changes, update the script selectors accordingly.


---

## Script B — `eleven_sfx_downloader.py` (public category **library** mode only)

This script attempts to download audio directly linked on public category pages (no JavaScript rendering). If the site populates items via JS, this may yield few or no files. Prefer Script A for the Explore feed.

### Usage

```bash
# Auto-discover categories (best-effort; may find few)
python eleven_sfx_downloader.py --mode library --out ./public_library

# Or specify categories explicitly (more reliable)
python eleven_sfx_downloader.py --mode library   --categories booms,whooshes,bass,braams --max-per-cat 150   --out ./public_library
```

**Outputs**

- Audio saved under `./public_library/<category>/…`
- `./public_library/metadata.csv` with the same columns as above:  
  `id, filename, text, source_url, file_size_bytes, sha256, downloaded_at_iso`

---

## Troubleshooting

- **Empty folders (library mode):** The category page likely uses JavaScript to inject content. Use `sfx_explore_downloader.py` instead.
- **No downloads in Explore:** Run without `--headless` once, log in, wait a few seconds for the script to capture state, then try again with `--headless`.
- **Rate limiting / throttling:** Reduce `--max-items` or wait and retry.
- **Hashes & sizes missing:** Ensure you’re on the latest version of the scripts (they compute file size and SHA‑256 after saving).

---

## Open Questions (please confirm)

1. The additional metadata columns chosen are: **`file_size_bytes`, `sha256`, `downloaded_at_iso`**. Do you also want **duration**? *(Would require an extra dependency like `mutagen` or `pydub`.)*
2. Should the Explore downloader filter by keywords or categories (e.g., only “whoosh”, “boom”)?
3. Is the default `--max-items 60` acceptable, or do you want a different default?
