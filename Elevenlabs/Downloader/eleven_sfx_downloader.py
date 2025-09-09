#!/usr/bin/env python3
"""
eleven_sfx_downloader.py

Two modes:
  1) history  – uses ElevenLabs API to download YOUR generated audio + metadata
  2) library  – best-effort crawler of public SFX category pages on elevenlabs.io

Outputs:
  - audio files in ./downloads/
  - metadata.csv with: id, filename, text, model_id (if available), source_url/category

Usage examples:
  # 1) Your history (official API):
  python eleven_sfx_downloader.py --mode history --api-key YOUR_XI_API_KEY

  # 2) Public library crawl (best-effort):
  python eleven_sfx_downloader.py --mode library --categories foley,swipe,bass --max-per-cat 200
"""
import argparse, csv, hashlib, os, re, time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://api.elevenlabs.io/v1"
PUBLIC_BASE = "https://elevenlabs.io"

def sanitize_filename(s: str, maxlen: int = 120) -> str:
    s = re.sub(r"[^\w\s\-\.\(\)_]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen] if s else "untitled"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_row(meta_path: Path, row: dict, header):
    file_exists = meta_path.exists()
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def download_binary(url: str, out_path: Path, headers=None, session=None):
    sess = session or requests.Session()
    r = sess.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)

# --------------------------- MODE 1: HISTORY (official) ---------------------------
def run_history(api_key: str, out_dir: Path, page_size: int = 1000):
    ensure_dir(out_dir)
    meta_csv = out_dir / "metadata.csv"
    header = ["id", "filename", "text", "model_id", "date_unix", "content_type", "source_url_or_category"]

    sess = requests.Session()
    sess.headers.update({"xi-api-key": api_key})

    last_id = None
    total = 0
    while True:
        params = {"page_size": page_size}
        if last_id:
            params["start_after_history_item_id"] = last_id
        # List your generated items (TTS, SFX, etc.)
        resp = sess.get(f"{BASE}/history", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("history", []) or []
        if not items:
            break

        for it in items:
            hid = it.get("history_item_id")
            text = it.get("text") or ""
            model_id = it.get("model_id") or ""
            ctype = it.get("content_type") or ""
            date_unix = it.get("date_unix")

            # (Optional) narrow to SFX by heuristic on model_id; comment this line out to keep everything
            # if model_id and "sound" not in model_id.lower() and "sfx" not in model_id.lower():
            #     continue

            # pick extension by content-type; default mp3
            ext = ".mp3"
            if "wav" in ctype:
                ext = ".wav"
            elif "mpeg" in ctype or "mp3" in ctype:
                ext = ".mp3"

            base_name = f"{hid}_{sanitize_filename(text[:60])}{ext}"
            out_path = out_dir / base_name
            if not out_path.exists():
                # download audio for this history item
                audio_url = f"{BASE}/history/{hid}/audio"
                r = sess.get(audio_url, timeout=120)
                r.raise_for_status()
                out_path.write_bytes(r.content)
                time.sleep(0.25)  # be polite

            write_row(
                meta_csv,
                {
                    "id": hid,
                    "filename": str(out_path.name),
                    "text": text,
                    "model_id": model_id,
                    "date_unix": date_unix,
                    "content_type": ctype,
                    "source_url_or_category": "history",
                },
                header,
            )
            total += 1

        if not data.get("has_more"):
            break
        last_id = data.get("last_history_item_id")

    print(f"[history] downloaded + indexed {total} items to {out_dir}")

# --------------------------- MODE 2: LIBRARY (public crawl) ---------------------------
from urllib.parse import urlparse

def _norm_href(hval) -> str:
    """Return a clean string href from bs4 .get('href') that may be str|list|None."""
    if isinstance(hval, list):
        hval = next((x for x in hval if isinstance(x, str) and x), "")  # first string item
    if not isinstance(hval, str):
        hval = "" if hval is None else str(hval)
    hval = hval.split("?", 1)[0].rstrip("/")  # strip query + trailing slash
    return hval

def discover_categories(session: requests.Session):
    """
    Scrape category slugs from the public SFX pages.
    Supports both /sound-effects/<slug> and /app/sound-effects/<slug>.
    """
    candidates = [
        f"{PUBLIC_BASE}/sound-effects",
        f"{PUBLIC_BASE}/app/sound-effects",
    ]

    cats: set[str] = set()
    for url in candidates:
        r = session.get(url, timeout=60)
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = _norm_href(a.get("href"))
            if not href:
                continue

            # accept both paths
            if href.startswith("/sound-effects/") or href.startswith("/app/sound-effects/"):
                parts = href.strip("/").split("/")
                if parts and parts[-1] not in ("sound-effects",):
                    cats.add(parts[-1])

    return sorted(cats)


def scrape_category(session: requests.Session, slug: str, out_dir: Path, max_per_cat: int, meta_header):
    url = f"{PUBLIC_BASE}/sound-effects/{slug}"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Heuristics: find any links to audio files or <audio> tags.
    audio_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(href.lower().endswith(ext) for ext in [".mp3", ".wav", ".ogg"]):
            audio_links.append((href, a))

    # also scan <audio> tags
    for aud in soup.find_all("audio"):
        src = aud.get("src") or ""
        if src and any(src.lower().endswith(ext) for ext in [".mp3", ".wav", ".ogg"]):
            audio_links.append((src, aud))

    # Dedup + make absolute
    seen = set()
    cleaned = []
    for href, node in audio_links:
        full = href if href.startswith("http") else f"{PUBLIC_BASE}{href}"
        if full not in seen:
            seen.add(full)
            cleaned.append((full, node))

    ensure_dir(out_dir / slug)
    meta_csv = out_dir / "metadata.csv"

    downloaded = 0
    for full, node in cleaned:
        if downloaded >= max_per_cat:
            break

        # grab a nearby description/title if present
        # try previous sibling text blocks or aria/alt/title
        text = ""
        # find closest text
        label_node = node
        for _ in range(4):
            if not label_node:
                break
            # combine text from siblings/parent
            txt = (label_node.get_text(" ", strip=True) or "").strip()
            if txt and len(txt) > 3:
                text = txt
                break
            label_node = label_node.parent

        # derive a stable id from URL hash (best-effort)
        hid = hashlib.sha1(full.encode("utf-8")).hexdigest()[:16]
        fname = f"{slug}_{hid}_{sanitize_filename(text[:60])}"
        ext = ".mp3"
        if ".wav" in full.lower():
            ext = ".wav"
        elif ".ogg" in full.lower():
            ext = ".ogg"

        out_path = out_dir / slug / f"{fname}{ext}"
        if not out_path.exists():
            try:
                download_binary(full, out_path, session=session)
                time.sleep(0.5)
            except Exception as e:
                print(f"[skip] {full} -> {e}")
                continue

        write_row(
            meta_csv,
            {
                "id": hid,
                "filename": str(Path(slug) / out_path.name),
                "text": text,
                "model_id": "",
                "date_unix": "",
                "content_type": Path(out_path).suffix.lstrip("."),
                "source_url_or_category": slug,
            },
            meta_header,
        )
        downloaded += 1

    print(f"[library] {slug}: saved {downloaded} items")

def run_library(out_dir: Path, categories: list[str] | None, max_per_cat: int):
    ensure_dir(out_dir)
    meta_header = ["id", "filename", "text", "model_id", "date_unix", "content_type", "source_url_or_category"]
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; sfx-downloader/1.0)"})

    if not categories:
        categories = discover_categories(sess)
        print(f"[library] discovered {len(categories)} categories")

    for slug in categories:
        scrape_category(sess, slug.strip(), out_dir, max_per_cat, meta_header)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["history", "library"], required=True)
    ap.add_argument("--api-key", help="Your xi-api-key (required for --mode history)")
    ap.add_argument("--out", default="downloads", help="Output directory")
    ap.add_argument("--max-per-cat", type=int, default=200, help="Max files per category (library mode)")
    ap.add_argument("--categories", default="", help="Comma-separated category slugs, e.g., foley,swipe,bass. Leave empty to auto-discover.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    if args.mode == "history":
        if not args.api_key:
            raise SystemExit("--api-key is required for --mode history")
        run_history(args.api_key, out_dir)
    else:
        cats = [c for c in args.categories.split(",") if c.strip()] if args.categories else None
        run_library(out_dir, cats, args.max_per_cat)

if __name__ == "__main__":
    main()
