import asyncio, csv, hashlib, os, re, time
from pathlib import Path
from urllib.parse import urlparse
from playwright.async_api import async_playwright

EXPLORE_URL = "https://elevenlabs.io/app/sound-effects"

def sanitize_filename(s: str, maxlen: int = 100) -> str:
    s = re.sub(r"[^\w\s\-\.\(\)_]+", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen] or "untitled"

async def scroll_to_load(page, target_count: int, max_scrolls: int = 200, step: int = 1200):
    last_height = 0
    for _ in range(max_scrolls):
        await page.evaluate(f"window.scrollBy(0, {step});")
        await page.wait_for_timeout(300)
        # stop early if we’ve loaded enough Download buttons
        count = await page.locator("a[download], button[aria-label*='download' i], a[aria-label*='download' i]").count()
        if count >= target_count:
            break
        # small safeguard to avoid endless loops when nothing changes
        height = await page.evaluate("document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height

async def run(out_dir="public_library", max_items=60, headless=False, storage_state="state.json"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_csv = out / "metadata.csv"
    meta_header = ["id", "filename", "text", "source_url"]

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context_kwargs = {}
        if Path(storage_state).exists():
            context_kwargs["storage_state"] = storage_state
        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()

        await page.goto(EXPLORE_URL, wait_until="networkidle")

        # If login is required and you haven't saved state yet, let user log in once
        if not Path(storage_state).exists():
            # Give you time to sign in if needed
            print("If a login is required, please sign in in the opened window. Waiting 15s...")
            await page.wait_for_timeout(15000)
            await context.storage_state(path=storage_state)

        # Load enough items
        await scroll_to_load(page, target_count=max_items)

        # We try multiple selectors to be resilient
        download_buttons = page.locator("a[download], button[aria-label*='download' i], a[aria-label*='download' i]")
        n = await download_buttons.count()
        print(f"Found ~{n} potential download buttons")

        taken = 0
        for i in range(n):
            if taken >= max_items:
                break

            btn = download_buttons.nth(i)

            # Try to find a nearby card/title text for metadata
            # Heuristic: get accessible name + nearest text container
            label = (await btn.get_attribute("aria-label")) or ""
            # Dive up a bit in DOM and grab text
            text = ""
            try:
                card = btn.locator("xpath=ancestor::*[self::div or self::article][1]")
                text = (await card.inner_text()).strip()
                # keep it short
                text = " ".join(text.split())[:200]
            except:
                text = label or ""

            # Click & capture the download
            try:
                with page.expect_download(timeout=15000) as dl_info:
                    await btn.click(force=True)
                d = await dl_info.value
                suggested = d.suggested_filename or "clip.mp3"
                # create a stable id from the download URL or filename
                url = d.url or suggested
                clip_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

                fname = f"{clip_id}_{sanitize_filename(text[:60])}"
                # use original extension if present
                ext = os.path.splitext(suggested)[1] or ".mp3"
                out_path = out / f"{fname}{ext}"

                await d.save_as(str(out_path))

                # write metadata
                new_file = not meta_csv.exists()
                with meta_csv.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=meta_header)
                    if new_file:
                        w.writeheader()
                    w.writerow({
                        "id": clip_id,
                        "filename": out_path.name,
                        "text": text,
                        "source_url": EXPLORE_URL
                    })

                taken += 1
                print(f"[{taken}/{max_items}] saved {out_path.name}")
                await page.wait_for_timeout(250)  # be polite
            except Exception as e:
                # If a button wasn’t a real download or failed, skip it
                print(f"[skip] item {i}: {e}")

        await browser.close()
        print(f"Done. Saved {taken} files to {out.resolve()}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public_library")
    ap.add_argument("--max-items", type=int, default=60)
    ap.add_argument("--headless", action="store_true", help="Run without a visible browser window")
    ap.add_argument("--state", default="state.json", help="Storage state file for keeping login")
    args = ap.parse_args()
    asyncio.run(run(out_dir=args.out, max_items=args.max_items, headless=args.headless, storage_state=args.state))
