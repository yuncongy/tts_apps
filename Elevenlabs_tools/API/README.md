
# ElevenLabs Batch TTS (Streamlit)

A tiny web UI to batch-generate speech from **text-only** rows in a CSV/TSV using the ElevenLabs API.

## Quick start

1. Create and activate a Python 3.9+ environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. In the browser UI:
   - Paste your **API Key** and **Voice ID**.
   - Upload your CSV/TSV.
   - Choose the text column.
   - Set output folder and options.
   - Click **Start Batch**.

### Environment variables (optional)

You can pre-fill some fields via env vars:

- `ELEVEN_API_KEY`
- `ELEVEN_VOICE_ID`
- `ELEVEN_MODEL_ID`
- `ELEVEN_VOICE_SETTINGS_JSON` (JSON string)
- `OUTPUT_DIR`

Example on macOS/Linux:

```bash
export ELEVEN_API_KEY="xi-..."
export ELEVEN_VOICE_ID="YOUR_VOICE_ID"
export OUTPUT_DIR="./outputs"
streamlit run app.py
```

### File naming

Uses `{row}`, `{slug}`, `{hash8}`, and `{ext}` to build filenames, e.g.:

```
{row:05d}_{slug}.{ext}  ->  00012_hello-world.mp3
```

### Logging

A `generation_log.csv` is appended in your output directory with columns:
`row, text, status, filepath, error, latency_s`

### Notes

- This app sends one request at a time (sequential), with optional **Requests/Minute** throttling.
- Retries with exponential backoff on **429** and **5xx** errors.
- Use **Skip existing** to resume safely.
- No reference audio is used or required.
