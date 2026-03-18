"""
v2_vision_llm_bounds_detection.py  (was: zelda_translator.py / zelda_translator_1.py)
======================================================================================
ITERATION 2 — Replace Tesseract with a two-phase vision LLM pipeline.

What it does:
  Phase 1 — DETECTION: sends full frames to Qwen2.5-VL every 3s and asks it to
    locate the dialogue box, returning pixel bounds (x, y, w, h). Retries until
    bounds are found and persists them to bounds.json so Phase 2 can start fast
    on the next run.
  Phase 2 — TRANSLATION: crops every frame to the saved bounds, sends the image
    crop directly to the LLM for Japanese OCR + English translation in one shot.
    Pure constant polling — no stability check yet.

Key improvements over v1:
  - Eliminates Tesseract entirely for OCR; vision model reads the image directly
  - Automatic dialogue box location — no hardcoded crop fractions
  - Brightness gate (optional): skips LLM call if mean crop brightness is too
    high (gameplay, no dialogue box) or too low (fade/cutscene)
  - Furigana masking via connected-component blob-size filtering
  - MJPEG live preview endpoint (/video_feed) so you can watch the crop in browser
  - CSV logging of every LLM call (timestamp, japanese, translation, timing)
  - Timing stats shown in UI (OCR ms, LLM ms, total ms)

Limitation at this stage:
  - No stability/typewriter detection — still fires on every poll cycle
  - No vocab tracking or learning features
  - Uses lightweight qwen2.5:1.5b for translation (speed > quality trade-off)

Run:  python3 v2_vision_llm_bounds_detection.py
Open: http://localhost:5002
"""

import cv2
import numpy as np
import requests
import time
import threading
import base64
import json
import re
import csv
import os
from flask import Flask, render_template_string, jsonify, Response

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL         = "http://localhost:11434/api/generate"
TRANSLATION_MODEL  = "qwen2.5:1.5b"   # text-only, lightweight
VIDEO_SOURCE      = "http://192.168.1.107:8080/video"

LOG_FILE    = "pixel_llm_log.csv"
PREVIEW_PATH = os.path.expanduser("~/Downloads/preprocessed_crop.jpg")

# ── Furigana masking ─────────────────────────────────────────────────────────
# Furigana (ruby) characters are small text above main kanji.
# We isolate text blobs by size and mask out anything smaller than
# FURIGANA_RATIO * median_blob_height — those are furigana.
# Lower ratio = more aggressive masking. 0.55 is a safe starting point.
FURIGANA_RATIO   = 0.75
FURIGANA_ENABLED = True

# ── Brightness gate ───────────────────────────────────────────────────────────
# Dialogue boxes in Zelda are dark semi-transparent overlays.
# If the mean brightness of the crop exceeds this value it's almost certainly
# pure gameplay — skip the LLM call entirely. Tune by watching the terminal.
BRIGHTNESS_GATE    = 80.0   # 0–255; lower = stricter
BRIGHTNESS_ENABLED = False   # set False to disable the gate

# ── Prompts ─────────────────────────────────────────────────────────────────────
TRANSLATION_PROMPT = (
    "Translate the following Japanese text from a video game to English. "
    "Reply strictly in this format with no brackets or extra text:\n"
    "English: the english translation\n"
    "Japanese text to translate: {japanese}"
)

# ── Shared state ───────────────────────────────────────────────────────────────
BOUNDS_FILE = "bounds.json"

state = {
    "phase":              "TRANSLATING",
    "status":             "Starting up...",
    "japanese":           "",
    "translation":        "",
    "llm_calls":          0,
    "bounds":             None,
    "history":            [],
    "ocr_timing":         {"ocr_ms": 0},
    "translation_timing": {"llm_ms": 0, "total_ms": 0},
    "diff_value":         0.0,
    "brightness":         0.0,
    "error":              "",
}

# MJPEG preview buffer
latest_frame_jpg = None
frame_lock        = threading.Lock()

app = Flask(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_b64(frame, quality=90):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")

def encode_jpg(frame, quality=75):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def frame_diff(a, b):
    """Mean absolute pixel difference between two greyscale crops."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(ga.astype(np.float32) - gb.astype(np.float32))))

def log_entry(diff, brightness, japanese, english, raw):
    """Append one {timestamp, diff, brightness, llm_output} row to the CSV log."""
    file_exists = False
    try:
        with open(LOG_FILE, "r"):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "diff", "brightness", "japanese", "english", "raw"])
        writer.writerow([
            time.strftime("%H:%M:%S"),
            round(diff, 3),
            round(brightness, 1),
            japanese or "",
            english  or "",
            raw.strip().replace("\n", " "),
        ])

def update_preview(frame):
    global latest_frame_jpg
    if frame is None or frame.size == 0:
        frame = np.zeros((80, 320, 3), dtype=np.uint8)
    with frame_lock:
        latest_frame_jpg = encode_jpg(frame)

def mjpeg_generator():
    while True:
        with frame_lock:
            jpg = latest_frame_jpg
        if jpg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        time.sleep(0.08)

def push_history(japanese, english):
    state["history"].insert(0, {
        "japanese": japanese,
        "english":  english,
        "time":     time.strftime("%H:%M:%S"),
    })
    if len(state["history"]) > 10:
        state["history"].pop()

def vision_ocr(frame):
    """
    Use Apple Vision framework to OCR Japanese text from a cropped frame.
    Returns a single string of all detected text joined by spaces, or "" if none.
    """
    import Vision
    import Quartz
    import objc

    t0 = time.perf_counter()

    # Convert OpenCV BGR frame to PNG bytes
    _, buf = cv2.imencode(".png", frame)
    png_bytes = buf.tobytes()

    # Create CGImage from PNG bytes
    data_provider = Quartz.CGDataProviderCreateWithData(None, png_bytes, len(png_bytes), None)
    cg_image = Quartz.CGImageCreateWithPNGDataProvider(data_provider, None, False,
                                                        Quartz.kCGRenderingIntentDefault)

    results = []
    done    = threading.Event()

    def handler(request, error):
        if error:
            print(f"⚠️  Vision OCR error: {error}")
        else:
            for obs in request.results():
                results.append(obs.topCandidates_(1)[0].string())
        done.set()

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    request.setRecognitionLanguages_(["ja", "ja-JP"])
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)

    handler_obj = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
    handler_obj.performRequests_error_([request], None)
    done.wait(timeout=10)

    ocr_ms = round((time.perf_counter() - t0) * 1000)
    state["ocr_timing"] = {"ocr_ms": ocr_ms}

    text = " ".join(results).strip()
    print(f"⏱  [ocr] {ocr_ms}ms  →  {text[:80]}")
    return text


def translate_text(japanese):
    """Send Japanese text to Ollama text-only model for translation."""
    t0 = time.perf_counter()

    prompt = TRANSLATION_PROMPT.format(japanese=japanese)
    payload = {
        "model":  TRANSLATION_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    t1 = time.perf_counter()

    raw = r.json()["response"].strip()

    llm_ms   = round((t1 - t0) * 1000)
    state["translation_timing"] = {"llm_ms": llm_ms, "total_ms": llm_ms}
    state["llm_calls"] += 1

    print(f"⏱  [translate] llm={llm_ms}ms")
    print(f"🤖  {raw[:120]}")
    return raw

# ── Bounds loading ─────────────────────────────────────────────────────────────

def load_bounds():
    """Read bounds.json written by calibrate.py. Exits if missing or invalid."""
    try:
        with open(BOUNDS_FILE, "r") as f:
            data = json.load(f)
        assert all(k in data for k in ("x", "y", "w", "h")), "Missing keys"
        bounds = {k: int(data[k]) for k in ("x", "y", "w", "h")}
        print(f"📦  Bounds loaded: x={bounds['x']} y={bounds['y']} w={bounds['w']} h={bounds['h']}")
        return bounds
    except FileNotFoundError:
        print(f"❌  {BOUNDS_FILE} not found — run calibrate.py first.")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌  Failed to load {BOUNDS_FILE}: {e}")
        raise SystemExit(1)

# ── Translation phase ──────────────────────────────────────────────────────────

def crop_to_bounds(frame, bounds):
    h, w = frame.shape[:2]
    x  = max(0, bounds["x"])
    y  = max(0, bounds["y"])
    x2 = min(w, x + bounds["w"])
    y2 = min(h, y + bounds["h"])
    return frame[y:y2, x:x2]

def parse_translation(raw):
    """Extract English translation from text-only model response. Returns string or None."""
    if raw.strip().upper() == "NONE":
        return None
    # Try to find "English: ..." label
    m = re.search(r"english:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback — return the whole response if no label found
    return raw.strip() or None

def preprocess_crop(crop):
    """
    Mask out furigana (small ruby characters) using row-based density analysis.

    Steps:
      1. Threshold to isolate bright text pixels
      2. Sum bright pixels per row to get a vertical density profile
      3. Find peaks in the profile — each peak is a line of text
      4. For each text line peak, identify the furigana rows sitting above it
         (low-density rows between peaks or above the first peak)
      5. Black out those furigana rows in a copy of the crop
    """
    if not FURIGANA_ENABLED:
        return crop.copy()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

    # Row density profile: sum of bright pixels per row
    row_density = thresh.sum(axis=1).astype(np.float32) / 255.0

    # Smooth to reduce noise
    kernel_size = max(3, len(row_density) // 20) | 1  # must be odd
    smoothed = np.convolve(row_density, np.ones(kernel_size) / kernel_size, mode='same')

    # Find text line peaks — rows with density above threshold
    density_threshold = smoothed.max() * FURIGANA_RATIO
    in_peak   = False
    peaks     = []   # list of (start_row, end_row) for each text line band
    peak_start = 0

    for i, val in enumerate(smoothed):
        if not in_peak and val >= density_threshold:
            in_peak    = True
            peak_start = i
        elif in_peak and val < density_threshold:
            in_peak = False
            peaks.append((peak_start, i))

    if in_peak:
        peaks.append((peak_start, len(smoothed) - 1))

    if not peaks:
        return crop.copy()

    # Build row mask — True = keep, False = blank out (furigana)
    h = crop.shape[0]
    keep = np.zeros(h, dtype=bool)

    PEAK_PADDING = 10
    for peak_start, peak_end in peaks:
        keep[max(0, peak_start - PEAK_PADDING):min(h, peak_end + PEAK_PADDING + 1)] = True

    # Apply — black out non-kept rows
    cleaned = crop.copy()
    for row_idx in range(h):
        if not keep[row_idx]:
            cleaned[row_idx, :] = 0

    return cleaned

# Shared latest crop for pixel diff thread
latest_crop      = None
latest_crop_lock = threading.Lock()

def pixel_diff_thread(bounds):
    """Runs as a daemon — reads frames continuously, calculates diff, logs to state."""
    global latest_crop
    cap_diff = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap_diff.isOpened():
        print("⚠️  Pixel diff thread: cannot open camera")
        return

    prev_crop = None
    while True:
        ret, frame = cap_diff.read()
        if not ret:
            time.sleep(0.1)
            continue

        crop = crop_to_bounds(frame, bounds)
        if crop.size == 0:
            time.sleep(0.1)
            continue

        # Store latest crop for the translation loop to grab
        with latest_crop_lock:
            latest_crop = crop.copy()

        # Calculate and log diff
        if prev_crop is not None and prev_crop.shape == crop.shape:
            diff = frame_diff(prev_crop, crop)
            state["diff_value"] = round(diff, 2)
            # print(f"📊  pixel_diff={diff:.2f}")

        prev_crop = crop.copy()
        update_preview(crop)

def translation_loop(cap, bounds):
    print("🌐  TRANSLATION — constant polling started.")
    print(f"📝  Logging pairs to {LOG_FILE}")
    state["phase"]  = "TRANSLATING"
    state["status"] = "Translating..."

    # Start pixel diff as a background thread on its own camera connection
    t = threading.Thread(target=pixel_diff_thread, args=(bounds,), daemon=True)
    t.start()

    while True:
        # Grab the latest crop from the diff thread
        with latest_crop_lock:
            crop = latest_crop.copy() if latest_crop is not None else None

        if crop is None:
            time.sleep(0.1)
            continue

        if crop.size == 0:
            print("⚠️  Crop is empty — check bounds")
            state["status"] = "Crop empty — check bounds"
            time.sleep(1)
            continue

        # Snapshot the current diff at the moment this frame is sent
        diff_at_capture = state["diff_value"]

        # ── Brightness gate ───────────────────────────────────────────────────
        brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        state["brightness"] = round(brightness, 1)
        if BRIGHTNESS_ENABLED and brightness > BRIGHTNESS_GATE:
            print(f"💡  Skipping — brightness={brightness:.1f} > {BRIGHTNESS_GATE} (gameplay)")
            state["status"] = f"Skipped (brightness={brightness:.1f})"
            continue

        # ── Furigana preprocessing + save preview ────────────────────────────
        cleaned = preprocess_crop(crop)
        cv2.imwrite(PREVIEW_PATH, cleaned)

        try:
            # Step 1: Apple Vision OCR — read all Japanese text from image
            jp = vision_ocr(cleaned)

            if not jp:
                state["status"] = "No text detected"
                print("💤  OCR: no text found\n")
                log_entry(diff_at_capture, brightness, "", "", "NONE")
                continue

            # Step 2: Text-only LLM — translate Japanese to English
            raw = translate_text(jp)
            en  = parse_translation(raw)

            # Log the pair
            log_entry(diff_at_capture, brightness, jp, en or "", raw)
            print(f"🔗  logged → diff={diff_at_capture:.2f}  brightness={brightness:.1f}  jp={jp[:40]}")

            state["japanese"]    = jp
            state["translation"] = en or ""
            state["status"]      = "Live"
            state["error"]       = ""
            push_history(jp, en or "")
            print(f"📺  {jp}")
            print(f"✅  {en}\n")

        except Exception as e:
            print(f"❌  Error: {e}")
            state["error"]  = str(e)
            state["status"] = "Error"
            time.sleep(1)

# ── Main capture orchestrator ──────────────────────────────────────────────────

def capture_loop():
    bounds = load_bounds()
    state["bounds"] = bounds
    state["status"] = f"Bounds: x={bounds['x']} y={bounds['y']} w={bounds['w']} h={bounds['h']}"

    print(f"📡  Connecting to {VIDEO_SOURCE} ...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌  Cannot connect. Check VIDEO_SOURCE.")
        state["status"] = "Cannot connect to camera"
        state["error"]  = "Check VIDEO_SOURCE and IP Webcam app"
        return
    print("✅  Connected.")

    translation_loop(cap, bounds)
    cap.release()

# ── Flask routes ───────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zelda Translator</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #080b10;
    --surface:   #0d1117;
    --border:    #1c2333;
    --border2:   #252d3d;
    --text:      #cdd6f4;
    --subtext:   #6c7a96;
    --dim:       #2a3245;
    --accent:    #89b4fa;
    --green:     #a6e3a1;
    --yellow:    #f9e2af;
    --red:       #f38ba8;
    --mauve:     #cba6f7;
    --peach:     #fab387;
    --teal:      #94e2d5;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --jp:        'Noto Sans JP', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    display: grid;
    grid-template-columns: 1fr 300px;
    grid-template-rows: auto 1fr;
    gap: 0;
  }

  /* ── Header ── */
  header {
    grid-column: 1 / -1;
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface);
  }
  .logo {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .logo span { color: var(--subtext); }
  .phase-pill {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid currentColor;
    transition: all 0.3s;
  }
  .phase-pill.DETECTING  { color: var(--yellow); border-color: var(--yellow); background: rgba(249,226,175,0.06); }
  .phase-pill.TRANSLATING { color: var(--green); border-color: var(--green); background: rgba(166,227,161,0.06); }
  .header-status {
    margin-left: auto;
    font-size: 12px;
    color: var(--subtext);
    font-family: var(--mono);
    max-width: 420px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .header-status.error { color: var(--red); }

  /* ── Main panel ── */
  main {
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    overflow-y: auto;
  }

  /* Translation card */
  .trans-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
  }
  .trans-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 10px;
  }
  .japanese-text {
    font-family: var(--jp);
    font-size: 28px;
    font-weight: 400;
    color: var(--text);
    line-height: 1.7;
    min-height: 44px;
    margin-bottom: 20px;
  }
  .japanese-text.placeholder { color: var(--dim); font-size: 18px; }
  .divider { border: none; border-top: 1px solid var(--border); margin-bottom: 20px; }
  .english-text {
    font-size: 17px;
    font-weight: 300;
    color: #a8b5d4;
    line-height: 1.65;
    min-height: 28px;
  }
  .english-text.placeholder { color: var(--dim); font-size: 14px; }

  /* Metrics row */
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }
  .metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
  }
  .metric-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 700;
    color: var(--accent);
  }
  .metric-value.green { color: var(--green); }
  .metric-value.yellow { color: var(--yellow); }

  /* Timing bars */
  .timing-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .timing-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
  }
  .timing-title {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 16px;
  }
  .t-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
  }
  .t-row:last-child { margin-bottom: 0; }
  .t-name {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--subtext);
    width: 52px;
    flex-shrink: 0;
  }
  .t-track {
    flex: 1;
    height: 4px;
    background: var(--border2);
    border-radius: 4px;
    overflow: hidden;
  }
  .t-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
    min-width: 2px;
  }
  .t-ms {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--subtext);
    width: 52px;
    text-align: right;
  }

  /* Bounds info */
  .bounds-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 24px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--subtext);
    display: none;
  }
  .bounds-card.visible { display: block; }
  .bounds-card strong { color: var(--teal); }

  /* ── Sidebar ── */
  aside {
    border-left: 1px solid var(--border);
    background: var(--surface);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .aside-section {
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }
  .aside-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--subtext);
    margin-bottom: 10px;
  }

  /* Preview */
  .preview-wrap {
    background: #050709;
    border-radius: 8px;
    overflow: hidden;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .preview-wrap img {
    width: 100%;
    display: block;
  }
  .preview-placeholder {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    text-align: center;
    padding: 20px;
  }

  /* Detection scanning animation */
  .scan-bar {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--yellow), transparent);
    background-size: 60% 100%;
    animation: scan 1.8s ease-in-out infinite;
    display: none;
  }
  .scan-bar.active { display: block; }
  @keyframes scan {
    0%   { background-position: -60% 0; }
    100% { background-position: 160% 0; }
  }

  /* History */
  .history-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
  }
  .h-entry {
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
  }
  .h-entry:last-child { border-bottom: none; }
  .h-time {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--dim);
    margin-bottom: 4px;
  }
  .h-jp {
    font-family: var(--jp);
    font-size: 13px;
    color: var(--text);
    margin-bottom: 3px;
    line-height: 1.5;
  }
  .h-en {
    font-size: 11px;
    color: var(--subtext);
    line-height: 1.4;
  }
  .h-empty {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--dim);
    text-align: center;
    padding: 24px 0;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
</style>
</head>
<body>

<header>
  <div class="logo">ZELDA <span>/</span> TRANSLATOR</div>
  <div class="phase-pill" id="phase-pill">DETECTING</div>
  <div class="header-status" id="header-status">Initializing...</div>
</header>

<main>
  <!-- Translation -->
  <div class="trans-card">
    <div class="trans-label">Japanese</div>
    <div class="japanese-text placeholder" id="japanese">Waiting for dialogue...</div>
    <hr class="divider">
    <div class="trans-label">English</div>
    <div class="english-text placeholder" id="english">Translation will appear here</div>
  </div>

  <!-- Metrics -->
  <div class="metrics-row">
    <div class="metric-box">
      <div class="metric-label">LLM Calls</div>
      <div class="metric-value" id="llm-calls">0</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Pixel Diff</div>
      <div class="metric-value yellow" id="diff-value">—</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Brightness</div>
      <div class="metric-value" id="brightness-value">—</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Last Translate</div>
      <div class="metric-value green" id="total-ms">—</div>
    </div>
  </div>

  <!-- Timing breakdown -->
  <div class="timing-grid">
    <div class="timing-card">
      <div class="timing-title">OCR timing (Apple Vision)</div>
      <div class="t-row">
        <div class="t-name">OCR</div>
        <div class="t-track"><div class="t-fill" id="ot-ocr" style="background:var(--teal)"></div></div>
        <div class="t-ms" id="otv-ocr">—</div>
      </div>
    </div>
    <div class="timing-card">
      <div class="timing-title">Translation timing (LLM)</div>
      <div class="t-row">
        <div class="t-name">LLM</div>
        <div class="t-track"><div class="t-fill" id="tt-llm" style="background:var(--mauve)"></div></div>
        <div class="t-ms" id="ttv-llm">—</div>
      </div>
      <div class="t-row">
        <div class="t-name">Total</div>
        <div class="t-track"><div class="t-fill" id="tt-total" style="background:var(--subtext)"></div></div>
        <div class="t-ms" id="ttv-total">—</div>
      </div>
    </div>
  </div>

  <!-- Bounds info -->
  <div class="bounds-card" id="bounds-card">
    Dialogue bounds locked — <strong id="bounds-text"></strong>
  </div>
</main>

<aside>
  <div class="aside-section">
    <div class="aside-label">Live crop</div>
    <div class="preview-wrap">
      <img src="/preview" id="preview-img" alt="Live crop">
    </div>
  </div>
  <div class="aside-label" style="padding:12px 16px 0">History</div>
  <div class="history-list" id="history-list">
    <div class="h-empty">No translations yet</div>
  </div>
</aside>

<script>
async function poll() {
  try {
    const d = await (await fetch('/state')).json();

    // Phase pill
    const pill = document.getElementById('phase-pill');
    pill.textContent = d.phase;
    pill.className = 'phase-pill ' + d.phase;

    // Status
    const hs = document.getElementById('header-status');
    hs.textContent = d.error ? d.error : (d.status || '');
    hs.className = 'header-status' + (d.error ? ' error' : '');

    // Japanese
    const jpEl = document.getElementById('japanese');
    if (d.japanese) {
      jpEl.textContent = d.japanese;
      jpEl.className = 'japanese-text';
    } else {
      jpEl.innerHTML = '<span>Waiting for dialogue...</span>';
      jpEl.className = 'japanese-text placeholder';
    }

    // English
    const enEl = document.getElementById('english');
    if (d.translation) {
      enEl.textContent = d.translation;
      enEl.className = 'english-text';
    } else {
      enEl.innerHTML = '<span>Translation will appear here</span>';
      enEl.className = 'english-text placeholder';
    }

    // Metrics
    document.getElementById('llm-calls').textContent  = d.llm_calls || 0;
    document.getElementById('diff-value').textContent = d.diff_value != null ? d.diff_value.toFixed(2) : '—';
    const bv = document.getElementById('brightness-value');
    bv.textContent = d.brightness != null ? d.brightness.toFixed(1) : '—';
    bv.style.color = (d.brightness > 80) ? 'var(--red)' : 'var(--green)';
    // OCR timing
    if (d.ocr_timing && d.ocr_timing.ocr_ms) {
      document.getElementById('ot-ocr').style.width  = '100%';
      document.getElementById('otv-ocr').textContent = d.ocr_timing.ocr_ms + 'ms';
    }
    // Translation timing
    if (d.translation_timing && d.translation_timing.llm_ms) {
      const t = d.translation_timing;
      document.getElementById('tt-llm').style.width    = '100%';
      document.getElementById('tt-total').style.width  = '100%';
      document.getElementById('ttv-llm').textContent   = t.llm_ms + 'ms';
      document.getElementById('ttv-total').textContent = t.total_ms + 'ms';
      document.getElementById('total-ms').textContent  = t.total_ms + 'ms';
    }

    // Bounds
    if (d.bounds) {
      const bc = document.getElementById('bounds-card');
      bc.className = 'bounds-card visible';
      const b = d.bounds;
      document.getElementById('bounds-text').textContent =
        `x=${b.x} y=${b.y} w=${b.w} h=${b.h}`;
    }

    // History
    const hl = document.getElementById('history-list');
    if (d.history && d.history.length) {
      hl.innerHTML = d.history.map(h =>
        `<div class="h-entry">
          <div class="h-time">${h.time}</div>
          <div class="h-jp">${h.japanese}</div>
          <div class="h-en">${h.english}</div>
        </div>`
      ).join('');
    } else {
      hl.innerHTML = '<div class="h-empty">No translations yet</div>';
    }

  } catch(e) {}
  setTimeout(poll, 500);
}
poll();
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/preview')
def preview():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def unload_model():
    """Tell Ollama to evict the model from RAM immediately."""
    try:
        requests.post(OLLAMA_URL, json={"model": TRANSLATION_MODEL, "keep_alive": 0}, timeout=10)
        print(f"\n🧹  Model {TRANSLATION_MODEL} unloaded from RAM.")
    except Exception as e:
        print(f"\n⚠️  Could not unload model: {e}")

if __name__ == '__main__':
    print("🎮  Zelda Translator")
    print(f"📱  Camera: {VIDEO_SOURCE}")
    print(f"🤖  OCR:    Apple Vision (on-device)")
    print(f"🤖  Translate: {TRANSLATION_MODEL}")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        unload_model()
