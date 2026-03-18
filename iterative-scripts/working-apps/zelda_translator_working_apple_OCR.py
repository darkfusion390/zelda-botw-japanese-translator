"""
zelda_translator.py
===================
Two-step pipeline:
  Step 1 — OCR: Apple Vision framework (macOS built-in, M1 neural engine)
           Reads the cropped dialogue box and returns raw Japanese text.
           No model download, no extra RAM, runs natively on Apple Silicon.

  Step 2 — Translation: qwen2.5:1.5b via Ollama
           Takes the Japanese text string and returns a natural English translation.
           Pure text call, no image, fast and lightweight.

UI: http://localhost:5002

Install deps:
  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
"""

import cv2
import numpy as np
import requests
import time
import threading
import json
import re
import csv
import os
import tempfile
from flask import Flask, render_template_string, jsonify, Response

import Vision
import Quartz

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL         = "http://localhost:11434/api/generate"
TRANSLATION_MODEL  = "qwen2.5:7b"
IP_WEBCAM_URL      = "http://192.168.1.107:8080/video"

LOG_FILE     = "pixel_llm_log.csv"
PREVIEW_PATH = os.path.expanduser("~/Downloads/preprocessed_crop.jpg")

# ── Brightness gate ───────────────────────────────────────────────────────────
# Dialogue boxes in Zelda are dark semi-transparent overlays.
# Skip the pipeline if the crop is too bright (gameplay) or too dark (fade/cutscene).
BRIGHTNESS_GATE_HIGH = 80.0   # above this = gameplay, no dialogue box
BRIGHTNESS_GATE_LOW  = 10.0   # below this = fade/cutscene, nothing to read
BRIGHTNESS_ENABLED   = False  # set True to enable both gates

# ── Prompts ──────────────────────────────────────────────────────────────────
TRANSLATION_PROMPT = (
    "Translate this Japanese text into natural English. "
    "Output the translation only. No explanations, no notes, no preamble.\n\n{japanese}"
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

def apple_vision_ocr(frame):
    """
    Run Apple Vision Japanese OCR on a BGR numpy frame.
    Saves a temp JPEG, loads it as a CGImage, runs VNRecognizeTextRequest
    with Japanese language hint, returns concatenated text string.
    """
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    try:
        img_url = Quartz.CFURLCreateFromFileSystemRepresentation(
            None, tmp_path.encode(), len(tmp_path), False
        )
        src = Quartz.CGImageSourceCreateWithURL(img_url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

        results = []
        def handler(request, error):
            if error:
                return
            for obs in request.results():
                results.append(obs.topCandidates_(1)[0].string())

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLanguages_(["ja"])
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)

        handler_obj = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}
        )
        handler_obj.performRequests_error_([request], None)

        japanese = " ".join(results).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

def ollama_translate(japanese):
    """Send Japanese text to qwen2.5:1.5b for translation. Returns English string."""
    t0 = time.perf_counter()
    payload = {
        "model":  TRANSLATION_MODEL,
        "prompt": TRANSLATION_PROMPT.format(japanese=japanese),
        "stream": False,
        "options": {
            "num_ctx":     512,
            "num_batch":   64,
            "num_keep":    0,
            "temperature": 0,
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    english = r.json()["response"].strip()
    state["llm_calls"] += 1
    print(f"⏱  [translate] {elapsed_ms}ms  →  {english[:80]}")
    return english, elapsed_ms

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
    """Returns (japanese, english) or (None, None)."""
    if raw.strip().upper() == "NONE":
        return None, None

    # Split on the English: keyword — everything before is Japanese, after is English.
    # This handles both single-line and multi-line model responses correctly.
    parts = re.split(r"english:", raw, flags=re.IGNORECASE, maxsplit=1)

    if len(parts) == 2:
        japanese_part = parts[0]
        english_part  = parts[1].strip()

        # Strip the "Japanese:" label from the japanese part
        japanese = re.sub(r"^japanese:\s*", "", japanese_part, flags=re.IGNORECASE).strip()
        english  = english_part

        if japanese and english:
            return japanese, english

    # Last resort: try line by line
    japanese = english = None
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("japanese:"):
            japanese = line.split(":", 1)[1].strip()
        elif line.lower().startswith("english:"):
            english = line.split(":", 1)[1].strip()

    return japanese, english

def preprocess_crop(crop):
    """
    1. Extract white text pixels onto black background
    2. Black out sparse rows (furigana) using median-based density threshold
    3. Upscale with Lanczos for sharper dakuten dots

    - Threshold 160 (was 200): catches grunge-eroded BotW font pixels
    - median * 0.42 (was max * 0.20): uses median row density as reference
      so short centred lines aren't wiped alongside furigana
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Lower threshold catches grunge-eroded character pixels
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    if mask.max() == 0:
        return crop.copy()

    row_density = mask.sum(axis=1) / 255.0

    # Build white-on-black result
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)

    # Use median of non-zero rows as reference — more stable than max
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density = float(np.median(non_zero_densities))
        furigana_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furigana_threshold:
                result[i, :] = 0

    # Upscale 2x with Lanczos — preserves fine details like dakuten dots
    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Solid black border
    result = cv2.copyMakeBorder(
        result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return result

def clean_ocr(text):
    """
    Strip non-Japanese characters from Vision OCR output.
    Vision occasionally picks up edge noise as stray symbols or kana.
    Keep only hiragana, katakana, kanji, full-width chars, and Japanese punctuation.
    """
    text = re.sub(
        r'[^\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\s、。！？「」『』・…]',
        '', text
    )
    return re.sub(r'\s+', ' ', text).strip()

def normalize_for_dedup(text):
    """Strip spaces and punctuation variations before comparing — prevents
    minor OCR inconsistencies (spacing, commas) from bypassing the dedup check."""
    return re.sub(r'[\s、。・…]', '', text)

# Shared latest crop for pixel diff thread
latest_crop      = None
latest_crop_lock = threading.Lock()

def pixel_diff_thread(bounds):
    """Runs as a daemon — reads frames continuously, calculates diff, logs to state."""
    global latest_crop
    cap_diff = cv2.VideoCapture(IP_WEBCAM_URL)
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
        # Preview updated in translation_loop with processed frame

def clean_ocr(text):
    text = re.sub(
        r'[^\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\s、。！？「」『』・…]',
        '', text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove isolated single kana surrounded by spaces or at start/end
    # These are always furigana noise, never real sentence content
    text = re.sub(r'(?<!\S)\S(?!\S)', '', text).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    return text      

def translation_loop(cap, bounds):
    """
    Main pipeline loop — runs on the main thread after camera connects.
    Continuously grabs the latest crop from the pixel diff thread, preprocesses it,
    runs OCR, applies stability/dedup gates, and fires the LLM when new stable
    dialogue is detected. A single camera connection is used here; a second
    independent connection is used by pixel_diff_thread to avoid frame contention.

    Stability gates (in order):
      Gate 1: Empty / NONE output — skip, reset stability counter
      Gate 2: Too short — fewer than MIN_JAPANESE_CHARS real characters
      Gate 3: Stability — text must be identical for STABLE_THRESHOLD consecutive
              frames. Prevents firing the LLM mid-scroll while text is typing in.
      Gate 4: Dedup — fuzzy edit-distance check against last translated text.
              Tolerates 1-character OCR noise so the same line doesn't re-fire.
    """
    print(f"📝  Logging pairs to {LOG_FILE}")
    state["phase"]  = "TRANSLATING"
    state["status"] = "Translating..."

    # Stability tracker — only translate once dialogue finishes typing
    STABLE_THRESHOLD  = 4   # consecutive identical reads required
    MIN_JAPANESE_CHARS = 3  # minimum chars to be considered real dialogue

    text_stable = {"text": "", "stable_count": 0}

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

        diff_at_capture = state["diff_value"]

        # ── Brightness gate ───────────────────────────────────────────────────
        brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        state["brightness"] = round(brightness, 1)
        if BRIGHTNESS_ENABLED:
            if brightness > BRIGHTNESS_GATE_HIGH:
                print(f"💡  Skipping — brightness={brightness:.1f} too bright (gameplay)")
                state["status"] = f"Skipped (too bright={brightness:.1f})"
                continue
            if brightness < BRIGHTNESS_GATE_LOW:
                print(f"🌑  Skipping — brightness={brightness:.1f} too dark (fade/cutscene)")
                state["status"] = f"Skipped (too dark={brightness:.1f})"
                continue

        # ── Preprocessing ─────────────────────────────────────────────────────
        cleaned = preprocess_crop(crop)
        cv2.imwrite(PREVIEW_PATH, cleaned)
        update_preview(cleaned)

        try:
            # Step 1: OCR with Apple Vision
            cv2.imwrite(os.path.expanduser("~/Downloads/ocr_input.jpg"), cleaned)
            t0 = time.perf_counter()
            jp, ocr_ms = apple_vision_ocr(cleaned)
            jp = clean_ocr(jp)
            state["ocr_timing"] = {"ocr_ms": ocr_ms}

            # Gate 1: empty — reset stability and skip
            if not jp or jp.upper() == "NONE":
                text_stable["text"] = ""
                text_stable["stable_count"] = 0
                state["status"] = "No text in frame"
                print("💤  OCR returned empty\n")
                continue

            # Gate 2: minimum length — noise/hallucination is usually very short
            if len(jp.replace(" ", "")) < MIN_JAPANESE_CHARS:
                text_stable["text"] = ""
                text_stable["stable_count"] = 0
                print(f"🗑   Too short, likely noise: {jp}")
                state["status"] = "Noise detected, skipping"
                continue

            # Gate 3: stability — wait for text to stop changing (dialogue done typing)
            normalized = normalize_for_dedup(jp)
            if normalized == normalize_for_dedup(text_stable["text"]):
                text_stable["stable_count"] += 1
            else:
                # Text changed — still being typed, reset and skip immediately
                text_stable["text"] = jp
                text_stable["stable_count"] = 1
                print(f"📝  Text changing... {jp}")
                state["status"] = "Dialogue typing..."
                continue

            if text_stable["stable_count"] < STABLE_THRESHOLD:
                print(f"⏳  Stabilizing ({text_stable['stable_count']}/{STABLE_THRESHOLD}): {jp}")
                state["status"] = f"Stabilizing... ({text_stable['stable_count']}/{STABLE_THRESHOLD})"
                continue

            # Gate 4: dedup — already translated this exact text
            if normalize_for_dedup(jp) == normalize_for_dedup(state.get("japanese", "")):
                print(f"⏭   Already translated, skipping.")
                continue

            state["japanese"] = jp

            # Step 2: Translate
            en, translate_ms = ollama_translate(jp)
            state["translation_timing"] = {
                "llm_ms":   translate_ms,
                "total_ms": ocr_ms + translate_ms,
            }

            log_entry(diff_at_capture, brightness, jp, en, en)
            print(f"🔗  logged → diff={diff_at_capture:.2f}  brightness={brightness:.1f}")

            state["translation"] = en
            state["status"]      = "Live"
            state["error"]       = ""
            push_history(jp, en)
            print(f"📺  {jp}")
            print(f"✅  {en}\n")

        except Exception as e:
            print(f"❌  Error: {e}")
            state["error"]  = str(e)
            state["status"] = "Error"
            time.sleep(1)

# ── Main capture orchestrator ──────────────────────────────────────────────────

def capture_loop():
    """
    Entry point for the background thread. Loads bounds, opens the camera
    connection, and hands off to translation_loop. Runs for the lifetime of
    the process — camera is released only on clean exit.
    """
    bounds = load_bounds()
    state["bounds"] = bounds
    state["status"] = f"Bounds: x={bounds['x']} y={bounds['y']} w={bounds['w']} h={bounds['h']}"

    print(f"📡  Connecting to {IP_WEBCAM_URL} ...")
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print("❌  Cannot connect. Check IP_WEBCAM_URL.")
        state["status"] = "Cannot connect to camera"
        state["error"]  = "Check IP_WEBCAM_URL and IP Webcam app"
        return
    print("✅  Connected.")
    print("🍎  Using Apple Vision OCR — no model load needed.")

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
  <div class="timing-card">
    <div class="timing-title">Pipeline timing</div>
    <div class="t-row">
      <div class="t-name">OCR</div>
      <div class="t-track"><div class="t-fill" id="tt-ocr" style="background:var(--accent)"></div></div>
      <div class="t-ms" id="ttv-ocr">—</div>
    </div>
    <div class="t-row">
      <div class="t-name">Translate</div>
      <div class="t-track"><div class="t-fill" id="tt-llm" style="background:var(--mauve)"></div></div>
      <div class="t-ms" id="ttv-llm">—</div>
    </div>
    <div class="t-row">
      <div class="t-name">Total</div>
      <div class="t-track"><div class="t-fill" id="tt-total" style="background:var(--subtext)"></div></div>
      <div class="t-ms" id="ttv-total">—</div>
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
    // Timing
    const t = d.translation_timing;
    const o = d.ocr_timing;
    if (t && o && t.total_ms) {
      const max = t.total_ms || 1;
      document.getElementById('tt-ocr').style.width   = Math.min(100, o.ocr_ms   / max * 100) + '%';
      document.getElementById('tt-llm').style.width   = Math.min(100, t.llm_ms   / max * 100) + '%';
      document.getElementById('tt-total').style.width = '100%';
      document.getElementById('ttv-ocr').textContent   = o.ocr_ms   + 'ms';
      document.getElementById('ttv-llm').textContent   = t.llm_ms   + 'ms';
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
    """Tell Ollama to evict the translation model from RAM immediately."""
    try:
        requests.post(OLLAMA_URL, json={"model": TRANSLATION_MODEL, "keep_alive": 0}, timeout=10)
        print(f"\n🧹  Model {TRANSLATION_MODEL} unloaded from RAM.")
    except Exception as e:
        print(f"\n⚠️  Could not unload model: {e}")

if __name__ == '__main__':
    print("🎮  Zelda Translator")
    print(f"📱  Camera: {IP_WEBCAM_URL}")
    print(f"🤖  Model:  {TRANSLATION_MODEL}")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        unload_model()
