"""
v1_tesseract_ollama_basic.py  (was: zelda_simple.py)
=====================================================
ITERATION 1 — The simplest working proof of concept.

What it does:
  - Captures video from an Android phone running IP Webcam (Samsung S21)
  - Crops a hardcoded region of the screen to isolate the Zelda dialogue box
  - Uses Tesseract OCR (jpn) to extract Japanese text from the cropped frame
  - Sends the extracted text to a local Ollama LLM (qwen2.5:7b) for translation
  - Serves a minimal dark-themed Flask web UI at localhost:5002 showing
    the Japanese text and English translation, polled every 1.5 seconds

Key design choices / limitations at this stage:
  - Pure Tesseract OCR: aggressive white-pixel threshold + connected-component
    furigana removal, no vision model involved
  - Hardcoded crop fractions for both the TV bezel and the dialogue box region
  - No stability detection — polls on a fixed 5-second interval regardless of
    whether the dialogue has changed
  - No vocab tracking, no learning features, no history
  - Translation is a plain Ollama text call: "translate this, reply only with English"

Run:  python3 v1_tesseract_ollama_basic.py
Open: http://localhost:5002
"""

import cv2
import pytesseract
import requests
import time
import threading
import re
from flask import Flask, render_template_string, jsonify
from PIL import Image
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "qwen2.5:7b"
POLL_INTERVAL = 5
IP_WEBCAM_URL = "http://192.168.1.107:8080/video"

# ── Crop tuning ───────────────────────────────────────────────────────────────
# These are fractions of the TV crop (after the outer bezel crop below).
# Dialogue box sits in the bottom-left portion of the screen.
# Adjust if text is being cut off or noise creeps back in.
DLGBOX_TOP    = 0.62   # top edge of dialogue bubble
DLGBOX_BOTTOM = 1.00   # bottom edge
DLGBOX_LEFT   = 0.00   # left edge
DLGBOX_RIGHT  = 0.72   # right edge (excludes the response option buttons)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
state = {"japanese": "", "translation": "", "processing": False}

def preprocess_frame(frame):
    """
    frame is already cropped to just the dialogue box region.
    It has a dark semi-transparent background with pure-white text.
    We just need to threshold for white and clean up noise.
    """
    # Upscale for better OCR
    frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # White text >200; dark box background is ~30-80; no game world pixels survive the crop
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Small dilation to reconnect strokes broken by JPEG/camera compression
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Remove furigana and tiny noise blobs via connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    if heights:
        max_h = max(heights)
        min_main_h = max_h * 0.45
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            if h >= min_main_h and w < frame.shape[1] * 0.8 and area > 30:
                filtered[labels == i] = 255
        mask = filtered

    # Tesseract wants black text on white background
    return cv2.bitwise_not(mask)

def extract_japanese(frame):
    processed = preprocess_frame(frame)
    cv2.imwrite("ocr_processed.jpg", processed)

    raw = pytesseract.image_to_string(
        Image.fromarray(processed),
        lang='jpn',
        config='--oem 1 --psm 6'
    )

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        ja_chars = re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', line)
        if len(ja_chars) < 3:
            continue
        total = len(line.replace(" ", ""))
        if total > 0 and len(ja_chars) / total < 0.4:
            continue
        lines.append(line)

    result = " ".join(lines).strip()
    print(f"🔍  OCR raw: {result}")
    return result

def translate(japanese):
    prompt = (
        f"Translate this Japanese to English. "
        f"Reply with only the English translation, nothing else.\n\n"
        f"Japanese: {japanese}"
    )
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=30)
    return r.json()["response"].strip()

def capture_loop():
    print(f"📡  Connecting to {IP_WEBCAM_URL} ...")
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print("❌  Cannot connect. Check IP_WEBCAM_URL and that IP Webcam app is running.")
        return
    print("✅  Connected. Open http://localhost:5002\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Lost connection — retrying...")
            time.sleep(3)
            cap = cv2.VideoCapture(IP_WEBCAM_URL)
            continue

        # Step 1: crop out phone bezel / surroundings to get just the TV
        h, w = frame.shape[:2]
        tv = frame[int(h * 0.15):int(h * 0.80), int(w * 0.05):int(w * 0.95)]
        cv2.imwrite("ocr_preview.jpg", tv)

        # Step 2: hard-crop to the dialogue box region within the TV frame
        th, tw = tv.shape[:2]
        dlg = tv[
            int(th * DLGBOX_TOP)   : int(th * DLGBOX_BOTTOM),
            int(tw * DLGBOX_LEFT)  : int(tw * DLGBOX_RIGHT)
        ]
        cv2.imwrite("ocr_preview_box.jpg", dlg)

        japanese = extract_japanese(dlg)
        if not japanese:
            time.sleep(POLL_INTERVAL)
            continue

        print(f"📺  {japanese}")
        state["japanese"]   = japanese
        state["processing"] = True

        try:
            translation = translate(japanese)
            state["translation"] = translation
            print(f"✅  {translation}\n")
        except Exception as e:
            print(f"❌  {e}")
        finally:
            state["processing"] = False

        time.sleep(POLL_INTERVAL)

    cap.release()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Zelda Translator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0a0a0f;
    font-family: 'Helvetica Neue', sans-serif;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
  }
  .card {
    background: #12121a;
    border: 1px solid #2a2a3a;
    border-radius: 16px;
    padding: 48px;
    max-width: 760px;
    width: 100%;
    text-align: center;
  }
  .japanese {
    font-size: 38px;
    color: #f0e6c8;
    line-height: 1.6;
    min-height: 56px;
    margin-bottom: 32px;
  }
  .divider { border: none; border-top: 1px solid #2a2a3a; margin-bottom: 32px; }
  .translation {
    font-size: 26px;
    color: #e8d9a0;
    line-height: 1.5;
    min-height: 40px;
  }
  .placeholder { color: #333344; }
  .status {
    margin-top: 32px;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #444460;
  }
  .dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #333;
    margin-right: 6px;
    vertical-align: middle;
  }
  .dot.live { background: #4caf50; }
  .dot.thinking { background: #f0a500; animation: pulse 0.8s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }
</style>
</head>
<body>
<div class="card">
  <div class="japanese" id="japanese"><span class="placeholder">Waiting for dialogue...</span></div>
  <hr class="divider">
  <div class="translation" id="translation"><span class="placeholder">Translation will appear here</span></div>
  <div class="status"><span class="dot" id="dot"></span><span id="status">Idle</span></div>
</div>
<script>
async function poll() {
  try {
    const d = await (await fetch('/state')).json();
    const dot = document.getElementById('dot');
    const status = document.getElementById('status');
    dot.className = 'dot' + (d.processing ? ' thinking' : d.japanese ? ' live' : '');
    status.textContent = d.processing ? 'Translating...' : d.japanese ? 'Live' : 'Idle';
    if (d.japanese) {
      document.getElementById('japanese').textContent = d.japanese;
    } else {
      document.getElementById('japanese').innerHTML = '<span class="placeholder">Waiting for dialogue...</span>';
    }
    if (d.translation) {
      document.getElementById('translation').textContent = d.translation;
    } else {
      document.getElementById('translation').innerHTML = '<span class="placeholder">Translation will appear here</span>';
    }
  } catch(e) {}
  setTimeout(poll, 1500);
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

if __name__ == '__main__':
    print("🎮  Zelda Simple Translator")
    print(f"📱  {IP_WEBCAM_URL}")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5002, debug=False)
