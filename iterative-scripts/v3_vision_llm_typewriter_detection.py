"""
v3_vision_llm_typewriter_detection.py  (was: zelda_vision.py)
==============================================================
ITERATION 3 — Add typewriter/animation detection so the LLM fires only once
per complete dialogue, not on every frame.

What it does:
  - Same vision LLM pipeline as v2 (Qwen2.5-VL reads the crop directly)
  - Adds a frame-diff state machine with three states:
      IDLE        → no dialogue box visible, waiting
      ANIMATING   → dialogue box detected but text is still typing in
      STABLE      → text has stopped changing; fires the LLM exactly once
  - Stability is defined as STABLE_FRAMES_NEEDED (6) consecutive frames
    where mean pixel diff < DIFF_THRESHOLD (8)
  - MIN_ACTIVE_FRAMES guard prevents triggering on transitional/empty screens
  - FRAME_SKIP reduces CPU load by comparing every Nth frame

Key improvements over v2:
  - Single LLM call per dialogue instead of firing every poll cycle
  - State machine visible in the UI (IDLE / ANIMATING / STABLE)
  - Typewriter animation no longer causes duplicate or mid-sentence translations
  - Translation history panel (last N translations shown in UI)
  - Stable progress bar in UI showing approach to stability threshold

Limitation at this stage:
  - Still no vocab tracking or learning features
  - Hardcoded crop fractions (no auto-detection from v2)
  - macOS/Linux only (no Apple Vision yet)

Run:  python3 v3_vision_llm_typewriter_detection.py
Open: http://localhost:5002
"""

import cv2
import numpy as np
import requests
import time
import threading
import base64
from flask import Flask, render_template_string, jsonify, Response

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "qwen2.5vl:7b"       # ollama pull qwen2.5vl:7b
IP_WEBCAM_URL = "http://192.168.1.107:8080/video"

# ── Typewriter detection tuning ───────────────────────────────────────────────
# How different two frames must be to count as "still animating" (0-255 scale)
# Lower = more sensitive. Raise if flickering causes false triggers.
DIFF_THRESHOLD      = 8

# How many consecutive stable frames before we consider text "done"
# At ~10fps from IP Webcam, 6 frames ≈ 0.6 seconds of stability
STABLE_FRAMES_NEEDED = 6

# How many frames to skip between comparisons (reduces CPU load)
FRAME_SKIP = 2

# Minimum number of frames that must show a dialogue box before we bother
# waiting for stability — avoids triggering on empty/transitioning screens
MIN_ACTIVE_FRAMES = 3
# ─────────────────────────────────────────────────────────────────────────────

# ── Crop tuning ───────────────────────────────────────────────────────────────
TV_TOP    = 0.15
TV_BOTTOM = 0.80
TV_LEFT   = 0.05
TV_RIGHT  = 0.95

DLGBOX_TOP    = 0.62
DLGBOX_BOTTOM = 1.00
DLGBOX_LEFT   = 0.00
DLGBOX_RIGHT  = 0.72
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = (
    "This is a cropped screenshot from a Japanese video game. "
    "There is a dialogue box containing Japanese text. "
    "Read the Japanese text exactly as written, then translate it to English. "
    "Reply in this exact format:\n"
    "Japanese: <the japanese text>\n"
    "English: <the english translation>\n"
    "If there is no dialogue text visible, reply with exactly: NONE"
)

app = Flask(__name__)
state = {
    "japanese":       "",
    "translation":    "",
    "processing":     False,
    "llm_calls":      0,
    "stable_progress": 0,
    "status_detail":  "Idle",
    "diff_value":     0.0,
    "state_machine":  "IDLE",
    "history":        [],
    "last_timing":    {"encode_ms": 0, "llm_ms": 0, "parse_ms": 0, "total_ms": 0},
}

# Shared MJPEG frame buffer
latest_frame_jpg = None
frame_lock        = threading.Lock()

SM_BGR = {
    "IDLE":        (60,  60,  50),
    "ANIMATING":   (0,  108, 224),
    "STABILISING": (23, 160, 212),
    "LOCKED":      (217, 144,  74),
    "TRANSLATING": (221, 120, 198),
    "LIVE":        (79, 175,  76),
    "ERROR":       (85,  85, 224),
}

def crop_dialogue(frame):
    """Crop camera frame down to just the dialogue box."""
    h, w = frame.shape[:2]
    tv = frame[int(h * TV_TOP):int(h * TV_BOTTOM), int(w * TV_LEFT):int(w * TV_RIGHT)]
    th, tw = tv.shape[:2]
    dlg = tv[
        int(th * DLGBOX_TOP)   : int(th * DLGBOX_BOTTOM),
        int(tw * DLGBOX_LEFT)  : int(tw * DLGBOX_RIGHT)
    ]
    return tv, dlg

def frame_diff(a, b):
    """Mean absolute pixel difference between two greyscale frames."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return np.mean(np.abs(ga.astype(np.float32) - gb.astype(np.float32)))

def image_to_base64(frame):
    """Encode an OpenCV frame as a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buf).decode("utf-8")

def encode_jpg(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()

def update_preview(frame, sm):
    global latest_frame_jpg
    if frame is None or frame.size == 0:
        frame = np.zeros((80, 320, 3), dtype=np.uint8)
    preview = frame.copy()
    color = SM_BGR.get(sm, (60, 60, 50))
    cv2.rectangle(preview, (0, 0), (preview.shape[1]-1, preview.shape[0]-1), color, 4)
    with frame_lock:
        latest_frame_jpg = encode_jpg(preview)

def mjpeg_generator():
    while True:
        with frame_lock:
            jpg = latest_frame_jpg
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(0.08)

def push_history(japanese, english):
    state["history"].insert(0, {
        "japanese": japanese,
        "english":  english,
        "time":     time.strftime("%H:%M:%S"),
    })
    if len(state["history"]) > 8:
        state["history"].pop()

def query_vision_model(frame):
    """
    Send the cropped image to Ollama vision model.
    Returns (japanese, english) tuple or (None, None) if no dialogue found.
    """
    t0 = time.perf_counter()
    b64 = image_to_base64(frame)
    t1 = time.perf_counter()

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [b64],
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    t2 = time.perf_counter()

    response = r.json()["response"].strip()
    t3 = time.perf_counter()

    encode_ms = (t1 - t0) * 1000
    llm_ms    = (t2 - t1) * 1000
    parse_ms  = (t3 - t2) * 1000
    total_ms  = (t3 - t0) * 1000
    print(f"⏱️   encode={encode_ms:.0f}ms  llm={llm_ms:.0f}ms  parse={parse_ms:.0f}ms  total={total_ms:.0f}ms")
    state["last_timing"] = {
        "encode_ms": round(encode_ms),
        "llm_ms":    round(llm_ms),
        "parse_ms":  round(parse_ms),
        "total_ms":  round(total_ms),
    }

    print(f"🤖  Model raw: {response}")

    if response.strip().upper() == "NONE":
        return None, None

    japanese = None
    english  = None
    for line in response.splitlines():
        if line.lower().startswith("japanese:"):
            japanese = line.split(":", 1)[1].strip()
        elif line.lower().startswith("english:"):
            english = line.split(":", 1)[1].strip()

    return japanese, english

def capture_loop():
    print(f"📡  Connecting to {IP_WEBCAM_URL} ...")
    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print("❌  Cannot connect. Check IP_WEBCAM_URL and that IP Webcam app is running.")
        return
    print(f"✅  Connected. Using model: {MODEL}")
    print("    Open http://localhost:5002\n")

    prev_dlg           = None          # last dialogue frame for diff comparison
    stable_count       = 0             # consecutive stable frames so far
    active_count       = 0             # consecutive frames with a non-empty box
    last_japanese      = None          # last translated text — avoid re-translating same box
    frame_counter      = 0             # for FRAME_SKIP
    waiting_for_change = False         # True after firing — must see movement before re-arming

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Lost connection — retrying...")
            time.sleep(3)
            cap = cv2.VideoCapture(IP_WEBCAM_URL)
            prev_dlg = None
            stable_count = 0
            active_count = 0
            continue

        frame_counter += 1
        if frame_counter % FRAME_SKIP != 0:
            continue

        tv, dlg = crop_dialogue(frame)

        # Save debug images
        cv2.imwrite("ocr_preview.jpg", tv)
        cv2.imwrite("ocr_preview_box.jpg", dlg)

        if dlg.size == 0:
            print("⚠️   Dialogue crop is empty — check DLGBOX constants")
            prev_dlg = None
            stable_count = 0
            active_count = 0
            state["stable_progress"] = 0
            state["status_detail"] = "Crop empty"
            continue

        # ── Typewriter detection ──────────────────────────────────────────────
        if prev_dlg is None or prev_dlg.shape != dlg.shape:
            # First frame or size changed — reset
            prev_dlg     = dlg
            stable_count = 0
            active_count = 1
            state["stable_progress"] = 0
            state["status_detail"] = "Watching..."
            continue

        diff = frame_diff(prev_dlg, dlg)
        prev_dlg = dlg.copy()
        state["diff_value"] = round(float(diff), 2)

        is_stable = diff < DIFF_THRESHOLD
        active_count += 1

        # Update preview with current dialogue crop
        update_preview(dlg, state["state_machine"])

        if not is_stable:
            stable_count       = 0
            waiting_for_change = False
            last_japanese      = None   # screen changed — allow re-translation
            state["stable_progress"] = 0
            state["status_detail"]   = "Animating..."
            state["state_machine"]   = "ANIMATING"
            print(f"   diff={diff:.2f}  ✍️  animating  (re-armed)")
            continue

        if waiting_for_change:
            state["status_detail"]   = "Waiting for next dialogue..."
            state["stable_progress"] = 100
            state["state_machine"]   = "LOCKED"
            print(f"   diff={diff:.2f}  🔒 locked (already fired, waiting for change)")
            continue

        stable_count += 1
        progress = int(100 * min(stable_count, STABLE_FRAMES_NEEDED) / STABLE_FRAMES_NEEDED)
        state["stable_progress"] = progress
        state["state_machine"]   = "STABILISING"

        print(f"   diff={diff:.2f}  stable={stable_count}/{STABLE_FRAMES_NEEDED}  active={active_count}")

        # ── Fire LLM once stability threshold is reached ──────────────────────
        if stable_count >= STABLE_FRAMES_NEEDED and active_count >= MIN_ACTIVE_FRAMES:
            state["status_detail"]  = "Translating..."
            state["processing"]     = True
            state["stable_progress"] = 100
            state["state_machine"]  = "TRANSLATING"

            try:
                japanese, english = query_vision_model(dlg)
                state["llm_calls"] += 1

                if japanese:
                    if japanese == last_japanese:
                        # Same dialogue as last time — skip update, don't push history
                        print(f"♻️   Duplicate dialogue — skipping")
                        state["status_detail"] = "Duplicate — skipped"
                        state["state_machine"] = "LOCKED"
                    else:
                        print(f"📺  {japanese}")
                        print(f"✅  {english}\n")
                        state["japanese"]      = japanese
                        state["translation"]   = english or ""
                        state["status_detail"] = "Live"
                        state["state_machine"] = "LIVE"
                        push_history(japanese, english or "")
                        last_japanese = japanese
                else:
                    print("💤  No dialogue detected")
                    state["status_detail"] = "No dialogue"
                    state["state_machine"] = "IDLE"

            except Exception as e:
                print(f"❌  {e}")
                state["status_detail"] = f"Error: {e}"
                state["state_machine"] = "ERROR"
            finally:
                state["processing"] = False

            waiting_for_change = True
            stable_count       = 0

        else:
            state["status_detail"] = f"Stabilising ({stable_count}/{STABLE_FRAMES_NEEDED})"

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
    align-items: flex-start;
    justify-content: center;
    padding: 32px 20px;
    gap: 20px;
  }
  .card {
    background: #12121a; border: 1px solid #2a2a3a;
    border-radius: 16px; padding: 36px; width: 580px; flex-shrink: 0;
  }
  .japanese { font-size: 32px; color: #f0e6c8; line-height: 1.6; min-height: 50px; margin-bottom: 20px; }
  .divider  { border: none; border-top: 1px solid #2a2a3a; margin-bottom: 20px; }
  .translation { font-size: 20px; color: #e8d9a0; line-height: 1.5; min-height: 34px; margin-bottom: 28px; }
  .placeholder { color: #222230; }
  .badge-row { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .badge {
    padding: 3px 11px; border-radius: 20px; font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase;
    background: #1e1e2e; color: #555577; transition: all 0.2s;
  }
  .badge.ANIMATING   { background: #2a1500; color: #e06c00; }
  .badge.STABILISING { background: #1e1800; color: #d4a017; }
  .badge.LOCKED      { background: #0a1525; color: #4a90d9; }
  .badge.TRANSLATING { background: #1a0825; color: #c678dd; }
  .badge.LIVE        { background: #081a0a; color: #4caf50; }
  .badge.ERROR       { background: #250808; color: #e05555; }
  .status-text { font-size: 12px; color: #444460; }
  .meter-row { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
  .meter-label { font-size: 10px; color: #444460; text-transform: uppercase;
                 letter-spacing: 0.08em; width: 62px; flex-shrink: 0; }
  .meter-bg   { flex: 1; background: #1a1a28; border-radius: 4px; height: 5px; overflow: hidden; }
  .meter-fill { height: 100%; border-radius: 4px; transition: width 0.15s, background 0.15s; }
  .meter-val  { font-size: 10px; color: #444460; width: 34px; text-align: right;
                font-variant-numeric: tabular-nums; }
  .bottom-row { display: flex; justify-content: flex-end; margin-top: 8px; }
  .llm-counter { font-size: 11px; color: #444460; letter-spacing: 0.08em; text-transform: uppercase; }
  .llm-counter span { color: #a0a0cc; font-weight: 700; font-size: 13px; }
  .timing-row { margin-top: 16px; border-top: 1px solid #1a1a28; padding-top: 14px; display: none; }
  .timing-title { font-size: 10px; color: #444460; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 10px; }
  .t-bar-wrap { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
  .t-bar-wrap.total { margin-top: 6px; border-top: 1px solid #1a1a28; padding-top: 8px; }
  .t-label { font-size: 10px; color: #444460; width: 46px; flex-shrink: 0; }
  .t-bg    { flex: 1; background: #1a1a28; border-radius: 3px; height: 4px; overflow: hidden; }
  .t-fill  { height: 100%; border-radius: 3px; transition: width 0.4s; min-width: 2px; }
  .t-val   { font-size: 10px; color: #666688; width: 48px; text-align: right;
             font-variant-numeric: tabular-nums; }
  .side { display: flex; flex-direction: column; gap: 16px; width: 260px; }
  .preview-card { background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px; overflow: hidden; }
  .panel-title  { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
                  color: #444460; padding: 10px 12px 6px; }
  .preview-card img { width: 100%; display: block; }
  .history-card {
    background: #12121a; border: 1px solid #2a2a3a; border-radius: 12px;
    padding: 14px; overflow-y: auto; max-height: 500px;
  }
  .history-entry { margin-bottom: 12px; border-bottom: 1px solid #1a1a28; padding-bottom: 10px; }
  .history-entry:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
  .h-time { font-size: 9px; color: #2e2e46; margin-bottom: 3px; }
  .h-jp   { font-size: 14px; color: #c8b89a; margin-bottom: 2px; line-height: 1.4; }
  .h-en   { font-size: 12px; color: #887755; line-height: 1.4; }
  .no-history { color: #222230; font-size: 12px; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }
  .pulse { animation: pulse 0.9s infinite; }
</style>
</head>
<body>
<div class="card">
  <div class="japanese" id="japanese"><span class="placeholder">Waiting for dialogue...</span></div>
  <hr class="divider">
  <div class="translation" id="translation"><span class="placeholder">Translation will appear here</span></div>
  <div class="badge-row">
    <div class="badge" id="badge">IDLE</div>
    <span class="status-text" id="status-text"></span>
  </div>
  <div class="meter-row">
    <div class="meter-label">Pixel diff</div>
    <div class="meter-bg"><div class="meter-fill" id="diff-fill" style="width:0%;background:#555577"></div></div>
    <div class="meter-val" id="diff-val">0.0</div>
  </div>
  <div class="meter-row">
    <div class="meter-label">Stability</div>
    <div class="meter-bg"><div class="meter-fill" id="stab-fill" style="width:0%;background:#4caf50"></div></div>
    <div class="meter-val" id="stab-val">0%</div>
  </div>
  <div class="bottom-row">
    <div class="llm-counter">LLM calls &nbsp;<span id="llm-calls">0</span></div>
  </div>
  <div class="timing-row" id="timing-row">
    <div class="timing-title">Last call breakdown</div>
    <div class="t-bar-wrap">
      <div class="t-label">Encode</div>
      <div class="t-bg"><div class="t-fill" id="t-encode" style="background:#4a90d9"></div></div>
      <div class="t-val" id="tv-encode">—</div>
    </div>
    <div class="t-bar-wrap">
      <div class="t-label">LLM</div>
      <div class="t-bg"><div class="t-fill" id="t-llm" style="background:#c678dd"></div></div>
      <div class="t-val" id="tv-llm">—</div>
    </div>
    <div class="t-bar-wrap">
      <div class="t-label">Parse</div>
      <div class="t-bg"><div class="t-fill" id="t-parse" style="background:#4caf50"></div></div>
      <div class="t-val" id="tv-parse">—</div>
    </div>
    <div class="t-bar-wrap total">
      <div class="t-label">Total</div>
      <div class="t-bg"><div class="t-fill" id="t-total" style="background:#666688"></div></div>
      <div class="t-val" id="tv-total">—</div>
    </div>
  </div>
</div>
<div class="side">
  <div class="preview-card">
    <div class="panel-title">Live dialogue crop</div>
    <img src="/preview" alt="preview">
  </div>
  <div class="history-card">
    <div class="panel-title">History</div>
    <div id="history-list"><span class="no-history">No translations yet</span></div>
  </div>
</div>
<script>
const SM_COLORS = {
  IDLE: '#555577', ANIMATING: '#e06c00', STABILISING: '#d4a017',
  LOCKED: '#4a90d9', TRANSLATING: '#c678dd', LIVE: '#4caf50', ERROR: '#e05555',
};
const DIFF_MAX = 40;
function renderHistory(history) {
  const el = document.getElementById('history-list');
  if (!history || !history.length) {
    el.innerHTML = '<span class="no-history">No translations yet</span>';
    return;
  }
  el.innerHTML = history.map(h =>
    '<div class="history-entry">' +
      '<div class="h-time">' + h.time + '</div>' +
      '<div class="h-jp">'   + h.japanese + '</div>' +
      '<div class="h-en">'   + h.english  + '</div>' +
    '</div>'
  ).join('');
}
function updateTiming(t) {
  if (!t || t.total_ms === 0) return;
  document.getElementById('timing-row').style.display = 'block';
  const total = t.total_ms || 1;
  [['encode', t.encode_ms, '#4a90d9'],
   ['llm',    t.llm_ms,    '#c678dd'],
   ['parse',  t.parse_ms,  '#4caf50'],
   ['total',  t.total_ms,  '#666688']
  ].forEach(([id, ms, col]) => {
    document.getElementById('t-' + id).style.width = Math.min(100, (ms / total) * 100) + '%';
    document.getElementById('tv-' + id).textContent = ms + 'ms';
  });
}
async function poll() {
  try {
    const d = await (await fetch('/state')).json();
    const sm  = d.state_machine || 'IDLE';
    const col = SM_COLORS[sm] || '#555577';
    const badge = document.getElementById('badge');
    badge.textContent = sm;
    badge.className = 'badge ' + sm + (sm === 'TRANSLATING' ? ' pulse' : '');
    document.getElementById('status-text').textContent = d.status_detail || '';
    document.getElementById('japanese').innerHTML = d.japanese
      ? d.japanese : '<span class="placeholder">Waiting for dialogue...</span>';
    document.getElementById('translation').innerHTML = d.translation
      ? d.translation : '<span class="placeholder">Translation will appear here</span>';
    const diffPct = Math.min(100, ((d.diff_value || 0) / DIFF_MAX) * 100);
    const df = document.getElementById('diff-fill');
    df.style.width = diffPct + '%';
    df.style.background = col;
    document.getElementById('diff-val').textContent = (d.diff_value || 0).toFixed(1);
    const sf = document.getElementById('stab-fill');
    sf.style.width = (d.stable_progress || 0) + '%';
    sf.style.background = d.stable_progress >= 100 ? '#4caf50' : '#d4a017';
    document.getElementById('stab-val').textContent = (d.stable_progress || 0) + '%';
    document.getElementById('llm-calls').textContent = d.llm_calls || 0;
    updateTiming(d.last_timing);
    renderHistory(d.history);
  } catch(e) {}
  setTimeout(poll, 400);
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

if __name__ == '__main__':
    print("🎮  Zelda Vision Translator")
    print(f"📱  {IP_WEBCAM_URL}")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5002, debug=False)
