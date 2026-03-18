"""
v8_vision_llm_learning_companion_memory.py  (was: zelda_learn.py)
=================================================================
ITERATION 8 — Final form: vision LLM handles both OCR and lesson generation
in a single call, with persistent cross-session memory and a full learning UI.

What it does:
  - OCR + Lesson: Qwen2.5-VL (vision model) receives the cropped dialogue box
    image directly. No Tesseract, no separate OCR step. Two prompt modes:
      TRANSLATE — structured key:value response (JP, romaji, EN, response options)
      LEARN     — structured JSON lesson (words, kanji, grammar patterns, tip,
                  new vocabulary to commit to memory)
  - Memory (zelda_memory.json): vocabulary and grammar patterns accumulated across
    sessions. Injected into the LEARN prompt so the model knows what you've seen.
    Atomic safe-write (write-to-temp + os.replace) to prevent corruption on kill.
  - Stability detection: STABLE_FRAMES_REQUIRED consecutive frames with identical
    perceptual hash before firing the LLM — avoids mid-typewriter translations.
  - Empty frame detection: skips frames where < 8% pixels are dark (no dialogue box).
  - Acknowledge flow: lesson display pauses polling (waiting_ack = True) until user
    presses Space/Enter; ack_event resumes the capture loop.
  - Session stats: unique words seen, total dialogues, grammar patterns this session.

Key improvements over v7:
  - Vision model eliminates OCR as a separate failure point — reads dialogue
    directly from the image, handles fonts and furigana natively
  - Single model call does OCR + translation + lesson generation together
  - Cleaner memory model (vocabulary dict + grammar_patterns dict, no JLPT scoring)
  - Response option buttons (Yes/No/etc.) parsed and displayed alongside dialogue
  - LLM timing shown in UI (last call duration in seconds)
  - Mode switchable at runtime via /mode POST endpoint; synced to UI toggle

Run:  python3 v8_vision_llm_learning_companion_memory.py
Open: http://localhost:5002
Press SPACE or Enter in the browser to acknowledge a lesson and resume polling.
"""

import cv2
import requests
import time
import threading
import base64
import json
import os
import hashlib
import tempfile
from flask import Flask, render_template_string, jsonify, request as flask_request

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
MODEL           = "qwen2.5vl:7b"
VIDEO_SOURCE   = "http://192.168.1.107:8080/video"
MEMORY_FILE     = "zelda_memory.json"

# How many consecutive stable frames before text is considered "settled"
# At ~1 frame/sec polling this = ~3 seconds of no change
STABLE_FRAMES_REQUIRED = 3

# Minimum number of Japanese characters required to consider it real dialogue
# (filters out menus, HUD labels, single-word UI chrome)
MIN_JP_CHARS = 6

# ── Crop tuning ───────────────────────────────────────────────────────────────
TV_TOP    = 0.15
TV_BOTTOM = 0.80
TV_LEFT   = 0.05
TV_RIGHT  = 0.95

# Full width to capture response option buttons on the right
DLGBOX_TOP    = 0.62
DLGBOX_BOTTOM = 1.00
DLGBOX_LEFT   = 0.00
DLGBOX_RIGHT  = 1.00
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Shared state (written by capture thread, read by Flask) ───────────────────
state = {
    "mode":           "translate",   # "translate" | "learn"
    "japanese":       "",
    "romaji":         "",
    "translation":    "",
    "response_options": [],          # list of {jp, romaji, en}
    "lesson":         None,          # dict or None
    "processing":     False,
    "waiting_ack":    False,         # lesson on screen, polling paused
    "status":         "Idle",
    "last_llm_ms":    None,          # how long the last LLM call took
    "session": {
        "unique_words":    0,
        "total_dialogues": 0,
        "patterns_seen":   0,
    }
}
state_lock = threading.Lock()
ack_event  = threading.Event()   # set when user presses key to continue

# ── Memory helpers ────────────────────────────────────────────────────────────

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"vocabulary": {}, "grammar_patterns": {}, "total_sessions": 0}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        print("⚠️  Memory file corrupt — starting fresh")
        return {"vocabulary": {}, "grammar_patterns": {}, "total_sessions": 0}

def save_memory(memory):
    """Safe write: write to temp file then rename to avoid corruption on kill."""
    dir_ = os.path.dirname(os.path.abspath(MEMORY_FILE))
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False,
                                         suffix=".tmp", encoding="utf-8") as tf:
            json.dump(memory, tf, ensure_ascii=False, indent=2)
            tmp_path = tf.name
        os.replace(tmp_path, MEMORY_FILE)
    except Exception as e:
        print(f"⚠️  Failed to save memory: {e}")

# ── Image helpers ─────────────────────────────────────────────────────────────

def crop_frame(frame):
    h, w = frame.shape[:2]
    tv = frame[int(h * TV_TOP):int(h * TV_BOTTOM),
               int(w * TV_LEFT):int(w * TV_RIGHT)]
    th, tw = tv.shape[:2]
    dlg = tv[int(th * DLGBOX_TOP) :int(th * DLGBOX_BOTTOM),
             int(tw * DLGBOX_LEFT):int(tw * DLGBOX_RIGHT)]
    return tv, dlg

def frame_hash(frame):
    """Fast perceptual hash — resize to thumbnail and md5."""
    small = cv2.resize(frame, (64, 16))
    return hashlib.md5(small.tobytes()).hexdigest()

def frame_is_empty(frame):
    """True if the crop is mostly bright (no dark dialogue box present)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark_pixel_ratio = (gray < 80).sum() / gray.size
    return dark_pixel_ratio < 0.08   # less than 8% dark pixels → no box

def to_base64(frame):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf).decode("utf-8")

# ── LLM calls ─────────────────────────────────────────────────────────────────

TRANSLATE_PROMPT = """This is a screenshot of a dialogue box from a Japanese video game (The Legend of Zelda).

Your job:
1. Find the main dialogue text in the large dark rounded box on the left/centre.
2. Find any response option buttons (small pill-shaped boxes, usually on the right side).
3. Translate everything.

Reply in EXACTLY this format and nothing else:

DIALOGUE_JP: <full japanese dialogue text>
DIALOGUE_ROMAJI: <romaji reading>
DIALOGUE_EN: <english translation>
RESPONSE_COUNT: <number of response options, 0 if none>
RESPONSE_1_JP: <japanese>
RESPONSE_1_ROMAJI: <romaji>
RESPONSE_1_EN: <english>
RESPONSE_2_JP: <japanese>
RESPONSE_2_ROMAJI: <romaji>
RESPONSE_2_EN: <english>

Only include RESPONSE lines that exist. If there is no dialogue box visible, reply with exactly: NONE
This must be a real dialogue box with sentence-length Japanese text, not a menu, HUD, or single item label."""

def build_learn_prompt(memory):
    # Summarise known vocabulary for the model so it can flag repeats
    known = memory.get("vocabulary", {})
    known_summary = ""
    if known:
        top = sorted(known.items(), key=lambda x: x[1]["count"], reverse=True)[:30]
        known_summary = "Known vocabulary (word: times seen):\n" + \
            "\n".join(f"  {w}: {d['count']}x" for w, d in top)

    known_patterns = memory.get("grammar_patterns", {})
    pattern_summary = ""
    if known_patterns:
        pattern_summary = "Known grammar patterns:\n" + \
            "\n".join(f"  {p}: {d['count']}x" for p, d in list(known_patterns.items())[:10])

    return f"""This is a screenshot of a dialogue box from a Japanese video game (The Legend of Zelda).

{known_summary}
{pattern_summary}

Your job:
1. Find the main dialogue text and any response option buttons.
2. Create a structured Japanese language lesson for a beginner/intermediate learner.

Reply in EXACTLY this JSON format and nothing else (no markdown, no backticks):

{{
  "dialogue_jp": "full japanese text",
  "dialogue_romaji": "full romaji",
  "dialogue_en": "english translation",
  "response_options": [
    {{"jp": "...", "romaji": "...", "en": "..."}}
  ],
  "words": [
    {{
      "jp": "word",
      "romaji": "romaji",
      "en": "meaning",
      "type": "noun/verb/particle/etc",
      "seen_before": true/false,
      "times_seen": 0
    }}
  ],
  "kanji": [
    {{
      "character": "珍",
      "reading": "めずら / chin",
      "meaning": "rare, unusual",
      "seen_before": true/false
    }}
  ],
  "grammar_patterns": [
    {{
      "pattern": "〜とは",
      "explanation": "used to express surprise or define something",
      "example": "人がいるとは珍しいのう",
      "seen_before": true/false
    }}
  ],
  "tip": "one memorable insight about this phrase or cultural context",
  "new_vocabulary": ["list", "of", "new", "words", "to", "add", "to", "memory"],
  "new_patterns": ["list", "of", "new", "grammar", "patterns"]
}}

If no dialogue box is visible or the text is not sentence-length Japanese dialogue, return exactly: NONE"""

def parse_translate_response(text):
    """Parse the key:value translation response."""
    if text.strip().upper() == "NONE":
        return None
    lines = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
             for l in text.splitlines() if ":" in l}
    jp = lines.get("DIALOGUE_JP", "")
    if not jp or len([c for c in jp if '\u3040' <= c <= '\u9fff']) < MIN_JP_CHARS:
        return None
    options = []
    for i in range(1, 5):
        if f"RESPONSE_{i}_JP" in lines:
            options.append({
                "jp":     lines.get(f"RESPONSE_{i}_JP", ""),
                "romaji": lines.get(f"RESPONSE_{i}_ROMAJI", ""),
                "en":     lines.get(f"RESPONSE_{i}_EN", ""),
            })
    return {
        "japanese":        jp,
        "romaji":          lines.get("DIALOGUE_ROMAJI", ""),
        "translation":     lines.get("DIALOGUE_EN", ""),
        "response_options": options,
    }

def parse_learn_response(text):
    """Parse the JSON lesson response."""
    text = text.strip()
    if text.upper() == "NONE":
        return None
    # Strip accidental markdown fences
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        jp = data.get("dialogue_jp", "")
        if not jp or len([c for c in jp if '\u3040' <= c <= '\u9fff']) < MIN_JP_CHARS:
            return None
        return data
    except Exception as e:
        print(f"⚠️  Failed to parse lesson JSON: {e}\nRaw: {text[:300]}")
        return None

def call_ollama(prompt, image_b64):
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json={
        "model":  MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }, timeout=90)
    r.raise_for_status()
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"⏱️   LLM call took {elapsed_ms}ms ({elapsed_ms/1000:.1f}s)")
    with state_lock:
        state["last_llm_ms"] = elapsed_ms
    return r.json()["response"].strip()

def update_memory(memory, lesson):
    """Merge new words and patterns from lesson into memory."""
    vocab = memory.setdefault("vocabulary", {})
    for word in lesson.get("words", []):
        jp = word.get("jp", "")
        if not jp:
            continue
        if jp not in vocab:
            vocab[jp] = {"en": word.get("en", ""), "count": 0, "romaji": word.get("romaji", "")}
        vocab[jp]["count"] += 1

    patterns = memory.setdefault("grammar_patterns", {})
    for p in lesson.get("grammar_patterns", []):
        key = p.get("pattern", "")
        if not key:
            continue
        if key not in patterns:
            patterns[key] = {"explanation": p.get("explanation", ""), "count": 0}
        patterns[key]["count"] += 1

    return memory

# ── Capture loop ──────────────────────────────────────────────────────────────

def capture_loop():
    print(f"📡  Connecting to {VIDEO_SOURCE} ...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("❌  Cannot connect.")
        return
    print(f"✅  Connected. Model: {MODEL}")
    print("    Open http://localhost:5002\n")

    memory = load_memory()
    memory["total_sessions"] = memory.get("total_sessions", 0) + 1
    save_memory(memory)

    stable_count   = 0
    last_hash      = None
    last_taught    = None   # hash of last dialogue we generated a lesson for
    llm_busy       = False  # block duplicate requests

    while True:
        # ── Paused waiting for user acknowledgement ──
        with state_lock:
            waiting = state["waiting_ack"]
        if waiting:
            ack_event.wait()        # block until SPACE pressed
            ack_event.clear()
            with state_lock:
                state["waiting_ack"] = False
                state["status"]      = "Polling..."
            time.sleep(0.5)
            continue

        ret, frame = cap.read()
        if not ret:
            print("⚠️  Lost connection — retrying...")
            time.sleep(3)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            stable_count = 0
            last_hash    = None
            continue

        tv, dlg = crop_frame(frame)
        cv2.imwrite("ocr_preview.jpg", tv)
        cv2.imwrite("ocr_preview_box.jpg", dlg)

        if dlg.size == 0:
            time.sleep(1)
            continue

        # ── Ignore frames where dialogue box has disappeared (between lines) ──
        if frame_is_empty(dlg):
            stable_count = 0
            with state_lock:
                state["status"] = "Watching..."
            time.sleep(1)
            continue

        current_hash = frame_hash(dlg)

        # ── Track stability ──
        if current_hash == last_hash:
            stable_count += 1
        else:
            stable_count = 1
            last_hash    = current_hash

        with state_lock:
            mode = state["mode"]

        settled = stable_count >= STABLE_FRAMES_REQUIRED
        already_taught = (current_hash == last_taught)

        if not settled or already_taught or llm_busy:
            with state_lock:
                if not settled:
                    state["status"] = f"Waiting for text to settle... ({stable_count}/{STABLE_FRAMES_REQUIRED})"
                elif already_taught:
                    state["status"] = "Lesson complete — move dialogue to continue"
            time.sleep(1)
            continue

        # ── Text is new and settled — call the LLM ──
        llm_busy = True
        with state_lock:
            state["processing"] = True
            state["status"]     = "Reading dialogue..."

        try:
            b64 = to_base64(dlg)
            memory = load_memory()

            if mode == "translate":
                raw = call_ollama(TRANSLATE_PROMPT, b64)
                print(f"🤖  Raw: {raw[:200]}")
                result = parse_translate_response(raw)
                if result:
                    with state_lock:
                        state["japanese"]        = result["japanese"]
                        state["romaji"]          = result["romaji"]
                        state["translation"]     = result["translation"]
                        state["response_options"]= result["response_options"]
                        state["lesson"]          = None
                        state["status"]          = "Translated"
                        state["session"]["total_dialogues"] += 1
                    last_taught = current_hash
                    print(f"📺  {result['japanese']}")
                    print(f"✅  {result['translation']}\n")
                else:
                    print("💤  No valid dialogue detected")

            else:  # learn mode
                with state_lock:
                    state["status"] = "Generating lesson..."
                raw = call_ollama(build_learn_prompt(memory), b64)
                print(f"🤖  Raw: {raw[:200]}")
                lesson = parse_learn_response(raw)
                if lesson:
                    memory = update_memory(memory, lesson)
                    save_memory(memory)

                    vocab_count = len(memory["vocabulary"])
                    pattern_count = len(memory["grammar_patterns"])

                    with state_lock:
                        state["japanese"]         = lesson.get("dialogue_jp", "")
                        state["romaji"]           = lesson.get("dialogue_romaji", "")
                        state["translation"]      = lesson.get("dialogue_en", "")
                        state["response_options"] = lesson.get("response_options", [])
                        state["lesson"]           = lesson
                        state["waiting_ack"]      = True
                        state["status"]           = "Lesson ready — press Space to continue"
                        state["session"]["unique_words"]    = vocab_count
                        state["session"]["total_dialogues"] += 1
                        state["session"]["patterns_seen"]   = pattern_count
                    last_taught = current_hash
                    print(f"📚  Lesson generated: {lesson.get('dialogue_jp', '')}\n")
                else:
                    print("💤  No valid dialogue detected")

        except Exception as e:
            print(f"❌  LLM error: {e}")
            with state_lock:
                state["status"] = f"Error: {e}"
        finally:
            llm_busy = False
            with state_lock:
                state["processing"] = False

        time.sleep(1)

    cap.release()

# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def get_state():
    with state_lock:
        return jsonify(state)

@app.route('/ack', methods=['POST'])
def acknowledge():
    ack_event.set()
    return jsonify({"ok": True})

@app.route('/mode', methods=['POST'])
def set_mode():
    data = flask_request.get_json()
    with state_lock:
        state["mode"] = data.get("mode", "translate")
    return jsonify({"mode": state["mode"]})

# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Zelda 日本語</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@400;600&family=Syne:wght@400;700;800&family=Noto+Sans+JP&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080b10;
    --surface:  #0e1219;
    --border:   #1e2535;
    --gold:     #c9a84c;
    --gold-dim: #7a6230;
    --text:     #e8dfc8;
    --muted:    #4a5068;
    --green:    #4a9e6b;
    --amber:    #d48b2a;
    --red:      #c45c5c;
    --learn-accent: #5b7fa6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 20px 60px;
  }

  /* Subtle grain overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0;
  }

  .wrap { position: relative; z-index: 1; width: 100%; max-width: 820px; }

  /* ── Header ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 28px;
  }
  .logo {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold);
  }
  .logo span { color: var(--muted); font-weight: 400; }

  /* ── Mode toggle ── */
  .toggle {
    display: flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 4px;
    gap: 4px;
  }
  .toggle button {
    border: none; cursor: pointer;
    padding: 7px 20px;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    transition: all 0.2s;
    background: transparent;
    color: var(--muted);
  }
  .toggle button.active {
    background: var(--gold);
    color: #080b10;
  }

  /* ── Status bar ── */
  .statusbar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--muted);
    flex-shrink: 0;
  }
  .dot.live     { background: var(--green); }
  .dot.thinking { background: var(--amber); animation: pulse 0.8s infinite; }
  .dot.waiting  { background: var(--learn-accent); animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

  /* ── Main card ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 16px;
  }

  .jp-text {
    font-family: 'Noto Serif JP', serif;
    font-size: 34px;
    line-height: 1.7;
    color: #f5edd8;
    margin-bottom: 6px;
    min-height: 48px;
  }
  .romaji {
    font-size: 14px;
    color: var(--gold-dim);
    margin-bottom: 18px;
    font-style: italic;
    min-height: 20px;
  }
  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
  }
  .en-text {
    font-size: 22px;
    line-height: 1.55;
    color: var(--text);
    min-height: 34px;
  }
  .placeholder { color: var(--muted); font-style: italic; }

  /* ── Response options ── */
  .responses-label {
    margin-top: 28px;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }
  .responses {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .resp-item {
    background: #12151e;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 16px;
    display: flex;
    gap: 16px;
    align-items: baseline;
  }
  .resp-jp   { font-family: 'Noto Serif JP', serif; font-size: 16px; color: #f5edd8; }
  .resp-rom  { font-size: 12px; color: var(--gold-dim); font-style: italic; }
  .resp-en   { font-size: 14px; color: var(--muted); margin-left: auto; }

  /* ── Lesson sections ── */
  .lesson { margin-top: 28px; display: flex; flex-direction: column; gap: 20px; }

  .section-title {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--learn-accent);
    margin-bottom: 10px;
  }

  /* Words table */
  .words { display: flex; flex-direction: column; gap: 6px; }
  .word-row {
    display: grid;
    grid-template-columns: 90px 110px 1fr 70px;
    gap: 8px;
    align-items: center;
    padding: 8px 12px;
    background: #0c1018;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .word-jp   { font-family: 'Noto Serif JP', serif; font-size: 17px; }
  .word-rom  { font-size: 12px; color: var(--gold-dim); font-style: italic; }
  .word-en   { font-size: 13px; color: var(--muted); }
  .word-type {
    font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
    color: #2a3550; background: #141926; border-radius: 4px;
    padding: 2px 6px; text-align: center;
  }
  .seen-badge {
    font-size: 10px; color: var(--amber); margin-left: 6px;
  }

  /* Kanji grid */
  .kanji-grid { display: flex; flex-wrap: wrap; gap: 10px; }
  .kanji-card {
    background: #0c1018;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    min-width: 90px;
  }
  .kanji-char { font-family: 'Noto Serif JP', serif; font-size: 36px; line-height: 1; margin-bottom: 6px; }
  .kanji-read { font-size: 11px; color: var(--gold-dim); margin-bottom: 4px; font-style: italic; }
  .kanji-mean { font-size: 12px; color: var(--muted); }

  /* Grammar patterns */
  .pattern-card {
    background: #0c1018;
    border: 1px solid var(--border);
    border-left: 3px solid var(--learn-accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .pattern-key  { font-family: 'Noto Serif JP', serif; font-size: 17px; margin-bottom: 4px; }
  .pattern-exp  { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  .pattern-ex   { font-size: 13px; color: var(--gold-dim); font-style: italic; }

  /* Tip */
  .tip-card {
    background: #0c1018;
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.6;
    color: var(--text);
  }

  /* ── Stats bar ── */
  .stats {
    display: flex;
    gap: 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 16px;
  }
  .stat { flex: 1; text-align: center; }
  .stat-val {
    font-size: 26px;
    font-weight: 800;
    color: var(--gold);
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
  }

  /* ── Ack hint ── */
  .ack-hint {
    text-align: center;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 8px;
    animation: fadeIn 0.6s ease;
  }
  .ack-hint kbd {
    display: inline-block;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: monospace;
    font-size: 12px;
    color: var(--text);
  }

  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }
  .card { animation: fadeIn 0.3s ease; }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo">Zelda <span>日本語</span></div>
    <div class="toggle">
      <button id="btn-translate" class="active" onclick="setMode('translate')">Translate</button>
      <button id="btn-learn"                    onclick="setMode('learn')">Learn</button>
    </div>
  </header>

  <div class="stats">
    <div class="stat"><div class="stat-val" id="s-words">0</div><div class="stat-label">Words Seen</div></div>
    <div class="stat"><div class="stat-val" id="s-dial">0</div><div class="stat-label">Dialogues</div></div>
    <div class="stat"><div class="stat-val" id="s-pat">0</div><div class="stat-label">Patterns</div></div>
  </div>

  <div class="statusbar">
    <div class="dot" id="dot"></div>
    <span id="status-text">Idle</span>
    <span id="llm-time" style="margin-left:auto;display:none"></span>
  </div>

  <div class="card" id="main-card">
    <div class="jp-text"  id="jp-text"><span class="placeholder">Waiting for dialogue...</span></div>
    <div class="romaji"   id="romaji"></div>
    <hr class="divider">
    <div class="en-text"  id="en-text"><span class="placeholder">Translation will appear here</span></div>
    <div id="responses-wrap" style="display:none">
      <div class="responses-label">Response Options</div>
      <div class="responses" id="responses"></div>
    </div>
    <div class="lesson" id="lesson" style="display:none"></div>
  </div>

  <div class="ack-hint" id="ack-hint" style="display:none">
    Press <kbd>Space</kbd> or <kbd>Enter</kbd> to continue
  </div>

</div>

<script>
let currentMode = 'translate';

function setMode(m) {
  currentMode = m;
  document.getElementById('btn-translate').classList.toggle('active', m === 'translate');
  document.getElementById('btn-learn').classList.toggle('active', m === 'learn');
  fetch('/mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode: m})});
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' || e.code === 'Enter') {
    e.preventDefault();
    fetch('/ack', {method:'POST'});
    document.getElementById('ack-hint').style.display = 'none';
  }
});

function renderResponses(options) {
  const wrap = document.getElementById('responses-wrap');
  const cont = document.getElementById('responses');
  if (!options || options.length === 0) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  cont.innerHTML = options.map(o => `
    <div class="resp-item">
      <span class="resp-jp">${o.jp}</span>
      <span class="resp-rom">${o.romaji}</span>
      <span class="resp-en">${o.en}</span>
    </div>`).join('');
}

function renderLesson(lesson) {
  const el = document.getElementById('lesson');
  if (!lesson) { el.style.display = 'none'; return; }
  el.style.display = 'flex';

  let html = '';

  // Words
  if (lesson.words && lesson.words.length) {
    html += `<div>
      <div class="section-title">Words</div>
      <div class="words">`;
    for (const w of lesson.words) {
      const badge = w.seen_before ? `<span class="seen-badge">★ ${w.times_seen}x</span>` : '';
      html += `<div class="word-row">
        <span class="word-jp">${w.jp}${badge}</span>
        <span class="word-rom">${w.romaji}</span>
        <span class="word-en">${w.en}</span>
        <span class="word-type">${w.type}</span>
      </div>`;
    }
    html += `</div></div>`;
  }

  // Kanji
  if (lesson.kanji && lesson.kanji.length) {
    html += `<div>
      <div class="section-title">Kanji</div>
      <div class="kanji-grid">`;
    for (const k of lesson.kanji) {
      const seen = k.seen_before ? '★' : '';
      html += `<div class="kanji-card">
        <div class="kanji-char">${k.character}</div>
        <div class="kanji-read">${k.reading}</div>
        <div class="kanji-mean">${k.meaning} ${seen}</div>
      </div>`;
    }
    html += `</div></div>`;
  }

  // Grammar
  if (lesson.grammar_patterns && lesson.grammar_patterns.length) {
    html += `<div><div class="section-title">Grammar</div>`;
    for (const p of lesson.grammar_patterns) {
      const seen = p.seen_before ? ' <span class="seen-badge">★ seen before</span>' : '';
      html += `<div class="pattern-card">
        <div class="pattern-key">${p.pattern}${seen}</div>
        <div class="pattern-exp">${p.explanation}</div>
        <div class="pattern-ex">${p.example}</div>
      </div>`;
    }
    html += `</div>`;
  }

  // Tip
  if (lesson.tip) {
    html += `<div>
      <div class="section-title">Tip</div>
      <div class="tip-card">${lesson.tip}</div>
    </div>`;
  }

  el.innerHTML = html;
}

async function poll() {
  try {
    const d = await (await fetch('/state')).json();

    // Mode sync
    if (d.mode !== currentMode) {
      currentMode = d.mode;
      document.getElementById('btn-translate').classList.toggle('active', d.mode === 'translate');
      document.getElementById('btn-learn').classList.toggle('active', d.mode === 'learn');
    }

    // Dot + status
    const dot = document.getElementById('dot');
    dot.className = 'dot' + (d.processing ? ' thinking' : d.waiting_ack ? ' waiting' : d.japanese ? ' live' : '');
    document.getElementById('status-text').textContent = d.status || 'Idle';

    // LLM timing
    const llmEl = document.getElementById('llm-time');
    if (d.last_llm_ms != null) {
      llmEl.style.display = 'inline';
      llmEl.textContent = `last LLM: ${(d.last_llm_ms/1000).toFixed(1)}s`;
    }

    // Stats
    document.getElementById('s-words').textContent = d.session.unique_words;
    document.getElementById('s-dial').textContent  = d.session.total_dialogues;
    document.getElementById('s-pat').textContent   = d.session.patterns_seen;

    // Main text
    if (d.japanese) {
      document.getElementById('jp-text').textContent = d.japanese;
      document.getElementById('romaji').textContent  = d.romaji || '';
      document.getElementById('en-text').textContent = d.translation || '';
    } else {
      document.getElementById('jp-text').innerHTML = '<span class="placeholder">Waiting for dialogue...</span>';
      document.getElementById('romaji').textContent = '';
      document.getElementById('en-text').innerHTML  = '<span class="placeholder">Translation will appear here</span>';
    }

    renderResponses(d.response_options);
    renderLesson(d.lesson);

    // Ack hint
    document.getElementById('ack-hint').style.display = d.waiting_ack ? 'block' : 'none';

  } catch(e) {}
  setTimeout(poll, 1500);
}
poll();
</script>
</body>
</html>"""

if __name__ == '__main__':
    print("🎮  Zelda 日本語 — Vision Learning Companion")
    print(f"📱  {VIDEO_SOURCE}")
    print(f"🧠  Memory: {MEMORY_FILE}")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5002, debug=False)
