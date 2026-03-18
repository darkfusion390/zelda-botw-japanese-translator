"""
v6_apple_vision_ocr_vocab_tracking.py  (was: zelda_translator.py)
=================================================================
ITERATION 6 — Replace Manga OCR with Apple Vision (macOS neural engine)
and introduce vocabulary familiarity tracking for the first time.

What it does:
  Step 1 — OCR: Apple Vision framework (VNRecognizeTextRequest) running on
    macOS M1+ neural engine. Fast, accurate on printed Japanese, no model
    download required. Requires pyobjc-framework-Vision + Quartz.
  Step 2 — LLM: qwen2.5:7b via Ollama. Two prompt modes:
    TRANSLATE — returns romaji + translation only (fast)
    LEARN     — returns vocab breakdown, grammar pattern, dialect notes,
                progress summary (slower, more educational)

Vocab tracking (vocab.json):
  Every word/kanji/grammar pattern seen is stored with a familiarity level:
    new      (0 seen)    → blue underline in UI
    learning (1–9 seen)  → yellow underline
    familiar (10+ seen)  → green underline
  Known words are injected into the LEARN prompt so the LLM can personalise
  the lesson (skip words you already know, highlight new ones).

Key improvements over v4/v5:
  - macOS Apple Vision OCR — fastest and most accurate OCR tested so far
  - First version with dual translate/learn UI modes (toggle in header)
  - First version with persistent vocabulary memory across sessions
  - Word familiarity highlighted inline in the Japanese text
  - CSV logging, MJPEG preview, bounds detection all retained

Install deps:
  pip install pyobjc-framework-Vision pyobjc-framework-Quartz

Run:  python3 v6_apple_vision_ocr_vocab_tracking.py
Open: http://localhost:5002
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
from datetime import date
from flask import Flask, render_template_string, jsonify, Response, request as flask_request

import Vision
import Quartz

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434/api/generate"
TRANSLATION_MODEL = "qwen2.5:7b"
VIDEO_SOURCE     = "http://192.168.1.107:8080/video"

LOG_FILE     = "pixel_llm_log.csv"
VOCAB_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.json")
PREVIEW_PATH = os.path.expanduser("~/Downloads/preprocessed_crop.jpg")
BOUNDS_FILE  = "bounds.json"

# ── Brightness gate ───────────────────────────────────────────────────────────
BRIGHTNESS_GATE_HIGH = 80.0
BRIGHTNESS_GATE_LOW  = 10.0
BRIGHTNESS_ENABLED   = False

# ── Prompts ───────────────────────────────────────────────────────────────────
TRANSLATE_PROMPT = """Translate this Japanese video game dialogue to natural English.
Respond ONLY with valid JSON, no markdown, no extra text:
{{"romaji": "...", "translation": "..."}}

Japanese: {japanese}"""

LEARN_PROMPT = """You are a Japanese tutor helping a complete beginner learn Japanese through a video game.
Analyze this dialogue line and respond ONLY with valid JSON, no markdown, no extra text.

Japanese: {japanese}
Words the student has already seen (with seen count): {known_words}

Respond with this exact JSON structure:
{{
  "romaji": "full romaji reading of the sentence",
  "translation": "natural English translation",
  "grammar_note": "one key grammar pattern explained simply for a beginner",
  "dialect_note": "any dialect, archaic, or casual speech notes (empty string if none)",
  "progress_note": "brief encouraging note about their progress based on known_words data",
  "vocab": [
    {{"word": "japanese word", "reading": "romaji", "meaning": "english meaning", "is_new": true}},
    ...
  ],
  "kanji": [
    {{"kanji": "single kanji", "reading": "reading", "meaning": "meaning", "is_new": true}},
    ...
  ],
  "grammar_patterns": [
    {{"pattern": "pattern like 〜かのう", "meaning": "explanation", "is_new": true}},
    ...
  ]
}}"""

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "mode":               "TRANSLATE",   # TRANSLATE or LEARN
    "phase":              "TRANSLATING",
    "status":             "Starting up...",
    "japanese":           "",
    "romaji":             "",
    "translation":        "",
    "lesson":             None,          # full lesson object in LEARN mode
    "llm_calls":          0,
    "translate_calls":    0,
    "learn_calls":        0,
    "translate_ms":       0,
    "learn_ms":           0,
    "bounds":             None,
    "history":            [],
    "ocr_timing":         {"ocr_ms": 0},
    "translation_timing": {"llm_ms": 0, "total_ms": 0},
    "brightness":         0.0,
    "error":              "",
    "vocab_stats":        {"total_words": 0, "total_kanji": 0, "total_patterns": 0, "new_today": 0},
}

latest_frame_jpg = None
frame_lock        = threading.Lock()
app = Flask(__name__)

# ── Vocab manager ─────────────────────────────────────────────────────────────

def load_vocab():
    try:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "words":    {},
            "kanji":    {},
            "grammar":  {},
            "stats":    {"total_lines": 0, "new_today": 0, "last_session": ""},
        }

def save_vocab(vocab):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def get_familiarity(times_seen):
    if times_seen == 0:
        return "new"
    elif times_seen < 10:
        return "learning"
    else:
        return "familiar"

def build_known_words_context(vocab):
    """Build a compact context string for the LLM showing what's been seen."""
    known = {}
    for w, d in vocab["words"].items():
        known[w] = d.get("times_seen", 0)
    # Only send top 30 most seen words to keep prompt size manageable
    top = sorted(known.items(), key=lambda x: x[1], reverse=True)[:30]
    return ", ".join(f"{w}({c})" for w, c in top) if top else "none yet"

def update_vocab(vocab, lesson):
    """Update vocab.json with new words/kanji/patterns from a lesson."""
    today = str(date.today())
    if vocab["stats"].get("last_session") != today:
        vocab["stats"]["new_today"] = 0
        vocab["stats"]["last_session"] = today

    for item in lesson.get("vocab", []):
        w = item.get("word", "")
        if not w:
            continue
        if w not in vocab["words"]:
            vocab["words"][w] = {
                "reading":    item.get("reading", ""),
                "meaning":    item.get("meaning", ""),
                "times_seen": 0,
                "first_seen": today,
                "last_seen":  today,
            }
            vocab["stats"]["new_today"] = vocab["stats"].get("new_today", 0) + 1
        vocab["words"][w]["times_seen"] += 1
        vocab["words"][w]["last_seen"] = today

    for item in lesson.get("kanji", []):
        k = item.get("kanji", "")
        if not k:
            continue
        if k not in vocab["kanji"]:
            vocab["kanji"][k] = {
                "reading":    item.get("reading", ""),
                "meaning":    item.get("meaning", ""),
                "times_seen": 0,
                "first_seen": today,
            }
            vocab["stats"]["new_today"] = vocab["stats"].get("new_today", 0) + 1
        vocab["kanji"][k]["times_seen"] += 1

    for item in lesson.get("grammar_patterns", []):
        p = item.get("pattern", "")
        if not p:
            continue
        if p not in vocab["grammar"]:
            vocab["grammar"][p] = {
                "meaning":    item.get("meaning", ""),
                "times_seen": 0,
                "first_seen": today,
            }
        vocab["grammar"][p]["times_seen"] += 1

    vocab["stats"]["total_lines"] = vocab["stats"].get("total_lines", 0) + 1
    save_vocab(vocab)

    # Update state stats
    state["vocab_stats"] = {
        "total_words":    len(vocab["words"]),
        "total_kanji":    len(vocab["kanji"]),
        "total_patterns": len(vocab["grammar"]),
        "new_today":      vocab["stats"].get("new_today", 0),
    }

def annotate_japanese(japanese, vocab):
    """Return list of {char_or_word, familiarity} for rendering in UI."""
    # Simple character-level annotation for kanji, word-level for known words
    annotated = []
    i = 0
    text = japanese
    words_sorted = sorted(vocab["words"].keys(), key=len, reverse=True)
    while i < len(text):
        matched = False
        for w in words_sorted:
            if text[i:i+len(w)] == w:
                times = vocab["words"][w].get("times_seen", 0)
                annotated.append({"text": w, "familiarity": get_familiarity(times)})
                i += len(w)
                matched = True
                break
        if not matched:
            c = text[i]
            if c in vocab["kanji"]:
                times = vocab["kanji"][c].get("times_seen", 0)
                annotated.append({"text": c, "familiarity": get_familiarity(times)})
            else:
                annotated.append({"text": c, "familiarity": "none"})
            i += 1
    return annotated

# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_jpg(frame, quality=75):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def frame_diff(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(ga.astype(np.float32) - gb.astype(np.float32))))

def log_entry(brightness, japanese, english):
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "brightness", "japanese", "english"])
        writer.writerow([time.strftime("%H:%M:%S"), round(brightness, 1), japanese or "", english or ""])

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

def push_history(entry):
    state["history"].insert(0, entry)
    if len(state["history"]) > 1:
        state["history"].pop()

# ── OCR ───────────────────────────────────────────────────────────────────────

def apple_vision_ocr(frame):
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        img_url = Quartz.CFURLCreateFromFileSystemRepresentation(
            None, tmp_path.encode(), len(tmp_path), False)
        src = Quartz.CGImageSourceCreateWithURL(img_url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
        results = []
        def handler(request, error):
            if error: return
            for obs in request.results():
                results.append(obs.topCandidates_(1)[0].string())
        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLanguages_(["ja"])
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)
        japanese = " ".join(results).strip()
    finally:
        os.unlink(tmp_path)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

def clean_ocr(text):
    text = re.sub(
        r'[^\u3000-\u9fff\u3040-\u309f\u30a0-\u30ff\uff00-\uffef\s、。！？「」『』・…]',
        '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(?<!\S)\S(?!\S)', '', text).strip()
    return re.sub(r'\s+', ' ', text).strip()

def normalize_for_dedup(text):
    return re.sub(r'[\s、。・…]', '', text)

# ── LLM calls ─────────────────────────────────────────────────────────────────

def ollama_call(prompt):
    t0 = time.perf_counter()
    payload = {
        "model":  TRANSLATION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 2048, "num_batch": 64, "num_keep": 0, "temperature": 0},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    state["llm_calls"] += 1
    return r.json()["response"].strip(), elapsed_ms

def call_translate(japanese):
    raw, elapsed_ms = ollama_call(TRANSLATE_PROMPT.format(japanese=japanese))
    state["translate_calls"] += 1
    state["translate_ms"] = elapsed_ms
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data.get("romaji", ""), data.get("translation", raw), elapsed_ms
    except Exception:
        return "", raw, elapsed_ms

def call_learn(japanese, vocab):
    known = build_known_words_context(vocab)
    raw, elapsed_ms = ollama_call(LEARN_PROMPT.format(japanese=japanese, known_words=known))
    state["learn_calls"] += 1
    state["learn_ms"] = elapsed_ms
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data, elapsed_ms
    except Exception as e:
        print(f"⚠️  Failed to parse learn JSON: {e}\n{raw[:200]}")
        return {"romaji": "", "translation": raw, "grammar_note": "", "dialect_note": "",
                "progress_note": "", "vocab": [], "kanji": [], "grammar_patterns": []}, elapsed_ms

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop.copy()
    row_density = mask.sum(axis=1) / 255.0
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    furigana_threshold = row_density.max() * 0.20
    for i, d in enumerate(row_density):
        if 0 < d < furigana_threshold:
            result[i, :] = 0
    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0,0,0))
    return result

# ── Bounds loading ─────────────────────────────────────────────────────────────

def load_bounds():
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

def crop_to_bounds(frame, bounds):
    h, w = frame.shape[:2]
    x  = max(0, bounds["x"])
    y  = max(0, bounds["y"])
    x2 = min(w, x + bounds["w"])
    y2 = min(h, y + bounds["h"])
    return frame[y:y2, x:x2]

# ── Camera threads ─────────────────────────────────────────────────────────────

latest_crop      = None
latest_crop_lock = threading.Lock()

def pixel_diff_thread(bounds):
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
        with latest_crop_lock:
            latest_crop = crop.copy()
        if prev_crop is not None and prev_crop.shape == crop.shape:
            state["diff_value"] = round(frame_diff(prev_crop, crop), 2)
        prev_crop = crop.copy()

# ── Translation loop ───────────────────────────────────────────────────────────

def translation_loop(cap, bounds):
    print("🌐  Pipeline started.")
    state["phase"]  = "TRANSLATING"
    state["status"] = "Listening..."

    STABLE_THRESHOLD   = 4
    MIN_JAPANESE_CHARS = 5
    text_stable = {"text": "", "stable_count": 0}

    vocab = load_vocab()
    state["vocab_stats"] = {
        "total_words":    len(vocab["words"]),
        "total_kanji":    len(vocab["kanji"]),
        "total_patterns": len(vocab["grammar"]),
        "new_today":      vocab["stats"].get("new_today", 0),
    }

    threading.Thread(target=pixel_diff_thread, args=(bounds,), daemon=True).start()

    while True:
        with latest_crop_lock:
            crop = latest_crop.copy() if latest_crop is not None else None

        if crop is None:
            time.sleep(0.1)
            continue
        if crop.size == 0:
            state["status"] = "Crop empty — check bounds"
            time.sleep(1)
            continue

        brightness = float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        state["brightness"] = round(brightness, 1)

        if BRIGHTNESS_ENABLED:
            if brightness > BRIGHTNESS_GATE_HIGH:
                state["status"] = f"Gameplay (brightness={brightness:.1f})"
                continue
            if brightness < BRIGHTNESS_GATE_LOW:
                state["status"] = f"Fade/cutscene (brightness={brightness:.1f})"
                continue

        cleaned = preprocess_crop(crop)
        cv2.imwrite(PREVIEW_PATH, cleaned)
        update_preview(cleaned)

        try:
            cv2.imwrite(os.path.expanduser("~/Downloads/ocr_input.jpg"), cleaned)
            t0 = time.perf_counter()
            jp, ocr_ms = apple_vision_ocr(cleaned)
            jp = clean_ocr(jp)
            state["ocr_timing"] = {"ocr_ms": ocr_ms}

            # Gate 1: empty
            if not jp or jp.upper() == "NONE":
                text_stable["text"] = ""
                text_stable["stable_count"] = 0
                state["status"] = "Listening..."
                continue

            # Gate 2: too short
            if len(jp.replace(" ", "")) < MIN_JAPANESE_CHARS:
                text_stable["text"] = ""
                text_stable["stable_count"] = 0
                state["status"] = "Listening..."
                continue

            # Gate 3: stability
            normalized = normalize_for_dedup(jp)
            if normalized == normalize_for_dedup(text_stable["text"]):
                text_stable["stable_count"] += 1
            else:
                text_stable["text"] = jp
                text_stable["stable_count"] = 1
                state["status"] = "Dialogue typing..."
                continue

            if text_stable["stable_count"] < STABLE_THRESHOLD:
                state["status"] = f"Reading... ({text_stable['stable_count']}/{STABLE_THRESHOLD})"
                continue

            # Gate 4: dedup
            if normalize_for_dedup(jp) == normalize_for_dedup(state.get("japanese", "")):
                continue

            state["japanese"] = jp
            state["status"]   = "Translating..."

            # Annotate known words for UI highlighting
            vocab = load_vocab()
            annotated = annotate_japanese(jp, vocab)
            state["annotated"] = annotated

            if state["mode"] == "LEARN":
                lesson, llm_ms = call_learn(jp, vocab)
                # Enrich each vocab item with actual times_seen from vocab.json
                # so the UI can show correct familiarity (0=new, 1-9=learning, 10+=familiar)
                for v in lesson.get("vocab", []):
                    word = v.get("word", "")
                    v["times_seen"] = vocab["words"].get(word, {}).get("times_seen", 0)
                state["romaji"]      = lesson.get("romaji", "")
                state["translation"] = lesson.get("translation", "")
                state["lesson"]      = lesson
                state["translation_timing"] = {"llm_ms": llm_ms, "total_ms": ocr_ms + llm_ms}
                update_vocab(vocab, lesson)
                history_entry = {
                    "time":        time.strftime("%H:%M:%S"),
                    "japanese":    jp,
                    "romaji":      lesson.get("romaji", ""),
                    "translation": lesson.get("translation", ""),
                    "grammar":     lesson.get("grammar_note", ""),
                    "dialect":     lesson.get("dialect_note", ""),
                    "progress":    lesson.get("progress_note", ""),
                    "vocab":       lesson.get("vocab", []),
                }
            else:
                romaji, translation, llm_ms = call_translate(jp)
                state["romaji"]      = romaji
                state["translation"] = translation
                state["lesson"]      = None
                state["translation_timing"] = {"llm_ms": llm_ms, "total_ms": ocr_ms + llm_ms}
                history_entry = {
                    "time":        time.strftime("%H:%M:%S"),
                    "japanese":    jp,
                    "romaji":      romaji,
                    "translation": translation,
                    "grammar":     "",
                    "dialect":     "",
                    "progress":    "",
                    "vocab":       [],
                }

            push_history(history_entry)
            log_entry(brightness, jp, state["translation"])
            state["status"] = "Live"
            state["error"]  = ""
            print(f"📺  {jp}")
            print(f"✅  {state['translation']}\n")

        except Exception as e:
            print(f"❌  Error: {e}")
            state["error"]  = str(e)
            state["status"] = "Error"
            time.sleep(1)

# ── Capture loop ───────────────────────────────────────────────────────────────

def capture_loop():
    bounds = load_bounds()
    state["bounds"] = bounds
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        state["error"] = "Cannot connect to camera"
        print("❌  Cannot connect. Check VIDEO_SOURCE.")
        return
    print("✅  Connected.")
    translation_loop(cap, bounds)
    cap.release()

# ── Flask ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zelda Translator</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080b10;
    --surface:  #0d1117;
    --border:   #1c2333;
    --border2:  #252d3d;
    --text:     #cdd6f4;
    --subtext:  #6c7a96;
    --dim:      #2a3245;
    --accent:   #89b4fa;
    --green:    #a6e3a1;
    --yellow:   #f9e2af;
    --red:      #f38ba8;
    --mauve:    #cba6f7;
    --teal:     #94e2d5;
    --mono:     'Space Mono', monospace;
    --sans:     'DM Sans', sans-serif;
    --jp:       'Noto Sans JP', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--sans);
    min-height: 100vh;
    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
  }
  header {
    grid-column: 1 / -1;
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    display: flex; align-items: center; gap: 16px;
  }
  .logo { font-family: var(--mono); font-size: 12px; color: var(--accent); letter-spacing: 0.12em; text-transform: uppercase; }
  .logo span { color: var(--subtext); }

  /* Mode toggle */
  .mode-toggle { display: flex; gap: 0; border: 1px solid var(--border2); border-radius: 8px; overflow: hidden; margin-left: 8px; }
  .mode-btn {
    font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 6px 16px; cursor: pointer; border: none;
    background: transparent; color: var(--subtext); transition: all 0.2s;
  }
  .mode-btn.active { background: var(--accent); color: var(--bg); }
  .mode-btn:hover:not(.active) { color: var(--text); background: var(--border); }

  /* Vocab stats */
  .vocab-stats { margin-left: auto; display: flex; gap: 16px; align-items: center; }
  .stat { font-family: var(--mono); font-size: 10px; color: var(--subtext); }
  .stat span { color: var(--text); font-weight: 700; }
  .stat .new { color: var(--accent); }

  .header-status { font-family: var(--mono); font-size: 10px; color: var(--subtext); }
  .header-status.error { color: var(--red); }

  /* Main panel */
  main { padding: 24px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }

  /* Translation card */
  .trans-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 24px; }
  .card-label { font-family: var(--mono); font-size: 9px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--subtext); margin-bottom: 8px; }

  /* Japanese with highlights */
  .japanese-wrap { font-family: var(--jp); font-size: 30px; font-weight: 400; line-height: 1.7; margin-bottom: 16px; min-height: 44px; }
  .japanese-wrap.placeholder { color: var(--dim); font-size: 18px; }
  .w-none     { color: var(--text); }
  .w-new      { color: var(--text); text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: 4px; }
  .w-learning { color: var(--text); text-decoration: underline; text-decoration-color: var(--yellow); text-underline-offset: 4px; }
  .w-familiar { color: var(--text); text-decoration: underline; text-decoration-color: var(--green); text-underline-offset: 4px; }

  .romaji { font-family: var(--mono); font-size: 13px; color: var(--subtext); margin-bottom: 14px; letter-spacing: 0.04em; }
  .divider { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
  .english { font-size: 18px; font-weight: 300; color: #a8b5d4; line-height: 1.65; }
  .placeholder-text { color: var(--dim); font-size: 14px; }

  /* Legend */
  .legend { display: flex; gap: 16px; margin-top: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 9px; color: var(--subtext); }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-new     { background: var(--accent); }
  .dot-learning { background: var(--yellow); }
  .dot-familiar { background: var(--green); }

  /* Metrics row */
  .metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .metric-box { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .metric-label { font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--subtext); margin-bottom: 6px; }
  .metric-value { font-family: var(--mono); font-size: 22px; font-weight: 700; color: var(--accent); }
  .metric-value.green { color: var(--green); }
  .metric-sub { font-family: var(--mono); font-size: 10px; color: var(--dim); margin-top: 2px; }

  /* Lesson panel */
  .lesson-panel { display: flex; flex-direction: column; gap: 12px; }
  .lesson-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .lesson-card.hidden { display: none; }

  /* Vocab table */
  .vocab-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  .vocab-table th { font-family: var(--mono); font-size: 9px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--subtext); padding: 0 8px 8px 0; text-align: left; }
  .vocab-table td { padding: 6px 8px 6px 0; font-size: 13px; border-top: 1px solid var(--border); vertical-align: top; }
  .vocab-table .jp { font-family: var(--jp); font-size: 16px; }
  .vocab-table .reading { font-family: var(--mono); font-size: 11px; color: var(--subtext); }
  .vocab-table .meaning { color: var(--text); }
  .vocab-new      { border-left: 2px solid var(--accent) !important; padding-left: 6px !important; }
  .vocab-learning { border-left: 2px solid var(--yellow) !important; padding-left: 6px !important; }
  .vocab-familiar { border-left: 2px solid var(--green) !important; padding-left: 6px !important; }
  .familiarity-badge { font-family: var(--mono); font-size: 9px; padding: 2px 6px; border-radius: 4px; }
  .badge-new      { background: rgba(137,180,250,0.15); color: var(--accent); }
  .badge-learning { background: rgba(249,226,175,0.15); color: var(--yellow); }
  .badge-familiar { background: rgba(166,227,161,0.15); color: var(--green); }

  /* Notes */
  .note-text { font-size: 13px; color: var(--subtext); line-height: 1.6; }
  .note-text strong { color: var(--text); font-weight: 500; }
  .progress-note { font-size: 13px; color: var(--teal); line-height: 1.6; font-style: italic; }

  /* Preview crop */
  .preview-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  .preview-card img { width: 100%; display: block; }

  /* Last dialogue card at bottom */
  .history-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; }
  .h-entry { }
  .h-time { font-family: var(--mono); font-size: 9px; color: var(--dim); margin-bottom: 4px; }
  .h-jp { font-family: var(--jp); font-size: 16px; color: var(--text); margin-bottom: 3px; line-height: 1.5; }
  .h-romaji { font-family: var(--mono); font-size: 11px; color: var(--subtext); margin-bottom: 4px; }
  .h-en { font-size: 13px; color: var(--subtext); line-height: 1.4; margin-bottom: 6px; }
  .h-grammar { font-size: 11px; color: var(--dim); line-height: 1.5; font-style: italic; }
  .h-expand { font-family: var(--mono); font-size: 9px; color: var(--accent); cursor: pointer; background: none; border: none; padding: 0; margin-top: 4px; }
  .h-lesson { display: none; margin-top: 10px; border-top: 1px solid var(--border); padding-top: 10px; }
  .h-lesson.open { display: block; }
  .h-vocab-item { font-size: 11px; color: var(--subtext); padding: 3px 0; }
  .h-vocab-item .hjp { font-family: var(--jp); color: var(--text); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
</style>
</head>
<body>
<header>
  <div class="logo">ZELDA <span>/</span> TRANSLATOR</div>
  <div class="mode-toggle">
    <button class="mode-btn active" id="btn-translate" onclick="setMode('TRANSLATE')">Translate</button>
    <button class="mode-btn" id="btn-learn" onclick="setMode('LEARN')">Learn</button>
  </div>
  <div class="vocab-stats" id="vocab-stats">
    <div class="stat">Words <span id="stat-words">0</span></div>
    <div class="stat">Kanji <span id="stat-kanji">0</span></div>
    <div class="stat">Patterns <span id="stat-patterns">0</span></div>
    <div class="stat">New today <span class="new" id="stat-new">0</span></div>
  </div>
  <div class="header-status" id="header-status">Initializing...</div>
</header>

<main>
  <!-- Translation card -->
  <div class="trans-card">
    <div class="card-label">Japanese</div>
    <div class="japanese-wrap placeholder" id="japanese-wrap">Waiting for dialogue...</div>
    <div class="romaji" id="romaji"></div>
    <hr class="divider">
    <div class="card-label">English</div>
    <div class="english placeholder-text" id="english">Translation will appear here</div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot dot-new"></div>New</div>
      <div class="legend-item"><div class="legend-dot dot-learning"></div>Learning (1-9×)</div>
      <div class="legend-item"><div class="legend-dot dot-familiar"></div>Familiar (10+×)</div>
    </div>
  </div>

  <!-- Metrics row -->
  <div class="metrics-row">
    <div class="metric-box">
      <div class="metric-label">Translate Calls</div>
      <div class="metric-value" id="translate-calls">0</div>
      <div class="metric-sub" id="translate-ms">—</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Last Translate</div>
      <div class="metric-value green" id="translate-time">—</div>
      <div class="metric-sub">ms</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Learn Calls</div>
      <div class="metric-value" id="learn-calls">0</div>
      <div class="metric-sub" id="learn-ms">—</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Last Lesson</div>
      <div class="metric-value green" id="learn-time">—</div>
      <div class="metric-sub">ms</div>
    </div>
  </div>

  <!-- Lesson panel — only shown in LEARN mode -->
  <div class="lesson-panel" id="lesson-panel" style="display:none">

    <!-- Vocab breakdown -->
    <div class="lesson-card" id="vocab-card">
      <div class="card-label">Vocabulary</div>
      <table class="vocab-table">
        <thead><tr><th>Word</th><th>Reading</th><th>Meaning</th><th></th></tr></thead>
        <tbody id="vocab-tbody"></tbody>
      </table>
    </div>

    <!-- Grammar note -->
    <div class="lesson-card" id="grammar-card">
      <div class="card-label">Grammar Pattern</div>
      <div class="note-text" id="grammar-note">—</div>
    </div>

    <!-- Dialect note -->
    <div class="lesson-card" id="dialect-card">
      <div class="card-label">Dialect / Nuance</div>
      <div class="note-text" id="dialect-note">—</div>
    </div>

    <!-- Progress note -->
    <div class="lesson-card" id="progress-card">
      <div class="card-label">Your Progress</div>
      <div class="progress-note" id="progress-note">—</div>
    </div>

  </div>

  <!-- Live crop preview -->
  <div class="preview-card">
    <img src="/preview" alt="Live crop">
  </div>

  <!-- Last dialogue — always at the bottom -->
  <div class="history-card" id="history-card" style="display:none">
    <div class="card-label">Previous Dialogue</div>
    <div id="history-list"></div>
  </div>

</main>

<script>
let currentMode = 'TRANSLATE';

function setMode(mode) {
  currentMode = mode;
  document.getElementById('btn-translate').classList.toggle('active', mode === 'TRANSLATE');
  document.getElementById('btn-learn').classList.toggle('active', mode === 'LEARN');
  document.getElementById('lesson-panel').style.display = mode === 'LEARN' ? 'flex' : 'none';
  fetch('/set_mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode})});
}

function getFamiliarity(v) {
  // Use times_seen count if available, fall back to is_new flag
  if (typeof v.times_seen === 'number') {
    if (v.times_seen === 0) return 'new';
    if (v.times_seen < 10)  return 'learning';
    return 'familiar';
  }
  return v.is_new ? 'new' : 'learning';
}

function famBadge(fam) {
  const labels = {new:'New', learning:'Learning', familiar:'Familiar'};
  return `<span class="familiarity-badge badge-${fam}">${labels[fam]||fam}</span>`;
}

function renderJapanese(annotated) {
  if (!annotated || !annotated.length) return '';
  return annotated.map(a => `<span class="w-${a.familiarity}">${a.text}</span>`).join('');
}

async function poll() {
  try {
    const d = await (await fetch('/state')).json();

    // Status
    const hs = document.getElementById('header-status');
    hs.textContent = d.error ? d.error : (d.status || '');
    hs.className = 'header-status' + (d.error ? ' error' : '');

    // Vocab stats
    if (d.vocab_stats) {
      document.getElementById('stat-words').textContent    = d.vocab_stats.total_words || 0;
      document.getElementById('stat-kanji').textContent    = d.vocab_stats.total_kanji || 0;
      document.getElementById('stat-patterns').textContent = d.vocab_stats.total_patterns || 0;
      document.getElementById('stat-new').textContent      = d.vocab_stats.new_today || 0;
    }

    // Japanese with highlights
    const jpWrap = document.getElementById('japanese-wrap');
    if (d.japanese && d.annotated) {
      jpWrap.innerHTML = renderJapanese(d.annotated);
      jpWrap.className = 'japanese-wrap';
    } else if (d.japanese) {
      jpWrap.textContent = d.japanese;
      jpWrap.className = 'japanese-wrap';
    } else {
      jpWrap.textContent = 'Waiting for dialogue...';
      jpWrap.className = 'japanese-wrap placeholder';
    }

    // Romaji
    const romajiEl = document.getElementById('romaji');
    romajiEl.textContent = d.romaji || '';

    // English
    const enEl = document.getElementById('english');
    if (d.translation) {
      enEl.textContent = d.translation;
      enEl.className = 'english';
    } else {
      enEl.textContent = 'Translation will appear here';
      enEl.className = 'english placeholder-text';
    }

    // Call counters and timers
    document.getElementById('translate-calls').textContent = d.translate_calls || 0;
    document.getElementById('translate-time').textContent  = d.translate_ms ? d.translate_ms + 'ms' : '—';
    document.getElementById('learn-calls').textContent     = d.learn_calls || 0;
    document.getElementById('learn-time').textContent      = d.learn_ms ? d.learn_ms + 'ms' : '—';

    // Lesson panel — update whenever lesson data exists, not just current mode
    // This ensures switching to LEARN shows fresh data immediately after re-run
    if (d.lesson) {
      const l = d.lesson;
      const tbody = document.getElementById('vocab-tbody');
      if (l.vocab && l.vocab.length) {
        tbody.innerHTML = l.vocab.map(v => {
          const fam = getFamiliarity(v);
          return `<tr>
            <td class="jp vocab-${fam}">${v.word}</td>
            <td class="reading">${v.reading}</td>
            <td class="meaning">${v.meaning}</td>
            <td>${famBadge(fam)}</td>
          </tr>`;
        }).join('');
      } else {
        document.getElementById('vocab-tbody').innerHTML = '<tr><td colspan="4" style="color:var(--dim);font-size:12px;padding-top:8px">No vocab extracted</td></tr>';
      }
      document.getElementById('grammar-note').innerHTML  = l.grammar_note  || '—';
      document.getElementById('dialect-note').innerHTML  = l.dialect_note  || '—';
      document.getElementById('progress-note').innerHTML = l.progress_note || '—';
    } else if (currentMode === 'LEARN') {
      // In learn mode but no lesson yet — clear stale data
      document.getElementById('vocab-tbody').innerHTML   = '';
      document.getElementById('grammar-note').innerHTML  = '—';
      document.getElementById('dialect-note').innerHTML  = '—';
      document.getElementById('progress-note').innerHTML = '—';
    }

    // Last dialogue card
    const hcard = document.getElementById('history-card');
    const hl = document.getElementById('history-list');
    if (d.history && d.history.length) {
      hcard.style.display = 'block';
      const h = d.history[0];
      hl.innerHTML = `
        <div class="h-entry">
          <div class="h-time">${h.time}</div>
          <div class="h-jp">${h.japanese}</div>
          ${h.romaji ? `<div class="h-romaji">${h.romaji}</div>` : ''}
          <div class="h-en">${h.translation}</div>
          ${h.grammar ? `
            <button class="h-expand" onclick="toggleLesson()">▸ show lesson</button>
            <div class="h-lesson" id="hl-0">
              <div class="h-grammar">${h.grammar}</div>
              ${h.dialect ? `<div class="h-grammar" style="margin-top:6px">${h.dialect}</div>` : ''}
              ${h.progress ? `<div class="h-grammar" style="margin-top:6px;color:var(--teal)">${h.progress}</div>` : ''}
              ${(h.vocab||[]).map(v =>
                `<div class="h-vocab-item"><span class="hjp">${v.word}</span> ${v.reading} — ${v.meaning}</div>`
              ).join('')}
            </div>
          ` : ''}
        </div>`;
    } else {
      hcard.style.display = 'none';
    }

  } catch(e) {}
  setTimeout(poll, 500);
}

function toggleLesson() {
  const el = document.getElementById('hl-0');
  if (!el) return;
  const isOpen = el.classList.contains('open');
  el.classList.toggle('open');
  const btn = el.previousElementSibling;
  if (btn) btn.textContent = isOpen ? '▸ show lesson' : '▾ hide lesson';
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

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = flask_request.get_json()
    if data and data.get('mode') in ('TRANSLATE', 'LEARN'):
        old_mode = state['mode']
        state['mode'] = data['mode']
        # Clear lesson and reset dedup so current text gets re-processed in new mode
        if old_mode != state['mode']:
            state['lesson'] = None
            state['romaji'] = ''
            state['translation'] = ''
            # Force re-translation by clearing japanese so dedup doesn't block it
            state['japanese'] = ''
        print(f"🔄  Mode switched to {state['mode']}")
    return jsonify({"mode": state["mode"]})

@app.route('/preview')
def preview():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def unload_model():
    try:
        requests.post(OLLAMA_URL, json={"model": TRANSLATION_MODEL, "keep_alive": 0}, timeout=10)
        print(f"\n🧹  {TRANSLATION_MODEL} unloaded from RAM.")
    except Exception as e:
        print(f"\n⚠️  Could not unload: {e}")

if __name__ == '__main__':
    print("🎮  Zelda Translator")
    print(f"📱  Camera:  {VIDEO_SOURCE}")
    print(f"🤖  Model:   {TRANSLATION_MODEL}")
    print(f"📚  Vocab:   {VOCAB_FILE}")
    print(f"🌐  UI:      http://localhost:5002")
    print("─" * 40)
    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        unload_model()
