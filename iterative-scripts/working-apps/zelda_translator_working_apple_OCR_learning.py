"""
zelda_translator.py
===================
Two-step pipeline:
  Step 1 — OCR: Apple Vision framework (macOS built-in, M1 neural engine)
  Step 2 — LLM: qwen2.5:7b via Ollama

Two UI modes (toggle in header):
  TRANSLATE — Japanese (known words highlighted), romaji, English. Fast.
  LEARN     — Full lesson: word-by-word breakdown, grammar note, kanji.

Vocab tracking: vocab.json saved next to this script.
  Words/kanji/grammar patterns tracked with familiarity levels:
    new      (0 seen)   → blue underline
    learning (1-9 seen) → yellow underline
    familiar (10+ seen) → green underline

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
LESSONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")
PREVIEW_PATH = os.path.expanduser("~/Downloads/preprocessed_crop.jpg")
BOUNDS_FILE  = "bounds.json"
# How many lessons the user must acknowledge before a review quiz is triggered.
# Lower = more frequent quizzes, more interruptions but better retention.
# This value is injected into the frontend JS via render_template_string so the
# 'Until quiz' counter in the sidebar always matches the Python trigger logic.
# Change only this constant — both backend and UI stay in sync automatically.
QUIZ_EVERY   = 3    # trigger a quiz after every N acknowledged lessons

# ── Brightness gate ───────────────────────────────────────────────────────────
# Zelda dialogue boxes are dark overlays. Gates can filter out gameplay frames
# (too bright) and fade/cutscene frames (too dark). Disabled by default — the
# crop region alone is usually sufficient to avoid false positives.
BRIGHTNESS_GATE_HIGH = 80.0
BRIGHTNESS_GATE_LOW  = 10.0
BRIGHTNESS_ENABLED   = False

# ── Prompts ───────────────────────────────────────────────────────────────────
TRANSLATE_PROMPT = """Translate this Japanese video game dialogue to natural English.
Respond ONLY with valid JSON, no markdown, no extra text:
{{"romaji": "...", "translation": "..."}}

Japanese: {japanese}"""

LEARN_PROMPT = """You are a Japanese tutor for a complete beginner learning through a video game.
Analyze this dialogue and respond ONLY with valid JSON, no markdown, no extra text.

Japanese: {japanese}
Words already seen (seen count): {known_words}

Respond with this exact JSON structure:
{{
  "romaji": "full romaji reading of the entire sentence",
  "translation": "natural English translation",
  "breakdown": [
    {{
      "word": "japanese word or particle",
      "reading": "romaji",
      "meaning": "english meaning",
      "role": "grammatical role, e.g. subject, verb, object, topic marker, particle"
    }}
  ],
  "grammar_note": "if a real grammar pattern exists explain it in one simple sentence, otherwise empty string",
  "kanji": [
    {{
      "kanji": "single kanji character",
      "reading": "reading used here",
      "meaning": "core english meaning",
      "example": "one common word using this kanji"
    }}
  ]
}}

Rules:
- breakdown must cover every meaningful word left to right
- kanji array must include every kanji character present in the sentence
- keep all meanings short and beginner-friendly
- only include grammar_note if there is a real pattern worth teaching"""

# ── Shared state ───────────────────────────────────────────────────────────────
state = {
    "mode":                "TRANSLATE",   # TRANSLATE or LEARN
    "phase":               "TRANSLATING",
    "status":              "Starting up...",  # translate pipeline status
    "learn_status":        "—",               # learn pipeline status (independent)
    "japanese":            "",
    "romaji":              "",
    "translation":         "",
    "translate_romaji":    "",              # live translate panel romaji
    "translate_translation": "",           # live translate panel translation
    "lesson":              None,          # full lesson object in LEARN mode
    "lesson_pending_ack":  False,         # True = lesson shown, not yet acknowledged
    "lesson_japanese":     "",            # japanese text the pending lesson is for
    "llm_calls":           0,
    "translate_calls":     0,
    "learn_calls":         0,             # only incremented on acknowledge
    "translate_ms":        0,
    "translate_ocr_ms":    0,             # OCR portion of last translate
    "translate_llm_ms":    0,             # LLM portion of last translate
    "learn_ms":            0,             # updated immediately when learn call returns
    "learn_ocr_ms":        0,             # OCR portion, updated immediately
    "learn_llm_ms":        0,             # LLM portion, updated immediately
    "_pending_learn_ms":   0,             # kept for ack to use in learn_calls increment
    "_pending_ocr_ms":     0,             # kept for ack to use in learn_calls increment
    "bounds":              None,
    "history":             [],
    "ocr_timing":          {"ocr_ms": 0},
    "translation_timing":  {"llm_ms": 0, "total_ms": 0},
    "brightness":          0.0,
    "error":               "",
    "vocab_stats":         {"total_words": 0, "total_kanji": 0, "new_today": 0},
    # Quiz
    "quiz_active":         False,          # True = quiz in progress, blocks learn calls
    "quiz_data":           None,           # current quiz object sent to UI
    "lessons_since_quiz":  0,              # resets to 0 after each quiz
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
            "stats":    {"total_lines": 0, "new_today": 0, "last_session": ""},
        }

def save_vocab(vocab):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def get_familiarity(entry):
    """
    entry can be a vocab word/kanji dict or a bare times_seen int (legacy).
    Familiarity is based on recall accuracy from quizzes.
    Falls back to times_seen tiers if no quiz data yet.
    """
    if isinstance(entry, int):
        ts = entry
        correct = 0
        total   = 0
    else:
        ts      = entry.get("times_seen", 0)
        correct = entry.get("correct_recalls", 0)
        total   = entry.get("total_recalls", 0)

    if total >= 3:
        ratio = correct / total
        if ratio >= 0.8:
            return "familiar"
        elif ratio >= 0.4:
            return "learning"
        else:
            return "new"
    # Not enough quiz data yet — use passive exposure tiers
    if ts == 0:    return "new"
    if ts < 5:     return "learning"
    return "familiar"

# ── Lessons store ─────────────────────────────────────────────────────────────

def load_lessons():
    try:
        with open(LESSONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_lessons(lessons):
    with open(LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(lessons, f, ensure_ascii=False, indent=2)

def append_lesson(japanese, lesson):
    """Append acknowledged lesson to lessons.json, keep last 50."""
    lessons = load_lessons()
    lessons.append({
        "time":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "japanese":     japanese,
        "romaji":       lesson.get("romaji", ""),
        "translation":  lesson.get("translation", ""),
        "grammar_note": lesson.get("grammar_note", ""),
        "breakdown":    lesson.get("breakdown", []),
        "kanji":        lesson.get("kanji", []),
    })
    lessons = lessons[-50:]
    save_lessons(lessons)

def build_quiz(recent_lessons):
    """
    Build a quiz deck from the last QUIZ_EVERY acknowledged lessons.
    All unique words and kanji across those lessons are collected into a pool,
    then randomly sampled down to n/4 (minimum 1 card). Sampling keeps quizzes
    short and varied — showing every word every time would be tedious, especially
    as the vocab grows. Random selection means each quiz feels different even for
    the same set of lessons.
    Only Japanese/kana characters qualify as word cards — punctuation is excluded.
    Cards are shuffled after sampling so order is unpredictable.
    """
    import random
    cards = []
    seen_words = set()
    seen_kanji = set()

    for lesson in recent_lessons:
        for item in lesson.get("breakdown", []):
            w = item.get("word", "")
            if w and w not in seen_words and any('\u3040' <= c <= '\u9fff' for c in w):
                cards.append({
                    "type":    "word",
                    "prompt":  w,
                    "reading": item.get("reading", ""),
                    "answer":  item.get("meaning", ""),
                })
                seen_words.add(w)
        for item in lesson.get("kanji", []):
            k = item.get("kanji", "")
            if k and k not in seen_kanji:
                cards.append({
                    "type":    "kanji",
                    "prompt":  k,
                    "reading": item.get("reading", ""),
                    "answer":  item.get("meaning", ""),
                    "example": item.get("example", ""),
                })
                seen_kanji.add(k)

    # Randomly sample n/4, minimum 1
    target  = max(1, len(cards) // 4)
    sampled = random.sample(cards, min(target, len(cards)))
    random.shuffle(sampled)

    return {
        "cards":     sampled,
        "index":     0,
        "total":     len(sampled),
        "correct":   0,
        "incorrect": 0,
        "done":      False,
    }

def build_known_words_context(vocab):
    """Build compact context string for the LLM."""
    known = {w: d.get("times_seen", 0) for w, d in vocab["words"].items()}
    top = sorted(known.items(), key=lambda x: x[1], reverse=True)[:30]
    return ", ".join(f"{w}({c})" for w, c in top) if top else "none yet"

def update_vocab(vocab, lesson):
    """Commit a lesson's words/kanji to vocab.json. Called only on acknowledge."""
    today = str(date.today())
    if vocab["stats"].get("last_session") != today:
        vocab["stats"]["new_today"] = 0
        vocab["stats"]["last_session"] = today

    for item in lesson.get("breakdown", []):
        w = item.get("word", "")
        if not w or not any('\u3040' <= c <= '\u9fff' for c in w):
            continue
        if w not in vocab["words"]:
            vocab["words"][w] = {
                "reading":         item.get("reading", ""),
                "meaning":         item.get("meaning", ""),
                "times_seen":      0,
                "correct_recalls": 0,
                "total_recalls":   0,
                "first_seen":      today,
                "last_seen":       today,
            }
            vocab["stats"]["new_today"] = vocab["stats"].get("new_today", 0) + 1
        vocab["words"][w]["times_seen"] = vocab["words"][w].get("times_seen", 0) + 1
        vocab["words"][w]["last_seen"] = today
        # ensure recall fields exist on older entries
        vocab["words"][w].setdefault("correct_recalls", 0)
        vocab["words"][w].setdefault("total_recalls",   0)

    for item in lesson.get("kanji", []):
        k = item.get("kanji", "")
        if not k:
            continue
        if k not in vocab["kanji"]:
            vocab["kanji"][k] = {
                "reading":         item.get("reading", ""),
                "meaning":         item.get("meaning", ""),
                "times_seen":      0,
                "correct_recalls": 0,
                "total_recalls":   0,
                "first_seen":      today,
            }
            vocab["stats"]["new_today"] = vocab["stats"].get("new_today", 0) + 1
        vocab["kanji"][k]["times_seen"] = vocab["kanji"][k].get("times_seen", 0) + 1
        vocab["kanji"][k].setdefault("correct_recalls", 0)
        vocab["kanji"][k].setdefault("total_recalls",   0)

    vocab["stats"]["total_lines"] = vocab["stats"].get("total_lines", 0) + 1
    save_vocab(vocab)

    state["vocab_stats"] = {
        "total_words": len(vocab["words"]),
        "total_kanji": len(vocab["kanji"]),
        "new_today":   vocab["stats"].get("new_today", 0),
    }

def annotate_japanese(japanese, vocab):
    """Return list of {text, familiarity} for rendering in UI."""
    annotated = []
    i = 0
    text = japanese
    words_sorted = sorted(vocab["words"].keys(), key=len, reverse=True)
    while i < len(text):
        matched = False
        for w in words_sorted:
            if text[i:i+len(w)] == w:
                entry = vocab["words"][w]
                annotated.append({"text": w, "familiarity": get_familiarity(entry)})
                i += len(w)
                matched = True
                break
        if not matched:
            c = text[i]
            if c in vocab["kanji"]:
                entry = vocab["kanji"][c]
                annotated.append({"text": c, "familiarity": get_familiarity(entry)})
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

def fuzzy_same(a, b, max_diff=1):
    """True if a and b differ by at most max_diff characters (edit distance).
    Used for Gate 4 so minor OCR noise doesn't re-fire the LLM."""
    a, b = normalize_for_dedup(a), normalize_for_dedup(b)
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_diff:
        return False
    # Wagner-Fischer edit distance with early exit
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j] + (0 if ca == cb else 1),
                curr[j] + 1,
                prev[j + 1] + 1,
            ))
        if min(curr) > max_diff:
            return False
        prev = curr
    return prev[len(b)] <= max_diff

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

def call_translate(japanese, ocr_ms=0):
    raw, elapsed_ms = ollama_call(TRANSLATE_PROMPT.format(japanese=japanese))
    state["translate_calls"]  += 1
    state["translate_ms"]      = ocr_ms + elapsed_ms
    state["translate_ocr_ms"]  = ocr_ms
    state["translate_llm_ms"]  = elapsed_ms
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data.get("romaji", ""), data.get("translation", raw), elapsed_ms
    except Exception:
        return "", raw, elapsed_ms

def call_learn(japanese, vocab):
    known = build_known_words_context(vocab)
    raw, elapsed_ms = ollama_call(LEARN_PROMPT.format(japanese=japanese, known_words=known))
    # NOTE: learn_calls / learn_ms are NOT incremented here — only on /acknowledge
    state["_pending_learn_ms"] = elapsed_ms
    state["_pending_ocr_ms"]   = 0  # will be set by caller
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        return data, elapsed_ms
    except Exception as e:
        print(f"⚠️  Failed to parse learn JSON: {e}\n{raw[:200]}")
        return {
            "romaji": "", "translation": raw,
            "breakdown": [], "grammar_note": "", "kanji": [],
        }, elapsed_ms

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop.copy()
    row_density = mask.sum(axis=1) / 255.0
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density = float(np.median(non_zero_densities))
        furigana_threshold = median_density * 0.42
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

# ── Shared OCR output — written by OCR loop, read by translate + learn loops ───
latest_stable_jp     = {"text": "", "ocr_ms": 0}
latest_stable_lock   = threading.Lock()

# ── OCR loop — continuously reads camera, runs stability gates, publishes stable text ──

def ocr_loop(bounds):
    """Runs OCR continuously. When text passes all stability/dedup gates,
    writes to latest_stable_jp so both translate and learn loops can consume it."""
    STABLE_THRESHOLD   = 4
    MIN_JAPANESE_CHARS = 4
    text_stable = {"text": "", "stable_count": 0}

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

            # Gate 3: stability — text must be identical for STABLE_THRESHOLD frames
            if normalize_for_dedup(jp) == normalize_for_dedup(text_stable["text"]):
                text_stable["stable_count"] += 1
            else:
                text_stable["text"] = jp
                text_stable["stable_count"] = 1
                state["status"] = "Dialogue typing..."
                continue

            if text_stable["stable_count"] < STABLE_THRESHOLD:
                state["status"] = f"Reading... ({text_stable['stable_count']}/{STABLE_THRESHOLD})"
                continue

            # Gate 4: only publish if different from last published (fuzzy 1-char tolerance)
            with latest_stable_lock:
                last = latest_stable_jp["text"]
            if not fuzzy_same(jp, last):
                with latest_stable_lock:
                    latest_stable_jp["text"]   = jp
                    latest_stable_jp["ocr_ms"] = ocr_ms
                # Annotate for highlights — shared by both panels
                vocab = load_vocab()
                state["japanese"] = jp
                state["annotated"] = annotate_japanese(jp, vocab)

        except Exception as e:
            print(f"❌  OCR error: {e}")
            state["error"]  = str(e)
            state["status"] = "Error"
            time.sleep(1)

# ── Translate loop — always running, fires on every new stable text ─────────────

def translate_loop():
    """Independently watches for new stable text and fires translate LLM calls.
    Always running regardless of which UI tab is active."""
    last_translated = ""
    print("🔤  Translate pipeline started.")

    while True:
        with latest_stable_lock:
            jp     = latest_stable_jp["text"]
            ocr_ms = latest_stable_jp["ocr_ms"]

        if not jp or jp == last_translated:
            time.sleep(0.1)
            continue

        last_translated = jp
        try:
            state["status"] = "Translating..."
            romaji, translation, llm_ms = call_translate(jp, ocr_ms)
            state["translate_romaji"]      = romaji
            state["translate_translation"] = translation
            state["translation_timing"]    = {"llm_ms": llm_ms, "total_ms": ocr_ms + llm_ms}
            history_entry = {
                "time":        time.strftime("%H:%M:%S"),
                "japanese":    jp,
                "romaji":      romaji,
                "translation": translation,
            }
            push_history(history_entry)
            log_entry(state.get("brightness", 0), jp, translation)
            state["status"] = "Live"
            print(f"🔤  {jp} → {translation}")
        except Exception as e:
            print(f"❌  Translate error: {e}")
            state["error"] = str(e)
            time.sleep(1)

# ── Learn loop — fires once per new text, freezes until acknowledged ─────────────

def learn_loop():
    """Independently watches for new stable text and fires learn LLM calls.
    Freezes (lesson_pending_ack=True) until user acknowledges. Always running."""
    last_learned = ""
    print("📖  Learn pipeline started.")

    while True:
        # Freeze if lesson pending ack or quiz active
        if state["lesson_pending_ack"] or state["quiz_active"]:
            if state["lesson_pending_ack"]:
                state["learn_status"] = "Lesson ready — acknowledge to continue"
            time.sleep(0.2)
            continue

        with latest_stable_lock:
            jp     = latest_stable_jp["text"]
            ocr_ms = latest_stable_jp["ocr_ms"]

        if not jp or jp == last_learned:
            time.sleep(0.1)
            continue

        last_learned = jp
        try:
            state["learn_status"] = "Generating lesson..."
            vocab = load_vocab()
            lesson, llm_ms = call_learn(jp, vocab)
            # Enrich with familiarity
            for item in lesson.get("breakdown", []):
                w = item.get("word", "")
                entry = vocab["words"].get(w, {})
                item["times_seen"]  = entry.get("times_seen", 0)
                item["familiarity"] = get_familiarity(entry) if entry else "new"
            for item in lesson.get("kanji", []):
                k = item.get("kanji", "")
                entry = vocab["kanji"].get(k, {})
                item["times_seen"]  = entry.get("times_seen", 0)
                item["familiarity"] = get_familiarity(entry) if entry else "new"
            state["lesson"]             = lesson
            state["lesson_pending_ack"] = True
            state["lesson_japanese"]    = jp
            state["_pending_ocr_ms"]    = ocr_ms
            state["learn_ocr_ms"]       = ocr_ms
            state["learn_llm_ms"]       = llm_ms
            state["learn_ms"]           = ocr_ms + llm_ms
            state["learn_status"]       = "Lesson ready — acknowledge to continue"
            print(f"📖  Lesson generated for: {jp}")
        except Exception as e:
            print(f"❌  Learn error: {e}")
            state["error"] = str(e)
            last_learned = ""  # allow retry on next cycle
            time.sleep(1)

# ── Main pipeline entrypoint ────────────────────────────────────────────────────

def translation_loop(cap, bounds):
    print("🌐  Pipeline started.")
    state["phase"]        = "TRANSLATING"
    state["status"]       = "Listening..."
    state["learn_status"] = "Listening..."

    vocab = load_vocab()
    state["vocab_stats"] = {
        "total_words": len(vocab["words"]),
        "total_kanji": len(vocab["kanji"]),
        "new_today":   vocab["stats"].get("new_today", 0),
    }

    threading.Thread(target=ocr_loop,       args=(bounds,), daemon=True).start()
    threading.Thread(target=translate_loop, daemon=True).start()
    threading.Thread(target=learn_loop,     daemon=True).start()

    # Keep main thread alive
    while True:
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
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&family=Space+Mono:wght@400;700&family=Nunito:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #1a1612;
    --surface:  #211d18;
    --border:   #332d25;
    --border2:  #3d3529;
    --text:     #d4c9b8;
    --subtext:  #8a7f6e;
    --dim:      #3d3529;
    --accent:   #7bafd4;
    --green:    #8ec49a;
    --yellow:   #d4b87a;
    --red:      #c47a7a;
    --mauve:    #b89ac4;
    --teal:     #7ab8b0;
    --mono:     'Space Mono', monospace;
    --sans:     'Nunito', sans-serif;
    --jp:       'Noto Sans JP', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text); font-family: var(--sans);
    height: 100vh; overflow: hidden;
    display: grid;
    grid-template-rows: auto 1fr;
    grid-template-columns: 260px 1fr;
  }
  header {
    grid-column: 1 / -1;
    padding: 14px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    display: flex; align-items: center; gap: 16px;
  }
  .logo { font-family: var(--mono); font-size: 18px; color: var(--accent); letter-spacing: 0.12em; text-transform: uppercase; }
  .logo span { color: var(--subtext); }

  /* Mode toggle */
  .mode-toggle { display: flex; gap: 0; border: 1px solid var(--border2); border-radius: 8px; overflow: hidden; margin-left: 8px; }
  .mode-btn {
    font-family: var(--mono); font-size: 14px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 6px 16px; cursor: pointer; border: none;
    background: transparent; color: var(--subtext); transition: all 0.2s;
  }
  .mode-btn.active { background: var(--accent); color: var(--bg); }
  .mode-btn:hover:not(.active) { color: var(--text); background: var(--border); }

  /* Vocab stats */
  .vocab-stats { margin-left: auto; display: flex; gap: 16px; align-items: center; }
  .stat { font-family: var(--mono); font-size: 14px; color: var(--subtext); }
  .stat span { color: var(--text); font-weight: 700; }
  .stat .new { color: var(--accent); }

  .header-statuses { display: flex; flex-direction: column; gap: 3px; align-items: flex-end; }
  .header-status { font-family: var(--mono); font-size: 14px; color: var(--subtext); }
  .header-status.learn-line { color: var(--teal); }
  .header-status.error { color: var(--red); }

  /* Sidebar */
  .sidebar {
    grid-row: 2; grid-column: 1;
    border-right: 1px solid var(--border);
    background: var(--surface);
    display: flex; flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--subtext);
    padding: 14px 14px 10px; border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .sidebar-list { overflow-y: auto; flex: 1; padding: 8px 0; }
  .sidebar-item {
    display: block; padding: 8px 14px; cursor: pointer;
    border-bottom: 1px solid var(--border);
    transition: background 0.1s;
  }
  .sidebar-item:last-child { border-bottom: none; }
  .sidebar-item:hover { background: rgba(123,175,212,0.06); }
  .sidebar-item.active { background: rgba(123,175,212,0.1); border-left: 2px solid var(--accent); padding-left: 12px; }
  .sidebar-jp {
    font-family: var(--jp); font-size: 22px; color: var(--accent);
    line-height: 1.6; display: block;
    white-space: normal; overflow-wrap: break-word;
  }
  .sidebar-time {
    font-family: var(--mono); font-size: 11px; color: var(--dim);
    display: block; margin-top: 2px;
  }
  .sidebar-empty { padding: 14px; font-family: var(--mono); font-size: 11px; color: var(--dim); }
  .sidebar-section-label {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--subtext);
    padding: 12px 14px 8px; border-top: 1px solid var(--border);
    flex-shrink: 0;
  }
  /* Metrics inside sidebar — no card border, just flat rows */
  .sidebar-metrics { padding: 0 14px 12px; display: flex; flex-direction: column; gap: 0; flex-shrink: 0; }

  /* Main panel */
  main { grid-row: 2; grid-column: 2; padding: 24px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; will-change: scroll-position; -webkit-overflow-scrolling: touch; }

  /* Translation card */
  .trans-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 28px; }
  .card-label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--subtext); margin-bottom: 10px; }

  /* Japanese with highlights */
  .japanese-wrap { font-family: var(--jp); font-size: 56px; font-weight: 400; line-height: 1.55; margin-bottom: 18px; min-height: 68px; white-space: normal; overflow-wrap: break-word; text-rendering: optimizeSpeed; }
  .japanese-wrap.placeholder { color: var(--dim); font-size: 24px; }
  .w-none     { color: var(--text); }
  .w-new      { color: var(--accent); }
  .w-learning { color: var(--yellow); }
  .w-familiar { color: var(--green); }

  .romaji { font-family: 'Nunito', sans-serif; font-size: 45px; font-weight: 400; color: #d4c9b8; margin-bottom: 20px; line-height: 1.45; white-space: normal; overflow-wrap: break-word; text-rendering: optimizeSpeed; }
  .divider { border: none; border-top: 1px solid var(--border); margin: 18px 0; }
  .english { font-family: 'Nunito', sans-serif; font-size: 56px; font-weight: 300; color: #d4c9b8; line-height: 1.45; text-rendering: optimizeSpeed; }
  .placeholder-text { color: var(--dim); font-size: 20px; }

  /* Legend */
  .legend { display: flex; gap: 16px; margin-top: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-family: var(--mono); font-size: 11px; color: var(--subtext); }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-new     { background: var(--accent); }
  .dot-learning { background: var(--yellow); }
  .dot-familiar { background: var(--green); }

  /* Metrics panel */
  .metric-row { display: flex; justify-content: space-between; align-items: baseline; padding: 5px 0; border-bottom: 1px solid var(--border); }
  .metric-row:last-child { border-bottom: none; }
  .metric-sub-row .metric-label { padding-left: 8px; color: var(--dim); }
  .metric-label { font-family: var(--mono); font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--subtext); }
  .metric-value { font-family: var(--mono); font-size: 16px; font-weight: 700; color: var(--accent); }
  .metric-value.green { color: var(--green); }

  /* Acknowledge bar */
  .ack-bar { display: none; align-items: center; gap: 12px; justify-content: center; padding: 4px 0; }
  .ack-bar.visible { display: flex; }
  .ack-btn {
    font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 10px 28px; cursor: pointer;
    border: 1px solid var(--accent); border-radius: 8px;
    background: rgba(123,175,212,0.1); color: var(--accent); transition: all 0.15s;
  }
  .ack-btn:hover { background: var(--accent); color: var(--bg); }
  .ack-hint { font-family: var(--mono); font-size: 11px; color: var(--dim); }

  /* Lesson panel */
  .lesson-panel { display: flex; flex-direction: column; gap: 14px; }
  .lesson-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }
  .lesson-card.hidden { display: none; }

  /* Breakdown table */
  .breakdown-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
  .breakdown-table th { font-family: var(--mono); font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--subtext); padding: 0 10px 12px 0; text-align: left; }
  .breakdown-table td { padding: 10px 10px 10px 0; font-size: 22px; border-top: 1px solid var(--border); vertical-align: middle; }
  .breakdown-table .jp { font-family: var(--jp); font-size: 32px; }
  .breakdown-table .reading { font-family: var(--jp); font-size: 22px; color: var(--text); }
  .breakdown-table .meaning { color: var(--text); font-size: 22px; }
  .breakdown-table .role { font-family: var(--sans); font-size: 22px; color: var(--text); }
  .w-bd-new      { color: var(--accent); }
  .w-bd-learning { color: var(--yellow); }
  .w-bd-familiar { color: var(--green); }

  /* Kanji grid */
  .kanji-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
  .kanji-card { background: var(--bg); border: 1px solid var(--border2); border-radius: 10px; padding: 14px 18px; min-width: 96px; text-align: center; }
  .kanji-card.k-new      { border-color: var(--accent); }
  .kanji-card.k-learning { border-color: var(--yellow); }
  .kanji-card.k-familiar { border-color: var(--green); }
  .kanji-char    { font-family: var(--jp); font-size: 52px; display: block; line-height: 1.2; }
  .kanji-reading { font-family: var(--jp); font-size: 20px; color: var(--text); margin-top: 6px; }
  .kanji-meaning { font-size: 20px; color: var(--text); margin-top: 4px; font-weight: 500; }
  .kanji-example { font-family: var(--sans); font-size: 20px; color: var(--text); margin-top: 4px; }

  /* Grammar note */
  .grammar-text { font-size: 20px; color: var(--subtext); line-height: 1.7; }

  /* Preview — floating bottom-right, only in TRANSLATE mode */
  .preview-float {
    position: fixed; bottom: 16px; right: 16px; z-index: 50;
    width: 380px;
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
    background: var(--surface);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    display: none;
  }
  .preview-float.visible { display: block; }
  .preview-float img { width: 100%; display: block; }

  /* Quiz overlay — fullscreen, quittable */
  .quiz-overlay {
    display: none; position: fixed; inset: 0; z-index: 200;
    background: rgba(8,11,16,0.96);
    align-items: center; justify-content: center;
  }
  .quiz-overlay.active { display: flex; }
  .quiz-box {
    background: var(--surface); border: 1px solid var(--border2);
    border-radius: 16px; padding: 40px 44px; width: 480px; max-width: 95vw;
    text-align: center; display: flex; flex-direction: column; gap: 0;
  }
  .quiz-header { font-family: var(--mono); font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--subtext); margin-bottom: 28px; }
  .quiz-progress-bar { height: 2px; background: var(--border); border-radius: 1px; margin-bottom: 32px; }
  .quiz-progress-fill { height: 100%; background: var(--accent); border-radius: 1px; transition: width 0.3s; }
  .quiz-type { font-family: var(--mono); font-size: 11px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px; }
  .quiz-prompt { font-family: var(--jp); font-size: 56px; line-height: 1.15; margin-bottom: 8px; color: var(--text); }
  .quiz-prompt.kanji { font-size: 72px; }
  .quiz-reveal-btn {
    margin-top: 28px; font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 11px 32px;
    border: 1px solid var(--border2); border-radius: 8px; cursor: pointer;
    background: var(--border); color: var(--text); transition: all 0.15s;
  }
  .quiz-reveal-btn:hover { border-color: var(--accent); color: var(--accent); }
  .quiz-answer { display: none; margin-top: 24px; flex-direction: column; gap: 6px; }
  .quiz-answer.visible { display: flex; }
  .quiz-answer-text { font-size: 20px; color: var(--text); font-weight: 300; }
  .quiz-answer-reading { font-family: var(--mono); font-size: 13px; color: var(--subtext); }
  .quiz-grade-btns { display: flex; gap: 12px; justify-content: center; margin-top: 20px; }
  .quiz-grade-btn {
    font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 10px 28px; border-radius: 8px; cursor: pointer; border: 1px solid;
    transition: all 0.15s;
  }
  .quiz-wrong { border-color: var(--red);   color: var(--red);   background: rgba(196,122,122,0.08); }
  .quiz-wrong:hover  { background: var(--red);   color: var(--bg); }
  .quiz-right { border-color: var(--green); color: var(--green); background: rgba(142,196,154,0.08); }
  .quiz-right:hover  { background: var(--green); color: var(--bg); }
  .quiz-score-line { font-family: var(--mono); font-size: 12px; color: var(--dim); margin-top: 16px; }
  /* Done screen */
  .quiz-done { display: none; flex-direction: column; align-items: center; gap: 14px; }
  .quiz-done.visible { display: flex; }
  .quiz-done-score { font-family: var(--mono); font-size: 36px; font-weight: 700; color: var(--accent); }
  .quiz-done-label { font-family: var(--mono); font-size: 12px; color: var(--subtext); letter-spacing: 0.1em; text-transform: uppercase; }
  .quiz-done-btn {
    margin-top: 8px; font-family: var(--mono); font-size: 11px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 11px 32px;
    border: 1px solid var(--green); border-radius: 8px; cursor: pointer;
    background: rgba(142,196,154,0.08); color: var(--green); transition: all 0.15s;
  }
  .quiz-done-btn:hover { background: var(--green); color: var(--bg); }
  .quiz-quit-btn {
    position: absolute; top: 16px; right: 20px;
    font-family: var(--mono); font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 6px 14px;
    border: 1px solid var(--border2); border-radius: 6px; cursor: pointer;
    background: none; color: var(--dim); transition: all 0.15s;
  }
  .quiz-quit-btn:hover { border-color: var(--red); color: var(--red); }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
</style>
</head>
<body>

<!-- Quiz overlay — blocks learn pipeline until finished or quit -->
<div class="quiz-overlay" id="quiz-overlay">
  <div class="quiz-box" style="position:relative">
    <button class="quiz-quit-btn" onclick="quizQuit()">✕ Quit</button>
    <div class="quiz-header" id="quiz-header">Lesson Review Quiz</div>
    <div class="quiz-progress-bar"><div class="quiz-progress-fill" id="quiz-fill" style="width:0%"></div></div>

    <!-- Card area -->
    <div id="quiz-card">
      <div class="quiz-type" id="quiz-type"></div>
      <div class="quiz-prompt" id="quiz-prompt"></div>
      <button class="quiz-reveal-btn" id="quiz-reveal-btn" onclick="quizReveal()">Show Answer</button>
      <div class="quiz-answer" id="quiz-answer">
        <div class="quiz-answer-text" id="quiz-answer-text"></div>
        <div class="quiz-answer-reading" id="quiz-answer-reading"></div>
        <div class="quiz-grade-btns">
          <button class="quiz-grade-btn quiz-wrong" onclick="quizGrade(false)">✗ Didn't know</button>
          <button class="quiz-grade-btn quiz-right" onclick="quizGrade(true)">✓ Got it</button>
        </div>
      </div>
      <div class="quiz-score-line" id="quiz-score-line"></div>
    </div>

    <!-- Done screen -->
    <div class="quiz-done" id="quiz-done">
      <div class="quiz-done-label">Quiz complete</div>
      <div class="quiz-done-score" id="quiz-done-score"></div>
      <div class="quiz-done-label" id="quiz-done-sub"></div>
      <button class="quiz-done-btn" onclick="quizFinish()">Resume Learning</button>
    </div>
  </div>
</div>

<header>
  <div class="logo">ZELDA <span>/</span> TRANSLATOR</div>
  <div class="mode-toggle">
    <button class="mode-btn active" id="btn-translate" onclick="setMode('TRANSLATE')">Translate</button>
    <button class="mode-btn" id="btn-learn" onclick="setMode('LEARN')">Learn</button>
  </div>
  <div class="vocab-stats" id="vocab-stats">
    <div class="stat">Words <span id="stat-words">0</span></div>
    <div class="stat">Kanji <span id="stat-kanji">0</span></div>
    <div class="stat">New today <span class="new" id="stat-new">0</span></div>
  </div>
  <div class="header-statuses">
    <div class="header-status" id="translate-status">Initializing...</div>
    <div class="header-status learn-line" id="learn-status">—</div>
  </div>
</header>

<!-- Lesson history sidebar -->
<div class="sidebar">
  <div class="sidebar-header">Last 10 Lessons</div>
  <div class="sidebar-list" id="sidebar-list">
    <div class="sidebar-empty">No lessons yet</div>
  </div>
  <div class="sidebar-section-label">Session</div>
  <div class="sidebar-metrics">
    <div class="metric-row">
      <span class="metric-label">Translate calls</span>
      <span class="metric-value" id="translate-calls">0</span>
    </div>
    <div class="metric-row metric-sub-row">
      <span class="metric-label">↳ ocr</span>
      <span class="metric-value green" id="translate-ocr-ms">—</span>
    </div>
    <div class="metric-row metric-sub-row">
      <span class="metric-label">↳ llm</span>
      <span class="metric-value green" id="translate-llm-ms">—</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Learn calls (ack'd)</span>
      <span class="metric-value" id="learn-calls">0</span>
    </div>
    <div class="metric-row metric-sub-row">
      <span class="metric-label">↳ ocr</span>
      <span class="metric-value green" id="learn-ocr-ms">—</span>
    </div>
    <div class="metric-row metric-sub-row">
      <span class="metric-label">↳ llm</span>
      <span class="metric-value green" id="learn-llm-ms">—</span>
    </div>
    <div class="metric-row">
      <span class="metric-label">Until quiz</span>
      <span class="metric-value" id="lessons-until-quiz">—</span>
    </div>
  </div>
</div>

<main>
  <!-- Translation card -->
  <div class="trans-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
      <div class="card-label" style="margin-bottom:0">Japanese</div>
      <button id="live-btn" onclick="returnToLive()" style="display:none;font-family:var(--mono);font-size:9px;color:var(--subtext);background:none;border:1px solid var(--border);border-radius:4px;padding:3px 8px;cursor:pointer;letter-spacing:0.08em">↩ back to live</button>
    </div>
    <div class="japanese-wrap placeholder" id="japanese-wrap">Waiting for dialogue...</div>
    <div class="romaji" id="romaji"></div>
    <hr class="divider">
    <div class="card-label">English</div>
    <div class="english placeholder-text" id="english">Translation will appear here</div>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot dot-new"></div>New</div>
      <div class="legend-item"><div class="legend-dot dot-learning"></div>Learning (1–4×)</div>
      <div class="legend-item"><div class="legend-dot dot-familiar"></div>Familiar (5+×)</div>
    </div>
  </div>

  <!-- Acknowledge bar — visible in LEARN mode when a lesson is pending -->
  <div class="ack-bar" id="ack-bar">
    <button class="ack-btn" onclick="acknowledge()">✓ Got it — next dialogue</button>
    <span class="ack-hint">saves vocab &amp; unlocks next lesson</span>
  </div>

  <!-- Lesson panel — only shown in LEARN mode -->
  <div class="lesson-panel" id="lesson-panel" style="display:none">

    <!-- Sentence breakdown -->
    <div class="lesson-card" id="breakdown-card">
      <div class="card-label">Word Breakdown</div>
      <table class="breakdown-table">
        <thead><tr><th>Word</th><th>Reading</th><th>Meaning</th><th>Role</th></tr></thead>
        <tbody id="breakdown-tbody"></tbody>
      </table>
    </div>

    <!-- Grammar note — only shown when non-empty -->
    <div class="lesson-card" id="grammar-card" style="display:none">
      <div class="card-label">Grammar Note</div>
      <div class="grammar-text" id="grammar-note"></div>
    </div>

    <!-- Kanji breakdown — only shown when kanji present -->
    <div class="lesson-card" id="kanji-card" style="display:none">
      <div class="card-label">Kanji in This Sentence</div>
      <div class="kanji-grid" id="kanji-grid"></div>
    </div>

  </div>


</main>

<!-- Floating preview — fixed bottom-right, only shown in TRANSLATE mode -->
<div class="preview-float" id="preview-float">
  <img src="/preview" alt="Live crop">
</div>

<script>
const QUIZ_EVERY   = {{ quiz_every }};
let currentMode    = 'TRANSLATE';
let quizData       = null;
let activeSidebarIdx = null;

function setMode(mode) {
  currentMode = mode;
  document.getElementById('btn-translate').classList.toggle('active', mode === 'TRANSLATE');
  document.getElementById('btn-learn').classList.toggle('active', mode === 'LEARN');
  document.getElementById('lesson-panel').style.display = mode === 'LEARN' ? 'flex' : 'none';
  document.getElementById('preview-float').classList.toggle('visible', mode === 'TRANSLATE');
  fetch('/set_mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode})});
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
let sidebarLessons = [];

async function loadSidebar() {
  try {
    const res = await fetch('/lessons');
    sidebarLessons = await res.json();
    const list = document.getElementById('sidebar-list');
    if (!sidebarLessons.length) {
      list.innerHTML = '<div class="sidebar-empty">No lessons yet</div>';
      return;
    }
    list.innerHTML = sidebarLessons.map((l, i) => `
      <div class="sidebar-item${activeSidebarIdx === i ? ' active' : ''}" onclick="showSidebarLesson(${i})">
        <span class="sidebar-jp">${l.japanese}</span>
        <span class="sidebar-time">${l.time.slice(11, 16)}</span>
      </div>`).join('');
  } catch(e) {}
}

function showSidebarLesson(idx) {
  activeSidebarIdx = idx;
  document.getElementById('live-btn').style.display = 'inline-block';
  // Update active state
  document.querySelectorAll('.sidebar-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });
  const l = sidebarLessons[idx];
  if (!l) return;

  // Populate trans-card with this lesson's data
  const jpWrap = document.getElementById('japanese-wrap');
  jpWrap.textContent = l.japanese;
  jpWrap.className   = 'japanese-wrap';

  document.getElementById('romaji').textContent = l.romaji || '';
  const enEl = document.getElementById('english');
  enEl.textContent = l.translation || '';
  enEl.className   = 'english';

  // If in LEARN mode, also populate lesson panel
  if (currentMode === 'LEARN') {
    renderLessonDetail(l);
  }
}

function renderLessonDetail(l) {
  document.getElementById('lesson-panel').style.display = 'flex';

  // Breakdown
  const tbody = document.getElementById('breakdown-tbody');
  if (l.breakdown && l.breakdown.length) {
    tbody.innerHTML = l.breakdown.map(item => {
      const fam = item.familiarity || 'none';
      return `<tr>
        <td class="jp"><span class="w-bd-${fam}">${item.word}</span></td>
        <td class="reading">${item.reading || ''}</td>
        <td class="meaning">${item.meaning || ''}</td>
        <td class="role">${item.role || ''}</td>
      </tr>`;
    }).join('');
  } else {
    tbody.innerHTML = '<tr><td colspan="4" style="color:var(--dim);font-size:12px;padding-top:8px">No breakdown available</td></tr>';
  }

  // Grammar
  const gcard = document.getElementById('grammar-card');
  const gnote = document.getElementById('grammar-note');
  if (l.grammar_note) {
    gcard.style.display = 'block';
    gnote.textContent   = l.grammar_note;
  } else {
    gcard.style.display = 'none';
  }

  // Kanji
  const kcard = document.getElementById('kanji-card');
  const kgrid = document.getElementById('kanji-grid');
  if (l.kanji && l.kanji.length) {
    kcard.style.display = 'block';
    kgrid.innerHTML = l.kanji.map(k => {
      const fam = k.familiarity || 'new';
      return `<div class="kanji-card k-${fam}">
        <span class="kanji-char">${k.kanji}</span>
        <div class="kanji-reading">${k.reading || ''}</div>
        <div class="kanji-meaning">${k.meaning || ''}</div>
        ${k.example ? `<div class="kanji-example">${k.example}</div>` : ''}
      </div>`;
    }).join('');
  } else {
    kcard.style.display = 'none';
  }
}

function getFamiliarity(entry) {
  if (!entry || typeof entry !== 'object') {
    const ts = typeof entry === 'number' ? entry : 0;
    return ts === 0 ? 'new' : ts < 5 ? 'learning' : 'familiar';
  }
  const correct = entry.correct_recalls || 0;
  const total   = entry.total_recalls   || 0;
  const ts      = entry.times_seen      || 0;
  if (total >= 3) {
    const r = correct / total;
    return r >= 0.8 ? 'familiar' : r >= 0.4 ? 'learning' : 'new';
  }
  return ts === 0 ? 'new' : ts < 5 ? 'learning' : 'familiar';
}

function renderJapanese(annotated) {
  if (!annotated || !annotated.length) return '';
  return annotated.map(a => `<span class="w-${a.familiarity}">${a.text}</span>`).join('');
}

function returnToLive() {
  activeSidebarIdx = null;
  document.getElementById('live-btn').style.display = 'none';
  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
}
async function acknowledge() {
  const res  = await fetch('/acknowledge', {method: 'POST'});
  const data = await res.json();
  activeSidebarIdx = null;  // clear sidebar selection so live lesson takes over
  await loadSidebar();
  if (data.quiz_triggered && data.quiz) {
    quizData = data.quiz;
    openQuiz();
  }
}

// ── Quiz ──────────────────────────────────────────────────────────────────────
function openQuiz() {
  document.getElementById('quiz-overlay').classList.add('active');
  document.getElementById('quiz-done').classList.remove('visible');
  document.getElementById('quiz-card').style.display = 'block';
  renderQuizCard();
}

function renderQuizCard() {
  if (!quizData || quizData.done) return;
  const card  = quizData.cards[quizData.index];
  const pct   = Math.round((quizData.index / quizData.total) * 100);

  document.getElementById('quiz-header').textContent =
    `Review Quiz — ${quizData.index + 1} of ${quizData.total}`;
  document.getElementById('quiz-fill').style.width = pct + '%';
  document.getElementById('quiz-type').textContent  =
    card.type === 'kanji' ? 'kanji — what does this mean?' : 'word — what does this mean?';

  const promptEl = document.getElementById('quiz-prompt');
  promptEl.textContent = card.prompt;
  promptEl.className   = 'quiz-prompt' + (card.type === 'kanji' ? ' kanji' : '');

  document.getElementById('quiz-answer').classList.remove('visible');
  document.getElementById('quiz-reveal-btn').style.display = 'inline-block';
  document.getElementById('quiz-answer-text').textContent    = card.answer;
  document.getElementById('quiz-answer-reading').textContent =
    card.reading ? `(${card.reading}${card.example ? ' · e.g. ' + card.example : ''})` : '';
  document.getElementById('quiz-score-line').textContent =
    `${quizData.correct} correct · ${quizData.incorrect} missed so far`;
}

function quizReveal() {
  document.getElementById('quiz-answer').classList.add('visible');
  document.getElementById('quiz-reveal-btn').style.display = 'none';
}

async function quizGrade(correct) {
  const res  = await fetch('/quiz_answer', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({correct})
  });
  const data = await res.json();
  if (!data.ok) return;
  quizData = data.quiz;

  if (quizData.done) {
    document.getElementById('quiz-card').style.display = 'none';
    document.getElementById('quiz-done').classList.add('visible');
    document.getElementById('quiz-fill').style.width = '100%';
    document.getElementById('quiz-done-score').textContent =
      `${quizData.correct} / ${quizData.total}`;
    document.getElementById('quiz-done-sub').textContent =
      `${quizData.incorrect} missed · vocab colors updated`;
  } else {
    renderQuizCard();
  }
}

function quizFinish() {
  document.getElementById('quiz-overlay').classList.remove('active');
  quizData = null;
}

async function quizQuit() {
  await fetch('/quiz_quit', {method: 'POST'});
  document.getElementById('quiz-overlay').classList.remove('active');
  quizData = null;
}

// ── Poll ──────────────────────────────────────────────────────────────────────
let sidebarRefreshCounter = 0;

async function poll() {
  try {
    const d = await (await fetch('/state')).json();

    // Translate status (grey)
    const ts = document.getElementById('translate-status');
    ts.textContent = d.error ? d.error : (d.status || '');
    ts.className = 'header-status' + (d.error ? ' error' : '');

    // Learn status (teal)
    const ls = document.getElementById('learn-status');
    ls.textContent = d.learn_status || '—';
    ls.className = 'header-status learn-line';

    // Vocab stats
    if (d.vocab_stats) {
      document.getElementById('stat-words').textContent = d.vocab_stats.total_words || 0;
      document.getElementById('stat-kanji').textContent = d.vocab_stats.total_kanji || 0;
      document.getElementById('stat-new').textContent   = d.vocab_stats.new_today   || 0;
    }

    // Only update main display if user isn't reviewing a sidebar lesson
    if (activeSidebarIdx === null) {
      const jpWrap = document.getElementById('japanese-wrap');

      if (currentMode === 'TRANSLATE') {
        // Translate tab: always shows live OCR text with highlights
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
        document.getElementById('romaji').textContent = d.translate_romaji || '';
        const enEl = document.getElementById('english');
        if (d.translate_translation) {
          enEl.textContent = d.translate_translation;
          enEl.className = 'english';
        } else {
          enEl.textContent = 'Translation will appear here';
          enEl.className = 'english placeholder-text';
        }
      } else {
        // Learn tab: shows lesson_japanese (frozen to lesson text until acknowledged)
        const lessonJp = d.lesson_japanese || '';
        if (lessonJp) {
          jpWrap.textContent = lessonJp;
          jpWrap.className = 'japanese-wrap';
        } else {
          jpWrap.textContent = 'Waiting for lesson...';
          jpWrap.className = 'japanese-wrap placeholder';
        }
        const learnRomaji = d.lesson ? d.lesson.romaji : '';
        const learnTrans  = d.lesson ? d.lesson.translation : '';
        document.getElementById('romaji').textContent = learnRomaji || '';
        const enEl = document.getElementById('english');
        if (learnTrans) {
          enEl.textContent = learnTrans;
          enEl.className = 'english';
        } else {
          enEl.textContent = 'Lesson will appear here';
          enEl.className = 'english placeholder-text';
        }
      }

      // Lesson panel (learn tab only)
      if (d.lesson && currentMode === 'LEARN') {
        renderLessonDetail(d.lesson);
      } else if (currentMode === 'LEARN') {
        document.getElementById('breakdown-tbody').innerHTML = '';
        document.getElementById('grammar-card').style.display = 'none';
        document.getElementById('kanji-card').style.display   = 'none';
      }
    }

    // Metrics
    document.getElementById('translate-calls').textContent  = d.translate_calls || 0;
    document.getElementById('translate-ocr-ms').textContent = d.translate_ocr_ms ? d.translate_ocr_ms + 'ms' : '—';
    document.getElementById('translate-llm-ms').textContent = d.translate_llm_ms ? d.translate_llm_ms + 'ms' : '—';
    document.getElementById('learn-calls').textContent      = d.learn_calls || 0;
    document.getElementById('learn-ocr-ms').textContent     = d.learn_ocr_ms ? d.learn_ocr_ms + 'ms' : '—';
    document.getElementById('learn-llm-ms').textContent     = d.learn_llm_ms ? d.learn_llm_ms + 'ms' : '—';
    const since  = d.lessons_since_quiz ?? 0;
    const until  = Math.max(0, QUIZ_EVERY - since);
    const untilEl = document.getElementById('lessons-until-quiz');
    untilEl.textContent = d.quiz_active ? 'quiz!' : until;
    untilEl.style.color = until <= 1 && !d.quiz_active ? 'var(--yellow)' : '';

    // Acknowledge bar — only in LEARN mode when lesson pending and no quiz active
    const ackBar = document.getElementById('ack-bar');
    if (currentMode === 'LEARN' && d.lesson_pending_ack && !d.quiz_active && activeSidebarIdx === null) {
      ackBar.classList.add('visible');
    } else {
      ackBar.classList.remove('visible');
    }

    // Re-open quiz overlay if quiz is active but overlay is closed (e.g. page refresh)
    if (d.quiz_active && d.quiz_data && !quizData) {
      quizData = d.quiz_data;
      openQuiz();
    }

    // Refresh sidebar every ~10s
    sidebarRefreshCounter++;
    if (sidebarRefreshCounter >= 20) {
      sidebarRefreshCounter = 0;
      await loadSidebar();
    }

  } catch(e) {}
  setTimeout(poll, 500);
}

// Initial sidebar load
// Show preview on initial load (default mode is TRANSLATE)
document.getElementById('preview-float').classList.add('visible');
loadSidebar();
poll();
</script>
</body>
</html>"""

@app.route('/')
def index():
    # quiz_every is injected into the HTML template so the JS 'Until quiz'
    # counter uses the same value as the Python trigger — change QUIZ_EVERY
    # once and both the backend logic and UI display stay in sync.
    return render_template_string(HTML, quiz_every=QUIZ_EVERY)

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = flask_request.get_json()
    if data and data.get('mode') in ('TRANSLATE', 'LEARN'):
        state['mode'] = data['mode']
        print(f"🔄  Mode switched to {state['mode']}")
    return jsonify({"mode": state["mode"]})

@app.route('/acknowledge', methods=['POST'])
def acknowledge():
    """Commit the pending lesson: update vocab, save lesson, increment counters, maybe trigger quiz."""
    if not state['lesson_pending_ack'] or not state['lesson']:
        return jsonify({"ok": False, "reason": "no pending lesson"})

    vocab = load_vocab()
    update_vocab(vocab, state['lesson'])

    # Save lesson to lessons.json for quiz building
    append_lesson(state['lesson_japanese'], state['lesson'])

    state['learn_calls']        += 1
    state['lessons_since_quiz']  = state.get('lessons_since_quiz', 0) + 1

    history_entry = {
        "time":        time.strftime("%H:%M:%S"),
        "japanese":    state['lesson_japanese'],
        "romaji":      state['lesson'].get("romaji", ""),
        "translation": state['lesson'].get("translation", ""),
    }
    push_history(history_entry)
    log_entry(state.get('brightness', 0), state['lesson_japanese'], state['lesson'].get("translation", ""))

    state['lesson_pending_ack'] = False
    state['lesson_japanese']    = ''

    # Trigger quiz every QUIZ_EVERY lessons.
    # Using the last QUIZ_EVERY lessons (not all-time) keeps quiz content
    # relevant to what was just seen in the current play session.
    # If build_quiz returns 0 cards (e.g. all lessons had empty breakdowns),
    # the counter resets without showing a quiz to avoid blocking the pipeline.
    quiz_triggered = False
    if state['lessons_since_quiz'] >= QUIZ_EVERY:
        lessons = load_lessons()
        recent  = lessons[-QUIZ_EVERY:]
        quiz    = build_quiz(recent)
        if quiz['total'] > 0:
            state['quiz_active']        = True
            state['quiz_data']          = quiz
            state['lessons_since_quiz'] = 0
            state['status']             = 'Listening...'
            state['learn_status']       = f"Quiz time! {quiz['total']} cards to review."
            quiz_triggered = True
            print(f"🧪  Quiz triggered — {quiz['total']} cards")
        else:
            state['lessons_since_quiz'] = 0
            state['status']       = 'Listening...'
            state['learn_status'] = 'Listening...'
    else:
        state['status']       = 'Listening...'
        state['learn_status'] = 'Listening...'

    print(f"✅  Lesson acknowledged — vocab updated, pipeline unlocked")
    return jsonify({"ok": True, "quiz_triggered": quiz_triggered, "quiz": state.get('quiz_data')})

@app.route('/quiz_state')
def quiz_state():
    return jsonify(state.get('quiz_data'))

@app.route('/quiz_answer', methods=['POST'])
def quiz_answer():
    """
    Grade a single quiz card and advance the quiz index.
    Correct/incorrect counts are stored both on the quiz object (for the
    end-of-quiz score screen) and on the individual vocab entry (for long-term
    familiarity calculation via get_familiarity). This means quiz performance
    directly influences the colour coding of words in the Japanese text display.
    When the last card is answered, quiz_active is cleared to unblock learn_loop.
    """
    data    = flask_request.get_json()
    correct = data.get('correct', False)

    quiz = state.get('quiz_data')
    if not quiz or quiz.get('done'):
        return jsonify({"error": "no active quiz"})

    card = quiz['cards'][quiz['index']]

    # Update recall stats in vocab
    vocab = load_vocab()
    key   = card['prompt']
    kind  = card['type']
    store = vocab['words'] if kind == 'word' else vocab['kanji']
    if key in store:
        store[key]['total_recalls']   = store[key].get('total_recalls',   0) + 1
        store[key]['correct_recalls'] = store[key].get('correct_recalls', 0) + (1 if correct else 0)
    save_vocab(vocab)

    # Advance quiz
    if correct:
        quiz['correct']   += 1
    else:
        quiz['incorrect'] += 1
    quiz['index'] += 1
    if quiz['index'] >= quiz['total']:
        quiz['done'] = True
        state['quiz_active']  = False
        state['status']       = 'Listening...'
        state['learn_status'] = 'Listening...'
        print(f"🏁  Quiz done — {quiz['correct']}/{quiz['total']} correct")

    state['quiz_data'] = quiz
    return jsonify({"ok": True, "quiz": quiz})

@app.route('/lessons')
def get_lessons():
    """Return the last 10 acknowledged lessons for the sidebar."""
    lessons = load_lessons()
    return jsonify(lessons[-10:][::-1])  # newest first

@app.route('/quiz_quit', methods=['POST'])
def quiz_quit():
    """
    Abandon the active quiz early and unlock the learn pipeline.
    Behaves similarly to completing the quiz — quiz_active is cleared so
    learn_loop can resume. lessons_since_quiz resets to 0 so the next quiz
    triggers fresh after another QUIZ_EVERY lessons rather than immediately.
    lesson_pending_ack is also cleared in case there was a lesson waiting
    behind the quiz, preventing a double-freeze on resume.
    """
    state['quiz_active']        = False
    state['quiz_data']          = None
    state['lessons_since_quiz'] = 0   # reset so next quiz triggers fresh after QUIZ_EVERY more lessons
    state['lesson_pending_ack'] = False
    state['lesson_japanese']    = ''
    state['status']             = 'Listening...'
    state['learn_status']       = 'Listening...'
    print("⏭  Quiz quit — learn pipeline unlocked")
    return jsonify({"ok": True})

@app.route('/preview')
def preview():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

def unload_model():
    """
    Tell Ollama to evict the model from RAM on exit (keep_alive=0).
    Without this the model stays loaded in RAM indefinitely after the script exits,
    consuming memory for other processes. Called in the finally block so it runs
    on both clean exit and KeyboardInterrupt.
    """
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
