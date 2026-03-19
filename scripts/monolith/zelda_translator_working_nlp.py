"""
zelda_translator.py
===================
Pipeline:
  Step 1 — OCR:         Apple Vision framework (macOS built-in, M1 neural engine)
  Step 2 — NLP:         fugashi (MeCab) for word segmentation + POS
                        pykakasi for romaji conversion
                        jamdict for kanji/word dictionary lookups
  Step 3 — LLM (lean):  qwen2.5:7b via Ollama — translation ONLY
                        Skipped entirely on cache hit (write-through JSON cache)

Two UI modes (toggle in header):
  TRANSLATE — Japanese (known words highlighted), romaji, English. Fast.
  LEARN     — Full lesson: word-by-word breakdown, grammar note, kanji.
              Breakdown/romaji/kanji built locally via NLP libs (fast + accurate).
              Only the English translation hits the LLM.

Vocab tracking: vocab.json saved next to this script.
  Words/kanji/grammar patterns tracked with familiarity levels:
    new      (0 seen)   → blue underline
    learning (1-9 seen) → yellow underline
    familiar (10+ seen) → green underline

UI: http://localhost:5002

Install deps:
  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
  pip install fugashi unidic-lite pykakasi jamdict
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

# ── NLP libraries (romaji / segmentation / dictionary) ────────────────────────
import fugashi
import pykakasi
from jamdict import Jamdict

# Initialise once at module load — these are expensive to spin up per-call.
# fugashi wraps MeCab, the standard Japanese morphological analyser. It segments
# sentences into tokens and provides POS tags, readings, and lemma forms.
# pykakasi converts Japanese (kana/kanji) to romaji using hepburn romanisation.
# Jamdict wraps JMdict, the standard Japanese-English dictionary database.
# jamdict-data must be installed separately: pip install jamdict-data
_tagger   = fugashi.Tagger()
_kakasi   = pykakasi.kakasi()
# Jamdict wraps a SQLite connection which cannot be shared across threads.
# Use threading.local() so each thread gets its own Jamdict instance,
# created lazily on first use — avoids the "SQLite object created in a
# different thread" error when learn_loop calls lookup from a daemon thread.
_jmd_local = threading.local()

def _get_jmd():
    if not hasattr(_jmd_local, 'jmd'):
        _jmd_local.jmd = Jamdict()
    return _jmd_local.jmd

# MeCab POS tag → human-readable role for beginners.
# MeCab returns part-of-speech in Japanese (e.g. 名詞 = noun). This dict
# translates them to English labels for display in the breakdown table.
# Any tag not in this dict falls through as the raw Japanese string.
_POS_LABELS = {
    "名詞":     "noun",
    "代名詞":   "pronoun",
    "動詞":     "verb",
    "形容詞":   "adjective",
    "形状詞":   "na-adjective",
    "副詞":     "adverb",
    "助詞":     "particle",
    "助動詞":   "aux. verb",
    "接続詞":   "conjunction",
    "感動詞":   "interjection",
    "接頭辞":   "prefix",
    "接尾辞":   "suffix",
    "連体詞":   "pre-noun adj.",
    "記号":     "symbol",
    "補助記号": "punctuation",
    "空白":     "whitespace",
    "フィラー": "filler",
    "その他":   "other",
}

# Single-character grammar particles and conjugation suffixes to skip for vocab
# tracking, quizzes, and meaning lookup. Jamdict returns wrong homophones for
# these — blanking them is cleaner than showing a wrong definition.
_SKIP_VOCAB = {
    # Topic / subject / object / direction particles
    "は", "が", "を", "に", "で", "と", "も", "や", "の", "へ",
    # Conjunctive / connective particles
    "て", "ね", "よ", "か", "な", "わ", "さ", "ぞ", "ぜ", "し",
    "ば", "ど",
    # Conjugation suffixes that aren't standalone vocabulary
    "た", "だ", "ん",
    # Sentence-ending / dialectal particles that jamdict mismatches
    "じゃ", "のう",
}

# ── Config ─────────────────────────────────────────────────────────────────────
# Ollama runs locally. The LLM is only used for English translation in this
# version — all linguistic analysis (romaji, breakdown, kanji) is handled by
# the NLP libraries above, which are faster and more accurate than the 7b model.
OLLAMA_URL        = "http://localhost:11434/api/generate"
TRANSLATION_MODEL = "qwen3:8b"
# VIDEO_SOURCE     = "http://192.168.1.107:8080/video"
VIDEO_SOURCE = 0  # OpenCV webcam capture device index (default 0)

GAME_NAME    = "zelda_botw_"
VOCAB_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), GAME_NAME + "vocab.json")
LESSONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), GAME_NAME + "lessons.json")
CACHE_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), GAME_NAME + "translation_cache.json")
PREVIEW_PATH = os.path.expanduser("~/Downloads/preprocessed_crop.jpg")
BOUNDS_FILE  = "bounds.json"
QUIZ_EVERY   = 10    # trigger a quiz after every N acknowledged lessons

# ── OCR training data collection ──────────────────────────────────────────────
# When enabled, saves the raw (pre-preprocessed) crop as a JPEG and appends a
# row to ocr_training_log.csv every time Gate 4 passes — i.e. once per unique
# stable dialogue line that will trigger an LLM call. One image per new line,
# never duplicates. Both image save and CSV append share this single toggle.
OCR_TRAINING_ENABLED = True
OCR_TRAINING_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_training_data")
OCR_TRAINING_CSV     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_training_log.csv")

# ── Brightness gate ───────────────────────────────────────────────────────────
BRIGHTNESS_GATE_HIGH = 80.0
BRIGHTNESS_GATE_LOW  = 10.0
BRIGHTNESS_ENABLED   = False

# ── Prompts ───────────────────────────────────────────────────────────────────
# TRANSLATE mode: JSON with translation only — romaji is handled by NLP libs
# so the LLM never needs to produce it. /no_think disables Qwen3's reasoning
# chain for faster response — without it Qwen3 thinks before replying.
# Few-shot examples sourced from zeldatranslationproject.wordpress.com —
# cover all four champion registers plus Zelda's archaic formal speech.
TRANSLATE_PROMPT = """
You are translating Legend of Zelda: Breath of the Wild dialogue from Japanese to English.

RULES:
- Read the register from the text and preserve it exactly:
  - Archaic grammar (-reshi, -ken, sonata/anata, 授けん) → formal, elevated English
  - Heavy ellipses (……) with short fragments → keep as fragments, do not complete them
  - Casual male markers (やれやれ, みてえ, 行くぜ, 相棒, ぜ/ぞ endings) → gruff, direct English
  - Sharp/clipped speech (じゃないよ, あんた, 言っとくけど) → terse, pointed English
  - Warm informal (しょうがない, もうちょっと) → gentle, conversational English
- Never translate proper nouns (character names, place names)
- 退魔の剣 → "Blade of Evil's Bane"
- 厄災ガノン → "Calamity Ganon"
- 神獣 → "Divine Beast"
- シーカーストーン → "Sheikah Stone"
- 英傑 → "Champions"
- 勇者 → "Hero"
- 赤き月 → "red moon"
- 御ひい様 → "Princess"

FEW-SHOT EXAMPLES:
Japanese: 貴方は このハイラルを再び照らす光…今こそ 旅立つ時です…
English: You are the light that will shine on Hyrule once more… Now is the time to depart on your journey.

Japanese: 地上をさまよう魔物達の魂が 再び肉体を取り戻してしまうのです……
English: The spirits of all the monsters that wander the earth will end up recovering their bodies.

Japanese: さらなる力が そなたと そして退魔の剣に宿らんことを……
English: May further power dwell in you, as well as in the Blade of Evil's Bane.

Japanese: やれやれ 前途多難みてえだな
English: Good grief, she's making it sound like we've got a lot of difficulties ahead.

Japanese: 言っとくけど 君の為じゃないよ？ 僕は ガノンに借りを返したいだけだからね！
English: Just to be clear, this isn't for you, you got that? I just want to repay my debt to Ganon!

Japanese: 御ひい様にとっちゃ あいつの存在は…… そう コンプレックスの象徴みたいなもんだから
English: For the princess, that guy's existence is… well, it's like seeing a symbol of her insecurities.

Japanese: 行くぜ 相棒！ さあ こいつを喰らいな ガノン！！
English: Let's go, buddy! Now, take this, Ganon!!

Respond ONLY with valid JSON, no markdown, no extra text:
{{"translation": "..."}}

Japanese: {japanese}"""

# LEARN mode: plain English only — no JSON, no romaji.
# Romaji, word breakdown, and kanji are built locally via fugashi/pykakasi/jamdict.
# learn_loop polls translate_loop's cache for this translation — this prompt is
# only hit as a fallback if the cache hasn't populated within TRANSLATION_POLL_TIMEOUT.
# /no_think keeps latency low while the NLP lesson is being built in parallel.
LEARN_TRANSLATE_PROMPT = """
You are translating Legend of Zelda: Breath of the Wild dialogue from Japanese to English.

RULES:
- Read the register from the text and preserve it exactly:
  - Archaic grammar (-reshi, -ken, sonata/anata, 授けん) → formal, elevated English
  - Heavy ellipses (……) with short fragments → keep as fragments, do not complete them
  - Casual male markers (やれやれ, みてえ, 行くぜ, 相棒, ぜ/ぞ endings) → gruff, direct English
  - Sharp/clipped speech (じゃないよ, あんた, 言っとくけど) → terse, pointed English
  - Warm informal (しょうがない, もうちょっと) → gentle, conversational English
- Never translate proper nouns (character names, place names)
- 退魔の剣 → "Blade of Evil's Bane"
- 厄災ガノン → "Calamity Ganon"
- 神獣 → "Divine Beast"
- シーカーストーン → "Sheikah Stone"
- 英傑 → "Champions"
- 勇者 → "Hero"
- 赤き月 → "red moon"
- 御ひい様 → "Princess"

FEW-SHOT EXAMPLES:
Japanese: 貴方は このハイラルを再び照らす光…今こそ 旅立つ時です…
English: You are the light that will shine on Hyrule once more… Now is the time to depart on your journey.

Japanese: 地上をさまよう魔物達の魂が 再び肉体を取り戻してしまうのです……
English: The spirits of all the monsters that wander the earth will end up recovering their bodies.

Japanese: さらなる力が そなたと そして退魔の剣に宿らんことを……
English: May further power dwell in you, as well as in the Blade of Evil's Bane.

Japanese: やれやれ 前途多難みてえだな
English: Good grief, she's making it sound like we've got a lot of difficulties ahead.

Japanese: 言っとくけど 君の為じゃないよ？ 僕は ガノンに借りを返したいだけだからね！
English: Just to be clear, this isn't for you, you got that? I just want to repay my debt to Ganon!

Japanese: 御ひい様にとっちゃ あいつの存在は…… そう コンプレックスの象徴みたいなもんだから
English: For the princess, that guy's existence is… well, it's like seeing a symbol of her insecurities.

Japanese: 行くぜ 相棒！ さあ こいつを喰らいな ガノン！！
English: Let's go, buddy! Now, take this, Ganon!!

Respond ONLY with the English translation, no extra text.

Japanese: {japanese}"""

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
    "translate_cached":    False,         # True = last translate was a cache hit
    "learn_ms":            0,             # updated immediately when learn call returns
    "learn_ocr_ms":        0,             # OCR portion, passed from latest_stable_jp
    "learn_nlp_ms":        0,             # NLP libs portion (fugashi/pykakasi/jamdict)
    "learn_poll_ms":       0,             # ms spent polling cache for translate_loop's result
    "learn_llm_ms":        0,             # LLM ms — only non-zero on poll timeout fallback
    "learn_cached":        False,         # True = last learn translation was a cache hit
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
    "_skip_learn_text":    "",             # if set, learn_loop skips this exact text once (used by quiz_quit)
}

latest_frame_jpg = None
frame_lock        = threading.Lock()
_ollama_lock      = threading.Lock()  # serialises translate + learn LLM calls — prevents
                                      # concurrent requests garbling Ollama responses
_file_lock        = threading.Lock()  # serialises all vocab.json + lessons.json reads/writes
                                      # prevents partial writes from concurrent Flask + learn_loop access
app = Flask(__name__)

# ── Vocab manager ─────────────────────────────────────────────────────────────

def load_vocab():
    with _file_lock:
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
    with _file_lock:
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
    with _file_lock:
        try:
            with open(LESSONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

def save_lessons(lessons):
    with _file_lock:
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
    Build a quiz from up to the last QUIZ_EVERY lessons.
    Cards are: word meaning recall + kanji meaning recall.
    Deduped, particles skipped, then randomly sampled to n/4 (minimum 1).
    """
    import random
    cards = []
    seen_words = set()
    seen_kanji = set()

    for lesson in recent_lessons:
        for item in lesson.get("breakdown", []):
            w = item.get("word", "")
            if w and w not in seen_words and w not in _SKIP_VOCAB and any('\u3040' <= c <= '\u9fff' for c in w):
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
        if w in _SKIP_VOCAB:
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
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)
    try:
        img_url = Quartz.CFURLCreateFromFileSystemRepresentation(
            None, tmp_path.encode(), len(tmp_path), False)
        src = Quartz.CGImageSourceCreateWithURL(img_url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

        raw_observations = []

        def handler(request, error):
            if error:
                return
            for obs in request.results():
                cand = obs.topCandidates_(1)
                if cand:
                    raw_observations.append((cand[0].string(), obs.boundingBox()))

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLanguages_(["ja"])
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)

        img_h = frame.shape[0]
        japanese = ""

        if raw_observations:
            # Convert Vision bounding boxes (origin bottom-left, y increases up)
            # to image coordinates (origin top-left, y increases down).
            candidates = []
            for text, bbox in raw_observations:
                px_h  = bbox.size.height * img_h
                top_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
                cy    = top_y + px_h / 2.0
                candidates.append((text, px_h, cy))

            # Bimodal gap split to find furigana/main-text height boundary
            sorted_h    = sorted(h for _, h, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i)
                        for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            # Isolation guard: keep small boxes that sit close to a main-text line
            median_h      = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

            texts = []
            for text, px_h, cy in candidates:
                if px_h >= furi_thresh:
                    texts.append(text)
                elif large_centres and any(
                        abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    texts.append(text)

            japanese = " ".join(texts).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
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

# ── Translation cache ─────────────────────────────────────────────────────────
# Write-through cache: in-memory dict for fast lookups during a session,
# backed by translation_cache.json on disk so hits persist across restarts.
#
# Key:   _cache_key(japanese) — strips spaces/punctuation, matching the same
#        normalisation used by the OCR dedup gates. Two strings the dedup gate
#        treats as identical map to the same cache key, preventing duplicate
#        entries from minor OCR noise.
#
# Value: three fields —
#   "japanese"    — original raw OCR string, preserved exactly as captured.
#                   Used for future purposes (training pairs, export) where
#                   you want the real source text, not the normalised key.
#   "translation" — English translation produced by the LLM.
#   "romaji"      — NLP-produced romaji (accurate, MeCab word-boundary aware).
#
# Thread safety:
#   _translate_cache dict: Python GIL makes individual dict reads/writes atomic.
#   _cache_write_lock: serialises background file writes so two concurrent misses
#     don't interleave JSON output and corrupt the file.
#   _cache_pending set: prevents two threads (translate_loop + learn_loop) both
#     getting a miss on the same new phrase and burning two LLM calls. The first
#     thread to claim the key calls the LLM; the second gets None from cache_get
#     and falls through to its own LLM call (benign — last cache_set wins).

_translate_cache: dict[str, dict] = {}
_cache_write_lock = threading.Lock()
_cache_pending:   set[str]        = set()   # keys currently being translated

def _cache_key(text: str) -> str:
    """Stable lookup key — same normalisation as normalize_for_dedup."""
    return re.sub(r'[\s、。・…]', '', text).strip()

def load_translation_cache():
    """Load cache from disk into memory at startup. Non-fatal if file missing."""
    global _translate_cache
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _translate_cache = json.load(f)
        print(f"🗃  Translation cache loaded: {len(_translate_cache)} entries")
    except FileNotFoundError:
        _translate_cache = {}
        print("🗃  Translation cache: no file yet, starting fresh")
    except Exception as e:
        _translate_cache = {}
        print(f"⚠️  Translation cache load failed: {e} — starting fresh")

def _persist_cache():
    """Write in-memory cache to disk. Always called in a background thread."""
    with _cache_write_lock:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_translate_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Cache write failed: {e}")

def cache_get(japanese: str):
    """Return (romaji, translation) on hit, or None on miss.
    Also returns None if key is pending (being translated by another thread)."""
    key = _cache_key(japanese)
    if key in _cache_pending:
        return None
    entry = _translate_cache.get(key)
    if entry:
        return entry.get("romaji", ""), entry.get("translation", "")
    return None

def cache_set(japanese: str, romaji: str, translation: str):
    """Store entry in memory, clear pending flag, schedule async disk write.
    'japanese' is the original raw OCR string — stored as-is for future use."""
    key = _cache_key(japanese)
    _translate_cache[key] = {
        "japanese":    japanese,       # original OCR output, not normalised
        "translation": translation,
        "romaji":      romaji,
    }
    _cache_pending.discard(key)
    threading.Thread(target=_persist_cache, daemon=True).start()

def cache_claim(japanese: str) -> bool:
    """Mark key as in-flight before an LLM call on a cache miss.
    Returns True if this thread claimed it, False if already claimed."""
    key = _cache_key(japanese)
    if key in _cache_pending:
        return False
    _cache_pending.add(key)
    return True

# ── LLM calls ─────────────────────────────────────────────────────────────────

def ollama_call(prompt):
    t0 = time.perf_counter()
    payload = {
        "model":  TRANSLATION_MODEL,
        "prompt": prompt,
        "stream": False,
        "think":  False,         # disables Qwen3 reasoning — output goes to "response" not "thinking"
        "options": {
            "num_ctx": 768,      # bumped from 256 — prompt + few-shots ~350 tokens, needs headroom
            "num_predict": 120,   # cap output length — translations are never that long
            "temperature": 0,
            "num_batch": 256,
        }
    }
    with _ollama_lock:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        response = r.json()["response"].strip()
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    state["llm_calls"] += 1
    return response, elapsed_ms

def call_translate(japanese, ocr_ms=0):
    """Fast path translation.
    Romaji is always NLP-produced (build_romaji_only) — never from the LLM.
    Cache checked before NLP so a hit skips both the LLM and romaji work entirely.
    On a hit the cached romaji is returned directly (already NLP-quality from when
    it was first stored). On a miss, build_romaji_only runs then LLM is called.
    cache_claim() marks the key in-flight so learn_loop won't burn a second LLM call.
    state['translate_cached'] is set so the UI can display a cache indicator."""
    state["translate_calls"] += 1

    # ── Cache hit — skip both NLP romaji and LLM entirely ────────────────────
    hit = cache_get(japanese)
    if hit:
        romaji, translation = hit
        if not romaji:
            # Pre-seeded or legacy entry has blank romaji — generate it now
            # and write back so subsequent hits are fully populated.
            romaji = build_romaji_only(japanese)
            cache_set(japanese, romaji, translation)
            print(f"🗃  Cache hit (translate, romaji backfilled): {japanese}")
        else:
            print(f"🗃  Cache hit (translate): {japanese}")
        state["translate_ms"]     = ocr_ms
        state["translate_ocr_ms"] = ocr_ms
        state["translate_llm_ms"] = 0
        state["translate_cached"] = True
        return romaji, translation, 0

    # ── Cache miss — NLP romaji then LLM ─────────────────────────────────────
    romaji = build_romaji_only(japanese)
    cache_claim(japanese)
    raw, elapsed_ms = ollama_call(TRANSLATE_PROMPT.format(japanese=japanese))
    state["translate_ms"]     = ocr_ms + elapsed_ms
    state["translate_ocr_ms"] = ocr_ms
    state["translate_llm_ms"] = elapsed_ms
    state["translate_cached"] = False

    # LLM still returns JSON — we only use the translation field, not its romaji
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean)
        translation = data.get("translation", raw)
    except Exception:
        translation = raw

    cache_set(japanese, romaji, translation)
    return romaji, translation, elapsed_ms

def _to_romaji(text):
    """
    Convert a Japanese string to hepburn romaji using pykakasi.
    Called per-token (not on the whole sentence) so MeCab word boundaries
    are preserved — produces 'wa sono mama' instead of 'hasonomama'.
    Fullwidth punctuation (？！。、) is normalised to ASCII equivalents so
    it doesn't appear as wide characters in the romaji display or cache.
    """
    result = _kakasi.convert(text)
    romaji = " ".join(item["hepburn"] for item in result if item["hepburn"]).strip()
    romaji = romaji.replace("？", "?").replace("！", "!").replace("。", ".").replace("、", ",")
    return romaji

def build_romaji_only(japanese: str) -> str:
    """
    Fast NLP-only romaji for TRANSLATE mode — no jamdict lookups, no breakdown.
    Uses MeCab for per-token kana readings then converts to hepburn romaji via
    pykakasi. Same quality as build_lesson_nlp romaji but skips the expensive
    meaning/kanji lookups that TRANSLATE mode doesn't display.
    Falls back to direct pykakasi conversion if MeCab fails.
    """
    try:
        parts = []
        for word in _tagger(japanese):
            surface = word.surface
            if not surface.strip():
                continue
            try:
                reading_kana = word.feature.kana or surface
            except AttributeError:
                reading_kana = surface
            r = _to_romaji(reading_kana)
            if r:
                parts.append(r)
        return " ".join(parts).strip()
    except Exception:
        return _to_romaji(japanese)

def _gloss_text(gloss_item):
    """Safely extract text from a jamdict gloss — handles str or object."""
    if isinstance(gloss_item, str):
        return gloss_item
    return getattr(gloss_item, "text", "") or ""

def _lookup_meaning(surface, reading_kana):
    """
    Look up the English meaning of a word via JMdict.
    Tries the surface form first (e.g. 食材), then falls back to the kana reading
    (e.g. しょくざい) if the surface returns nothing. Returns the first gloss from
    the first matching entry, truncated to 40 chars for UI readability.
    Words in _SKIP_VOCAB are never passed here — they get an empty meaning directly
    in build_lesson_nlp to avoid jamdict returning wrong homophones (e.g. は=feather).
    Errors are logged with the specific word so they're easy to diagnose.
    """
    def _first_gloss(result):
        if not result.entries:
            return ""
        for entry in result.entries:
            for sense in (entry.senses or []):
                for g in (sense.gloss or []):
                    text = _gloss_text(g)
                    if text:
                        return text[:40]
        return ""

    try:
        gloss = _first_gloss(_get_jmd().lookup(surface))
        if gloss:
            return gloss
        if reading_kana and reading_kana != surface:
            gloss = _first_gloss(_get_jmd().lookup(reading_kana))
            if gloss:
                return gloss
    except Exception as e:
        print(f"⚠️  jamdict meaning lookup failed for '{surface}': {e}")
    return ""

def _lookup_kanji(char):
    """
    Look up a single kanji character in JMdict/Kanjidic via jamdict.
    Extracts: on'yomi or kun'yomi reading (converted to romaji), English meaning,
    and an example word from the dictionary entries for that character.
    Attribute names vary between jamdict versions so access is defensive throughout.
    Returns None if the character isn't found or an error occurs.
    """
    try:
        result = _get_jmd().lookup(char)
        if not result.chars:
            return None
        c = result.chars[0]

        # ── Reading ───────────────────────────────────────────────────────────
        reading = ""
        if c.rm_groups:
            for r in c.rm_groups[0].readings:
                # r_type varies by jamdict version — accept any on/kun reading
                r_type = getattr(r, "r_type", "") or ""
                if "on" in r_type.lower() or "kun" in r_type.lower():
                    reading = _to_romaji(r.value)
                    break
            if not reading and c.rm_groups[0].readings:
                # fallback: just use the first reading whatever the type
                reading = _to_romaji(c.rm_groups[0].readings[0].value)

        # ── Meaning ───────────────────────────────────────────────────────────
        meaning = ""
        if c.rm_groups:
            for m in c.rm_groups[0].meanings:
                # Accept meanings with no language tag OR explicitly English
                lang = getattr(m, "m_lang", None) or getattr(m, "lang", None) or ""
                if lang in ("", "en", None):
                    val = getattr(m, "value", "") or getattr(m, "text", "") or str(m)
                    if val:
                        meaning = val[:30]
                        break

        # ── Example word ──────────────────────────────────────────────────────
        example = ""
        if result.entries:
            entry = result.entries[0]
            if entry.kana_forms:
                example = entry.kana_forms[0].text
            elif entry.kanji_forms:
                example = entry.kanji_forms[0].text
            if not example:
                for e in result.entries[1:]:
                    if e.kana_forms:
                        example = e.kana_forms[0].text
                        break
                    elif e.kanji_forms:
                        example = e.kanji_forms[0].text
                        break

        return {
            "kanji":   char,
            "reading": reading,
            "meaning": meaning,
            "example": example,
        }
    except Exception as e:
        print(f"⚠️  jamdict kanji lookup failed for '{char}': {e}")
    return None

def build_lesson_nlp(japanese):
    """
    Build the full lesson breakdown using local NLP libraries — no LLM involved.
    Returns (romaji_str, breakdown_list, kanji_list).

    Process per token (MeCab segment):
      1. Get the kana reading from MeCab feature data (more reliable than pykakasi
         on the raw surface form for conjugated words).
      2. Convert kana reading to romaji via pykakasi. Collecting per-token romaji
         and joining with spaces gives clean word boundaries in the full sentence
         romaji (e.g. 'wa sono mama' not 'hasonomama').
      3. Map MeCab POS tag to English role label via _POS_LABELS.
      4. Look up meaning using the lemma (dictionary base form) rather than the
         surface form — e.g. あぶっ → あぶる so jamdict can find the entry.
         _SKIP_VOCAB words (particles, suffixes) skip the lookup entirely.
      5. Collect all kanji characters from the token for the kanji section.
    """
    # ── Word breakdown via MeCab ──────────────────────────────────────────────
    breakdown  = []
    seen_kanji = {}       # char → kanji dict, preserving order
    romaji_parts = []     # collect per-token romaji for full-sentence string

    for word in _tagger(japanese):
        surface  = word.surface
        if not surface.strip():
            continue

        # Reading from MeCab feature (index 7 in unidic), fall back to surface
        try:
            reading_kana = word.feature.kana or surface
        except AttributeError:
            reading_kana = surface

        reading_romaji = _to_romaji(reading_kana)

        # Accumulate per-token romaji for clean full-sentence string
        if reading_romaji:
            romaji_parts.append(reading_romaji)

        # POS from MeCab feature
        try:
            pos_ja = word.feature.pos1 or ""
        except AttributeError:
            pos_ja = ""
        role = _POS_LABELS.get(pos_ja, pos_ja)

        # Dictionary meaning — use lemma (dictionary form) so conjugated
        # forms like あぶっ resolve to あぶる instead of returning nothing.
        # Skip lookup entirely for grammar particles — jamdict returns wrong homophones.
        if surface in _SKIP_VOCAB:
            meaning = ""
        else:
            try:
                lemma = word.feature.lemma or surface
            except AttributeError:
                lemma = surface
            meaning = _lookup_meaning(lemma, reading_kana)
            if not meaning and lemma != surface:
                meaning = _lookup_meaning(surface, reading_kana)

        breakdown.append({
            "word":    surface,
            "reading": reading_romaji,
            "meaning": meaning,
            "role":    role,
        })

        # Collect kanji characters from this token
        for ch in surface:
            if '\u4e00' <= ch <= '\u9fff' and ch not in seen_kanji:
                k = _lookup_kanji(ch)
                if k:
                    seen_kanji[ch] = k

    kanji_list = list(seen_kanji.values())
    romaji = " ".join(romaji_parts)
    return romaji, breakdown, kanji_list

def call_learn(japanese, vocab, ocr_ms=0):
    """
    Build a full lesson for LEARN mode using the hybrid NLP + LLM approach.
    Step 1 (NLP, ~50-200ms): build_lesson_nlp handles romaji, word breakdown,
      and kanji entirely locally — deterministic, fast, accurate.
    Step 2 (translation): cache polled first. translate_loop owns LLM translation
      and will almost always have written the result to cache by the time NLP
      finishes (or be in-flight via _cache_pending). learn_loop polls the cache
      for up to TRANSLATION_POLL_TIMEOUT seconds rather than firing its own LLM
      call, eliminating the redundant duplicate translation. Falls back to its
      own LLM call if the cache hasn't populated within the timeout — keeps
      learn_loop resilient if translate_loop is behind or errors out.
    Metrics: nlp_ms, poll_ms (time spent waiting for cache), llm_ms (fallback only).
    """
    TRANSLATION_POLL_TIMEOUT  = 5.0    # seconds to wait for translate_loop's result
    TRANSLATION_POLL_INTERVAL = 0.05   # seconds between cache polls

    # Step 1: NLP-based analysis (instant, deterministic)
    t_nlp = time.perf_counter()
    romaji, breakdown, kanji_list = build_lesson_nlp(japanese)
    nlp_ms = round((time.perf_counter() - t_nlp) * 1000)

    # Step 2: poll cache for translate_loop's translation.
    # Bypass cache_get's pending check — we WANT to wait for an in-flight key.
    translation = None
    llm_ms      = 0
    t_poll      = time.perf_counter()
    deadline    = t_poll + TRANSLATION_POLL_TIMEOUT

    while time.perf_counter() < deadline:
        hit = _translate_cache.get(_cache_key(japanese))
        if hit:
            translation = hit.get("translation", "")
            poll_ms = round((time.perf_counter() - t_poll) * 1000)
            state["learn_cached"] = True
            print(f"🗃  Cache hit (learn, polled {poll_ms}ms): {japanese}")
            break
        time.sleep(TRANSLATION_POLL_INTERVAL)

    if translation is None:
        poll_ms = round((time.perf_counter() - t_poll) * 1000)
        print(f"⏱  Learn poll timeout ({poll_ms}ms) for: {japanese} — falling back to LLM")
        cache_claim(japanese)
        raw, llm_ms = ollama_call(LEARN_TRANSLATE_PROMPT.format(japanese=japanese))
        translation = raw.strip()
        state["learn_cached"] = False
        cache_set(japanese, romaji, translation)

    state["_pending_learn_ms"] = nlp_ms + poll_ms + llm_ms
    state["_pending_ocr_ms"]   = ocr_ms
    state["learn_nlp_ms"]      = nlp_ms
    state["learn_poll_ms"]     = poll_ms

    lesson = {
        "romaji":       romaji,
        "translation":  translation,
        "breakdown":    breakdown,
        "grammar_note": "",
        "kanji":        kanji_list,
    }
    print(f"📖  Lesson built — NLP {nlp_ms}ms + poll {poll_ms}ms + LLM {llm_ms}ms")
    return lesson, llm_ms

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

# ── OCR training data helpers ──────────────────────────────────────────────────

def _save_ocr_training_sample(raw_crop, japanese: str):
    """Save the raw (pre-preprocessed) crop as training_image_<timestamp>.jpg and
    append a row to ocr_training_log.csv.  Called in a background thread on every
    Gate 4 pass — one entry per unique dialogue line that triggers an LLM call.
    Both the image write and CSV append are controlled by OCR_TRAINING_ENABLED."""
    try:
        os.makedirs(OCR_TRAINING_DIR, exist_ok=True)
        ts        = time.strftime("%Y%m%d_%H%M%S")
        img_name  = f"training_image_{ts}.jpg"
        img_path  = os.path.join(OCR_TRAINING_DIR, img_name)
        cv2.imwrite(img_path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # Append row to shared CSV: image_name, ocr_text, source_file
        csv_exists = os.path.exists(OCR_TRAINING_CSV)
        with open(OCR_TRAINING_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(["image_name", "ocr_text", "source_file"])
            writer.writerow([img_name, japanese, os.path.basename(__file__)])
        print(f"📸  OCR training sample saved: {img_name}")
    except Exception as e:
        print(f"⚠️  OCR training save failed: {e}")

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
                # Save raw crop + OCR text for training data
                if OCR_TRAINING_ENABLED:
                    threading.Thread(
                        target=_save_ocr_training_sample,
                        args=(crop.copy(), jp),
                        daemon=True,
                    ).start()

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
            state["status"] = "Live"
            print(f"🔤  {jp} → {translation}")
        except Exception as e:
            print(f"❌  Translate error: {e}")
            state["error"] = str(e)
            time.sleep(1)

# ── Learn loop — fires once per new text, freezes until acknowledged ─────────────

def learn_loop():
    """
    Independently watches for new stable OCR text and generates lessons.
    Runs as a daemon thread alongside translate_loop — both consume from the
    same latest_stable_jp but operate independently so TRANSLATE tab is always
    live even while a lesson is being generated for LEARN tab.

    Freeze conditions (checked every 200ms):
      - lesson_pending_ack: a lesson is ready but the user hasn't acknowledged it.
        The pipeline won't generate the next lesson until the user clicks 'Got it'.
        This is intentional — forces the learner to actually engage with each lesson.
      - quiz_active: a review quiz is in progress. Blocked until quiz is done or quit.

    After unfreezing, last_learned is compared to latest text to avoid re-generating
    a lesson for a line that was already processed before the freeze.
    """
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

        # quiz_quit sets _skip_learn_text to the current on-screen text before
        # clearing quiz_active. This prevents the learn loop from immediately
        # re-generating a lesson for the same dialogue line that was on screen
        # when the user quit — the text hasn't changed so there's nothing new to learn.
        # The flag is consumed (cleared) after one skip so normal behaviour resumes.
        skip = state.get("_skip_learn_text", "")
        if skip and jp == skip:
            state["_skip_learn_text"] = ""
            last_learned = jp
            time.sleep(0.1)
            continue

        last_learned = jp
        try:
            state["learn_status"] = "Generating lesson..."
            vocab = load_vocab()
            lesson, llm_ms = call_learn(jp, vocab, ocr_ms)
            # Enrich each breakdown item and kanji with familiarity level from vocab.
            # This is done after call_learn (not inside it) so the lesson object
            # always reflects the current state of vocab at the moment it's displayed,
            # not the state when the LLM was called (which could differ slightly).
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
            state["learn_ocr_ms"]       = ocr_ms
            state["learn_nlp_ms"]       = state.get("learn_nlp_ms", 0)   # set inside call_learn
            state["learn_poll_ms"]      = state.get("learn_poll_ms", 0)  # set inside call_learn
            state["learn_llm_ms"]       = llm_ms
            state["learn_ms"]           = ocr_ms + state["learn_nlp_ms"] + state["learn_poll_ms"] + llm_ms
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
  .bd-dim td { opacity: 0.45; font-size: 0.85em; }

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
  .quiz-quit-btn {
    position: absolute; top: 16px; right: 20px;
    font-family: var(--mono); font-size: 10px; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; padding: 6px 14px;
    border: 1px solid var(--border2); border-radius: 6px; cursor: pointer;
    background: none; color: var(--dim); transition: all 0.15s;
  }
  .quiz-quit-btn:hover { border-color: var(--red); color: var(--red); }
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
      <span class="metric-label">↳ nlp</span>
      <span class="metric-value green" id="learn-nlp-ms">—</span>
    </div>
    <div class="metric-row metric-sub-row">
      <span class="metric-label">↳ poll</span>
      <span class="metric-value green" id="learn-poll-ms">—</span>
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

// Track the latest poll state so returnToLive can restore the current lesson
let _lastPollState = null;

function _liveBtnLabel() {
  if (_lastPollState && _lastPollState.lesson_pending_ack) return '↩ back to current lesson';
  return '↩ back to live';
}

function showSidebarLesson(idx) {
  activeSidebarIdx = idx;
  const liveBtn = document.getElementById('live-btn');
  liveBtn.style.display = 'inline-block';
  liveBtn.textContent = _liveBtnLabel();
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

  // Hide ack bar while browsing history — poll restores it on return to live
  document.getElementById('ack-bar').classList.remove('visible');

  // If in LEARN mode, also populate lesson panel
  if (currentMode === 'LEARN') {
    renderLessonDetail(l);
  }
}

function renderLessonDetail(l) {
  document.getElementById('lesson-panel').style.display = 'flex';

  // Breakdown
  const tbody = document.getElementById('breakdown-tbody');
  const SKIP_VOCAB = new Set(["は","が","を","に","で","と","も","や","の","へ","て","ね","よ","か","な","わ","さ","ぞ","ぜ","し","ば","ど","た","だ","ん","じゃ","のう"]);
  const HIDE_ROLES = new Set(["punctuation","symbol","whitespace"]);
  if (l.breakdown && l.breakdown.length) {
    tbody.innerHTML = l.breakdown
      .filter(item => !HIDE_ROLES.has(item.role))
      .map(item => {
        const fam    = item.familiarity || 'none';
        const dimCls = SKIP_VOCAB.has(item.word) ? ' bd-dim' : '';
        return `<tr class="${dimCls}">
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
  // poll() will fire within 500ms and restore the correct live/pending-lesson state
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
    _lastPollState = d;

    // If user is browsing a sidebar lesson, keep the back-button label current
    // (lesson_pending_ack may change while they're reading history)
    if (activeSidebarIdx !== null) {
      document.getElementById('live-btn').textContent = _liveBtnLabel();
    }

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
    document.getElementById('translate-llm-ms').textContent = d.translate_cached ? '⚡ cached' : (d.translate_llm_ms ? d.translate_llm_ms + 'ms' : '—');
    document.getElementById('learn-calls').textContent      = d.learn_calls || 0;
    document.getElementById('learn-ocr-ms').textContent     = d.learn_ocr_ms ? d.learn_ocr_ms + 'ms' : '—';
    document.getElementById('learn-nlp-ms').textContent     = d.learn_nlp_ms ? d.learn_nlp_ms + 'ms' : '—';
    document.getElementById('learn-poll-ms').textContent    = d.learn_poll_ms != null ? d.learn_poll_ms + 'ms' : '—';
    document.getElementById('learn-llm-ms').textContent     = d.learn_llm_ms ? d.learn_llm_ms + 'ms ⚠️' : '—';
    const since   = d.lessons_since_quiz ?? 0;
    const until   = Math.max(0, QUIZ_EVERY - since);
    const untilEl = document.getElementById('lessons-until-quiz');
    untilEl.textContent  = d.quiz_active ? 'quiz!' : until;
    untilEl.style.color  = until <= 1 && !d.quiz_active ? 'var(--yellow)' : '';

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

    state['lesson_pending_ack'] = False
    state['lesson_japanese']    = ''

    # Trigger quiz every QUIZ_EVERY lessons
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
    """Grade a quiz card, update vocab recall stats, advance quiz."""
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
    """Abandon the active quiz and unlock the learn pipeline.
    Behaves like an acknowledge but does NOT generate a new lesson
    if the current on-screen text is the same as the last lesson."""
    with latest_stable_lock:
        current_jp = latest_stable_jp["text"]
    state['_skip_learn_text']   = current_jp  # tell learn_loop to skip this text once
    state['quiz_active']        = False
    state['quiz_data']          = None
    state['lessons_since_quiz'] = 0
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
    print(f"🗃  Cache:   {CACHE_FILE}")
    print(f"🌐  UI:      http://localhost:5002")
    print("─" * 40)
    load_translation_cache()
    threading.Thread(target=capture_loop, daemon=True).start()
    try:
        app.run(host='0.0.0.0', port=5002, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        unload_model()
