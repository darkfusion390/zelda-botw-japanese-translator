"""
v7_tesseract_progressive_learning_jlpt.py  (was: zelda_translate.py)
=====================================================================
ITERATION 7 — Full progressive Japanese learning system with JLPT progression,
mastery tracking, and an adaptive prompt that responds to your struggles.

What it does:
  - OCR: back to Tesseract (jpn+jpn_vert) with Otsu threshold + sharpening kernel.
    This was a cross-platform compatibility step — Apple Vision is macOS-only.
  - LLM: qwen2.5:7b via Ollama with a rich adaptive LEARN prompt that includes
    the student's known vocabulary + struggle words so lessons personalise over time.

Memory system (zelda_memory.json):
  Full JLPT progression model:
    N5 → N4 → N3 → N2 → N1 based on unique words mastered (mastery = 5 sightings)
  Per-word tracking: times_seen, mastered (bool), struggle (bool), first/last seen
  Per-kanji tracking: times_seen, mastered
  Per-grammar-pattern tracking: times_seen, mastered
  "Struggle words" (seen 3+ times, not yet mastered) surface in the lesson with
  extra focus and a dedicated struggle note from the LLM.

UI features added:
  - JLPT level badge + progress bar toward next level
  - Vocabulary, kanji, grammar, mastered stats in header
  - Acknowledge button: user confirms they've read the lesson before it clears.
    Acknowledging increments vocab counts and unlocks the next dialogue.
  - Queue: if a new dialogue arrives while a lesson is locked, it waits and
    fires automatically after acknowledgement.
  - Translate / Learn mode toggle retained from v6.
  - /memory endpoint: JSON dump of full vocab/kanji/grammar store.
  - /forget/<word> endpoint: remove a word from memory (for corrections).

Run:  python3 v7_tesseract_progressive_learning_jlpt.py
Open: http://localhost:5001
"""

import cv2
import pytesseract
import requests
import time
import threading
import json
import os
import re
from flask import Flask, render_template_string, jsonify, request
from PIL import Image
import numpy as np
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434/api/generate"
MODEL             = "qwen2.5:7b"
CAPTURE_INTERVAL  = 3     # how often OCR runs (seconds)
SETTLE_WAIT       = 2     # seconds text must be stable before firing Ollama
MEMORY_FILE       = "zelda_memory.json"
MASTERY_THRESHOLD = 5
STRUGGLE_THRESHOLD = 3

# IP Webcam (Android app) — replace with your S21's IP shown in the app
VIDEO_SOURCE     = "http://192.168.1.107:8080/video"
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

state = {
    "mode":            "learn",
    "japanese":        "",
    "romaji":          "",
    "translation":     "",
    "lesson":          {},
    "vocab":           [],
    "kanji_breakdown": [],
    "grammar_patterns": [],
    "recalls":         [],
    "struggles":       [],
    "struggle_note":   "",
    "last_updated":    "",
    "processing":      False,
    "stats":           {},
    "locked":          False,   # True once a lesson loads — prevents new lesson replacing it
    "queued_japanese": "",      # next detected text waiting while locked
    "acknowledged":    False,   # True after user acknowledges current lesson
}
last_text = ""

# ── Memory ────────────────────────────────────────────────────────────────────

EMPTY_MEMORY = {
    "vocab":       {},
    "kanji":       {},
    "grammar":     {},
    "jlpt":        "N5",
    "jlpt_score":  0,
    "sessions":    0,
    "total_seen":  0,
}

JLPT_THRESHOLDS = {"N5": 50, "N4": 150, "N3": 350, "N2": 700, "N1": 9999}
JLPT_ORDER = ["N5", "N4", "N3", "N2", "N1"]

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            mem = json.load(f)
            for k, v in EMPTY_MEMORY.items():
                if k not in mem:
                    mem[k] = v
            return mem
    return dict(EMPTY_MEMORY)

def save_memory(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)

def bump_jlpt(mem):
    cur = mem["jlpt"]
    idx = JLPT_ORDER.index(cur)
    if idx < len(JLPT_ORDER) - 1:
        if mem["jlpt_score"] >= JLPT_THRESHOLDS[cur]:
            mem["jlpt"] = JLPT_ORDER[idx + 1]

def record_vocab(mem, items):
    recalls, struggles = [], []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    for item in items:
        w = item.get("word", "").strip()
        if not w:
            continue
        if w in mem["vocab"]:
            entry = mem["vocab"][w]
            entry["times_seen"] += 1
            entry["last_seen"] = now
            if not entry["mastered"]:
                entry["struggle"] = entry["times_seen"] >= STRUGGLE_THRESHOLD
                if entry["times_seen"] >= MASTERY_THRESHOLD:
                    entry["mastered"] = True
                    entry["struggle"] = False
                    mem["jlpt_score"] += 1
            if entry["mastered"]:
                recalls.append({**entry, "word": w, "status": "mastered"})
            elif entry["struggle"]:
                struggles.append({**entry, "word": w})
            else:
                recalls.append({**entry, "word": w, "status": "learning"})
        else:
            mem["vocab"][w] = {
                "reading":    item.get("reading", ""),
                "meaning":    item.get("meaning", ""),
                "times_seen": 1,
                "mastered":   False,
                "struggle":   False,
                "first_seen": now,
                "last_seen":  now,
            }
        for ch in w:
            if re.match(r'[\u4e00-\u9fff]', ch):
                if ch not in mem["kanji"]:
                    mem["kanji"][ch] = {"times_seen": 1, "mastered": False, "struggle": False}
                else:
                    mem["kanji"][ch]["times_seen"] += 1
                    if mem["kanji"][ch]["times_seen"] >= MASTERY_THRESHOLD:
                        mem["kanji"][ch]["mastered"] = True
    mem["total_seen"] += 1
    bump_jlpt(mem)
    return recalls, struggles

def record_grammar(mem, patterns):
    for p in patterns:
        k = p.get("key", p.get("pattern", "")).strip()
        if not k:
            continue
        if k not in mem["grammar"]:
            mem["grammar"][k] = {
                "pattern":     p.get("pattern", k),
                "explanation": p.get("explanation", ""),
                "times_seen":  1,
                "mastered":    False,
            }
        else:
            mem["grammar"][k]["times_seen"] += 1
            if mem["grammar"][k]["times_seen"] >= MASTERY_THRESHOLD:
                mem["grammar"][k]["mastered"] = True

def get_stats(mem):
    vocab   = mem["vocab"]
    kanji   = mem["kanji"]
    grammar = mem["grammar"]
    return {
        "jlpt":             mem["jlpt"],
        "score":            mem["jlpt_score"],
        "next_at":          JLPT_THRESHOLDS.get(mem["jlpt"], 9999),
        "vocab_total":      len(vocab),
        "vocab_mastered":   sum(1 for v in vocab.values() if v["mastered"]),
        "vocab_struggle":   sum(1 for v in vocab.values() if v["struggle"]),
        "kanji_total":      len(kanji),
        "kanji_mastered":   sum(1 for k in kanji.values() if k["mastered"]),
        "grammar_total":    len(grammar),
        "grammar_mastered": sum(1 for g in grammar.values() if g["mastered"]),
    }

# ── Image processing ──────────────────────────────────────────────────────────

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_japanese(frame):
    processed = preprocess_frame(frame)
    pil_img = Image.fromarray(processed)
    raw = pytesseract.image_to_string(pil_img, lang='jpn+jpn_vert', config='--psm 6')

    lines = []
    for l in raw.splitlines():
        l = l.strip()
        # Must contain actual Japanese characters
        if not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', l):
            continue
        # Skip lines that are mostly non-Japanese noise (symbols, dashes, slashes)
        noise_chars = re.findall(r'[^\u3040-\u30ff\u4e00-\u9fff\u0021-\u007e\s]', l)
        ja_chars    = re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', l)
        if len(ja_chars) < 2:
            continue
        lines.append(l)

    result = " ".join(lines).strip()

    # Reject if fewer than 4 real Japanese characters total — likely noise
    ja_total = re.findall(r'[\u3040-\u30ff\u4e00-\u9fff]', result)
    if len(ja_total) < 4:
        return ""

    # Reject if ratio of Japanese chars to total chars is too low (garbage mixed in)
    total_chars = len(result.replace(" ", ""))
    if total_chars > 0 and len(ja_total) / total_chars < 0.3:
        return ""

    return result

# ── Adaptive prompt ───────────────────────────────────────────────────────────

def build_prompt(japanese, mem, recalls, struggles):
    level = mem["jlpt"]
    known_grammar = [k for k, v in mem["grammar"].items() if v["mastered"]]

    if level == "N5":
        level_instruction = (
            "The learner is a COMPLETE BEGINNER — just started Japanese.\n"
            "- Explain every hiragana and katakana character seen (name, sound, example)\n"
            "- Explain every kanji with reading, meaning, and a visual memory tip\n"
            "- Always explain basic particles は が を に で as if first time seeing them\n"
            "- Use only simple English, zero jargon, zero assumptions"
        )
    elif level == "N4":
        level_instruction = (
            "The learner knows basic kana and N5 particles.\n"
            "- Skip re-explaining は が を unless used unusually\n"
            "- Focus on kanji readings, verb conjugation (て-form, ます, dictionary form)\n"
            "- Introduce compound words and how kanji combine to form new meanings"
        )
    elif level == "N3":
        level_instruction = (
            "The learner has solid N4 foundations.\n"
            "- Focus on nuanced grammar: なのに vs けど, ている vs てある, causative/passive\n"
            "- Challenge with kanji compound reading without spoon-feeding\n"
            "- Point out politeness levels (keigo) and register"
        )
    else:
        level_instruction = "The learner is intermediate-advanced. Focus on nuance, literary patterns, and register."

    struggle_hint = ""
    if struggles:
        struggle_hint = "\nSTRUGGLE WORDS — seen many times, not mastered. Give special memory tricks:\n"
        for s in struggles:
            struggle_hint += f"  • {s['word']} ({s.get('reading','')}) = {s.get('meaning','')} seen {s['times_seen']}×\n"

    recall_hint = ""
    if recalls:
        recall_hint = "\nWORDS ALREADY IN MEMORY — acknowledge briefly:\n"
        for r in recalls:
            tag = "mastered" if r.get("status") == "mastered" else "still learning"
            recall_hint += f"  • {r['word']} [{tag}] seen {r['times_seen']}×\n"

    known_note = ""
    if known_grammar:
        known_note = f"\nGrammar already mastered (skip re-explaining): {', '.join(known_grammar[:12])}\n"

    return f"""You are an expert, encouraging Japanese tutor. The learner is playing Zelda in Japanese. Current JLPT level: {level}.

{level_instruction}
{known_note}{struggle_hint}{recall_hint}

Japanese text on screen:
{japanese}

Output ONLY valid JSON. No markdown fences. No extra text before or after. Use this exact schema:

{{
  "romaji": "full sentence sounded out in romaji",
  "translation": "natural English translation",
  "vocab": [
    {{"word": "勇者", "reading": "yuusha", "meaning": "hero / brave warrior"}}
  ],
  "kanji_breakdown": [
    {{"char": "勇", "reading": "yuu", "meaning": "brave / courage", "memory_tip": "looks like a person flexing their muscles"}}
  ],
  "grammar_patterns": [
    {{"key": "wa-topic-marker", "pattern": "〜は", "explanation": "は marks the topic. Everything after は is said about that topic."}}
  ],
  "lesson": {{
    "remember_this": "Most important word/phrase + a fun mnemonic to remember it",
    "alphabet_note": "Interesting hiragana or katakana worth noting for a beginner. Empty string if N3+.",
    "pattern_watch": "One grammar/sentence structure pattern this sentence shows, with a simple rule",
    "try_it": "A fill-in-the-blank practice sentence using a pattern from this line",
    "try_it_answer": "The answer to the exercise above",
    "zelda_context": "Who is likely speaking, what Zelda moment this is, why it matters to the story"
  }},
  "struggle_note": "Special memory trick for any struggle words. Empty string if none."
}}"""

def call_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model":  MODEL,
        "prompt": prompt,
        "stream": False,
    }, timeout=90)
    raw = r.json()["response"].strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)

# ── Capture loop ──────────────────────────────────────────────────────────────

def capture_loop():
    global last_text, state
    memory = load_memory()
    memory["sessions"] += 1
    save_memory(memory)

    print(f"📡  Connecting to IP Webcam at {VIDEO_SOURCE} ...")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌  Cannot connect to IP Webcam.")
        print("    • Make sure IP Webcam app is running on your S21")
        print("    • Tap 'Start server' in the app")
        print(f"    • Update VIDEO_SOURCE in config (current: {VIDEO_SOURCE})")
        print("    • Both devices must be on the same WiFi network")
        return

    print(f"✅  Connected to S21 via IP Webcam")
    print(f"📚  Memory: {len(memory['vocab'])} vocab | {len(memory['kanji'])} kanji | JLPT {memory['jlpt']}")
    print("🌐  Open http://localhost:5001\n")

    last_change_time = 0   # tracks when text last changed

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Lost connection to IP Webcam — retrying in 3s...")
            time.sleep(3)
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            continue
        h, w = frame.shape[:2]
        tv = frame[int(h * 0.15):int(h * 0.80), int(w * 0.05):int(w * 0.95)]
        th = tv.shape[0]
        frame = tv[int(th * 0.63):, :]

        # Save preview frame — open ocr_preview.jpg in Finder to see what OCR is seeing (updates every loop)
        cv2.imwrite("ocr_preview.jpg", frame)

        japanese = extract_japanese(frame)

        if not japanese or state["processing"]:
            time.sleep(CAPTURE_INTERVAL)
            continue

        # Text changed — reset settle clock, update UI but don't fire Ollama yet
        if japanese != last_text:
            last_text         = japanese
            last_change_time  = time.time()
            state["japanese"] = japanese
            time.sleep(CAPTURE_INTERVAL)
            continue

        # Text unchanged — check if settled long enough
        if time.time() - last_change_time < SETTLE_WAIT:
            time.sleep(CAPTURE_INTERVAL)
            continue

        # Fully settled — check lock before firing
        last_change_time = float('inf')

        if state["locked"]:
            # Lesson is locked — queue this text, don't fire Ollama
            state["queued_japanese"] = japanese
            print(f"📋  Queued (locked): {japanese}")
            time.sleep(CAPTURE_INTERVAL)
            continue

        state["processing"] = True
        state["locked"]     = True
        state["acknowledged"] = False
        print(f"📺  Settled: {japanese}")

        def process(jp=japanese, mem=memory, mode=state["mode"]):
            try:
                if mode == "translate":
                    prompt = (
                        f"Translate this Japanese to English in one natural sentence.\n"
                        f"Reply with exactly two lines: line 1 = romaji, line 2 = English translation.\n"
                        f"Japanese: {jp}"
                    )
                    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=30)
                    lines = r.json()["response"].strip().splitlines()
                    state["romaji"]      = lines[0] if lines else ""
                    state["translation"] = lines[1] if len(lines) > 1 else lines[0] if lines else ""
                    state["lesson"]      = {}
                    state["vocab"]       = []
                    state["kanji_breakdown"] = []
                    state["grammar_patterns"] = []
                    state["recalls"]     = []
                    state["struggles"]   = []
                else:
                    known_pre    = [{"word": w, **mem["vocab"][w]} for w in mem["vocab"] if w in jp]
                    struggles_pre = [x for x in known_pre if x.get("struggle")]
                    prompt = build_prompt(jp, mem, known_pre, struggles_pre)
                    parsed = call_ollama(prompt)

                    vocab_items   = parsed.get("vocab", [])
                    grammar_items = parsed.get("grammar_patterns", [])
                    recalls, struggles = record_vocab(mem, vocab_items)
                    record_grammar(mem, grammar_items)
                    save_memory(mem)

                    state["romaji"]           = parsed.get("romaji", "")
                    state["translation"]      = parsed.get("translation", "")
                    state["lesson"]           = parsed.get("lesson", {})
                    state["vocab"]            = vocab_items
                    state["kanji_breakdown"]  = parsed.get("kanji_breakdown", [])
                    state["grammar_patterns"] = grammar_items
                    state["recalls"]          = recalls
                    state["struggles"]        = struggles
                    state["struggle_note"]    = parsed.get("struggle_note", "")
                    state["stats"]            = get_stats(mem)
                    print(f"✅  Done | JLPT {mem['jlpt']} | {len(mem['vocab'])} words")

                state["last_updated"] = datetime.now().strftime("%H:%M:%S")
            except Exception as e:
                print(f"❌  Error: {e}")
            finally:
                state["processing"] = False

        threading.Thread(target=process, daemon=True).start()
        time.sleep(CAPTURE_INTERVAL)

    cap.release()
    cv2.destroyAllWindows()

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>和訳 · Zelda Japanese Tutor</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+JP:wght@300;400;700&family=Sora:wght@300;400;600&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { margin:0; padding:0; box-sizing:border-box; }
:root {
  --ink:    #0b0b0d;
  --surf:   #111114;
  --gold:   #c9913d;
  --gdim:   rgba(201,145,61,0.13);
  --red:    #c0392b;
  --green:  #27ae60;
  --mist:   #60606a;
  --paper:  #ede8dc;
  --border: rgba(201,145,61,0.16);
  --panel:  rgba(255,255,255,0.025);
}
html, body { height:100%; overflow:hidden; background:var(--ink); color:var(--paper); font-family:'Sora',sans-serif; }

/* ══ GRID: 48px header | 25vh strip | rest lesson ══ */
.app { display:grid; grid-template-rows:48px 25vh 1fr; height:100vh; overflow:hidden; }

/* ── Header ── */
.hdr {
  display:flex; align-items:center; gap:14px; padding:0 20px;
  background:rgba(11,11,13,0.98); border-bottom:1px solid var(--border); z-index:50;
}
.logo { font-family:'Noto Serif JP',serif; font-size:17px; color:var(--gold); letter-spacing:2px; }
.logo-sub { font-size:8px; color:var(--mist); letter-spacing:3px; text-transform:uppercase; margin-top:1px; }
.spacer { flex:1; }

.stats { display:flex; gap:0; }
.sp {
  display:flex; flex-direction:column; align-items:center; padding:0 11px;
  border-right:1px solid rgba(255,255,255,0.05);
}
.sp:last-child { border-right:none; }
.sp-n { font-family:'DM Mono',monospace; font-size:16px; color:var(--paper); }
.sp-l { font-size:9px; color:var(--mist); letter-spacing:2px; text-transform:uppercase; margin-top:1px; }

.jlpt-wrap { display:flex; align-items:center; gap:7px; padding:0 14px; border-left:1px solid rgba(255,255,255,0.05); }
.jlpt-lbl { font-family:'DM Mono',monospace; font-size:12px; color:var(--gold); }
.jlpt-bar { width:54px; height:3px; background:rgba(255,255,255,0.07); border-radius:2px; overflow:hidden; }
.jlpt-fill { height:100%; background:var(--gold); transition:width .7s ease; border-radius:2px; }

.toggle { display:flex; background:rgba(255,255,255,0.04); border:1px solid var(--border); border-radius:5px; overflow:hidden; margin-left:8px; }
.tbtn {
  padding:4px 13px; font-size:9px; font-family:'Sora',sans-serif;
  letter-spacing:2px; text-transform:uppercase;
  background:none; border:none; color:var(--mist); cursor:pointer; transition:all .18s;
}
.tbtn.on { background:var(--gold); color:#000; font-weight:600; }

.dot { width:7px; height:7px; border-radius:50%; background:#2a2a2a; transition:all .3s; margin-left:6px; flex-shrink:0; }
.dot.live     { background:var(--green); box-shadow:0 0 8px #27ae6066; animation:bl 2s infinite; }
.dot.thinking { background:var(--gold); box-shadow:0 0 10px var(--gold); animation:bl .4s infinite; }
@keyframes bl { 0%,100%{opacity:1} 50%{opacity:.18} }

/* ── Translation strip (top 25vh) ── */
.tstrip {
  display:grid; grid-template-columns:1fr 1fr; overflow:hidden;
  background:var(--surf); border-bottom:2px solid var(--border);
}
.ts-half { padding:14px 22px; display:flex; flex-direction:column; justify-content:center; }
.ts-half:first-child { border-right:1px solid var(--border); }
.ts-lbl { font-size:8px; letter-spacing:3px; text-transform:uppercase; color:var(--mist); margin-bottom:7px; }
.ts-ja { font-family:'Noto Serif JP',serif; font-size:clamp(22px,3vw,38px); color:var(--gold); line-height:1.4; margin-bottom:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ts-ro { font-family:'DM Mono',monospace; font-size:clamp(14px,1.6vw,20px); color:var(--mist); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ts-en { font-size:clamp(17px,2.2vw,28px); font-weight:600; color:var(--paper); line-height:1.4; }
.tstrip-wait { grid-column:1/-1; display:flex; align-items:center; justify-content:center; gap:10px; font-size:10px; color:var(--mist); letter-spacing:2px; text-transform:uppercase; }

/* ── Lesson panel (bottom 75vh) ── */
.lpanel {
  overflow:hidden;
  display:grid;
  /* 3 rows × 2 cols, last row full width */
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr auto;
  gap:1px;
  background:var(--border);
}
.lpanel.gone { display:none; }

.lc { background:var(--ink); overflow-y:auto; padding:14px 18px; }
.lc.fw { grid-column:1/-1; }

/* Section title */
.stitle {
  font-size:11px; letter-spacing:3px; text-transform:uppercase; color:var(--gold);
  margin-bottom:9px; display:flex; align-items:center; gap:8px;
}
.stitle::after { content:''; flex:1; height:1px; background:var(--border); }

/* Vocab */
.vrow { display:flex; flex-wrap:wrap; gap:7px; }
.vc {
  background:var(--panel); border:1px solid var(--border); border-radius:7px;
  padding:8px 11px; min-width:110px; flex:1; max-width:170px; position:relative;
}
.vc-w { font-family:'Noto Serif JP',serif; font-size:26px; color:var(--gold); }
.vc-r { font-family:'DM Mono',monospace; font-size:14px; color:var(--mist); margin:2px 0 4px; }
.vc-m { font-size:15px; color:var(--paper); }
.bdg {
  position:absolute; top:5px; right:7px; font-size:7px; letter-spacing:1px;
  text-transform:uppercase; padding:1px 5px; border-radius:3px;
}
.bdg.M { background:rgba(39,174,96,.12); color:var(--green); border:1px solid rgba(39,174,96,.25); }
.bdg.S { background:rgba(192,57,43,.12); color:#e74c3c; border:1px solid rgba(192,57,43,.25); }
.bdg.L { background:var(--gdim); color:var(--gold); border:1px solid var(--border); }

/* Kanji */
.krow { display:flex; flex-wrap:wrap; gap:7px; }
.kc { background:var(--gdim); border:1px solid var(--border); border-radius:6px; padding:10px; text-align:center; min-width:78px; }
.kc-ch { font-family:'Noto Serif JP',serif; font-size:42px; color:var(--gold); }
.kc-rd { font-family:'DM Mono',monospace; font-size:13px; color:var(--mist); margin:2px 0; }
.kc-mn { font-size:14px; color:var(--paper); }
.kc-tp { font-size:12px; color:var(--mist); margin-top:4px; font-style:italic; }

/* Grammar */
.glist { display:flex; flex-direction:column; gap:8px; }
.gi { background:var(--panel); border-left:2px solid var(--gold); border-radius:0 6px 6px 0; padding:8px 11px; }
.gi-p { font-family:'Noto Serif JP',serif; font-size:19px; color:var(--gold); margin-bottom:3px; }
.gi-e { font-size:15px; color:#8a8070; line-height:1.7; }

/* Lesson box */
.lbox { display:flex; flex-direction:column; gap:10px; }
.li-lbl { font-size:11px; letter-spacing:2px; text-transform:uppercase; color:var(--gold); margin-bottom:4px; }
.li-txt { font-size:15px; line-height:1.85; color:#8a8070; }

.trybox { background:rgba(201,145,61,0.05); border:1px solid var(--border); border-radius:7px; padding:11px 14px; }
.try-q { font-size:16px; color:var(--paper); margin-bottom:7px; }
.try-a {
  font-size:14px; color:var(--mist); cursor:pointer;
  border-top:1px solid var(--border); padding-top:7px; transition:color .2s;
  user-select:none;
}
.try-a:hover { color:var(--gold); }
.try-a.shown { color:var(--green); cursor:default; }

/* Recall + struggle banners */
.recall-bar { background:rgba(39,174,96,.06); border:1px solid rgba(39,174,96,.18); border-radius:6px; padding:8px 12px; margin-bottom:9px; font-size:14px; color:#6ab88a; line-height:1.7; }
.struggle-bar { background:rgba(192,57,43,.07); border:1px solid rgba(192,57,43,.22); border-radius:6px; padding:9px 12px; }
.struggle-bar .stitle { color:#e74c3c; }
.struggle-bar .li-txt { color:#b09090; }

/* Waiting */
.wait { grid-column:1/-1; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:14px; }
.wait-k { font-family:'Noto Serif JP',serif; font-size:72px; color:rgba(201,145,61,0.09); animation:brth 3s ease-in-out infinite; }
@keyframes brth { 0%,100%{opacity:.09} 50%{opacity:.2} }
.wait-t { font-size:10px; color:var(--mist); letter-spacing:3px; text-transform:uppercase; }

::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-thumb { background:rgba(201,145,61,0.18); border-radius:3px; }

@keyframes fi { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:none} }
.fi { animation:fi .32s ease; }

/* ── Acknowledge button ── */
.ack-bar {
  grid-column:1/-1; display:flex; align-items:center; justify-content:space-between;
  padding:12px 22px; background:rgba(201,145,61,0.06);
  border-top:1px solid var(--border);
}
.ack-btn {
  padding:10px 28px; border-radius:7px; border:1px solid var(--gold);
  background:var(--gold); color:#000; font-family:'Sora',sans-serif;
  font-size:13px; font-weight:600; letter-spacing:2px; text-transform:uppercase;
  cursor:pointer; transition:all .2s;
}
.ack-btn:hover:not(:disabled) { background:#e0a84a; }
.ack-btn:disabled { opacity:0.35; cursor:default; background:transparent; color:var(--gold); }
.ack-status { font-size:11px; color:var(--mist); letter-spacing:1px; }
.ack-queue  { font-size:11px; color:var(--gold); letter-spacing:1px; }

/* ── Translate mode fullscreen strip ── */
.tstrip-full {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  height:100%; padding:40px 60px; gap:18px; text-align:center;
}
.tstrip-full .ts-ja { font-size:clamp(28px,4vw,52px); white-space:normal; }
.tstrip-full .ts-ro { font-size:clamp(16px,2vw,24px); }
.tstrip-full .ts-en { font-size:clamp(20px,2.8vw,36px); }
.tstrip-full .ts-lbl { font-size:12px; }
</style>
</head>
<body>
<div class="app">

  <!-- Header -->
  <header class="hdr">
    <div><div class="logo">和訳</div><div class="logo-sub">Zelda Tutor</div></div>

    <div class="stats">
      <div class="sp"><div class="sp-n" id="sv">0</div><div class="sp-l">vocab</div></div>
      <div class="sp"><div class="sp-n" id="sk">0</div><div class="sp-l">kanji</div></div>
      <div class="sp"><div class="sp-n" id="sg">0</div><div class="sp-l">grammar</div></div>
      <div class="sp"><div class="sp-n" id="sm">0</div><div class="sp-l">mastered</div></div>
    </div>

    <div class="jlpt-wrap">
      <div class="jlpt-lbl" id="jlpt-lbl">N5</div>
      <div class="jlpt-bar"><div class="jlpt-fill" id="jlpt-fill" style="width:0%"></div></div>
    </div>

    <div class="spacer"></div>

    <div class="toggle">
      <button class="tbtn on" id="btn-learn" onclick="setMode('learn')">Learn</button>
      <button class="tbtn" id="btn-tr" onclick="setMode('translate')">Translate</button>
    </div>
    <div class="dot" id="dot"></div>
  </header>

  <!-- Translation strip -->
  <div class="tstrip" id="tstrip">
    <div class="tstrip-wait">
      <div class="dot"></div>Waiting for text on screen...
    </div>
  </div>

  <!-- Lesson panel -->
  <div class="lpanel" id="lpanel">
    <div class="wait">
      <div class="wait-k">学</div>
      <div class="wait-t">Point your S21 at the TV</div>
    </div>
  </div>

</div>
<script>
let curMode = 'learn', lastJa = '', lastAck = false;

function setMode(m) {
  curMode = m;
  fetch('/set_mode', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({mode:m})});
  document.getElementById('btn-learn').classList.toggle('on', m==='learn');
  document.getElementById('btn-tr').classList.toggle('on', m==='translate');
  renderAll(window._lastState);
}

async function acknowledge() {
  const btn = document.getElementById('ack-btn');
  if (btn) btn.disabled = true;
  await fetch('/acknowledge', {method:'POST'});
}

function badge(word, recalls, struggles) {
  if (struggles.some(s=>s.word===word)) return '<span class="bdg S">struggle</span>';
  const r = recalls.find(r=>r.word===word);
  if (!r) return '';
  return r.status==='mastered'
    ? '<span class="bdg M">mastered</span>'
    : `<span class="bdg L">${r.times_seen}×</span>`;
}

function renderStrip(d) {
  if (!d || !d.japanese) return '<div class="tstrip-wait"><div class="dot"></div>Waiting for text on screen...</div>';
  if (curMode === 'translate') {
    // Fullscreen translate mode — big text, centered, no lesson
    return `<div class="tstrip-full">
      <div>
        <div class="ts-lbl">Japanese</div>
        <div class="ts-ja" style="font-family:'Noto Serif JP',serif;color:var(--gold)">${d.japanese}</div>
        <div class="ts-ro" style="font-family:'DM Mono',monospace;color:var(--mist);margin-top:8px">${d.romaji||''}</div>
      </div>
      <div>
        <div class="ts-lbl">Translation</div>
        <div class="ts-en" style="font-weight:600;color:var(--paper)">${d.translation}</div>
      </div>
    </div>`;
  }
  return `
    <div class="ts-half">
      <div class="ts-lbl">Japanese detected</div>
      <div class="ts-ja">${d.japanese}</div>
      <div class="ts-ro">${d.romaji||''}</div>
    </div>
    <div class="ts-half">
      <div class="ts-lbl">Translation</div>
      <div class="ts-en">${d.translation}</div>
    </div>`;
}

function renderLesson(d) {
  if (!d || !d.lesson || !Object.keys(d.lesson).length) return '';
  const rc = d.recalls||[], st = d.struggles||[];
  const L = d.lesson||{};

  const vocab = (d.vocab||[]).map(v=>`
    <div class="vc">${badge(v.word,rc,st)}
      <div class="vc-w">${v.word}</div>
      <div class="vc-r">${v.reading}</div>
      <div class="vc-m">${v.meaning}</div>
    </div>`).join('');

  const kanji = (d.kanji_breakdown||[]).map(k=>`
    <div class="kc">
      <div class="kc-ch">${k.char}</div>
      <div class="kc-rd">${k.reading}</div>
      <div class="kc-mn">${k.meaning}</div>
      ${k.memory_tip?`<div class="kc-tp">${k.memory_tip}</div>`:''}
    </div>`).join('');

  const grammar = (d.grammar_patterns||[]).map(g=>`
    <div class="gi">
      <div class="gi-p">${g.pattern}</div>
      <div class="gi-e">${g.explanation}</div>
    </div>`).join('');

  const mastered = rc.filter(r=>r.status==='mastered');
  const rcBanner = mastered.length
    ? `<div class="recall-bar">✓ Already know: ${mastered.map(r=>`<strong>${r.word}</strong>`).join(', ')} — nice recall!</div>` : '';
  const stBanner = d.struggle_note
    ? `<div class="struggle-bar"><div class="stitle">Words to focus on</div><div class="li-txt">${d.struggle_note}</div></div>` : '';

  const isAcked = d.acknowledged;
  const hasQueue = d.queued_japanese && d.queued_japanese.length > 0;
  const ackBar = `
    <div class="ack-bar">
      <div>
        ${isAcked
          ? '<span class="ack-status">✓ Acknowledged — playing will load the next lesson</span>'
          : '<span class="ack-status">Read the lesson, then acknowledge to save progress</span>'}
        ${hasQueue && !isAcked ? '<br><span class="ack-queue">⏳ Next dialogue waiting...</span>' : ''}
      </div>
      <button class="ack-btn" id="ack-btn" onclick="acknowledge()" ${isAcked?'disabled':''}>
        ${isAcked ? '✓ Done' : '確認 Acknowledge'}
      </button>
    </div>`;

  return `
    <div class="lc fi">
      <div class="stitle">Vocabulary</div>
      ${rcBanner}
      <div class="vrow">${vocab||'<span style="color:var(--mist)">Nothing extracted</span>'}</div>
    </div>
    <div class="lc fi">
      <div class="stitle">Kanji</div>
      <div class="krow">${kanji||'<span style="color:var(--mist)">No kanji in this line</span>'}</div>
    </div>
    <div class="lc fi">
      <div class="stitle">Grammar Patterns</div>
      <div class="glist">${grammar||'<span style="color:var(--mist)">No new patterns</span>'}</div>
    </div>
    <div class="lc fi">
      <div class="stitle">Mini Lesson</div>
      <div class="lbox">
        ${L.alphabet_note?`<div><div class="li-lbl">Alphabet note</div><div class="li-txt">${L.alphabet_note}</div></div>`:''}
        <div><div class="li-lbl">Remember this</div><div class="li-txt">${L.remember_this||''}</div></div>
        <div><div class="li-lbl">Pattern watch</div><div class="li-txt">${L.pattern_watch||''}</div></div>
        <div class="trybox">
          <div class="li-lbl">Try it</div>
          <div class="try-q">${L.try_it||''}</div>
          <div class="try-a" data-ans="${L.try_it_answer||''}" onclick="this.textContent=this.dataset.ans;this.classList.add('shown')">Tap to reveal answer ↓</div>
        </div>
        ${stBanner}
      </div>
    </div>
    <div class="lc fw fi">
      <div class="stitle">Zelda Story Context</div>
      <div class="li-txt">${L.zelda_context||''}</div>
    </div>
    ${ackBar}`;
}

function updateStats(s) {
  if (!s) return;
  document.getElementById('sv').textContent = s.vocab_total||0;
  document.getElementById('sk').textContent = s.kanji_total||0;
  document.getElementById('sg').textContent = s.grammar_total||0;
  document.getElementById('sm').textContent = s.vocab_mastered||0;
  document.getElementById('jlpt-lbl').textContent = s.jlpt||'N5';
  document.getElementById('jlpt-fill').style.width =
    Math.min(100, Math.round(((s.score||0)/(s.next_at||50))*100))+'%';
}

function renderAll(d) {
  if (!d) return;
  const tstrip = document.getElementById('tstrip');
  const lpanel = document.getElementById('lpanel');

  tstrip.innerHTML = renderStrip(d);

  if (curMode === 'translate') {
    // Translate mode: expand strip to fill all space, hide lesson
    tstrip.style.gridRow = '2 / 4';
    lpanel.style.display = 'none';
  } else {
    tstrip.style.gridRow = '';
    lpanel.style.display = '';
    if (d.lesson && Object.keys(d.lesson).length)
      lpanel.innerHTML = renderLesson(d);
  }
}

async function poll() {
  try {
    const d = await (await fetch('/state')).json();
    window._lastState = d;
    const dot = document.getElementById('dot');
    dot.className = 'dot'+(d.processing?' thinking':(d.japanese?' live':''));
    updateStats(d.stats);

    // Re-render strip always (acknowledge status can change without japanese changing)
    document.getElementById('tstrip').innerHTML = renderStrip(d);

    // Re-render lesson if text changed OR acknowledge status changed
    const ackChanged = d.acknowledged !== lastAck;
    lastAck = d.acknowledged;

    if (curMode === 'translate') {
      document.getElementById('tstrip').style.gridRow = '2 / 4';
      document.getElementById('lpanel').style.display = 'none';
    } else {
      document.getElementById('tstrip').style.gridRow = '';
      document.getElementById('lpanel').style.display = '';
      if ((d.japanese && d.japanese !== lastJa) || ackChanged) {
        if (d.lesson && Object.keys(d.lesson).length)
          document.getElementById('lpanel').innerHTML = renderLesson(d);
      }
    }

    if (d.japanese !== lastJa) lastJa = d.japanese;
  } catch(e) {}
  setTimeout(poll, 1500);
}
poll();
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/set_mode', methods=['POST'])
def set_mode():
    m = request.json.get('mode', 'learn')
    state['mode'] = m
    return jsonify({"ok": True, "mode": m})

@app.route('/memory')
def memory_view():
    mem = load_memory()
    return jsonify({"stats": get_stats(mem), "vocab": mem["vocab"],
                    "kanji": mem["kanji"], "grammar": mem["grammar"]})

@app.route('/acknowledge', methods=['POST'])
def acknowledge():
    """User acknowledged current lesson — increment vocab counts and unlock."""
    mem = load_memory()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    vocab_items = state.get("vocab", [])
    for item in vocab_items:
        w = item.get("word", "").strip()
        if not w or w not in mem["vocab"]:
            continue
        entry = mem["vocab"][w]
        entry["times_seen"] += 1
        entry["last_seen"]   = now
        if not entry["mastered"] and entry["times_seen"] >= MASTERY_THRESHOLD:
            entry["mastered"] = True
            entry["struggle"] = False
            mem["jlpt_score"] += 1
        elif not entry["mastered"] and entry["times_seen"] >= STRUGGLE_THRESHOLD:
            entry["struggle"] = True
        bump_jlpt(mem)
    save_memory(mem)
    state["stats"]       = get_stats(mem)
    state["acknowledged"] = True
    state["locked"]       = False   # unlock — allow next lesson to load

    # If something was queued while locked, fire it now
    queued = state.get("queued_japanese", "")
    state["queued_japanese"] = ""
    if queued:
        print(f"📬  Loading queued: {queued}")
        state["processing"]  = True
        state["locked"]      = True
        state["acknowledged"] = False
        state["japanese"]    = queued
        def fire_queued(jp=queued, mem=mem):
            try:
                known_pre     = [{"word": w, **mem["vocab"][w]} for w in mem["vocab"] if w in jp]
                struggles_pre = [x for x in known_pre if x.get("struggle")]
                prompt = build_prompt(jp, mem, known_pre, struggles_pre)
                parsed = call_ollama(prompt)
                vocab_items   = parsed.get("vocab", [])
                grammar_items = parsed.get("grammar_patterns", [])
                recalls, struggles = record_vocab(mem, vocab_items)
                record_grammar(mem, grammar_items)
                save_memory(mem)
                state["romaji"]           = parsed.get("romaji", "")
                state["translation"]      = parsed.get("translation", "")
                state["lesson"]           = parsed.get("lesson", {})
                state["vocab"]            = vocab_items
                state["kanji_breakdown"]  = parsed.get("kanji_breakdown", [])
                state["grammar_patterns"] = grammar_items
                state["recalls"]          = recalls
                state["struggles"]        = struggles
                state["struggle_note"]    = parsed.get("struggle_note", "")
                state["stats"]            = get_stats(mem)
                state["last_updated"]     = datetime.now().strftime("%H:%M:%S")
                print(f"✅  Queued lesson done | JLPT {mem['jlpt']}")
            except Exception as e:
                print(f"❌  Queued error: {e}")
            finally:
                state["processing"] = False
        threading.Thread(target=fire_queued, daemon=True).start()

    return jsonify({"ok": True, "unlocked": True, "queued_fired": bool(queued)})

@app.route('/forget/<word>', methods=['DELETE'])
def forget_word(word):
    mem = load_memory()
    removed = []
    if word in mem["vocab"]:
        del mem["vocab"][word]
        removed.append(f"vocab:{word}")
    if word in mem["kanji"]:
        del mem["kanji"][word]
        removed.append(f"kanji:{word}")
    save_memory(mem)
    return jsonify({"removed": removed})

if __name__ == '__main__':
    print("🎮  Zelda Progressive Japanese Tutor")
    print(f"📱  S21 → IP Webcam @ {VIDEO_SOURCE}")
    print(f"💾  Memory → {os.path.abspath(MEMORY_FILE)}")
    print("─" * 52)
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5001, debug=False)
