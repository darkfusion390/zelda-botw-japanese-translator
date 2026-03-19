# Calamity Ganon's Captions

Real-time Japanese dialogue translator and vocabulary trainer for Nintendo Switch games. Points a phone camera at your TV, reads the dialogue box, and gives you a live translation plus a full word-by-word breakdown to help you learn Japanese as you play.

Fully local — no cloud APIs and dependence on external LLMs. Works with any video source — phone camera, capture card, or webcam.

![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey) ![LLM](https://img.shields.io/badge/LLM-Ollama%20qwen2.5%3A7b-green) ![License](https://img.shields.io/badge/License-MIT-blue)

> This project was built through extensive iteration and experimentation — trying different OCR engines, vision models, LLM sizes, preprocessing approaches, and architectural patterns before arriving at the current design. The repo reflects the final state across three milestone versions. The full development story is documented in the [accompanying blog post](#). *(link coming soon)*

---

## Features

- **Translate mode** — live romaji + English translation as dialogue appears (~2.5s)
- **Learn mode** — word-by-word breakdown with readings, meanings, grammatical roles, and kanji analysis
- **Vocabulary tracking** — words colour-coded by familiarity (new / learning / familiar) based on exposure and quiz performance
- **Review quizzes** — triggered every N lessons, randomly sampled from recent vocabulary
- **All local** — runs entirely on your Mac via Ollama, no data leaves your machine

---

## Requirements

- macOS (Apple Silicon — M1/M2/M3)
- [Ollama](https://ollama.com) installed
- Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app
- Phone and Mac on the same WiFi network
- Python 3.9+

---

## Versions

There are four versions of the script representing different stages of development. They all share the same setup steps — the only difference is which file you run and which dependencies you need.

### `zelda_translator_working_apple_OCR.py` — Translation only
The simplest version. OCR extracts Japanese text, LLM translates it to English. No learn mode, no vocab tracking, no quizzes. Good if you just want a fast live translation with minimal setup.

**Dependencies:**
```bash
pip install opencv-python numpy requests flask pyobjc-framework-Vision pyobjc-framework-Quartz
ollama pull qwen2.5:7b
```

---

### `zelda_translator_working_apple_OCR_learning.py` — Translation + Learn mode (LLM only)
Adds a full Learn tab alongside Translate. The LLM handles everything in Learn mode — romaji, word breakdown, grammatical roles, grammar notes, and kanji analysis in a single prompt. Vocab tracking and quizzes included.

Note: Learn mode in this version is slow (~17,000–27,000ms per lesson) because the LLM generates the full lesson output. This was the direct predecessor to the NLP version below.

**Dependencies:**
```bash
pip install opencv-python numpy requests flask pyobjc-framework-Vision pyobjc-framework-Quartz
ollama pull qwen2.5:7b
```

---

### `zelda_translator_working_nlp.py` — Translation + Learn mode (NLP hybrid) ⭐ Recommended
The final and most capable version. Learn mode is rebuilt using local NLP libraries for everything except translation — romaji, word segmentation, POS tagging, meanings, and kanji are all handled deterministically by fugashi/MeCab, pykakasi, and jamdict. The LLM is only called for the English translation (~15–20 output tokens vs ~400–600 previously), bringing Learn mode down to ~3,000ms.

Also includes all quiz improvements: random n/4 card sampling, quit button, familiarity-aware particle filtering, and a sidebar counter showing lessons until next quiz.

**Additional dependencies (on top of the base set):**
```bash
pip install opencv-python numpy requests flask pyobjc-framework-Vision pyobjc-framework-Quartz
pip install fugashi unidic-lite pykakasi jamdict jamdict-data
ollama pull qwen2.5:7b
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/zelda-translator.git
cd zelda-translator
```

**2. Calibrate the dialogue box crop (first time only)**

With your game running and a dialogue box visible on screen:
```bash
python3 calibrate.py
```
Draw a rectangle over the dialogue box. Coordinates are saved to `bounds.json` and reused on every subsequent run.

**3. Install dependencies for your chosen version**

See the version table above.

**4. Run the startup script (recommended)**
```bash
chmod +x start_zelda_translator.sh   # first time only
./start_zelda_translator.sh
```
The startup script checks all dependencies, installs anything missing, starts Ollama, pulls the model if needed, and launches `zelda_translator_working_nlp.py` by default. Edit the `TRANSLATOR_SCRIPT` variable at the top of the script to point to a different version.

**Or run directly:**
```bash
python3 zelda_translator_working_nlp.py
```

**5. Open the UI**

Navigate to `http://localhost:5002` in your browser.

---

## Usage

- **Translate tab** — always live. Shows romaji and English translation as dialogue appears.
- **Learn tab** — generates a full lesson for each dialogue line. Hit **Got it** to acknowledge, save vocab, and unlock the next lesson.
- Quizzes trigger automatically every N lessons. `QUIZ_EVERY` in the config controls the frequency.

---

## File Structure

```
zelda_translator_working_apple_OCR.py           # Translation only
zelda_translator_working_apple_OCR_learning.py  # Translation + Learn (LLM only)
zelda_translator_working_nlp.py                 # Translation + Learn (NLP hybrid) ⭐
zelda_translator_working_quizzes_update.py      # Translation + Learn + quiz improvements (LLM only)
calibrate.py                                    # One-time setup: draw crop bounds
start_zelda_translator.sh                       # Startup script
```

---

## Stack

| Component | Technology |
|---|---|
| Camera feed | Any MJPEG video source (phone, capture card, webcam) — tested with Android + IP Webcam app |
| OCR | Apple Vision framework |
| Word segmentation | fugashi (MeCab) |
| Romaji | pykakasi |
| Dictionary | jamdict (JMdict + Kanjidic) |
| Translation | qwen2.5:7b via Ollama |
| Web UI | Flask |

---

## License

MIT
