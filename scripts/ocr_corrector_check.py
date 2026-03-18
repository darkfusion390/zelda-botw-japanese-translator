import re
import requests
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

INPUT_FILE = "erroneous_phrases.txt"
OUTPUT_FILE = "validated_output.txt"
API_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

# SYSTEM_PROMPT = """You are given a sentence in Japanese that may contain errors.
# Output ONLY the corrected phrase with no explanation, no arrows, no formatting, no quotes.
# It may contain errors such as:
# - Small kana written as large kana (e.g. じゃ written as じや, った written as つた)
# - Kanji replaced by its furigana reading in hiragana or katakana
# - Visually similar character swaps (e.g. ポ→ボ, が→カ, 栄→柴)
# - Punctuation errors (e.g. 。replaced by , or ! replaced by 1 or I)
# - Spurious spaces inserted mid-sentence
# - Kanji and furigana written side by side redundantly

# Do not change the speech style or formality level of the original.
# Do not add or remove sentence meaning.
# If the phrase is already correct, output it unchanged."""

SYSTEM_PROMPT = """
You are translating Legend of Zelda: Breath of the Wild dialogue from Japanese to English.

TASK:
1. The sentence MAY contain OCR errors. Validate the sentence for correctness and only if it is incorrect fix the errors in the Japanese sentence. Otherwise leave the sentence as it is.
OCR ERRORS MAY INCLUDE:
- じや → じゃ
- つた → った
- カ/が swaps
- ポ/ボ swaps
- Kanji replaced by kana
- wrong punctuation
- extra spaces

2. Translate the now processed Japanese sentence/phrase into natural English.

RULES:
- "corrected_japanese" must be valid Japanese with OCR errors fixed
- "translation_en" must be English only
- Never output Japanese inside translation_en
- If translation_en contains any Japanese characters, the answer is wrong

REGISTER RULES:
- Archaic grammar (-reshi, -ken, sonata/anata, 授けん) → formal, elevated English
- Heavy ellipses (……) → keep fragments
- Casual male speech → gruff English
- Sharp speech → terse English
- Warm speech → conversational English

DO NOT translate proper nouns:
退魔の剣 → Blade of Evil's Bane
厄災ガノン → Calamity Ganon
神獣 → Divine Beast
シーカーストーン → Sheikah Stone
英傑 → Champions
勇者 → Hero
赤き月 → red moon
御ひい様 → Princess

OUTPUT FORMAT (STRICT JSON):
{
  "fixed": "true" or "false",
  "corrected_japanese": "...",
  "translation_en": "..."
}

Return ONLY JSON. No explanation, no formatting, no quotes. One line only.
Japanese: {japanese}
"""

# SYSTEM_PROMPT = """
# You are translating Legend of Zelda: Breath of the Wild dialogue.

# STEP 1 — VALIDATE OCR
# You must first check if the Japanese sentence contains OCR errors.

# ONLY fix the sentence if there is a clear OCR mistake.

# OCR mistakes include ONLY:
# - じや → じゃ
# - つた → った
# - ポ ↔ ボ
# - カ ↔ が
# - wrong punctuation
# - kanji replaced by kana reading
# - broken spacing
# - visually wrong kanji (柴 instead of 栄 etc.)

# DO NOT change wording.
# DO NOT change grammar.
# DO NOT change kanji choice.
# DO NOT modernize.
# DO NOT rewrite.
# DO NOT paraphrase.

# If the sentence is valid Japanese, keep it EXACTLY the same.

# STEP 2 — TRANSLATE
# Translate the validated sentence to English.

# RULES
# - corrected_japanese must be identical to input if no OCR error
# - translation_en must contain only English
# - never output Japanese in translation_en

# OUTPUT JSON ONLY

# {
#   "fixed": "true" or "false",
#   "corrected_japanese": "...",
#   "translation_en": "..."
# }

# Japanese: {japanese}
# """

# OPTIONS = {
#     "temperature": 0.0,
#     "top_k": 1,
#     "top_p": 1.0,
#     "repeat_penalty": 1.0,
#     "num_gpu": 99,
#     "num_ctx": 512,
#     "num_predict": 200
# }

OPTIONS = {
    "temperature": 0,
    "top_k": 1,
    "top_p": 0,
    "repeat_penalty": 1.0,
    "num_ctx": 512,
    "num_predict": 200,
    "seed": 42
}

# Think mode regularly takes 7-11s per phrase; 60s gives real headroom
REQUEST_TIMEOUT = 60


def extract_content(message: dict) -> str:
    """
    When think=True, Ollama puts reasoning in message["thinking"] and the
    actual response in message["content"]. When think=False, only
    message["content"] exists and is already plain.

    This function cleans up content regardless of which mode is active.
    """
    raw = message.get("content", "").strip()

    # 1. Strip any <think> blocks that leaked into content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # 2. Strip LaTeX \boxed{...} — extract the inner text
    raw = re.sub(r"\\boxed\{([^}]*)\}", r"\1", raw).strip()

    # 3. Strip markdown bold/italic markers
    raw = re.sub(r"\*+", "", raw).strip()

    # 4. Single line — return it directly
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(lines) == 1:
        return lines[0].strip("「」『』\"'\u201c\u201d\u2018\u2019")

    # 5. Multi-line — pick the last line that looks like clean Japanese
    def is_clean_japanese(line: str) -> bool:
        # Must contain at least one Japanese character
        if not re.search(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]", line):
            return False
        # Reject markdown / commentary patterns
        if re.match(r"^(\*\*|##|[-*•]|\d+\.\s)", line):
            return False
        if any(tok in line for tok in ["→", "：", "**", "Explanation", "Corrected",
                                        "Answer", "Final", "\\", "http", "correction"]):
            return False
        # Reject lines that are mostly ASCII (English explanation text)
        ascii_ratio = sum(1 for c in line if ord(c) < 128) / max(len(line), 1)
        if ascii_ratio > 0.4:
            return False
        return True

    candidates = [l for l in lines if is_clean_japanese(l)]
    if candidates:
        return candidates[-1].strip("「」『』\"'\u201c\u201d\u2018\u2019")

    # Fallback — return everything cleaned up on one line
    return re.sub(r"\s+", " ", raw).strip("「」『』\"'\u201c\u201d\u2018\u2019")


def call_model(phrase: str) -> str:
    response = requests.post(API_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": phrase}
        ],
        "options": OPTIONS,
        "think": False,
        "stream": False
    }, timeout=REQUEST_TIMEOUT)
    message = response.json()["message"]
    return extract_content(message)

# tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

# def call_model(phrase: str) -> str:
#     inputs = tokenizer(phrase, return_tensors="pt")

#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
#         max_length=256
#     )

#     english = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

#     # Back to Japanese (correction effect)
#     inputs_back = tokenizer(english, return_tensors="pt")

#     translated_back = model.generate(
#         **inputs_back,
#         forced_bos_token_id=tokenizer.lang_code_to_id["jpn_Jpan"],
#         max_length=256
#     )

#     japanese = tokenizer.batch_decode(translated_back, skip_special_tokens=True)[0]

#     return japanese.strip()

def correct_phrase(phrase: str):
    start = time.time()
    try:
        corrected = call_model(phrase)
        elapsed = time.time() - start
        return corrected, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return phrase, elapsed


def warmup():
    print("Warming up model...")
    try:
        call_model("テスト")
        print("Model ready.\n")
    except Exception as e:
        print(f"Warmup failed — is Ollama running? Error: {e}\n")


def main():
    warmup()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    results = []
    total_start = time.time()

    print(f"Processing {len(lines)} phrases...\n")

    for i, line in enumerate(lines, 1):
        if ". " in line and line.split(". ")[0].isdigit():
            number, phrase = line.split(". ", 1)
        else:
            number = str(i)
            phrase = line

        corrected, elapsed = correct_phrase(phrase)
        elapsed_str = f"{elapsed:.2f}s"
        result_line = f"{number}. {corrected} | time: {elapsed_str}"
        results.append(result_line)

        changed = "✓ fixed" if corrected != phrase else "  ok"
        print(f"[{i:>3}/{len(lines)}] {changed} | {elapsed_str} | {corrected}")

    total_elapsed = time.time() - total_start
    avg = total_elapsed / len(lines) if lines else 0

    summary = (
        f"\n# ── Summary ──────────────────────────────\n"
        f"# Total phrases : {len(lines)}\n"
        f"# Total time    : {total_elapsed:.2f}s\n"
        f"# Average/phrase: {avg:.2f}s\n"
        f"# ──────────────────────────────────────────"
    )

    print(summary)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
        f.write("\n")
        f.write(summary)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
