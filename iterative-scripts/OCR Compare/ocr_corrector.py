import requests
import time

INPUT_FILE = "erroneous_phrases.txt"
OUTPUT_FILE = "validated_output.txt"
API_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "qwen3-1.7b"

SYSTEM_PROMPT = """You are a Japanese OCR correction assistant.
You will receive one Japanese phrase at a time.
It may contain OCR errors. Common error types:
- Small kana written as large kana: じゃ→じや, った→つた, ょ→よ, ッ→ツ
- Kanji swapped for furigana: 魔王→まおう, 扉→とびら, 絆→きずな
- Similar character swaps: ポ→ボ, が→カ, 栄→柴, ぞ→そ, だ→た
- Punctuation: 。→, or . or nothing, ！→! or 1 or I, ？→?
- Extra spaces inserted: 旅の 扉 → 旅の扉
- Kanji and furigana side by side: 栄えいよう→栄養

Respond with ONLY the corrected Japanese phrase.
No explanation. No quotes. No formatting. One line only.
Preserve the original casual/game speech style exactly."""

def call_model(phrase):
    response = requests.post(API_URL, json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": phrase}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
        "thinking": {"type": "disabled"}
    }, timeout=60)
    return response.json()["choices"][0]["message"]["content"].strip()

def correct_phrase(phrase):
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
        print(f"Warmup failed — is LM Studio server running? Error: {e}\n")

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
