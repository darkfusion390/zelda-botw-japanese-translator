"""
zelda_translator_working_nlp_paddle_ocr.py
===========================================
Variant: PaddleOCR v3 mobile  |  Preprocessing: denoise + fixed-threshold + furigana suppression
"""
import paddleocr
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import tempfile
import time
import zelda_core

import paddleocr
from paddleocr import PaddleOCR

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
# PaddleOCR — Japanese OCR engine. More robust than manga-ocr on italic/bold
# game fonts like Zelda BotW's dialogue. use_angle_cls detects rotated text.
# use_gpu=False for CPU-only; set True if CUDA is available for faster inference.
_tagger   = fugashi.Tagger()
_kakasi   = pykakasi.kakasi()
# Jamdict wraps a SQLite connection which cannot be shared across threads.
# Use threading.local() so each thread gets its own Jamdict instance,
# created lazily on first use — avoids the "SQLite object created in a
# different thread" error when learn_loop calls lookup from a daemon thread.
_jmd_local = threading.local()

def _get_jmd():
    """Return this thread's Jamdict instance (thread-local, created on first call)."""
    if not hasattr(_jmd_local, 'jmd'):
        _jmd_local.jmd = Jamdict()
    return _jmd_local.jmd
_mocr = PaddleOCR(
    lang='japan',                   # Sets language
    ocr_version='PP-OCRv3',         # FORCE v3 Mobile (avoids v5 Server bloat)
    use_textline_orientation=True,  # New argument name (replaces use_angle_cls)
    device='cpu'                    # New argument name (replaces use_gpu=False)
)
print(f"🔍  PaddleOCR {paddleocr.__version__} initialised — check 'Creating model:' lines above for active model names")


def apple_vision_ocr(frame):
    """PaddleOCR — replaces manga-ocr.

    PaddleOCR handles Zelda BotW's bold-italic dialogue font more reliably
    than manga-ocr, which was trained primarily on standard upright manga fonts.

    PaddleOCR returns a nested list: result[line][[bbox, (text, confidence)]].
    We concatenate all detected text lines in top-to-bottom order (they come
    out ordered by bbox y-coordinate already) and join with a space.

    The function signature and return value are identical to the original so
    no call-sites need to change.
    """
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    try:
        result = _mocr.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    # predict() returns a list of dicts, one per image.
    # Each dict has a 'rec_texts' key with a list of recognised text strings.
    japanese = ""
    if result:
        for item in result:
            texts = item.get('rec_texts', []) if isinstance(item, dict) else []
            parts = [t for t in texts if t and t.strip()]
            if parts:
                japanese = " ".join(parts)
                break  # single image — only one result item expected

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

do_ocr = apple_vision_ocr

def preprocess_crop(crop):
    """Preprocessing pipeline for manga-ocr on Zelda's noisy dark-background dialogue.

    Why Otsu failed:
      Otsu picks a threshold based on the image histogram. Zelda's dark textured
      background has enough contrast variation that Otsu mistakes grain for foreground,
      producing noisy white splotches that change every frame — preventing stabilization
      and confusing the OCR model.

    Why a high fixed threshold works here:
      The game text is rendered as bright white (180-255) on a dark grey/black
      background (0-120). After denoising kills the grain, a threshold of ~185
      cleanly isolates only genuine text strokes. The key is denoising FIRST so the
      background is flat dark before the threshold is applied.

    Steps:
      1. Greyscale.
      2. Denoise (fastNlMeans, aggressive h=15) — flattens background grain to near-black
         so the fixed threshold doesn't pick it up.
      3. Fixed threshold at 185 — isolates only bright white text strokes.
      4. Morphological closing (small kernel) — reconnects any strokes slightly broken
         by denoising, improves character shape integrity for the model.
      5. Furigana row suppression.
      6. 2x upscale + padding.
      7. Back to BGR.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Step 2: aggressive denoise — flattens background texture to near-black
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # Step 3: fixed high threshold — only captures bright white text strokes
    _, mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    if mask.max() == 0:
        # Nothing bright enough — return padded greyscale as fallback
        h, w = gray.shape[:2]
        result = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
        return cv2.bitwise_not(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

    # Step 4: morphological closing — reconnects strokes slightly broken by denoising
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 5: furigana row suppression
    row_density = mask.sum(axis=1) / 255.0
    non_zero = row_density[row_density > 0]
    if len(non_zero) > 0:
        median_density     = float(np.median(non_zero))
        furigana_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furigana_threshold:
                mask[i, :] = 0

    # Step 6: upscale + sharpen + padding
    # 3x instead of 2x — manga-ocr needs larger input to resolve thin-stroke
    # distinctions (の vs ん, が vs か, 味 vs 咲, リンゴ vs ソフ etc).
    h, w = mask.shape[:2]
    result = cv2.resize(mask, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    # Unsharp mask after upscale — Lanczos softens thin strokes slightly;
    # this restores edge crispness so the model sees clean stroke boundaries.
    # blur = cv2.GaussianBlur(result, (0, 0), 3)
    # result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)

    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)

    # Step 7: back to BGR
    # Step 8: invert — manga-ocr was trained on black-on-white manga pages.
    # Zelda's UI is white-on-black. Feeding inverted polarity causes the model
    # to hallucinate structurally similar but wrong characters (食→もう, 火→水 etc).
    return cv2.bitwise_not(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

# ── Bounds loading ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
