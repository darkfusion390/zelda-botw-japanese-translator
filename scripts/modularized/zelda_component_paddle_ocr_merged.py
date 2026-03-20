"""
zelda_component_paddle_ocr_merged.py
=====================================
Variant: PaddleOCR v5 mobile  |  Postprocessing: exact-string fixes + noise filter
Preprocessing: denoise → threshold-175 → morph-close → row-density furigana suppression

Merges the best of two prior variants:
  base_postprocessing — PP-OCRv5 model, bimodal furigana filter in postprocessing,
                        row-density furigana suppression in preprocessing.
  working_nlp         — fastNlMeansDenoising before threshold to flatten background
                        grain, morphological closing to reconnect denoised strokes.

Why each step is here:
  Denoise first (h=15) — Zelda's dialogue crops have game-world texture bleeding
    through the semi-transparent dialogue bar. Without denoising, a threshold of
    160-185 picks up grain as foreground, producing frame-to-frame noise that
    confuses the stability gate. Denoising flattens that to near-black before
    the threshold runs.
  Threshold 175 — base used 160 (too low, picks up grain on textured backgrounds),
    working_nlp used 185 (slightly aggressive, clips thinner strokes on Zelda's
    bold-italic font). 175 sits cleanly between both failure modes.
  Morph close (2x2) — fastNlMeansDenoising occasionally breaks thin stroke
    junctions. A small closing kernel reconnects them before the furigana
    suppression runs so row densities are accurate.
  Row-density furigana suppression — rows with fewer bright pixels than 42% of
    the median non-zero row density are blanked. Removes furigana rows before
    the image reaches the model without needing bounding-box analysis.
  No inversion — PP-OCRv5 handles white-on-black natively. working_nlp inverted
    because PP-OCRv3 with lang='japan' was trained on black-on-white manga pages;
    that inversion is not needed here.
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
import cv2
import numpy as np
import tempfile
import time
import threading
import zelda_core

# ── PaddleOCR initialisation ──────────────────────────────────────────────────
# Loaded once at module level — model init takes ~2s, reusing avoids that cost
# on every OCR call.
_paddle_ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="cpu",
)
# PaddleOCR's predict() is not thread-safe on a shared model instance.
# zelda_core runs OCR concurrently across regions — this lock serialises
# all calls into _paddle_ocr so only one runs at a time.
_paddle_lock = threading.Lock()

# ── Postprocessing ────────────────────────────────────────────────────────────

_EXACT_FIXES = {
    # Add zero-false-positive exact-string substitutions here as they are
    # discovered. Each entry must be verified against Japanese vocabulary
    # before adding — a wrong fix here silently corrupts correct output.
    # e.g. "誤認パターン": "正しい文字列",
}

def _fix_exact(text: str) -> str:
    """Apply targeted exact-string substitutions from _EXACT_FIXES.
    Each rule is zero-false-positive — verified against Japanese vocabulary."""
    for wrong, correct in _EXACT_FIXES.items():
        text = text.replace(wrong, correct)
    return text

def _fix_hira_before_kata_N(text: str) -> str:
    """Convert hiragana immediately before katakana ン to katakana (+0x60).
    ン only exists in katakana so any hiragana before it is a misread.
    Fixes e.g. りンゴ → リンゴ."""
    result = list(text)
    for i in range(len(result) - 1):
        if 'ぁ' <= result[i] <= 'ん' and result[i + 1] == 'ン':
            result[i] = chr(ord(result[i]) + 0x60)
    return ''.join(result)

def _postprocess_paddle(pairs: list) -> list:
    """Apply fixes to (text, score) pairs. Drops noise lines (len ≤ 3).
    Scores stay in sync — dropped lines remove their score too."""
    out = []
    for t, s in pairs:
        t = _fix_exact(t)
        t = _fix_hira_before_kata_N(t)
        if len(t.strip()) > 3:
            out.append((t, s))
    return out

# ── OCR ───────────────────────────────────────────────────────────────────────

def paddle_ocr(frame):
    """Run PaddleOCR on a preprocessed BGR frame. Returns (japanese_text, elapsed_ms).
    Detections are sorted top-to-bottom by vertical centre before filtering so
    reading order is always preserved regardless of how Paddle returns boxes."""
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)

    try:
        with _paddle_lock:
            result = _paddle_ocr.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_texts, all_scores, all_heights, all_centres = [], [], [], []
    for res in (result or []):
        polys  = res.get("rec_polys") if res.get("rec_polys") is not None else (res.get("rec_boxes") or [])
        t_list = res.get("rec_texts") or []
        s_list = res.get("rec_scores") or []
        for poly, t, s in zip(polys, t_list, s_list):
            pts   = np.array(poly)
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            all_texts.append(t)
            all_scores.append(s)
            all_heights.append(y_max - y_min)
            all_centres.append((y_min + y_max) / 2.0)

    # Sort top-to-bottom by vertical centre
    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres = map(list, zip(*combined))

    if all_heights:
        # Bimodal gap split to find furigana/main-text height boundary
        sorted_h    = sorted(all_heights)
        furi_thresh = sorted_h[0]
        if len(sorted_h) >= 2:
            gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > sorted_h[-1] * 0.20:
                furi_thresh = sorted_h[gap_idx + 1]
        median_h      = float(np.median(all_heights))
        large_centres = [c for h, c in zip(all_heights, all_centres) if h >= furi_thresh]
        filtered = []
        for t, s, h, cy in zip(all_texts, all_scores, all_heights, all_centres):
            if h >= furi_thresh:
                filtered.append((t, s))
            elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                filtered.append((t, s))
        texts  = [t for t, _ in filtered]
        scores = [s for _, s in filtered]
    else:
        texts, scores = all_texts, all_scores

    # Post-processing: fixes + noise filter, scores kept in sync
    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    texts = [t for t, _ in filtered_pairs]

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    japanese = "\n".join(texts)
    return japanese, elapsed_ms

do_ocr = paddle_ocr

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop):
    """Merged preprocessing pipeline.
    Steps:
      1. Greyscale.
      2. Denoise (fastNlMeans h=15) — flattens background grain before threshold.
      3. Threshold at 175 — isolates bright white text, rejects denoised background.
      4. Morph close (2x2) — reconnects stroke junctions broken by denoising.
      5. Row-density furigana suppression — blanks rows below 42% of median density.
      6. 2x Lanczos upscale + 20px black border.
    No inversion — PP-OCRv5 handles white-on-black natively."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Step 2: denoise — eliminates background texture before thresholding
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # Step 3: threshold — 175 avoids grain pickup (160 too low) without
    # clipping Zelda's bold-italic strokes (185 slightly aggressive)
    _, mask = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)

    if mask.max() == 0:
        # Nothing survived the threshold — crop is too dark to contain text.
        # Return a black frame at the upscaled size so downstream always gets
        # a consistent pure B&W image rather than a raw colour frame.
        h, w = crop.shape[:2]
        return np.zeros((h * 2 + 40, w * 2 + 40, 3), dtype=np.uint8)

    # Step 4: morph close — reconnects strokes broken by denoising
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 5: row-density furigana suppression
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    row_density        = mask.sum(axis=1) / 255.0
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density     = float(np.median(non_zero_densities))
        furigana_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furigana_threshold:
                result[i, :] = 0

    # Step 6: 2x upscale + border
    h, w   = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
