"""
zelda_translator_paddle_ocr_furigana_box.py
============================================
Variant: PaddleOCR v5 mobile  |  Postprocessing: exact-string fixes + noise filter
Preprocessing: CC furigana removal (PIL, before upscaling)
"""
from PIL import Image
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import tempfile
import time
import zelda_core


# ── PaddleOCR initialisation ─────────────────────────────────────────────────
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

def paddle_ocr(frame):
    """Run PaddleOCR on a preprocessed BGR frame. Returns (japanese_text, elapsed_ms).
    Detections are sorted top-to-bottom by vertical centre before filtering so
    reading order is always preserved regardless of how Paddle returns boxes."""
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)

    try:
        result = _paddle_ocr.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_texts, all_scores, all_heights, all_centres = [], [], [], []
    for res in (result or []):
        polys  = res.get("rec_polys") or res.get("rec_boxes") or []
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

def remove_furigana_components(pil_image):
    """CC-based furigana removal with isolation guard.
    Removes small glyphs that are isolated above/below main-text lines (furigana)
    while preserving small kana (ゃ/っ/ょ) that sit on the same line as large chars.
    Runs on the raw binary image BEFORE upscaling for accurate component sizing."""
    arr     = np.array(pil_image.convert("L"))
    dark_bg = np.mean(arr) < 127
    binary  = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return pil_image
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    if not heights:
        return pil_image
    median_h           = float(np.median(heights))
    furigana_threshold = median_h * 0.55
    centres = [
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        for i in range(1, num_labels)
    ]
    large_indices = [
        idx for idx in range(len(centres))
        if stats[idx + 1, cv2.CC_STAT_HEIGHT] >= furigana_threshold
    ]
    out      = arr.copy()
    bg_value = 255 if not dark_bg else 0
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        if h >= furigana_threshold or w >= median_h * 2:
            continue
        cy = centres[i - 1]
        has_large_neighbour = any(
            abs(centres[j] - cy) < median_h * 1.5
            for j in large_indices
        )
        if not has_large_neighbour:
            out[labels == i] = bg_value
    return Image.fromarray(out)

def preprocess_crop(crop):
    """Zeldacc preset preprocessing.
    Pipeline: threshold-160 → CC furigana removal at original res → 2x Lanczos → black border.
    CC removal runs before upscaling so component sizes are accurate (no Lanczos halo artifacts).
    Matches the 'zeldacc' preset in japanese_ocr_compare.py exactly."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop.copy()
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    # CC furigana removal at original resolution — before upscaling
    pil_pre   = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    pil_clean = remove_furigana_components(pil_pre)
    result    = cv2.cvtColor(np.array(pil_clean), cv2.COLOR_RGB2BGR)
    h, w   = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result

# ── Bounds loading ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
