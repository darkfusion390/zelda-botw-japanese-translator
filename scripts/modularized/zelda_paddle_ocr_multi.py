"""
zelda_translator_paddle_ocr_base_postprocessing.py
===================================================
Variant: PaddleOCR v5 mobile  |  Postprocessing: exact-string fixes + noise filter
Preprocessing: row-density furigana suppression
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
from paddleocr import PaddleOCR
import cv2
import numpy as np
import time
import zelda_core


import threading

# ── PaddleOCR instance pool ───────────────────────────────────────────────────
# One PaddleOCR instance per region, keyed by region name.
# Built in __main__ after load_bounds() reveals the region names.
# Each instance is fully independent — no shared state, no lock needed.
# zelda_core's ThreadPoolExecutor can then run all regions truly in parallel.
_ocr_pool: dict = {}

def _build_ocr_pool(region_names: list):
    """Create one PaddleOCR instance per region name. Called once at startup.
    Model init takes ~2s per instance — this runs sequentially at launch so
    startup is slower, but all subsequent OCR calls run without serialisation."""
    pool = {}
    for name in region_names:
        print(f"🔧  Initialising OCR instance for region '{name}'...")
        pool[name] = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,  # prevents MKLDNN/PIR crash
        )
    return pool
# ── Postprocessing fixes ─────────────────────────────────────────────────────
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

def paddle_ocr(frame, region_name="default"):
    """Run PaddleOCR on a preprocessed BGR frame. Returns (japanese_text, elapsed_ms).
    Uses the instance assigned to region_name from _ocr_pool — each instance is
    independent so concurrent calls from zelda_core's ThreadPoolExecutor run
    without serialisation. Falls back to the first available instance if
    region_name is not in the pool (e.g. during testing outside zelda_core).

    paddleocr 3.x API:
      result = predict(img)  — accepts numpy array directly, no temp file needed.
      Passing numpy array avoids Windows file locking and thread race conditions.
      result is a list of dicts with rec_polys, rec_texts, rec_scores keys.
    """
    t0 = time.perf_counter()

    instance = _ocr_pool.get(region_name) or next(iter(_ocr_pool.values()))
    result = instance.predict(frame)

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

def preprocess_crop(crop):
    """Zelda preset preprocessing.
    Pipeline: threshold-160 → row-density furigana suppression → 2x Lanczos → black border.
    Matches the 'zelda' preset in japanese_ocr_compare.py exactly."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        # Nothing survived the threshold — crop is too dark to contain text.
        # Return a black frame at the upscaled size so downstream always gets
        # a consistent pure B&W image rather than a raw color frame.
        h, w = crop.shape[:2]
        return np.zeros((h * 2 + 40, w * 2 + 40, 3), dtype=np.uint8)
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    # Row-density furigana suppression — rows with fewer bright pixels than
    # 42% of the median non-zero row density are blanked out.
    row_density = mask.sum(axis=1) / 255.0
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density     = float(np.median(non_zero_densities))
        furigana_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furigana_threshold:
                result[i, :] = 0
    h, w   = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result

# ── Bounds loading ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Load region names before registering so the pool can be sized correctly.
    # load_bounds() exits with a clear error if bounds.json is missing.
    regions, _ = zelda_core.load_bounds()
    _ocr_pool.update(_build_ocr_pool(list(regions.keys())))
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
