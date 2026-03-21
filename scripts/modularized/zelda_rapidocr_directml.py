"""
zelda_rapidocr_directml.py
===========================
Variant: RapidOCR (ONNX Runtime + DirectML)  |  Postprocessing: exact-string fixes + noise filter
Preprocessing: row-density furigana suppression

Replaces PaddleOCR with RapidOCR which runs the same PP-OCR models via ONNX
Runtime. On Windows, DirectML is used as the execution provider, offloading
inference to the GPU (RX 6800 in this case) via DirectX 12 — no ROCm needed.

Install requirements (order matters — plain onnxruntime conflicts with directml):
    pip uninstall onnxruntime onnxruntime-directml -y
    pip install rapidocr
    pip install onnxruntime-directml

Verify GPU is being used before running the main script:
    python zelda_rapidocr_directml.py --check-gpu
"""
import sys
import traceback
import threading as _threading

# ── Debug: patch threading BEFORE any other imports so every thread started
# by zelda_core or rapidocr during import also gets the patch.
# BaseException catches SystemExit and KeyboardInterrupt, not just Exception.
# Remove this block once the crash is identified and fixed.
_original_run = _threading.Thread.run
def _patched_run(self):
    try:
        _original_run(self)
    except BaseException as e:
        print(f"\n💥  Exception in thread '{self.name}': {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise  # re-raise so the thread still dies naturally
_threading.Thread.run = _patched_run

import argparse
import cv2
import numpy as np
import time
import zelda_core

from rapidocr import EngineType, ModelType, RapidOCR

# ── RapidOCR instance pool ────────────────────────────────────────────────────
# One RapidOCR instance per region, keyed by region name.
# Built in __main__ after load_bounds() reveals the region names.
# Each instance is independent — each has its own ONNX sessions internally.
# DirectML does not support concurrent calls on the same session, but since
# each region gets its own instance, zelda_core's ThreadPoolExecutor is safe.
_ocr_pool: dict = {}

# ── DirectML concurrency control ──────────────────────────────────────────────
# DirectML crashes with os._exit() when multiple ONNX sessions submit GPU work
# concurrently. _dml_sem serialises GPU calls — value of 1 = fully serial.
# Try Semaphore(2) once confirmed working to allow 2 concurrent GPU calls.
_dml_sem: _threading.Semaphore = _threading.Semaphore(1)

def _build_ocr_pool(region_names: list):
    """Create one RapidOCR instance per region name. Called once at startup.
    use_dml=True routes inference to the GPU via DirectML (Windows only).
    use_cls=False skips the angle classifier — Zelda dialogue is always upright.
    log_level=critical suppresses RapidOCR's per-call INFO logs."""
    pool = {}
    for name in region_names:
        print(f"🔧  Initialising RapidOCR instance for region '{name}'...")
        pool[name] = RapidOCR(
            params={
                "Global.log_level":                 "critical",
                "EngineConfig.onnxruntime.use_dml": True,
                "Det.engine_type":                  EngineType.ONNXRUNTIME,
                "Det.model_type":                   ModelType.MOBILE,
                "Rec.engine_type":                  EngineType.ONNXRUNTIME,
                "Rec.model_type":                   ModelType.MOBILE,
            }
        )
    return pool

# ── Postprocessing fixes ──────────────────────────────────────────────────────
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

def _postprocess(pairs: list) -> list:
    """Apply fixes to (text, score) pairs. Drops noise lines (len ≤ 1).
    Threshold is 1 (not 3 as in PaddleOCR variant) because RapidOCR on
    preprocessed white-on-black images returns shorter per-detection strings —
    dialogue text like '早取' (2 chars) is legitimate and must not be dropped.
    Single-character noise is still filtered since Japanese particles and
    punctuation alone are not meaningful detections."""
    out = []
    for t, s in pairs:
        t = _fix_exact(t)
        t = _fix_hira_before_kata_N(t)
        if len(t.strip()) > 1:
            out.append((t, s))
    return out

def rapid_ocr(frame, region_name="default"):
    """Run RapidOCR on a preprocessed BGR frame. Returns (japanese_text, elapsed_ms).
    Uses the instance assigned to region_name from _ocr_pool.
    Falls back to the first available instance if region_name not in pool.

    RapidOCR returns a RapidOCROutput object with named attributes:
      result.boxes:  list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or None
      result.txts:   list of str or None
      result.scores: list of float or None
    All three are None together when no text is detected.
    """
    t0 = time.perf_counter()

    instance = _ocr_pool.get(region_name) or next(iter(_ocr_pool.values()))
    with _dml_sem:
        result = instance(frame, use_cls=False)

    # ── Debug: print raw result per region — remove once detection confirmed
    n_boxes = len(result.boxes) if result.boxes is not None else 0
    print(f"🔬  [{region_name}] boxes={n_boxes} txts={result.txts}")

    all_texts, all_scores, all_heights, all_centres = [], [], [], []

    # RapidOCROutput has named attributes: boxes, txts, scores
    # boxes:  list of [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] or None if no detections
    # txts:   list of str or None
    # scores: list of float or None
    boxes  = result.boxes
    txts   = result.txts
    scores = result.scores

    if boxes is not None and txts is not None and scores is not None:
        for box, text, score in zip(boxes, txts, scores):
            box   = np.array(box)
            y_min = float(box[:, 1].min())
            y_max = float(box[:, 1].max())
            all_texts.append(text)
            all_scores.append(float(score))
            all_heights.append(y_max - y_min)
            all_centres.append((y_min + y_max) / 2.0)

    # Sort top-to-bottom by vertical centre (RapidOCR usually returns in order
    # already, but sort explicitly to match PaddleOCR variant's guarantee)
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

    # Post-processing: fixes + noise filter
    filtered_pairs = _postprocess(list(zip(texts, scores)))
    texts = [t for t, _ in filtered_pairs]

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    japanese = "\n".join(texts)
    return japanese, elapsed_ms

do_ocr = rapid_ocr

def preprocess_crop(crop):
    """Zelda preset preprocessing.
    Pipeline: threshold-160 → row-density furigana suppression → 2x Lanczos → black border.
    Identical to the PaddleOCR variant — preprocessing is backend-agnostic."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
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

def check_gpu():
    """Verify DirectML is available as an ONNX Runtime execution provider.
    Run with: python zelda_rapidocr_directml.py --check-gpu
    Expected output when working: ['DmlExecutionProvider', 'CPUExecutionProvider']"""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    if "DmlExecutionProvider" in providers:
        print("✅  DirectML available — GPU inference active.")
    else:
        print("❌  DirectML not found — CPU only.")
        print("    Fix: pip uninstall onnxruntime -y && pip install onnxruntime-directml")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-gpu", action="store_true")
    args, _ = parser.parse_known_args()

    if args.check_gpu:
        check_gpu()
    else:
        # Load region names before registering so the pool can be sized correctly.
        # load_bounds() exits with a clear error if bounds.json is missing.
        regions, _ = zelda_core.load_bounds()
        _ocr_pool.update(_build_ocr_pool(list(regions.keys())))
        zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
        zelda_core.main()
