"""
zelda_translator_working_av_furigana_box.py
============================================
Variant: Apple Vision OCR  |  Preprocessing: CC-based furigana removal (post-upscale)
"""
import Vision
import Quartz
import cv2
import numpy as np
import os
import tempfile
import time
import zelda_core


def apple_vision_ocr(frame):
    """Run Apple Vision OCR with a post-OCR isolation guard.
    Language correction is enabled (setUsesLanguageCorrection_ True) to help
    Vision read small kana correctly after CC furigana removal.
    Bounding-box heights are converted to pixel units and filtered with the
    same bimodal gap split used in the compare script.
    Returns (japanese_str, elapsed_ms)."""
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, frame)
    try:
        img_url = Quartz.CFURLCreateFromFileSystemRepresentation(
            None, tmp_path.encode(), len(tmp_path), False)
        src = Quartz.CGImageSourceCreateWithURL(img_url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)

        # Collect raw observations including bounding-box geometry for the
        # post-OCR isolation guard applied below.
        raw_observations = []
        def handler(request, error):
            if error: return
            for obs in request.results():
                cand = obs.topCandidates_(1)
                if cand:
                    raw_observations.append((cand[0].string(), obs.boundingBox()))

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLanguages_(["ja"])
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)   # helps Apple Vision read small kana correctly
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)

        # ── Post-OCR isolation guard (mirrors _run_apple_single in compare script) ──
        # Apple Vision bounding boxes use normalised coordinates (0–1, origin at
        # bottom-left). Convert box heights to pixel units so the same 1.5×median_h
        # threshold used in the image-level CC pass applies consistently here.
        img_h = frame.shape[0]

        if raw_observations:
            candidates = []
            for text, bbox in raw_observations:
                px_h = bbox.size.height * img_h
                # Vertical centre in pixel coords (VN origin is bottom-left → flip)
                cy   = (1.0 - (bbox.origin.y + bbox.size.height / 2.0)) * img_h
                candidates.append((text, px_h, cy))

            sorted_h    = sorted(h for _, h, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            median_h      = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

            kept = []
            for text, px_h, cy in candidates:
                if px_h >= furi_thresh:
                    kept.append(text)
                elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    kept.append(text)   # small kana on same line as main text — keep
                # else: isolated small observation → furigana, silently dropped

            japanese = " ".join(kept).strip()
        else:
            japanese = ""

    finally:
        os.unlink(tmp_path)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

do_ocr = apple_vision_ocr

def remove_furigana_components(arr_gray):
    """
    Connected-component furigana removal with vertical-isolation guard.

    Operates on a grayscale numpy array (uint8). Small glyphs (below 55% of
    the median component height) are blanked unless they share a vertical band
    with a large neighbour — protecting small kana like ゃ/っ/ょ that appear
    inside main-text words (じゃ, って) from being incorrectly removed.

    Works on both dark-bg/light-text and light-bg/dark-text images.
    Returns a grayscale numpy array with furigana pixels set to background.
    """
    dark_bg = np.mean(arr_gray) < 127
    binary  = cv2.threshold(arr_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return arr_gray

    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    if not heights:
        return arr_gray

    median_h           = float(np.median(heights))
    furigana_threshold = median_h * 0.55

    # Vertical centres for all components (index i-1 corresponds to label i)
    centres = [
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        for i in range(1, num_labels)
    ]

    large_indices = [
        idx for idx in range(len(centres))
        if stats[idx + 1, cv2.CC_STAT_HEIGHT] >= furigana_threshold
    ]

    out      = arr_gray.copy()
    bg_value = 255 if not dark_bg else 0

    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]

        if h >= furigana_threshold or w >= median_h * 2:
            continue  # large enough to be main text — keep unconditionally

        cy = centres[i - 1]

        # Isolation guard: keep small components that share a vertical band with
        # a large component — these are small kana embedded in main-text words.
        has_large_neighbour = any(
            abs(centres[j] - cy) < median_h * 1.5
            for j in large_indices
        )

        if not has_large_neighbour:
            out[labels == i] = bg_value

    return out

def preprocess_crop(crop):
    """
    CC-based preprocessor (replaces the original row-density version).

    Pipeline:
      1. Threshold at 160 → isolate bright pixels                (same as before)
      2. 2× Lanczos upscale + 20px black border                  (same as before)
      3. remove_furigana_components() — CC glyph-level removal   (replaces row-density pass)

    Improvements over the original row-density approach:
      • Glyph-level precision — individual components evaluated, not whole rows.
      • Isolation guard — small kana (ゃ/っ/ょ) next to large characters are kept,
        preventing corruptions like じゃ → じや.
      • Background-agnostic — works on dark-bg and light-bg UI screens alike.
    """
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
    # NOTE: row-density furigana pass omitted intentionally — replaced by CC below.

    h, w   = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # CC-based furigana removal on the upscaled grayscale image
    gray_up        = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_cleaned   = remove_furigana_components(gray_up)
    result_cleaned = cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR)

    return result_cleaned

# ── Bounds loading ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
