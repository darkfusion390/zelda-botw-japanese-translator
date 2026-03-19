"""
zelda_translator_working_nlp.py
================================
Variant: Apple Vision OCR  |  Preprocessing: row-density furigana suppression
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
    """Run Apple Vision OCR on a preprocessed BGR frame.
    Uses a bimodal bounding-box height split to separate furigana observations
    from main-text observations, keeping only main-text (plus small kana that
    sit close to a main-text line — isolation guard).
    Language correction is disabled so Vision does not "fix" unusual names.
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

        raw_observations = []

        def handler(request, error):
            if error:
                return
            for obs in request.results():
                cand = obs.topCandidates_(1)
                if cand:
                    raw_observations.append((cand[0].string(), obs.boundingBox()))

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLanguages_(["ja"])
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)

        img_h = frame.shape[0]
        japanese = ""

        if raw_observations:
            # Convert Vision bounding boxes (origin bottom-left, y increases up)
            # to image coordinates (origin top-left, y increases down).
            candidates = []
            for text, bbox in raw_observations:
                px_h  = bbox.size.height * img_h
                top_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
                cy    = top_y + px_h / 2.0
                candidates.append((text, px_h, cy))

            # Bimodal gap split to find furigana/main-text height boundary
            sorted_h    = sorted(h for _, h, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i)
                        for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            # Isolation guard: keep small boxes that sit close to a main-text line
            median_h      = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

            texts = []
            for text, px_h, cy in candidates:
                if px_h >= furi_thresh:
                    texts.append(text)
                elif large_centres and any(
                        abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    texts.append(text)

            japanese = " ".join(texts).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

# Rename to do_ocr so core hook is satisfied
do_ocr = apple_vision_ocr

def preprocess_crop(crop):
    """Row-density furigana suppression pipeline.
    1. Threshold at 160 → isolate bright text pixels
    2. Blank rows whose bright-pixel density is below 42% of the median
       non-zero row density — these are typically furigana rows
    3. 2× Lanczos upscale + 20px black border
    Returns a white-on-black BGR image ready for OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop.copy()
    row_density = mask.sum(axis=1) / 255.0
    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density = float(np.median(non_zero_densities))
        furigana_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furigana_threshold:
                result[i, :] = 0
    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0,0,0))
    return result

# ── Bounds loading ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
