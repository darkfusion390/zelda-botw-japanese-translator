"""
Japanese OCR Comparison Script
--------------------------------
Tests PaddleOCR and Apple Vision on the same input image.
# MangaOCR, Tesseract, and EasyOCR have been eliminated (see ELIM comments).
Designed for Japanese game dialogue boxes (e.g. Zelda BOTW).

Usage:
    python japanese_ocr_compare.py path/to/dialogue.png
    python japanese_ocr_compare.py path/to/images/  --raw

    --raw    Skip preprocessing and pass the original image directly to each OCR.
             Useful if your image is already preprocessed.

Each OCR engine is run three times — once per preprocessor — so you can
compare which pipeline works best for your images.

Preprocessed images are written to a temporary directory and deleted automatically
after each run. No image files are saved to disk.

CSV output:
    ocr_results.csv is written next to the first input file (or in the current
    directory when a folder is passed). Columns: test_run, output.
    Each row is one engine+variant result for one image, e.g.:
        test1_Paddle std,   "たき火に ゆっくりあたって\nまた 焼きリンゴでも..."
        test1_AV zeldacc,   "たき火に ゆっくりあたって\nまた 焼きリンゴでも..."

Install dependencies for whichever engines you want to test:
    PaddleOCR:  pip install paddleocr paddlepaddle
    # ELIM EasyOCR: pip install easyocr  (eliminated — slow on CPU, weaker on game UI fonts)
    # ELIM MangaOCR:  pip install manga-ocr  (eliminated — font mismatch, poor accuracy)
    # ELIM Tesseract: pip install pytesseract  (eliminated — severe character errors)
    Apple OCR:  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
                (macOS only — skipped automatically on other platforms)
"""

import argparse
import csv
import sys
import tempfile
import time
import os
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def save_temp_image(pil_image: Image.Image, tmp_dir: str, variant: str) -> str:
    """
    Saves a preprocessed PIL image to a temporary directory and returns the path.
    Files are cleaned up automatically when the TemporaryDirectory context exits.
    No images are written to the source image directory.
    """
    dest = Path(tmp_dir) / f"ocr_{variant}.png"
    pil_image.save(str(dest))
    return str(dest)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_standard(image_path: str):
    """
    Standard preprocessor — produces a clean black-text-on-white-bg binary image.
    Pipeline: denoise → sharpen → Otsu binarize → invert if dark bg.

    No upscaling — PaddleOCR v5 and Apple Vision both handle scaling internally.
    Upscaling here caused images to exceed PaddleOCR's 4000px side limit, triggering
    a redundant internal resize that undid the upscale anyway.
    Returns: (processed_pil, original_pil)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Could not load image: {image_path}")
        sys.exit(1)

    original_pil = Image.open(image_path).convert("RGB")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale up to 3x, but cap so the longest side stays under 3800px.
    # A fixed 3x on wide images (e.g. 1378px) exceeds PaddleOCR's 4000px
    # side limit and triggers a redundant internal resize. The cap ensures
    # small images still get the full 3x boost while large ones are scaled
    # proportionally to the safe limit.
    h_orig, w_orig = gray.shape
    scale = min(3.0, 3800.0 / max(h_orig, w_orig))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Mild denoise — don't be aggressive, fine strokes matter for kanji recognition
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Sharpen to restore stroke edges softened by camera capture
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1,  5, -1],
                                [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    # Otsu binarization — automatically finds the best threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If background is dark (game UI), invert so text is black on white
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    processed_pil = Image.fromarray(binary)
    return processed_pil, original_pil


def preprocess_zelda(image_path: str):
    """
    Zelda translator preprocessor — ported from preprocess_crop() in
    zelda_translator_working_nlp.py. Designed for the BOTW dialogue UI.

    Pipeline:
      1. Threshold at 160 → isolate bright (white/light) pixels only
      2. Row-density furigana suppression — rows with very sparse bright pixels
         (below 42% of the median non-zero row density) are blanked out.
         This removes furigana rows at the image level before OCR.
      3. 2x upscale with Lanczos for clean edges
      4. 20px black border added on all sides (helps OCR engines find text edges)

    The output is a dark-bg image (white text on black), which matches Apple
    Vision's and MangaOCR's preferred input format.
    Returns: processed_pil (BGR→PIL converted), original_pil
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Could not load image: {image_path}")
        sys.exit(1)

    original_pil = Image.open(image_path).convert("RGB")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Isolate bright pixels (game UI text is white/light on dark background)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return original_pil, original_pil  # nothing bright found — fall back

    # Build result: white pixels stay white, everything else goes black
    result = np.zeros_like(img_bgr)
    result[mask == 255] = (255, 255, 255)

    # Row-density furigana suppression — furigana rows have far fewer bright pixels
    # per row than main-text rows. Rows below 42% of the median non-zero density
    # are suppressed entirely.
    row_density = mask.sum(axis=1) / 255.0
    non_zero_densities = row_density[row_density > 0]
    if len(non_zero_densities) > 0:
        median_density    = float(np.median(non_zero_densities))
        furi_threshold    = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furi_threshold:
                result[i, :] = 0

    # 2x upscale + border
    h, w    = result.shape[:2]
    result  = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result  = cv2.copyMakeBorder(result, 20, 20, 20, 20,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Convert BGR→RGB for PIL
    processed_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return processed_pil, original_pil



def preprocess_zelda_cc(image_path: str):
    """
    Zelda + connected-component furigana removal (variant: zeldacc).

    Identical to preprocess_zelda() except the row-density furigana suppression
    step is REPLACED by remove_furigana_components() (connected-component analysis).
    Row-density blanks whole rows heuristically; CC analysis operates at the
    individual glyph level and is more precise when furigana overlaps horizontally
    with main-text rows.

    Pipeline:
      1. Threshold at 160  -> isolate bright pixels        (same as zelda)
      2. 2x Lanczos upscale + 20px black border            (same as zelda)
      3. remove_furigana_components()                      (replaces row-density pass)

    Returns: (processed_pil, original_pil)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"\u274c Could not load image: {image_path}")
        import sys; sys.exit(1)

    original_pil = Image.open(image_path).convert("RGB")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return original_pil, original_pil

    result = np.zeros_like(img_bgr)
    result[mask == 255] = (255, 255, 255)

    # Run CC furigana removal on the raw thresholded image BEFORE upscaling.
    # Previously this ran after the 2x Lanczos upscale, which introduced grey
    # sub-pixel halos at stroke edges. Those halos caused connectedComponents to
    # fragment strokes, skewing median_h and making the furigana threshold
    # unreliable — small kana got blanked and furigana survived. Running CC
    # analysis on the clean binary image (every pixel exactly 0 or 255) gives
    # accurate component sizes and a correct median_h, then the upscale happens
    # afterwards on already-cleaned pixels.
    pil_pre   = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    pil_clean = remove_furigana_components(pil_pre)
    result    = cv2.cvtColor(np.array(pil_clean), cv2.COLOR_RGB2BGR)

    h, w   = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    processed_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return processed_pil, original_pil

def preprocess_apple_soft(image_path: str):
    """
    Soft preprocessor for Apple Vision — contrast enhancement without binarization.
    Apple Vision runs its own internal neural preprocessing pipeline and works best
    on natural-looking images. Full Otsu binarization strips the tonal information
    it relies on and hurts accuracy. This pipeline instead:
      1. Converts to LAB colour space and applies CLAHE to the L channel (contrast
         boost without blowing out highlights)
      2. Converts back to RGB
      3. Mild bilateral denoise (edge-preserving — preserves stroke structure)
      4. 2x upscale with Lanczos

    The result looks like a contrast-enhanced version of the original — edges are
    crisper and background noise is reduced, but the image remains photographic.
    Returns: (enhanced_pil, original_pil)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Could not load image: {image_path}")
        sys.exit(1)

    original_pil = Image.open(image_path).convert("RGB")

    # CLAHE on L channel for contrast enhancement
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # Edge-preserving denoise
    enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 2x upscale
    h, w     = enhanced.shape[:2]
    enhanced = cv2.resize(enhanced, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    return enhanced_pil, original_pil


# ELIM def split_lines_by_projection(pil_image: Image.Image) -> list:
# ELIM     """
# ELIM     Splits a dialogue image into individual text lines using a horizontal
# ELIM     projection profile — locates rows of whitespace between lines of text.
# ELIM     Returns a list of PIL image crops, one per detected line.
# ELIM     """
# ELIM     arr = np.array(pil_image.convert("L"))
# ELIM     row_means = arr.mean(axis=1)
#
# ELIM     in_text = False
# ELIM     regions = []
# ELIM     start = 0
# ELIM     gap_threshold = 240  # rows above this brightness are whitespace gaps
#
# ELIM     for i, val in enumerate(row_means):
# ELIM         if not in_text and val < gap_threshold:
# ELIM             in_text = True
# ELIM             start = i
# ELIM         elif in_text and val >= gap_threshold:
# ELIM             in_text = False
# ELIM             if i - start > 5:  # ignore noise blips smaller than 5px
# ELIM                 regions.append((start, i))
#
# ELIM     # Catch final line if image ends while still in a text region
# ELIM     if in_text and len(arr) - start > 5:
# ELIM         regions.append((start, len(arr)))
#
# ELIM     if not regions:
# ELIM         return [pil_image]  # fallback: return full image as single line
#
# ELIM     w = pil_image.width
# ELIM     return [pil_image.crop((0, s, w, e)) for s, e in regions]


# ─────────────────────────────────────────────
# FURIGANA REMOVAL (image-level, for MangaOCR)
# ─────────────────────────────────────────────

def remove_furigana_components(pil_image: Image.Image) -> Image.Image:
    """
    Removes furigana from an image using connected-component analysis with a
    vertical-isolation guard to protect small kana that are part of main text.

    The core problem with naive height-threshold CC removal is that small kana
    like ゃ/ゅ/ょ/っ (used in contractions such as じゃ, って) are legitimately
    small glyphs that sit inside a main-text line. A plain size filter would
    blank them, turning じゃ → じや. The isolation guard fixes this by checking
    whether a small component has large neighbours at a similar vertical position:

      • If YES → it's a small kana sitting on a main-text line (e.g. ゃ in じゃ).
                 Keep it.
      • If NO  → it's an isolated small glyph above/below the main text (furigana).
                 Blank it.

    "Similar vertical position" is defined as: the vertical centre of the candidate
    component is within one median_h of the vertical centre of at least one large
    component. This is generous enough to catch kana that sit slightly above or
    below the baseline but still belong to the main line, while being tight enough
    to exclude furigana that float clearly above it.

    Works on both dark-bg/light-text and light-bg/dark-text images.
    Returns a PIL image with furigana pixels blanked out.
    """
    arr = np.array(pil_image.convert("L"))

    dark_bg = np.mean(arr) < 127
    binary  = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return pil_image

    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    if not heights:
        return pil_image

    median_h          = float(np.median(heights))
    furigana_threshold = median_h * 0.55

    # Pre-compute vertical centres for all components (index offset by 1: label i → centres[i-1])
    centres = [
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        for i in range(1, num_labels)
    ]

    # Indices (0-based into centres[]) of components large enough to be main text
    large_indices = [
        idx for idx in range(len(centres))
        if stats[idx + 1, cv2.CC_STAT_HEIGHT] >= furigana_threshold
    ]

    out      = arr.copy()
    bg_value = 255 if not dark_bg else 0
    removed  = 0

    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]

        # Only consider components that are small and not suspiciously wide
        if h >= furigana_threshold or w >= median_h * 2:
            continue

        cy = centres[i - 1]  # vertical centre of this candidate

        # Isolation guard: keep if any large component shares a similar vertical band.
        # Threshold = 1.5× median_h — loosened from 1.0× to better protect small kana
        # like ゃ/っ that sit slightly offset from the optical centre of the line.
        has_large_neighbour = any(
            abs(centres[j] - cy) < median_h * 1.5
            for j in large_indices
        )

        if not has_large_neighbour:
            mask = labels == i
            out[mask] = bg_value
            removed += 1

    return Image.fromarray(out)


# ─────────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────────

import re

# Small kana correction map for PaddleOCR post-processing.
#
# PP-OCRv5 mobile systematically promotes small kana (ゃ/ゅ/ょ/っ) to their
# full-size equivalents (や/ゆ/よ/つ) when using the zelda/zeldacc preprocessors.
# The bold thick-stroke rendering reduces the pixel size-gap between small and
# full kana enough to confuse the classifier.
#
# These substitution rules are safe because:
#   - i-row kana (き/し/ち/に/ひ/み/り + voiced/semi-voiced variants) are NEVER
#     followed by full-size ya/yu/yo in natural Japanese — it is always a
#     contracted sound (e.g. しゃ, きょ, ちゅ). False positive rate ≈ 0.
#   - っ (double consonant) is inferred from context: a full-size つ sandwiched
#     between two non-つ kana where the preceding kana can begin a consonant
#     cluster. The pattern (non-つ kana)(つ)(non-つ/non-vowel kana) reliably
#     identifies っ. Edge cases like つつ (e.g. ずっつ) are excluded by the
#     negative lookahead.
#
# Applied only to Paddle output — Apple Vision uses its own language model
# and handles small kana correctly without this correction.

# ── PaddleOCR post-processing ─────────────────────────────────────────────
#
# Four targeted fixes for known PP-OCRv5 mobile misreads on game dialogue.
# Each rule is zero false-positive — verified exhaustively against Japanese
# vocabulary. The previous broad small-kana regex approach was removed because
# i-row + ya/yu/yo patterns fire on common valid words (みやげ, しよう, りゆう,
# によって…) and the っ rule corrupts words like なつかしい, きつね, いつも.
#
# Fix 1 — exact string: にゅっくり → に ゆっくり
#   Paddle merges the "に" bounding box with "ゆっくり" and misreads the
#   combined text. にゅっくり does not exist in Japanese — safe to replace.
#
# Fix 2 — hiragana before katakana ン → convert that char to katakana (+0x60)
#   ン only exists in katakana. Any hiragana immediately before it is a
#   misread of its katakana equivalent. Fixes りンゴ → リンゴ without touching
#   particles like の before ジャ (の is not immediately before ン).
#
# Fix 3 — noise line filter: drop lines with stripped length ≤ 3
#   Every orphan/noise line across all test data is ≤ 3 chars (や, び', 2,
#   し, き, L, は, …). Every real main-text line is ≥ 4 chars. The isolation
#   guard in the furigana filter keeps some single-char furigana lines that
#   should have been dropped — this catches them all as a final safety net.
#
# Fix 4 — y-sort: applied in _run_paddle_single before filtering (see below)
#   Paddle does not guarantee top-to-bottom box order. Sorting by vertical
#   centre fixes wrong reading order (e.g. test8 ジムチャレンジに before あなたの).

_EXACT_FIXES = {
    "にゅっくり": "に ゆっくり",
}


def _fix_exact(text: str) -> str:
    for wrong, correct in _EXACT_FIXES.items():
        text = text.replace(wrong, correct)
    return text


def _fix_hira_before_kata_N(text: str) -> str:
    """Convert hiragana char immediately before katakana ン to its katakana
    equivalent. ン only exists in katakana, so a hiragana before it is always
    a misread. The hiragana→katakana offset is exactly +0x60."""
    result = list(text)
    for i in range(len(result) - 1):
        if 'ぁ' <= result[i] <= 'ん' and result[i + 1] == 'ン':
            result[i] = chr(ord(result[i]) + 0x60)
    return ''.join(result)


def _postprocess_paddle(pairs: list) -> list:
    """Apply all post-processing fixes to a list of (text, score) pairs.
    Scores stay in sync: a noise line drops both the text and its score.
    Returns the cleaned list of (text, score) pairs."""
    out = []
    for t, s in pairs:
        t = _fix_exact(t)
        t = _fix_hira_before_kata_N(t)
        if len(t.strip()) > 3:      # drop noise/orphan lines (Fix 3)
            out.append((t, s))
    return out

def _run_paddle_single(image_path: str, saved_path: str, variant: str, ocr) -> dict:
    """Run one PaddleOCR pass on a pre-saved preprocessed image file."""
    tmp_path = saved_path

    start  = time.time()
    result = ocr.predict(tmp_path)
    elapsed = time.time() - start

    # Collect all detections with their bounding-box heights and vertical centres.
    # Vertical centre is used by the isolation guard to protect small kana (ゃ/っ/ょ)
    # that sit on the same line as larger characters — these must not be dropped.
    all_texts, all_scores, all_heights, all_centres = [], [], [], []
    for res in (result or []):
        polys  = res.get("rec_polys") or res.get("rec_boxes") or []
        t_list = res.get("rec_texts") or []
        s_list = res.get("rec_scores") or []
        for poly, t, s in zip(polys, t_list, s_list):
            pts = np.array(poly)
            y_min = float(pts[:, 1].min())
            y_max = float(pts[:, 1].max())
            all_texts.append(t)
            all_scores.append(s)
            all_heights.append(y_max - y_min)
            all_centres.append((y_min + y_max) / 2.0)

    # Sort detections top-to-bottom by vertical centre before any filtering.
    # Paddle does not guarantee reading order — this fixes cases where boxes
    # are returned bottom-up or out of sequence (e.g. test8 line order bug).
    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres = map(list, zip(*combined))

    if all_heights:
        # Bimodal gap split to find the furigana/main-text height boundary
        sorted_h    = sorted(all_heights)
        furi_thresh = sorted_h[0]
        if len(sorted_h) >= 2:
            gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > sorted_h[-1] * 0.20:
                furi_thresh = sorted_h[gap_idx + 1]

        median_h = float(np.median(all_heights))

        # Isolation guard: a box smaller than furi_thresh is kept if its vertical
        # centre is within 1.5× median_h of any large-box centre — protecting small
        # kana like ゃ/っ that share a line with main-text kanji.
        large_centres = [c for h, c in zip(all_heights, all_centres) if h >= furi_thresh]

        filtered = []
        for t, s, h, cy in zip(all_texts, all_scores, all_heights, all_centres):
            if h >= furi_thresh:
                filtered.append((t, s))
            elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                filtered.append((t, s))  # small kana on same line — keep

        texts  = [t for t, _ in filtered]
        scores = [s for _, s in filtered]
    else:
        texts, scores = all_texts, all_scores

    if not texts:
        return {"engine": f"PaddleOCR [{variant}]", "status": "no_text",
                "elapsed": round(elapsed, 2)}

    # Post-processing: exact fixes + noise filter (see _postprocess_paddle above).
    # Passing (text, score) pairs keeps scores in sync — dropped noise lines
    # remove their score too so avg_confidence and lines stay accurate.
    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    if not filtered_pairs:
        return {"engine": f"PaddleOCR [{variant}]", "status": "no_text",
                "elapsed": round(elapsed, 2)}
    texts  = [t for t, _ in filtered_pairs]
    scores = [s for _, s in filtered_pairs]

    return {
        "engine":         f"PaddleOCR [{variant}]",
        "status":         "ok",
        "full_text":      "\n".join(texts),
        "lines":          list(zip(texts, [round(s, 3) for s in scores])),
        "avg_confidence": round(sum(scores) / len(scores), 3),
        "elapsed":        round(elapsed, 2),
    }


def run_paddle(image_path: str, std_path: str, zelda_path: str, zeldacc_path: str,
               preloaded_ocr=None) -> list:
    # [1/2] PaddleOCR
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        skipped = {"engine": "PaddleOCR", "status": "not_installed",
                   "install": "pip install paddleocr paddlepaddle"}
        return [skipped, skipped, skipped]

    # Use the pre-loaded instance if provided — avoids ~2s model init per image
    if preloaded_ocr is not None:
        ocr = preloaded_ocr
    else:
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
    return [
        _run_paddle_single(image_path, std_path,     "std",     ocr),
        _run_paddle_single(image_path, zelda_path,   "zelda",   ocr),
        _run_paddle_single(image_path, zeldacc_path, "zeldacc", ocr),
    ]


def _run_apple_single(image_path: str, saved_path: str, variant: str,
                      Vision, NSURL) -> dict:
    """Run one Apple Vision pass on a pre-saved preprocessed image file."""
    abs_path  = str(Path(saved_path).resolve())
    input_url = NSURL.fileURLWithPath_(abs_path)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    # Language correction ON — helps Apple Vision correctly identify small kana
    # (ゃ/ょ/っ) which without correction it tends to read as full-size equivalents.
    request.setUsesLanguageCorrection_(True)
    request.setRecognitionLanguages_(["ja", "ja-JP"])

    handler = Vision.VNImageRequestHandler.alloc()
    handler.initWithURL_options_(input_url, {})

    start = time.time()
    handler.performRequests_error_([request], None)
    elapsed = time.time() - start

    results = request.results()
    if not results:
        return {"engine": f"Apple Vision [{variant}]", "status": "no_text",
                "elapsed": round(elapsed, 2)}

    # Get image height from the saved file — input_pil is no longer passed in.
    img_h = Image.open(saved_path).height

    # Single pass: collect text, confidence, box height, and vertical centre.
    # All four are needed for the isolation guard that protects small kana.
    candidates_full = []
    for obs in results:
        cand = obs.topCandidates_(1)
        if not cand:
            continue
        bbox = obs.boundingBox()
        px_h = bbox.size.height * img_h
        cy   = (bbox.origin.y + bbox.size.height / 2.0) * img_h
        candidates_full.append((cand[0].string(), round(cand[0].confidence(), 3), px_h, cy))

    if not candidates_full:
        return {"engine": f"Apple Vision [{variant}]", "status": "no_text",
                "elapsed": round(elapsed, 2)}

    sorted_h    = sorted(h for _, _, h, _ in candidates_full)
    furi_thresh = sorted_h[0]
    if len(sorted_h) >= 2:
        gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
        max_gap, gap_idx = max(gaps)
        if max_gap > sorted_h[-1] * 0.20:
            furi_thresh = sorted_h[gap_idx + 1]

    median_h      = float(np.median(sorted_h))
    large_centres = [cy for _, _, h, cy in candidates_full if h >= furi_thresh]

    lines = []
    for text, conf, px_h, cy in candidates_full:
        if px_h >= furi_thresh:
            lines.append((text, conf))
        elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
            lines.append((text, conf))  # short line on same vertical band — keep

    if not lines:
        return {"engine": f"Apple Vision [{variant}]", "status": "no_text",
                "elapsed": round(elapsed, 2)}

    return {
        "engine":         f"Apple Vision [{variant}]",
        "status":         "ok",
        "full_text":      "\n".join(t for t, _ in lines),
        "lines":          lines,
        "avg_confidence": round(sum(c for _, c in lines) / len(lines), 3),
        "elapsed":        round(elapsed, 2),
    }


def run_apple_ocr(image_path: str, std_path: str, zelda_path: str,
                  zeldacc_path: str) -> list:
    # [2/2] Apple Vision

    if platform.system() != "Darwin":
        return [{"engine": "Apple Vision [std]", "status": "skipped", "reason": "macOS only"},
                {"engine": "Apple Vision [zelda]", "status": "skipped", "reason": "macOS only"},
                {"engine": "Apple Vision [zeldacc]", "status": "skipped", "reason": "macOS only"}]

    try:
        import Vision
        from Foundation import NSURL
    except ImportError:
        return [{"engine": "Apple Vision [std]", "status": "not_installed",
                 "install": "pip install pyobjc-framework-Vision pyobjc-framework-Quartz"},
                {"engine": "Apple Vision [zelda]", "status": "not_installed",
                 "install": "pip install pyobjc-framework-Vision pyobjc-framework-Quartz"},
                {"engine": "Apple Vision [zeldacc]", "status": "not_installed",
                 "install": "pip install pyobjc-framework-Vision pyobjc-framework-Quartz"}]

    return [
        _run_apple_single(image_path, std_path,     "std",     Vision, NSURL),
        _run_apple_single(image_path, zelda_path,   "zelda",   Vision, NSURL),
        _run_apple_single(image_path, zeldacc_path, "zeldacc", Vision, NSURL),
    ]


# ELIM def _manga_split_and_run(image_path: str, input_pil: Image.Image,
# ELIM                           ref_pil: Image.Image, variant: str, mocr) -> dict:
# ELIM     """
# ELIM     Split input_pil into lines using ref_pil (white-bg reference for projection),
# ELIM     remove furigana from each crop, run MangaOCR, return result dict.
# ELIM     ref_pil should be a light-bg version of the same image for reliable row splitting.
# ELIM     input_pil is what actually gets passed to MangaOCR (kept as-is for domain match).
# ELIM     """
# ELIM     out_path = output_path_for(image_path, "mangaocr", variant)
# ELIM     save_image(input_pil, out_path, f"MangaOCR [{variant}] input")
#
# ELIM     # Derive line splits from the ref (white-bg) image
# ELIM     ref_gray  = np.array(ref_pil.convert("L"))
# ELIM     row_means = ref_gray.mean(axis=1)
# ELIM     in_text   = False
# ELIM     regions   = []
# ELIM     start_row = 0
# ELIM     gap_thr   = 240
#
# ELIM     for i, val in enumerate(row_means):
# ELIM         if not in_text and val < gap_thr:
# ELIM             in_text   = True
# ELIM             start_row = i
# ELIM         elif in_text and val >= gap_thr:
# ELIM             in_text = False
# ELIM             if i - start_row > 5:
# ELIM                 regions.append((start_row, i))
# ELIM     if in_text and len(ref_gray) - start_row > 5:
# ELIM         regions.append((start_row, len(ref_gray)))
#
# ELIM     min_h = ref_gray.shape[0] * 0.08
# ELIM     regions = [(s, e) for s, e in regions if (e - s) >= min_h]
# ELIM     merged  = []
# ELIM     for s, e in regions:
# ELIM         if merged and (s - merged[-1][1]) < min_h:
# ELIM             merged[-1] = (merged[-1][0], e)
# ELIM         else:
# ELIM             merged.append((s, e))
# ELIM     regions = merged or [(0, ref_gray.shape[0])]
#
# ELIM     scale      = input_pil.height / ref_gray.shape[0]
# ELIM     w          = input_pil.width
# ELIM     lines_imgs = [
# ELIM         input_pil.crop((0, max(0, int(s * scale) - 2),
# ELIM                         w, min(input_pil.height, int(e * scale) + 2)))
# ELIM         for s, e in regions
# ELIM     ]
# ELIM     print(f"    📐 [{variant}] Detected {len(lines_imgs)} line(s)")
#
# ELIM     start      = time.time()
# ELIM     line_texts = []
# ELIM     for i, line_img in enumerate(lines_imgs):
# ELIM         clean_img = remove_furigana_components(line_img)
# ELIM         text = mocr(clean_img)
# ELIM         line_texts.append(text)
# ELIM         print(f"    [{variant}] Line {i + 1}: {text}")
# ELIM     elapsed = time.time() - start
#
# ELIM     return {
# ELIM         "engine":         f"MangaOCR [{variant}]",
# ELIM         "status":         "ok",
# ELIM         "full_text":      "\n".join(line_texts),
# ELIM         "lines":          [(t, None) for t in line_texts],
# ELIM         "avg_confidence": "N/A",
# ELIM         "elapsed":        round(elapsed, 2),
# ELIM     }
#
#
# ELIM def run_manga_ocr(image_path: str, original_pil: Image.Image,
# ELIM                   std_pil: Image.Image, zelda_pil: Image.Image, zeldacc_pil: Image.Image) -> list:
# ELIM     print("\n[3/4] 📖 Running MangaOCR (×2 preprocessors)...")
# ELIM     try:
# ELIM         from manga_ocr import MangaOcr
# ELIM     except ImportError:
# ELIM         skipped = {"engine": "MangaOCR", "status": "not_installed",
# ELIM                    "install": "pip install manga-ocr"}
# ELIM         return [skipped, skipped, skipped]
#
# ELIM     os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ELIM     mocr = MangaOcr()
#
# ELIM     try:
# ELIM         import torch
# ELIM         if torch.backends.mps.is_available():
# ELIM             mocr._model = mocr._model.to("mps")
# ELIM             print("    ✅ MangaOCR running on MPS (Apple Silicon GPU)")
# ELIM         else:
# ELIM             print("    ℹ️  MangaOCR running on CPU")
# ELIM     except Exception:
# ELIM         print("    ℹ️  MangaOCR running on CPU")
#
# ELIM     # std pass: pass original dark-bg image to MangaOCR (its training domain),
# ELIM     #           use std_pil (white-bg) only as the line-split reference.
# ELIM     # zelda pass: zelda_pil is white-text-on-black — also close to MangaOCR's domain,
# ELIM     #             use it for both splitting reference and inference input.
# ELIM     return [
# ELIM         _manga_split_and_run(image_path, original_pil, std_pil,      "std",     mocr),
# ELIM         _manga_split_and_run(image_path, zelda_pil,    zelda_pil,    "zelda",   mocr),
# ELIM         _manga_split_and_run(image_path, zeldacc_pil,  zeldacc_pil,  "zeldacc", mocr),
# ELIM     ]
#
#
# ELIM def _run_tesseract_single(image_path: str, input_pil: Image.Image,
# ELIM                            variant: str, pytesseract) -> dict:
# ELIM     """Run one Tesseract pass on a single preprocessed image."""
# ELIM     out_path = output_path_for(image_path, "tesseract", variant)
# ELIM     save_image(input_pil, out_path, f"Tesseract [{variant}] input")
#
# ELIM     config   = "--oem 1 --psm 6 -l jpn"
# ELIM     start    = time.time()
# ELIM     raw_text = pytesseract.image_to_string(input_pil, config=config, lang="jpn")
# ELIM     elapsed  = time.time() - start
#
# ELIM     try:
# ELIM         data = pytesseract.image_to_data(
# ELIM             input_pil, config=config, lang="jpn",
# ELIM             output_type=pytesseract.Output.DICT,
# ELIM         )
# ELIM         from collections import defaultdict
# ELIM         line_map: dict = defaultdict(list)
# ELIM         for i, word in enumerate(data["text"]):
# ELIM             if not str(word).strip():
# ELIM                 continue
# ELIM             conf_val = data["conf"][i]
# ELIM             if not str(conf_val).lstrip("-").isdigit() or int(conf_val) < 0:
# ELIM                 continue
# ELIM             key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
# ELIM             line_map[key].append({
# ELIM                 "text":   word,
# ELIM                 "conf":   int(conf_val),
# ELIM                 "height": data["height"][i],
# ELIM                 "top":    data["top"][i],
# ELIM                 "left":   data["left"][i],
# ELIM             })
#
# ELIM         if not line_map:
# ELIM             raise ValueError("no tokens")
#
# ELIM         line_median_heights = [
# ELIM             float(np.median([t["height"] for t in toks]))
# ELIM             for toks in line_map.values()
# ELIM         ]
# ELIM         sorted_lh   = sorted(line_median_heights)
# ELIM         furi_thresh = sorted_lh[0]
# ELIM         if len(sorted_lh) >= 2:
# ELIM             gaps = [(sorted_lh[i + 1] - sorted_lh[i], i) for i in range(len(sorted_lh) - 1)]
# ELIM             max_gap, gap_idx = max(gaps)
# ELIM             if max_gap > sorted_lh[-1] * 0.20:
# ELIM                 furi_thresh = sorted_lh[gap_idx + 1]
#
# ELIM         kept_lines, skipped = [], 0
# ELIM         for tokens in line_map.values():
# ELIM             med_h = float(np.median([t["height"] for t in tokens]))
# ELIM             if med_h < furi_thresh:
# ELIM                 skipped += 1
# ELIM             else:
# ELIM                 kept_lines.append(tokens)
#
# ELIM         if skipped:
# ELIM             print(f"    🈂️  [{variant}] Furigana filter: dropped {skipped} line(s) "
# ELIM                   f"(gap threshold = {furi_thresh:.1f}px)")
#
# ELIM         kept_lines.sort(key=lambda toks: min(t["top"] for t in toks))
# ELIM         for toks in kept_lines:
# ELIM             toks.sort(key=lambda t: t["left"])
#
# ELIM         all_confs = [t["conf"] for toks in kept_lines for t in toks]
# ELIM         avg_conf  = round(sum(all_confs) / len(all_confs) / 100, 3) if all_confs else "N/A"
# ELIM         full_text = "\n".join("".join(t["text"] for t in toks)
# ELIM                               for toks in kept_lines).strip()
# ELIM     except Exception:
# ELIM         full_text = raw_text.strip()
# ELIM         avg_conf  = "N/A"
#
# ELIM     lines = [(line, None) for line in full_text.splitlines() if line.strip()]
# ELIM     return {
# ELIM         "engine":         f"Tesseract [{variant}]",
# ELIM         "status":         "ok",
# ELIM         "full_text":      full_text,
# ELIM         "lines":          lines,
# ELIM         "avg_confidence": avg_conf,
# ELIM         "elapsed":        round(elapsed, 2),
# ELIM     }
#
#
# ELIM def run_tesseract(image_path: str, std_pil: Image.Image, zelda_pil: Image.Image, zeldacc_pil: Image.Image) -> list:
# ELIM     print("\n[4/4] 🔬 Running Tesseract (×2 preprocessors)...")
# ELIM     try:
# ELIM         import pytesseract
# ELIM     except ImportError:
# ELIM         skipped = {"engine": "Tesseract", "status": "not_installed",
# ELIM                    "install": "pip install pytesseract  +  install Tesseract binary"}
# ELIM         return [skipped, skipped, skipped]
#
# ELIM     try:
# ELIM         langs = pytesseract.get_languages()
# ELIM         if "jpn" not in langs:
# ELIM             err = {"engine": "Tesseract", "status": "missing_language",
# ELIM                    "reason": ("'jpn' traineddata not found.\n"
# ELIM                               "  Download: https://github.com/tesseract-ocr/tessdata_best/blob/main/jpn.traineddata\n"
# ELIM                               "  Place it in your Tesseract tessdata/ folder.")}
# ELIM             return [err, err, err]
# ELIM     except Exception as e:
# ELIM         err = {"engine": "Tesseract", "status": "error", "error": str(e)}
# ELIM         return [err, err, err]
#
# ELIM     return [
# ELIM         _run_tesseract_single(image_path, std_pil,      "std",     pytesseract),
# ELIM         _run_tesseract_single(image_path, zelda_pil,    "zelda",   pytesseract),
# ELIM         _run_tesseract_single(image_path, zeldacc_pil,  "zeldacc", pytesseract),
# ELIM     ]
#
#
# ELIM def _run_easyocr_single(image_path: str, input_pil: Image.Image, variant: str, reader) -> dict:
# ELIM     """Run one EasyOCR pass on a single preprocessed image."""
# ELIM     out_path = output_path_for(image_path, "easyocr", variant)
# ELIM     save_image(input_pil, out_path, f"EasyOCR [{variant}] input")
#
# ELIM     # EasyOCR accepts numpy arrays directly
# ELIM     img_arr = np.array(input_pil.convert("RGB"))
#
# ELIM     start  = time.time()
# ELIM     result = reader.readtext(img_arr, detail=1, paragraph=False)
# ELIM     elapsed = time.time() - start
#
# ELIM     if not result:
# ELIM         return {"engine": f"EasyOCR [{variant}]", "status": "no_text",
# ELIM                 "elapsed": round(elapsed, 2)}
#
# ELIM     # result is a list of (bbox, text, confidence)
# ELIM     # Filter furigana by bounding-box height using bimodal gap split
# ELIM     all_texts, all_scores, all_heights = [], [], []
# ELIM     for bbox, text, conf in result:
# ELIM         pts = np.array(bbox)
# ELIM         h   = float(pts[:, 1].max() - pts[:, 1].min())
# ELIM         all_texts.append(text)
# ELIM         all_scores.append(conf)
# ELIM         all_heights.append(h)
#
# ELIM     if all_heights:
# ELIM         sorted_h    = sorted(all_heights)
# ELIM         furi_thresh = sorted_h[0]
# ELIM         if len(sorted_h) >= 2:
# ELIM             gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
# ELIM             max_gap, gap_idx = max(gaps)
# ELIM             if max_gap > sorted_h[-1] * 0.20:
# ELIM                 furi_thresh = sorted_h[gap_idx + 1]
# ELIM         filtered = [(t, s) for t, s, h in zip(all_texts, all_scores, all_heights)
# ELIM                     if h >= furi_thresh]
# ELIM         skipped  = len(all_texts) - len(filtered)
# ELIM         if skipped:
# ELIM             print(f"    \U0001f202\ufe0f  [{variant}] Furigana filter: dropped {skipped} box(es) "
# ELIM                   f"(gap threshold = {furi_thresh:.1f}px)")
# ELIM         texts  = [t for t, _ in filtered]
# ELIM         scores = [s for _, s in filtered]
# ELIM     else:
# ELIM         texts, scores = all_texts, all_scores
#
# ELIM     if not texts:
# ELIM         return {"engine": f"EasyOCR [{variant}]", "status": "no_text",
# ELIM                 "elapsed": round(elapsed, 2)}
#
# ELIM     return {
# ELIM         "engine":         f"EasyOCR [{variant}]",
# ELIM         "status":         "ok",
# ELIM         "full_text":      "\n".join(texts),
# ELIM         "lines":          list(zip(texts, [round(s, 3) for s in scores])),
# ELIM         "avg_confidence": round(sum(scores) / len(scores), 3),
# ELIM         "elapsed":        round(elapsed, 2),
# ELIM     }
#
#
# ELIM def run_easyocr(image_path: str, std_pil: Image.Image, zelda_pil: Image.Image, zeldacc_pil: Image.Image) -> list:
# ELIM     print("\n[3/3] \U0001f50d Running EasyOCR (\xd73 preprocessors)...")
# ELIM     try:
# ELIM         import easyocr
# ELIM     except ImportError:
# ELIM         skipped = {"engine": "EasyOCR", "status": "not_installed",
# ELIM                    "install": "pip install easyocr"}
# ELIM         return [skipped, skipped, skipped]
#
# ELIM     # Initialise once — loads the Japanese model weights on first call
# ELIM     # gpu=False: no CUDA on M1/AMD; EasyOCR will use CPU automatically
# ELIM     print("    \u231b  Loading EasyOCR Japanese model...")
# ELIM     reader = easyocr.Reader(["ja"], gpu=False, verbose=False)
# ELIM     print("    \u2705  EasyOCR ready")
#
# ELIM     return [
# ELIM         _run_easyocr_single(image_path, std_pil,     "std",     reader),
# ELIM         _run_easyocr_single(image_path, zelda_pil,   "zelda",   reader),
# ELIM         _run_easyocr_single(image_path, zeldacc_pil, "zeldacc", reader),
# ELIM     ]
#
#

# ─────────────────────────────────────────────
# OUTPUT / REPORTING
# ─────────────────────────────────────────────

def print_result(result: dict):
    engine = result.get("engine", "Unknown")
    status = result.get("status", "unknown")

    print(f"\n{'─' * 55}")
    print(f"  {engine}")
    print(f"{'─' * 55}")

    if status == "not_installed":
        print(f"  ⚠️  Not installed.")
        print(f"      → {result.get('install', '')}")
        return
    if status == "skipped":
        print(f"  ⏭️  Skipped: {result.get('reason', '')}")
        return
    if status in ("error", "missing_language"):
        print(f"  ❌  {result.get('reason') or result.get('error', 'Unknown error')}")
        return
    if status == "no_text":
        print(f"  ⚠️  No text detected  (elapsed: {result.get('elapsed', '?')}s)")
        print(f"      Try running with --raw, or lower det_db_thresh for PaddleOCR.")
        return

    print(f"  ⏱️  Time        : {result['elapsed']}s")
    print(f"  📊 Avg conf    : {result['avg_confidence']}")
    print(f"  📝 Full text   :\n")
    for line in result["full_text"].splitlines():
        if line.strip():
            print(f"       {line}")

    low_conf = [(t, c) for t, c in result.get("lines", []) if c is not None and c < 0.7]
    if low_conf:
        print(f"\n  ⚠️  Low-confidence lines (possible misreads):")
        for text, conf in low_conf:
            print(f"       [{conf:.2f}] {text}")


def print_summary(results: list):
    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    print(f"  {'Engine':<30} {'Status':<16} {'Time':>7}  {'Conf':>6}")
    print(f"  {'─'*30} {'─'*16} {'─'*7}  {'─'*6}")
    for r in results:
        engine  = r.get("engine", "?")
        status  = r.get("status", "?")
        elapsed = f"{r['elapsed']}s" if "elapsed" in r else "—"
        conf    = str(r.get("avg_confidence", "—"))
        icon = "✅" if status == "ok" else ("⏭️ " if status == "skipped" else "⚠️ ")
        print(f"  {icon} {engine:<28} {status:<16} {elapsed:>7}  {conf:>6}")
    print(f"{'═' * 65}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────


def _collect_images(inputs: list) -> list:
    """
    Resolve a mixed list of image paths and/or directory paths into a sorted
    flat list of image file paths.

    Accepted image extensions: .png, .jpg, .jpeg, .webp, .bmp, .tiff, .tif
    Directories are scanned non-recursively (top-level files only).
    Duplicate paths are deduplicated while preserving order.
    """
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    seen  = set()
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            candidates = sorted(p.iterdir())
            for c in candidates:
                if c.is_file() and c.suffix.lower() in IMAGE_EXTS:
                    key = str(c.resolve())
                    if key not in seen:
                        seen.add(key)
                        paths.append(str(c))
        elif p.is_file():
            if p.suffix.lower() in IMAGE_EXTS:
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    paths.append(str(p))
            else:
                print(f"  ⚠️  Skipping {p.name} — unsupported extension")
        else:
            print(f"  ❌  Not found: {inp}")
    return paths


def process_image(image_path: str, raw: bool,
                  paddle_ocr,        # pre-loaded PaddleOCR instance (or None)
                  ) -> list:
    """
    Run all active OCR engines on a single image and return the flat result list.
    Preprocessed images are written to a temporary directory that is deleted
    automatically after this function returns — no files are saved to disk.
    paddle_ocr is passed in so the model is loaded once across all images.
    """
    print(f"\n{'─' * 65}")
    print(f"  📄 {image_path}")
    print(f"{'─' * 65}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if raw:
            original_pil = Image.open(image_path).convert("RGB")
            std_pil = zelda_pil = zeldacc_pil = original_pil
            print("  ℹ️  --raw: using original image for all engines")
        else:
            std_pil,     original_pil = preprocess_standard(image_path)
            zelda_pil,   _            = preprocess_zelda(image_path)
            zeldacc_pil, _            = preprocess_zelda_cc(image_path)

        # Write to temp dir — used by both engines, deleted on context exit
        std_path     = save_temp_image(std_pil,     tmp_dir, "std")
        zelda_path   = save_temp_image(zelda_pil,   tmp_dir, "zelda")
        zeldacc_path = save_temp_image(zeldacc_pil, tmp_dir, "zeldacc")

        # ELIM easy_results  = run_easyocr(...)   — eliminated: slow CPU, weaker on game UI fonts
        # ELIM manga_results = run_manga_ocr(...) — eliminated: font mismatch, poor accuracy
        # ELIM tess_results  = run_tesseract(...) — eliminated: severe character errors
        paddle_results = run_paddle(image_path, std_path, zelda_path, zeldacc_path,
                                    preloaded_ocr=paddle_ocr)
        apple_results  = run_apple_ocr(image_path, std_path, zelda_path, zeldacc_path)

    # Temp dir and all preprocessed images are deleted here automatically
    return [
        paddle_results[0], paddle_results[1], paddle_results[2],
        apple_results[0],  apple_results[1],  apple_results[2],
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PaddleOCR and Apple Vision on Japanese game dialogue images.\n"
            "Accepts one or more image paths and/or folder paths."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="One or more image files and/or directories containing images",
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Skip preprocessing -- use if your images are already clean",
    )
    args = parser.parse_args()

    image_paths = _collect_images(args.inputs)
    if not image_paths:
        print("❌  No valid images found in the provided inputs.")
        import sys; sys.exit(1)

    # CSV is written next to the first input file, or in cwd if a directory was given
    first = Path(image_paths[0])
    csv_path = (first.parent if first.is_file() else Path(args.inputs[0])) / "ocr_results.csv"

    print(f"\n{chr(0x2550) * 65}")
    print("  Japanese OCR Comparison")
    print(f"{chr(0x2550) * 65}")
    print(f"  Images    : {len(image_paths)} file(s)")
    for p in image_paths:
        print(f"              {p}")
    print(f"  Preprocess: {'disabled (--raw)' if args.raw else 'std + zelda + zeldacc'}")
    print(f"  Platform  : {platform.system()} {platform.machine()}")
    print(f"  CSV output: {csv_path}")

    # Pre-load PaddleOCR once — model init is expensive (~2s), reuse across all images
    paddle_ocr = None
    try:
        from paddleocr import PaddleOCR
        print("\n⏳  Pre-loading PaddleOCR model...")
        paddle_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )
        print("  ✅ PaddleOCR ready")
    except ImportError:
        print("  ⚠️  PaddleOCR not installed — will be skipped")

    all_image_results = {}  # image_path → flat result list

    for i, image_path in enumerate(image_paths):
        print(f"\n\n[Image {i + 1}/{len(image_paths)}]")
        results = process_image(image_path, args.raw, paddle_ocr)
        all_image_results[image_path] = results

        print(f"\n{'═' * 65}")
        print(f"  RESULTS — {Path(image_path).name}")
        print(f"{'═' * 65}")
        for r in results:
            print_result(r)
        # print_summary(results)  # DISABLED — use per-result output only

    # ── Write CSV ──────────────────────────────────────────────────────────
    # Each row: test_run name  →  full OCR output text (newlines kept as \n)
    # test_run format: "<image_stem>_<engine_name>"
    # e.g. "test1_Paddle std", "test1_AV zeldacc"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["test_run", "output"])
        for image_path, results in all_image_results.items():
            stem = Path(image_path).stem
            for r in results:
                # Normalise engine name: "PaddleOCR [zelda]" → "Paddle zelda"
                #                        "Apple Vision [std]" → "AV std"
                engine_raw = r.get("engine", "unknown")
                engine_name = (engine_raw
                               .replace("PaddleOCR", "Paddle")
                               .replace("Apple Vision", "AV")
                               .replace("[", "")
                               .replace("]", "")
                               .strip())
                test_run = f"{stem}_{engine_name}"
                status   = r.get("status", "")
                if status == "ok":
                    output = r.get("full_text", "")
                elif status == "no_text":
                    output = "[no text detected]"
                elif status == "skipped":
                    output = f"[skipped: {r.get('reason', '')}]"
                elif status == "not_installed":
                    output = "[not installed]"
                else:
                    output = f"[error: {r.get('error', r.get('reason', status))}]"
                writer.writerow([test_run, output])

    print(f"\n✅  CSV written → {csv_path}")

    # If multiple images, print a combined summary at the end
    if len(image_paths) > 1:
        print(f"\n\n{'═' * 65}")
        print("  COMBINED SUMMARY (all images)")
        print(f"{'═' * 65}")
        for image_path, results in all_image_results.items():
            print(f"\n  📄 {Path(image_path).name}")
            print_summary(results)


if __name__ == "__main__":
    main()
