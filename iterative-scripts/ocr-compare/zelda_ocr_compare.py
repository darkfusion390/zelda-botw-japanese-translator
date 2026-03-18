"""
zelda_ocr_compare.py
--------------------
Compares the four OCR pipelines used across the Zelda translator variants:

  1. av_rowdensity   — Apple Vision + row-density furigana suppression
                       (zelda_translator_working_nlp.py)
  2. av_cc           — Apple Vision + CC furigana removal + post-OCR isolation guard
                       (zelda_translator_working_av_furigana_box.py)
  3. paddle_rowdensity — PaddleOCR + row-density furigana suppression
                       (zelda_translator_paddle_ocr_base.py)
  4. paddle_cc       — PaddleOCR + CC furigana removal pre-upscale
                       (zelda_translator_paddle_ocr_furigana_box.py)

Usage:
    python zelda_ocr_compare.py path/to/dialogue.png
    python zelda_ocr_compare.py path/to/images/

CSV output:
    zelda_ocr_results.csv is written next to the first input file (or cwd for folders).
    Columns: image, engine, output

Install deps:
    PaddleOCR:  pip install paddleocr paddlepaddle
    Apple OCR:  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
                (macOS only — skipped automatically on other platforms)
"""

import argparse
import csv
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_row_density(crop_bgr):
    """
    'zelda' preset — row-density furigana suppression.
    Pipeline: threshold-160 → blank sparse rows (42% of median density)
              → 2x Lanczos → 20px black border.
    Source: zelda_translator_working_nlp.py / zelda_translator_paddle_ocr_base.py
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop_bgr.copy()

    result = np.zeros_like(crop_bgr)
    result[mask == 255] = (255, 255, 255)

    row_density = mask.sum(axis=1) / 255.0
    non_zero = row_density[row_density > 0]
    if len(non_zero) > 0:
        median_density = float(np.median(non_zero))
        furi_threshold = median_density * 0.42
        for i, d in enumerate(row_density):
            if 0 < d < furi_threshold:
                result[i, :] = 0

    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


def _remove_furigana_components_bgr(bgr_image):
    """
    CC-based furigana removal operating on a BGR image.
    Converts to grayscale internally, removes isolated small components,
    returns a grayscale array.
    Source: zelda_translator_paddle_ocr_furigana_box.py (PIL-based version adapted here)
    """
    pil = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    arr = np.array(pil.convert("L"))

    dark_bg = np.mean(arr) < 127
    binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return arr

    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
    if not heights:
        return arr

    median_h = float(np.median(heights))
    furigana_threshold = median_h * 0.55

    centres = [
        stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
        for i in range(1, num_labels)
    ]
    large_indices = [
        idx for idx in range(len(centres))
        if stats[idx + 1, cv2.CC_STAT_HEIGHT] >= furigana_threshold
    ]

    out = arr.copy()
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

    return out


def preprocess_cc_pre_upscale(crop_bgr):
    """
    'zeldacc' preset — CC furigana removal at original resolution BEFORE upscaling.
    Pipeline: threshold-160 → CC removal on raw binary → 2x Lanczos → 20px black border.
    Source: zelda_translator_paddle_ocr_furigana_box.py
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop_bgr.copy()

    result = np.zeros_like(crop_bgr)
    result[mask == 255] = (255, 255, 255)

    # CC removal at original resolution (avoids Lanczos halo artifacts)
    gray_cleaned = _remove_furigana_components_bgr(result)
    result = cv2.cvtColor(gray_cleaned, cv2.COLOR_GRAY2BGR)

    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return result


def preprocess_cc_post_upscale(crop_bgr):
    """
    'zeldacc (post-upscale)' variant — CC furigana removal AFTER upscaling.
    Pipeline: threshold-160 → 2x Lanczos → 20px black border → CC removal on gray.
    Source: zelda_translator_working_av_furigana_box.py
    Note: runs CC on upscaled grayscale, unlike the Paddle version which runs pre-upscale.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        return crop_bgr.copy()

    result = np.zeros_like(crop_bgr)
    result[mask == 255] = (255, 255, 255)

    h, w = result.shape[:2]
    result = cv2.resize(result, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    result = cv2.copyMakeBorder(result, 20, 20, 20, 20,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # CC removal on the upscaled grayscale (AV variant behaviour)
    gray_up = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    dark_bg = np.mean(gray_up) < 127
    binary = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels > 1:
        heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, num_labels)]
        median_h = float(np.median(heights))
        furigana_threshold = median_h * 0.55
        centres = [
            stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2.0
            for i in range(1, num_labels)
        ]
        large_indices = [
            idx for idx in range(len(centres))
            if stats[idx + 1, cv2.CC_STAT_HEIGHT] >= furigana_threshold
        ]
        bg_value = 255 if not dark_bg else 0
        for i in range(1, num_labels):
            h_s = stats[i, cv2.CC_STAT_HEIGHT]
            w_s = stats[i, cv2.CC_STAT_WIDTH]
            if h_s >= furigana_threshold or w_s >= median_h * 2:
                continue
            cy = centres[i - 1]
            has_large_neighbour = any(
                abs(centres[j] - cy) < median_h * 1.5
                for j in large_indices
            )
            if not has_large_neighbour:
                gray_up[labels == i] = bg_value

    return cv2.cvtColor(gray_up, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING (PaddleOCR)
# ─────────────────────────────────────────────────────────────────────────────

_EXACT_FIXES = {
    "にゅっくり": "に ゆっくり",
}

def _fix_exact(text):
    for wrong, correct in _EXACT_FIXES.items():
        text = text.replace(wrong, correct)
    return text

def _fix_hira_before_kata_N(text):
    result = list(text)
    for i in range(len(result) - 1):
        if 'ぁ' <= result[i] <= 'ん' and result[i + 1] == 'ン':
            result[i] = chr(ord(result[i]) + 0x60)
    return ''.join(result)

def _postprocess_paddle(pairs):
    out = []
    for t, s in pairs:
        t = _fix_exact(t)
        t = _fix_hira_before_kata_N(t)
        if len(t.strip()) > 3:
            out.append((t, s))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def _paddle_ocr_on_frame(bgr_frame, ocr_instance):
    """
    Run PaddleOCR on a preprocessed BGR frame. Applies bimodal gap split
    furigana filter on bounding boxes + isolation guard, then post-processing.
    Returns (japanese_text, elapsed_ms).
    Source: zelda_translator_paddle_ocr_base.py / zelda_translator_paddle_ocr_furigana_box.py
    """
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, bgr_frame)

    try:
        result = ocr_instance.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_texts, all_scores, all_heights, all_centres = [], [], [], []
    for res in (result or []):
        polys = res.get("rec_polys") or res.get("rec_boxes") or []
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

    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres = map(list, zip(*combined))

    if all_heights:
        sorted_h = sorted(all_heights)
        furi_thresh = sorted_h[0]
        if len(sorted_h) >= 2:
            gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > sorted_h[-1] * 0.20:
                furi_thresh = sorted_h[gap_idx + 1]
        median_h = float(np.median(all_heights))
        large_centres = [c for h, c in zip(all_heights, all_centres) if h >= furi_thresh]
        filtered = []
        for t, s, h, cy in zip(all_texts, all_scores, all_heights, all_centres):
            if h >= furi_thresh:
                filtered.append((t, s))
            elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                filtered.append((t, s))
        texts = [t for t, _ in filtered]
        scores = [s for _, s in filtered]
    else:
        texts, scores = all_texts, all_scores

    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    texts = [t for t, _ in filtered_pairs]

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return "\n".join(texts), elapsed_ms


def _apple_vision_ocr_on_frame(bgr_frame, use_language_correction, Vision, Quartz):
    """
    Run Apple Vision OCR on a preprocessed BGR frame. Applies post-OCR
    bimodal gap split isolation guard on bounding boxes.
    Returns (japanese_text, elapsed_ms).
    Source: zelda_translator_working_av_furigana_box.py (with language_correction flag)
    """
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, bgr_frame)

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
        request.setUsesLanguageCorrection_(use_language_correction)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)

        img_h = bgr_frame.shape[0]

        if raw_observations:
            candidates = []
            for text, bbox in raw_observations:
                px_h = bbox.size.height * img_h
                cy = (1.0 - (bbox.origin.y + bbox.size.height / 2.0)) * img_h
                candidates.append((text, px_h, cy))

            sorted_h = sorted(h for _, h, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            median_h = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

            kept = []
            for text, px_h, cy in candidates:
                if px_h >= furi_thresh:
                    kept.append(text)
                elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    kept.append(text)

            japanese = " ".join(kept).strip()
        else:
            japanese = ""

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return japanese, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_av_rowdensity(image_path, Vision, Quartz):
    """
    Pipeline 1: Apple Vision + row-density furigana suppression.
    setUsesLanguageCorrection_(False) — matches working_nlp.py exactly.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_rowdensity", "status": "load_error"}

    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, use_language_correction=False, Vision=Vision, Quartz=Quartz)
    return {
        "engine":   "av_rowdensity",
        "status":   "ok",
        "text":     text,
        "elapsed":  elapsed_ms,
    }


def run_av_cc(image_path, Vision, Quartz):
    """
    Pipeline 2: Apple Vision + CC furigana removal post-upscale + post-OCR isolation guard.
    setUsesLanguageCorrection_(True) — matches working_av_furigana_box.py exactly.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_cc", "status": "load_error"}

    preprocessed = preprocess_cc_post_upscale(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, use_language_correction=True, Vision=Vision, Quartz=Quartz)
    return {
        "engine":   "av_cc",
        "status":   "ok",
        "text":     text,
        "elapsed":  elapsed_ms,
    }


def run_paddle_rowdensity(image_path, paddle_ocr):
    """
    Pipeline 3: PaddleOCR + row-density furigana suppression.
    Matches zelda_translator_paddle_ocr_base.py exactly.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_rowdensity", "status": "load_error"}

    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(preprocessed, paddle_ocr)
    return {
        "engine":   "paddle_rowdensity",
        "status":   "ok",
        "text":     text,
        "elapsed":  elapsed_ms,
    }


def run_paddle_cc(image_path, paddle_ocr):
    """
    Pipeline 4: PaddleOCR + CC furigana removal pre-upscale.
    Matches zelda_translator_paddle_ocr_furigana_box.py exactly.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_cc", "status": "load_error"}

    preprocessed = preprocess_cc_pre_upscale(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(preprocessed, paddle_ocr)
    return {
        "engine":   "paddle_cc",
        "status":   "ok",
        "text":     text,
        "elapsed":  elapsed_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT / REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result):
    engine = result.get("engine", "?")
    status = result.get("status", "?")

    print(f"\n{'─' * 55}")
    print(f"  {engine}")
    print(f"{'─' * 55}")

    if status == "skipped":
        print(f"  ⏭️  Skipped: {result.get('reason', '')}")
        return
    if status == "not_installed":
        print(f"  ⚠️  Not installed — {result.get('install', '')}")
        return
    if status == "load_error":
        print(f"  ❌  Could not load image")
        return
    if status == "ok" and not result.get("text", "").strip():
        print(f"  ⚠️  No text detected  ({result['elapsed']}ms)")
        return

    print(f"  ⏱️  {result['elapsed']}ms")
    print(f"  📝 Output:")
    for line in result["text"].splitlines():
        if line.strip():
            print(f"       {line}")


def print_summary(results):
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  {'Engine':<22} {'Status':<12} {'Time':>7}  Output preview")
    print(f"  {'─'*22} {'─'*12} {'─'*7}  {'─'*20}")
    for r in results:
        engine  = r.get("engine", "?")
        status  = r.get("status", "?")
        elapsed = f"{r['elapsed']}ms" if "elapsed" in r else "—"
        text    = r.get("text", "")
        preview = text.replace("\n", " / ")[:30] if text else "—"
        icon    = "✅" if (status == "ok" and text.strip()) else ("⏭️ " if status == "skipped" else "⚠️ ")
        print(f"  {icon} {engine:<20} {status:<12} {elapsed:>7}  {preview}")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def _collect_images(inputs):
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    seen, paths = set(), []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for c in sorted(p.iterdir()):
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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare the four Zelda translator OCR pipelines on dialogue images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="One or more image files and/or directories containing images",
    )
    args = parser.parse_args()

    image_paths = _collect_images(args.inputs)
    if not image_paths:
        print("❌  No valid images found.")
        sys.exit(1)

    first = Path(image_paths[0])
    csv_path = (first.parent if first.is_file() else Path(args.inputs[0])) / "zelda_ocr_results.csv"

    print(f"\n{'═' * 60}")
    print("  Zelda OCR Pipeline Comparison")
    print(f"{'═' * 60}")
    print(f"  Images   : {len(image_paths)} file(s)")
    for p in image_paths:
        print(f"             {p}")
    print(f"  Platform : {platform.system()} {platform.machine()}")
    print(f"  CSV      : {csv_path}")

    # ── Load PaddleOCR once ───────────────────────────────────────────────────
    paddle_ocr = None
    try:
        from paddleocr import PaddleOCR
        print("\n⏳  Loading PaddleOCR...")
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
        print("  ⚠️  PaddleOCR not installed — paddle pipelines will be skipped")

    # ── Load Apple Vision once ────────────────────────────────────────────────
    Vision = None
    Quartz = None
    if platform.system() == "Darwin":
        try:
            import Vision as _Vision
            import Quartz as _Quartz
            Vision = _Vision
            Quartz = _Quartz
            print("  ✅ Apple Vision ready")
        except ImportError:
            print("  ⚠️  pyobjc not installed — Apple Vision pipelines will be skipped")
    else:
        print("  ⚠️  Not macOS — Apple Vision pipelines will be skipped")

    # ── Process images ────────────────────────────────────────────────────────
    all_results = {}

    for idx, image_path in enumerate(image_paths):
        print(f"\n\n[Image {idx + 1}/{len(image_paths)}] {Path(image_path).name}")
        results = []

        # Pipeline 1: Apple Vision + row-density
        if Vision and Quartz:
            r = run_av_rowdensity(image_path, Vision, Quartz)
        else:
            r = {"engine": "av_rowdensity", "status": "skipped", "reason": "Apple Vision unavailable"}
        results.append(r)

        # Pipeline 2: Apple Vision + CC
        if Vision and Quartz:
            r = run_av_cc(image_path, Vision, Quartz)
        else:
            r = {"engine": "av_cc", "status": "skipped", "reason": "Apple Vision unavailable"}
        results.append(r)

        # Pipeline 3: Paddle + row-density
        if paddle_ocr:
            r = run_paddle_rowdensity(image_path, paddle_ocr)
        else:
            r = {"engine": "paddle_rowdensity", "status": "skipped", "reason": "PaddleOCR unavailable"}
        results.append(r)

        # Pipeline 4: Paddle + CC
        if paddle_ocr:
            r = run_paddle_cc(image_path, paddle_ocr)
        else:
            r = {"engine": "paddle_cc", "status": "skipped", "reason": "PaddleOCR unavailable"}
        results.append(r)

        all_results[image_path] = results

        print(f"\n{'═' * 60}")
        print(f"  RESULTS — {Path(image_path).name}")
        print(f"{'═' * 60}")
        for r in results:
            print_result(r)
        print_summary(results)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "engine", "status", "elapsed_ms", "output"])
        for image_path, results in all_results.items():
            stem = Path(image_path).stem
            for r in results:
                status = r.get("status", "")
                if status == "ok":
                    output = r.get("text", "")
                elif status == "skipped":
                    output = f"[skipped: {r.get('reason', '')}]"
                elif status == "not_installed":
                    output = "[not installed]"
                else:
                    output = f"[{status}]"
                writer.writerow([
                    stem,
                    r.get("engine", "?"),
                    status,
                    r.get("elapsed", ""),
                    output,
                ])

    print(f"\n✅  CSV written → {csv_path}")


if __name__ == "__main__":
    main()
