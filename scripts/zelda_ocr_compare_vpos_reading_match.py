"""
zelda_ocr_compare.py
--------------------
Compares six OCR pipelines across two engines (Apple Vision, PaddleOCR),
each paired with three post-OCR furigana filtering strategies:

  Apple Vision:
    1. av_rowdensity    — row-density image preprocessing (baseline)
    2. av_reading_match — row-density preproc + reading-match output filter
    3. av_vpos          — row-density preproc + vertical-position-gate filter

  PaddleOCR:
    4. paddle_rowdensity    — row-density image preprocessing (baseline)
    5. paddle_reading_match — row-density preproc + reading-match output filter
    6. paddle_vpos          — row-density preproc + vertical-position-gate filter

All six pipelines share the same row-density preprocessing. The two new
filters operate purely on OCR output (text strings + bounding boxes) with
no changes to the image sent to the OCR engine.

Reading-match filter:
  Uses MeCab (fugashi) to segment the full OCR output and predict the kana
  reading of every kanji-containing token. Any detected text string that is
  pure kana AND exactly matches one of those predicted readings is dropped
  as furigana. Catches e.g. しょくさい (reading of 食材) while protecting
  ゆっくり (not a kanji reading) and grammatical particles.

Vertical-position gate:
  Computes the median top-edge y-coordinate of all detected boxes. Any box
  whose top edge sits more than VPOS_THRESHOLD (15%) of the image height
  above the median is dropped as furigana. Furigana always floats above the
  main text line regardless of glyph size or content.

Dependencies:
    PaddleOCR:  pip install paddleocr paddlepaddle
    Apple OCR:  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
                (macOS only)
    MeCab/NLP:  pip install fugashi unidic-lite
                (reading-match pipelines only; others work without it)

Usage:
    python zelda_ocr_compare.py path/to/dialogue.png
    python zelda_ocr_compare.py path/to/images/

CSV output:
    zelda_ocr_results.csv written next to the first input file (or cwd).
    Columns: image, engine, status, elapsed_ms, output
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
# NLP — MeCab initialisation (lazy, graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

_tagger = None
_tagger_loaded = False


def _get_tagger():
    """Return a fugashi MeCab tagger, or None if fugashi is not installed."""
    global _tagger, _tagger_loaded
    if _tagger_loaded:
        return _tagger
    _tagger_loaded = True
    try:
        import fugashi
        _tagger = fugashi.Tagger()
        print("  ✅ fugashi (MeCab) ready — reading-match pipelines enabled")
    except ImportError:
        _tagger = None
        print("  ⚠️  fugashi not installed — reading-match pipelines will skip the filter")
    return _tagger


def _is_pure_kana(text: str) -> bool:
    """True if text contains only hiragana, katakana, and/or prolonged sound mark."""
    if not text:
        return False
    for ch in text:
        if not ('\u3040' <= ch <= '\u309f'    # hiragana
                or '\u30a0' <= ch <= '\u30ff'  # katakana
                or ch == '\u30fc'):             # prolonged sound mark ー
            return False
    return True


def _has_kanji(text: str) -> bool:
    """True if text contains at least one CJK unified ideograph."""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def _kata_to_hira(text: str) -> str:
    """Convert katakana to hiragana for normalised comparison."""
    return "".join(
        chr(ord(c) - 0x60) if '\u30a1' <= c <= '\u30f6' else c
        for c in text
    )


def filter_reading_match(texts: list, tagger) -> list:
    """
    Remove furigana strings from a list of OCR text tokens using reading-match.

    Joins all texts, segments with MeCab, and collects the predicted kana
    readings of every kanji-containing token. Any text item that is pure kana
    and exactly matches one of those readings (after katakana->hiragana
    normalisation) is dropped as furigana.

    Protected by design:
      - Content kana words like ゆっくり, もしかして — not kanji readings
      - Grammar particles は, が, の, etc. — too short to match a reading
      - Contracted small kana ゃ in じゃ — single char, not a full reading
    """
    if not tagger or not texts:
        return texts

    joined = " ".join(texts)

    # Collect predicted kana readings of kanji tokens (both katakana and hiragana forms)
    kanji_readings: set = set()
    try:
        for word in tagger(joined):
            if _has_kanji(word.surface):
                try:
                    kana = word.feature.kana
                except AttributeError:
                    kana = None
                if kana:
                    kanji_readings.add(kana)
                    kanji_readings.add(_kata_to_hira(kana))
    except Exception:
        return texts

    if not kanji_readings:
        return texts

    kept = []
    for t in texts:
        t_stripped = t.strip()
        t_hira = _kata_to_hira(t_stripped)
        if _is_pure_kana(t_stripped) and t_hira in kanji_readings:
            # Pure kana that exactly matches a kanji reading — drop as furigana
            continue
        kept.append(t)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (shared by all six pipelines)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_row_density(crop_bgr):
    """
    Row-density furigana suppression.
    Pipeline: threshold-160 -> blank sparse rows (42% of median density)
              -> 2x Lanczos -> 20px black border.
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


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING (PaddleOCR — shared fixes)
# ─────────────────────────────────────────────────────────────────────────────

_EXACT_FIXES = {
    "にゅっくり": "に ゆっくり",
}


def _fix_exact(text: str) -> str:
    for wrong, correct in _EXACT_FIXES.items():
        text = text.replace(wrong, correct)
    return text


def _fix_hira_before_kata_N(text: str) -> str:
    """Convert hiragana immediately before katakana N to its katakana equivalent.
    N only exists in katakana; hiragana before it is a recognition error.
    Fixes e.g. riNgo -> RiNgo."""
    result = list(text)
    for i in range(len(result) - 1):
        if 'ぁ' <= result[i] <= 'ん' and result[i + 1] == 'ン':
            result[i] = chr(ord(result[i]) + 0x60)
    return ''.join(result)


def _postprocess_paddle(pairs: list) -> list:
    """Apply targeted fixes to (text, score) pairs; drop noise lines (len <= 3)."""
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

# Vertical position gate: drop boxes whose top edge is more than this fraction
# of image height above the median top edge of all surviving boxes.
VPOS_THRESHOLD = 0.15


def _paddle_ocr_on_frame(bgr_frame, ocr_instance,
                          filter_mode: str = "baseline",
                          tagger=None):
    """
    Run PaddleOCR on a preprocessed BGR frame.

    filter_mode:
      "baseline"      — bimodal gap split + isolation guard (original)
      "reading_match" — baseline then reading-match filter
      "vpos"          — baseline then vertical-position gate

    Returns (japanese_text, elapsed_ms).
    """
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, bgr_frame)

    try:
        result = ocr_instance.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    all_texts, all_scores, all_heights, all_centres, all_tops = \
        [], [], [], [], []
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
            all_tops.append(y_min)

    # Sort top-to-bottom by vertical centre
    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres, all_tops),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres, all_tops = \
            map(list, zip(*combined))

    # ── Baseline: bimodal gap split + isolation guard ─────────────────────────
    if all_heights:
        sorted_h = sorted(all_heights)
        furi_thresh = sorted_h[0]
        if len(sorted_h) >= 2:
            gaps = [(sorted_h[i + 1] - sorted_h[i], i)
                    for i in range(len(sorted_h) - 1)]
            max_gap, gap_idx = max(gaps)
            if max_gap > sorted_h[-1] * 0.20:
                furi_thresh = sorted_h[gap_idx + 1]
        median_h = float(np.median(all_heights))
        large_centres = [c for h, c in zip(all_heights, all_centres)
                         if h >= furi_thresh]

        filtered_with_tops = []
        for t, s, h, cy, top in zip(
                all_texts, all_scores, all_heights, all_centres, all_tops):
            if h >= furi_thresh:
                filtered_with_tops.append((t, s, top))
            elif large_centres and any(
                    abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                filtered_with_tops.append((t, s, top))

        texts  = [t for t, _, _ in filtered_with_tops]
        scores = [s for _, s, _ in filtered_with_tops]
        tops   = [top for _, _, top in filtered_with_tops]
    else:
        texts, scores, tops = all_texts, all_scores, all_tops

    # Paddle postprocessing (exact string fixes + noise-length filter)
    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    # Keep tops in sync with surviving texts
    surviving_texts_set = [t for t, _ in filtered_pairs]
    surviving_tops = []
    ti = 0
    for t in surviving_texts_set:
        while ti < len(texts) and texts[ti] != t:
            ti += 1
        surviving_tops.append(tops[ti] if ti < len(tops) else 0.0)
        ti += 1
    texts = surviving_texts_set
    tops  = surviving_tops

    # ── Additional filter modes ───────────────────────────────────────────────
    if filter_mode == "reading_match" and tagger and texts:
        texts = filter_reading_match(texts, tagger)

    elif filter_mode == "vpos" and tops and texts:
        median_top = float(np.median(tops))
        img_h = bgr_frame.shape[0]
        vpos_cutoff = median_top - VPOS_THRESHOLD * img_h
        texts = [
            t for t, top in zip(texts, tops)
            if top >= vpos_cutoff
        ]

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return "\n".join(texts), elapsed_ms


def _apple_vision_ocr_on_frame(bgr_frame, Vision, Quartz,
                                filter_mode: str = "baseline",
                                tagger=None):
    """
    Run Apple Vision OCR on a preprocessed BGR frame.

    filter_mode:
      "baseline"      — bimodal gap split + isolation guard (original)
      "reading_match" — baseline then reading-match filter
      "vpos"          — baseline then vertical-position gate

    setUsesLanguageCorrection_(False) throughout — matches the best-performing
    AV baseline (zelda_translator_working_nlp.py).

    Returns (japanese_text, elapsed_ms).
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
        request.setUsesLanguageCorrection_(False)
        Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, {}).performRequests_error_([request], None)

        img_h = bgr_frame.shape[0]
        japanese = ""

        if raw_observations:
            # Convert AV bounding boxes (origin bottom-left, y up) to image coords
            # (origin top-left, y down).
            # top_y in image coords = (1 - (origin.y + height)) * img_h
            candidates = []
            for text, bbox in raw_observations:
                px_h  = bbox.size.height * img_h
                top_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
                cy    = top_y + px_h / 2.0
                candidates.append((text, px_h, cy, top_y))

            # ── Baseline: bimodal gap split + isolation guard ─────────────────
            sorted_h = sorted(h for _, h, _, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i)
                        for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            median_h = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy, _ in candidates if h >= furi_thresh]

            kept = []  # (text, top_y)
            for text, px_h, cy, top_y in candidates:
                if px_h >= furi_thresh:
                    kept.append((text, top_y))
                elif large_centres and any(
                        abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    kept.append((text, top_y))

            texts = [t for t, _ in kept]
            tops  = [top for _, top in kept]

            # ── Additional filter modes ───────────────────────────────────────
            if filter_mode == "reading_match" and tagger and texts:
                filtered = filter_reading_match(texts, tagger)
                texts = filtered

            elif filter_mode == "vpos" and tops and texts:
                median_top = float(np.median(tops))
                vpos_cutoff = median_top - VPOS_THRESHOLD * img_h
                texts = [
                    t for t, top in zip(texts, tops)
                    if top >= vpos_cutoff
                ]

            japanese = " ".join(texts).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return japanese, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_av_rowdensity(image_path, Vision, Quartz):
    """Pipeline 1: Apple Vision + row-density. Baseline."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_rowdensity", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, Vision, Quartz, filter_mode="baseline")
    return {"engine": "av_rowdensity", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_av_reading_match(image_path, Vision, Quartz, tagger):
    """Pipeline 2: Apple Vision + row-density + reading-match filter."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_reading_match", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, Vision, Quartz, filter_mode="reading_match", tagger=tagger)
    return {"engine": "av_reading_match", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_av_vpos(image_path, Vision, Quartz):
    """Pipeline 3: Apple Vision + row-density + vertical-position gate."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_vpos", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, Vision, Quartz, filter_mode="vpos")
    return {"engine": "av_vpos", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_paddle_rowdensity(image_path, paddle_ocr):
    """Pipeline 4: PaddleOCR + row-density. Baseline."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_rowdensity", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(
        preprocessed, paddle_ocr, filter_mode="baseline")
    return {"engine": "paddle_rowdensity", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_paddle_reading_match(image_path, paddle_ocr, tagger):
    """Pipeline 5: PaddleOCR + row-density + reading-match filter."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_reading_match", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(
        preprocessed, paddle_ocr, filter_mode="reading_match", tagger=tagger)
    return {"engine": "paddle_reading_match", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_paddle_vpos(image_path, paddle_ocr):
    """Pipeline 6: PaddleOCR + row-density + vertical-position gate."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_vpos", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(
        preprocessed, paddle_ocr, filter_mode="vpos")
    return {"engine": "paddle_vpos", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


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
    print(f"\n{'═' * 65}")
    print("  SUMMARY")
    print(f"{'═' * 65}")
    print(f"  {'Engine':<26} {'Status':<12} {'Time':>7}  Output preview")
    print(f"  {'─'*26} {'─'*12} {'─'*7}  {'─'*20}")
    for r in results:
        engine  = r.get("engine", "?")
        status  = r.get("status", "?")
        elapsed = f"{r['elapsed']}ms" if "elapsed" in r else "—"
        text    = r.get("text", "")
        preview = text.replace("\n", " / ")[:28] if text else "—"
        icon    = "✅" if (status == "ok" and text.strip()) else \
                  ("⏭️ " if status == "skipped" else "⚠️ ")
        print(f"  {icon} {engine:<24} {status:<12} {elapsed:>7}  {preview}")
    print(f"{'═' * 65}\n")


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
        description="Compare six Zelda OCR pipelines (2 engines x 3 filters).",
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
    csv_path = (
        (first.parent if first.is_file() else Path(args.inputs[0]))
        / "zelda_ocr_results.csv"
    )

    print(f"\n{'═' * 65}")
    print("  Zelda OCR Pipeline Comparison — 6 pipelines")
    print(f"{'═' * 65}")
    print(f"  Images        : {len(image_paths)} file(s)")
    for p in image_paths:
        print(f"                  {p}")
    print(f"  Platform      : {platform.system()} {platform.machine()}")
    print(f"  CSV           : {csv_path}")
    print(f"  Pipelines     : baseline | reading_match | vpos  (x2 engines)")
    print(f"  Vpos threshold: {VPOS_THRESHOLD*100:.0f}% of image height above median top edge")

    # ── Load MeCab once ───────────────────────────────────────────────────────
    tagger = _get_tagger()

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

        # Apple Vision pipelines (1–3)
        if Vision and Quartz:
            results.append(run_av_rowdensity(image_path, Vision, Quartz))
            results.append(run_av_reading_match(image_path, Vision, Quartz, tagger))
            results.append(run_av_vpos(image_path, Vision, Quartz))
        else:
            for engine in ("av_rowdensity", "av_reading_match", "av_vpos"):
                results.append({"engine": engine, "status": "skipped",
                                 "reason": "Apple Vision unavailable"})

        # PaddleOCR pipelines (4–6)
        if paddle_ocr:
            results.append(run_paddle_rowdensity(image_path, paddle_ocr))
            results.append(run_paddle_reading_match(image_path, paddle_ocr, tagger))
            results.append(run_paddle_vpos(image_path, paddle_ocr))
        else:
            for engine in ("paddle_rowdensity", "paddle_reading_match", "paddle_vpos"):
                results.append({"engine": engine, "status": "skipped",
                                 "reason": "PaddleOCR unavailable"})

        all_results[image_path] = results

        print(f"\n{'═' * 65}")
        print(f"  RESULTS — {Path(image_path).name}")
        print(f"{'═' * 65}")
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
