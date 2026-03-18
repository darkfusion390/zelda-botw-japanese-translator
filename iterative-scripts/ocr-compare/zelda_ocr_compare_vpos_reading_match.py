"""
zelda_ocr_compare.py
--------------------
Compares four OCR pipelines across two engines (Apple Vision, PaddleOCR),
each with and without the reading-match furigana output filter:

  Apple Vision:
    1. av_rowdensity    — row-density image preprocessing (baseline)
    2. av_reading_match — row-density preproc + reading-match output filter

  PaddleOCR:
    3. paddle_rowdensity    — row-density image preprocessing (baseline)
    4. paddle_reading_match — row-density preproc + reading-match output filter

Why vpos was removed:
  After inspecting the test images, furigana in all test games sits within
  the vertical extent of the main text line (directly above its kanji, inside
  the same crop row), not in a separate band above the dialogue box. A
  vertical-position gate based on image height or median top-edge cannot
  distinguish furigana boxes from line-1 main-text boxes because they share
  the same y-region. vpos was removed to avoid first-line content drops.

Reading-match filter (two-stage):
  Stage 1 — MeCab token readings (fugashi):
    Segments the full OCR text with MeCab and collects the predicted kana
    reading of every kanji-containing token. Handles multi-kanji compound
    words correctly (e.g. 食材 -> しょくざい, 栄養 -> えいよう).

  Stage 2 — Per-character jamdict readings:
    For every individual kanji character in the OCR text, looks up its
    on'yomi and kun'yomi readings in JMdict. This catches furigana for kanji
    inside compound verbs that MeCab segments as a unit:
      e.g. 見つかる -> MeCab reading みつかる; the standalone furigana み
           (above 見) would not match the compound reading without this stage.
           jamdict adds み (kun'yomi of 見) to the candidate set.
    Also ensures single-kanji furigana (び above 火, や above 焼, い above 行,
    ぬし above 主, えら above 選) are always caught.
    Okurigana suffixes are stripped (み.る -> み, い.く -> い).

  A pure-kana OCR string is dropped as furigana if it exactly matches any
  reading collected by either stage (after katakana->hiragana normalisation).

  Protected by design:
    - Multi-kana content words (ゆっくり, もしかして, いたずら) — not kanji readings
    - Grammar particles (は, が, の) — too short or not in reading sets
    - Kana embedded inside kanji+kana OCR strings — never candidates since the
      whole mixed string is not pure kana

Dependencies:
    PaddleOCR:  pip install paddleocr paddlepaddle
    Apple OCR:  pip install pyobjc-framework-Vision pyobjc-framework-Quartz
                (macOS only — skipped automatically on other platforms)
    MeCab/NLP:  pip install fugashi unidic-lite
    JMdict:     pip install jamdict jamdict-data
                (reading-match pipelines degrade gracefully if either missing)

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
# NLP — MeCab + jamdict initialisation (lazy, graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

_tagger = None
_tagger_loaded = False
_jmd = None
_jmd_loaded = False


def _get_tagger():
    """Return a fugashi MeCab tagger, or None if fugashi is not installed."""
    global _tagger, _tagger_loaded
    if _tagger_loaded:
        return _tagger
    _tagger_loaded = True
    try:
        import fugashi
        _tagger = fugashi.Tagger()
        print("  ✅ fugashi (MeCab) ready")
    except ImportError:
        _tagger = None
        print("  ⚠️  fugashi not installed — reading-match will skip MeCab stage")
    return _tagger


def _get_jmd():
    """Return a Jamdict instance, or None if jamdict is not installed."""
    global _jmd, _jmd_loaded
    if _jmd_loaded:
        return _jmd
    _jmd_loaded = True
    try:
        from jamdict import Jamdict
        _jmd = Jamdict()
        print("  ✅ jamdict ready — per-character kanji readings enabled")
    except ImportError:
        _jmd = None
        print("  ⚠️  jamdict not installed — reading-match will skip per-character stage")
    return _jmd


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


def _collect_kanji_readings(text: str, tagger, jmd) -> set:
    """
    Collect all kana readings of kanji that appear in the given text string.

    Stage 1 — MeCab token readings:
      For each MeCab token that contains kanji, add its predicted kana reading
      (both katakana and hiragana forms) to the candidate set.

    Stage 2 — Per-character jamdict readings:
      For each individual kanji character in the text, look up all on'yomi and
      kun'yomi readings in JMdict. Strip okurigana suffixes (e.g. み.る -> み)
      so that the standalone furigana character matches.

    Returns a set of hiragana strings that, if found as a standalone pure-kana
    OCR box, should be treated as furigana and dropped.
    """
    readings: set = set()

    # ── Stage 1: MeCab compound-word readings ────────────────────────────────
    if tagger:
        try:
            for word in tagger(text):
                if _has_kanji(word.surface):
                    try:
                        kana = word.feature.kana
                    except AttributeError:
                        kana = None
                    if kana:
                        readings.add(_kata_to_hira(kana))
                        readings.add(kana)
        except Exception:
            pass

    # ── Stage 2: per-character on'yomi / kun'yomi via jamdict ────────────────
    if jmd:
        seen_chars: set = set()
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff' and ch not in seen_chars:
                seen_chars.add(ch)
                try:
                    result = jmd.lookup(ch)
                    for char_entry in (result.chars or []):
                        for rm_group in (char_entry.rm_groups or []):
                            for reading in (rm_group.readings or []):
                                r_val = getattr(reading, 'value', '') or ''
                                if not r_val:
                                    continue
                                # Strip okurigana (み.る -> み, い.く -> い)
                                # and irregular reading markers (む―す -> む)
                                r_base = r_val.split('.')[0].split('―')[0].strip()
                                if r_base and _is_pure_kana(r_base):
                                    readings.add(_kata_to_hira(r_base))
                except Exception:
                    pass

    return readings


def filter_reading_match(texts: list, tagger, jmd) -> list:
    """
    Remove furigana strings from a list of OCR text tokens using reading-match.

    Collects kana readings of all kanji in the joined OCR text (MeCab + jamdict),
    then drops any token that is pure kana and exactly matches one of those
    readings (after katakana->hiragana normalisation).

    The full joined text is used for reading collection so MeCab has sentence
    context for accurate compound-word segmentation. Drop decisions are made
    per-token.
    """
    if (not tagger and not jmd) or not texts:
        return texts

    joined = " ".join(texts)
    readings = _collect_kanji_readings(joined, tagger, jmd)

    if not readings:
        return texts

    kept = []
    for t in texts:
        t_stripped = t.strip()
        t_hira = _kata_to_hira(t_stripped)
        if _is_pure_kana(t_stripped) and t_hira in readings:
            continue  # pure kana matching a kanji reading -> furigana, drop
        kept.append(t)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING (shared by all four pipelines)
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
    N only exists in katakana; any hiragana before it is a recognition error.
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

def _paddle_ocr_on_frame(bgr_frame, ocr_instance,
                          use_reading_match: bool = False,
                          tagger=None, jmd=None):
    """
    Run PaddleOCR on a preprocessed BGR frame.

    use_reading_match: if True, applies the reading-match filter after the
      baseline bimodal gap split + isolation guard + postprocessing.

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

    # Sort top-to-bottom by vertical centre
    if all_texts:
        combined = sorted(
            zip(all_texts, all_scores, all_heights, all_centres),
            key=lambda x: x[3]
        )
        all_texts, all_scores, all_heights, all_centres = map(list, zip(*combined))

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
        filtered = []
        for t, s, h, cy in zip(all_texts, all_scores, all_heights, all_centres):
            if h >= furi_thresh:
                filtered.append((t, s))
            elif large_centres and any(
                    abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                filtered.append((t, s))
        texts  = [t for t, _ in filtered]
        scores = [s for _, s in filtered]
    else:
        texts, scores = all_texts, all_scores

    # Paddle postprocessing: exact string fixes + noise-length filter
    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    texts = [t for t, _ in filtered_pairs]

    # ── Reading-match filter ──────────────────────────────────────────────────
    if use_reading_match and texts:
        texts = filter_reading_match(texts, tagger, jmd)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return "\n".join(texts), elapsed_ms


def _apple_vision_ocr_on_frame(bgr_frame, Vision, Quartz,
                                use_reading_match: bool = False,
                                tagger=None, jmd=None):
    """
    Run Apple Vision OCR on a preprocessed BGR frame.

    use_reading_match: if True, applies the reading-match filter after the
      baseline bimodal gap split + isolation guard.

    setUsesLanguageCorrection_(False) — matches the best-performing AV
    baseline (zelda_translator_working_nlp.py).

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
            # Convert AV bounding boxes (origin bottom-left, y increases up)
            # to image coordinates (origin top-left, y increases down).
            candidates = []
            for text, bbox in raw_observations:
                px_h  = bbox.size.height * img_h
                top_y = (1.0 - (bbox.origin.y + bbox.size.height)) * img_h
                cy    = top_y + px_h / 2.0
                candidates.append((text, px_h, cy))

            # ── Baseline: bimodal gap split + isolation guard ─────────────────
            sorted_h = sorted(h for _, h, _ in candidates)
            furi_thresh = sorted_h[0]
            if len(sorted_h) >= 2:
                gaps = [(sorted_h[i + 1] - sorted_h[i], i)
                        for i in range(len(sorted_h) - 1)]
                max_gap, gap_idx = max(gaps)
                if max_gap > sorted_h[-1] * 0.20:
                    furi_thresh = sorted_h[gap_idx + 1]

            median_h = float(np.median(sorted_h))
            large_centres = [cy for _, h, cy in candidates if h >= furi_thresh]

            texts = []
            for text, px_h, cy in candidates:
                if px_h >= furi_thresh:
                    texts.append(text)
                elif large_centres and any(
                        abs(cy - lc) < median_h * 1.5 for lc in large_centres):
                    texts.append(text)

            # ── Reading-match filter ──────────────────────────────────────────
            if use_reading_match and texts:
                texts = filter_reading_match(texts, tagger, jmd)

            japanese = " ".join(texts).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    return japanese, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_av_rowdensity(image_path, Vision, Quartz):
    """Pipeline 1: Apple Vision + row-density preprocessing. Baseline."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_rowdensity", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, Vision, Quartz, use_reading_match=False)
    return {"engine": "av_rowdensity", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_av_reading_match(image_path, Vision, Quartz, tagger, jmd):
    """Pipeline 2: Apple Vision + row-density + reading-match filter."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "av_reading_match", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _apple_vision_ocr_on_frame(
        preprocessed, Vision, Quartz,
        use_reading_match=True, tagger=tagger, jmd=jmd)
    return {"engine": "av_reading_match", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_paddle_rowdensity(image_path, paddle_ocr):
    """Pipeline 3: PaddleOCR + row-density preprocessing. Baseline."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_rowdensity", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(
        preprocessed, paddle_ocr, use_reading_match=False)
    return {"engine": "paddle_rowdensity", "status": "ok",
            "text": text, "elapsed": elapsed_ms}


def run_paddle_reading_match(image_path, paddle_ocr, tagger, jmd):
    """Pipeline 4: PaddleOCR + row-density + reading-match filter."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"engine": "paddle_reading_match", "status": "load_error"}
    preprocessed = preprocess_row_density(img_bgr)
    text, elapsed_ms = _paddle_ocr_on_frame(
        preprocessed, paddle_ocr,
        use_reading_match=True, tagger=tagger, jmd=jmd)
    return {"engine": "paddle_reading_match", "status": "ok",
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
        description="Compare four Zelda OCR pipelines (2 engines x baseline + reading-match).",
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
    print("  Zelda OCR Pipeline Comparison — 4 pipelines")
    print(f"{'═' * 65}")
    print(f"  Images    : {len(image_paths)} file(s)")
    for p in image_paths:
        print(f"              {p}")
    print(f"  Platform  : {platform.system()} {platform.machine()}")
    print(f"  CSV       : {csv_path}")
    print(f"  Pipelines : av_rowdensity | av_reading_match | "
          f"paddle_rowdensity | paddle_reading_match")

    # ── Load NLP libraries once ───────────────────────────────────────────────
    print("\n⏳  Loading NLP libraries...")
    tagger = _get_tagger()
    jmd    = _get_jmd()
    if not tagger and not jmd:
        print("  ⚠️  No NLP libraries available — reading-match pipelines will "
              "produce identical output to baselines")

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

        # Apple Vision pipelines (1–2)
        if Vision and Quartz:
            results.append(run_av_rowdensity(image_path, Vision, Quartz))
            results.append(run_av_reading_match(
                image_path, Vision, Quartz, tagger, jmd))
        else:
            for engine in ("av_rowdensity", "av_reading_match"):
                results.append({"engine": engine, "status": "skipped",
                                 "reason": "Apple Vision unavailable"})

        # PaddleOCR pipelines (3–4)
        if paddle_ocr:
            results.append(run_paddle_rowdensity(image_path, paddle_ocr))
            results.append(run_paddle_reading_match(
                image_path, paddle_ocr, tagger, jmd))
        else:
            for engine in ("paddle_rowdensity", "paddle_reading_match"):
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
