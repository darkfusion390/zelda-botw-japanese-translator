"""
zelda_translator_av_reading_match.py
======================================
Variant: Apple Vision OCR  |  Preprocessing: row-density + reading-match furigana filter
Reading-match uses MeCab + jamdict to drop pure-kana tokens that are kanji readings.
"""
import Vision
import Quartz
import cv2
import numpy as np
import os
import tempfile
import time
import zelda_core

from zelda_core import _tagger, _get_jmd

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

def _collect_kanji_readings(text: str) -> set:
    """
    Collect all kana readings of kanji in text using _tagger (MeCab) and _jmd (jamdict).

    Stage 1 — MeCab compound-word readings:
      For each token containing kanji, add its predicted kana reading to the set.
      Handles multi-kanji words correctly (e.g. 食材 → しょくざい, 栄養 → えいよう).

    Stage 2 — Per-character jamdict readings:
      For each individual kanji character, look up all on'yomi and kun'yomi entries.
      Strips okurigana suffixes (み.る → み, い.く → い) so single-mora furigana
      like び (above 火) or み (above 見) match even when MeCab embeds them in a
      compound reading.

    Returns a set of hiragana strings; any pure-kana OCR token that exactly
    matches one of these should be treated as furigana and dropped.
    """
    readings: set = set()

    # Stage 1: MeCab compound-word readings
    try:
        for word in _tagger(text):
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

    # Stage 2: per-character on'yomi / kun'yomi via jamdict
    seen_chars: set = set()
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' and ch not in seen_chars:
            seen_chars.add(ch)
            try:
                result = _get_jmd().lookup(ch)
                for char_entry in (result.chars or []):
                    for rm_group in (char_entry.rm_groups or []):
                        for reading in (rm_group.readings or []):
                            r_val = getattr(reading, 'value', '') or ''
                            if not r_val:
                                continue
                            # Strip okurigana (み.る → み) and irregular markers
                            r_base = r_val.split('.')[0].split('―')[0].strip()
                            if r_base and _is_pure_kana(r_base):
                                readings.add(_kata_to_hira(r_base))
            except Exception:
                pass

    return readings

def filter_reading_match(texts: list) -> list:
    """
    Drop furigana tokens from a list of OCR text strings using reading-match.

    Joins all tokens to give MeCab full sentence context for compound-word
    segmentation, collects kanji readings via _collect_kanji_readings, then
    removes any token that is pure kana and exactly matches a collected reading
    (after katakana→hiragana normalisation).

    Protected by design:
      - Multi-kana content words (ゆっくり, もしかして) — not kanji readings
      - Grammar particles — too short or absent from any kanji reading set
      - Mixed kanji+kana tokens — never pure kana, never candidates for removal
    """
    if not texts:
        return texts

    joined = " ".join(texts)
    readings = _collect_kanji_readings(joined)

    if not readings:
        return texts

    kept = []
    for t in texts:
        t_stripped = t.strip()
        t_hira = _kata_to_hira(t_stripped)
        if _is_pure_kana(t_stripped) and t_hira in readings:
            continue  # pure kana matching a kanji reading → furigana, drop
        kept.append(t)
    return kept


# ── OCR ───────────────────────────────────────────────────────────────────────

def apple_vision_ocr(frame):
    """Run Apple Vision OCR with bimodal height filtering + reading-match furigana filter.
    First pass: bounding-box height split drops observations whose height is
    below the bimodal gap threshold (furigana are shorter than main text).
    Second pass: filter_reading_match drops any surviving pure-kana token
    that exactly matches a kanji reading collected from the full text.
    Language correction is disabled. Returns (japanese_str, elapsed_ms)."""
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

            # Reading-match filter: drop any surviving pure-kana token that
            # exactly matches a kanji reading collected from the full text.
            texts = filter_reading_match(texts)

            japanese = " ".join(texts).strip()

    finally:
        os.unlink(tmp_path)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    # print(f"⏱  [ocr] {elapsed_ms}ms  →  {japanese}")
    return japanese, elapsed_ms

do_ocr = apple_vision_ocr

def preprocess_crop(crop):
    """Row-density furigana suppression — identical to the working_nlp baseline.
    The reading-match filter in apple_vision_ocr handles furigana at the
    OCR level, so preprocessing remains the same simple row-density pass.
    Returns a white-on-black BGR image ready for OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        # Nothing survived the threshold — crop is too dark to contain text.
        # Return a black frame at the upscaled size so downstream always gets
        # a consistent pure B&W image rather than a raw color frame.
        h, w = crop.shape[:2]
        return np.zeros((h * 2 + 40, w * 2 + 40, 3), dtype=np.uint8)
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
