"""
zelda_translator_paddle_reading_match.py
=========================================
Variant: PaddleOCR v5 mobile  |  Postprocessing: reading-match furigana filter
Preprocessing: row-density furigana suppression
"""
import re
from PIL import Image
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import tempfile
import time
import zelda_core

from zelda_core import _tagger, _get_jmd

# ── PaddleOCR initialisation ──────────────────────────────────────────────────
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

# ── Reading-match furigana filter ─────────────────────────────────────────────
# Drops pure-kana OCR tokens that exactly match a kanji reading collected from
# the same text, identifying them as furigana that survived row-density + the
# bimodal height filter.  Two-stage reading collection:
#   Stage 1 — MeCab compound-word readings (handles e.g. 食材 → しょくざい)
#   Stage 2 — jamdict per-character on/kun'yomi (handles single-kanji furigana
#              like び above 火, み above 見, ぬし above 主)
# Both stages reuse the already-initialised _tagger and _jmd instances.

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

    # Postprocessing: exact string fixes + noise-length filter, scores in sync
    filtered_pairs = _postprocess_paddle(list(zip(texts, scores)))
    texts = [t for t, _ in filtered_pairs]

    # Reading-match filter: drop any surviving pure-kana token that exactly
    # matches a kanji reading collected from the full text.
    texts = filter_reading_match(texts)

    elapsed_ms = round((time.perf_counter() - t0) * 1000)
    japanese = "\n".join(texts)
    return japanese, elapsed_ms

do_ocr = paddle_ocr

def preprocess_crop(crop):
    """Row-density furigana suppression — same as the working_nlp baseline.
    The reading-match filter inside paddle_ocr handles furigana at the
    text-token level, so image preprocessing stays as the simple row-density pass.
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
