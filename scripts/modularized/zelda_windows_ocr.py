"""
zelda_windows_ocr.py
====================
Variant: Windows.Media.Ocr via PowerShell subprocess
Preprocessing: Row-density furigana suppression (Zelda Classic)

Drop-in replacement for zelda_apple_ocr.py on Windows.
Registers the same do_ocr / preprocess_crop interface that zelda_core expects,
so the entire pipeline — multi-region concurrent OCR, stability gates, translation,
learn mode, quiz — works identically.

Usage:
    python zelda_windows_ocr.py

Requirements:
  - Windows 10 / 11
  - PowerShell 5.1+ (built into Windows — no install)
  - Japanese language pack installed:
      Settings → Time & Language → Language & Region → Add Japanese

Threading model — see note at bottom of file.
"""

import cv2
import numpy as np
import os
import tempfile
import subprocess
import time
import threading
import zelda_core


# ── PowerShell OCR script ─────────────────────────────────────────────────────
# Written to a fresh temp .ps1 file on each call to avoid any path/quoting
# issues with Japanese characters. The script outputs one line per detected
# text line in the format:  Y_TOP|Y_BOTTOM|TEXT
# Python parses these, sorts by Y, and applies the bimodal furigana filter.

_PS_SCRIPT = r"""
param([string]$ImagePath)

# Force UTF-8 throughout the subprocess pipe
$OutputEncoding            = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding  = [System.Text.Encoding]::UTF8
[Console]::InputEncoding   = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Add-Type -AssemblyName System.Runtime.WindowsRuntime

# Helper: block synchronously on a WinRT IAsyncOperation<T>
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and
    $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($WinRtTask, $ResultType) {
    $NetTask = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask))
    $NetTask.Wait() | Out-Null
    $NetTask.Result
}

# Load WinRT namespaces into the session
[Windows.Storage.StorageFile,            Windows.Storage,          ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine,            Windows.Media.Ocr,        ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Globalization.Language,         Windows.Globalization,    ContentType=WindowsRuntime] | Out-Null

# Create Japanese OCR engine
$lang   = [Windows.Globalization.Language]::new('ja')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)

if ($null -eq $engine) {
    Write-Error "NOPACK"
    exit 1
}

# Load image → SoftwareBitmap
$file    = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
$stream  = Await ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap  = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

# Run OCR
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

# Emit Y_TOP|Y_BOTTOM|TEXT for each detected line
foreach ($line in $result.Lines) {
    $r    = $line.BoundingRect
    $text = ($line.Words | ForEach-Object { $_.Text }) -join ''
    Write-Output ("{0}|{1}|{2}" -f [int]$r.Y, [int]($r.Y + $r.Height), $text)
}
"""

# ── Thread safety ─────────────────────────────────────────────────────────────
# zelda_core's ocr_loop fires _ocr_fn concurrently via ThreadPoolExecutor —
# one thread per region. Each Windows OCR call is a separate PowerShell child
# process, so they are fully independent OS processes. Unlike PaddleOCR (which
# shares a single model object and needs a lock) or Apple Vision (which runs
# in-process), Windows OCR has no shared state between calls. No lock is needed.
#
# The only shared resource is the temp file system. Each call creates its own
# uniquely-named temp PNG and PS1, so concurrent calls never collide.
#
# PowerShell startup cost: ~150-250ms per call on a warm system. This is paid
# once per region per frame. With 2 regions and 2 threads the wall time is
# ~250ms (parallel), not 500ms (serial). Faster than PaddleOCR on CPU.
#
# GPU usage: Windows.Media.Ocr uses the Windows ML runtime which will use
# DirectML (GPU-accelerated) if a compatible GPU is available. On your system
# with an RX 6800, DirectML should accelerate the recognition model
# automatically — no configuration needed. You cannot directly control this
# but you will see it in Task Manager under GPU compute when OCR is running.
# This is a meaningful advantage over PaddleOCR which was strictly CPU-only.


def windows_ocr(frame, region_name="default"):
    """
    Run Windows.Media.Ocr on a preprocessed BGR frame via PowerShell subprocess.
    Returns (japanese_str, elapsed_ms) — same signature as apple_vision_ocr.

    Each call:
      1. Writes the frame to a temp PNG
      2. Writes the PS script to a temp PS1
      3. Spawns a PowerShell subprocess
      4. Parses Y_TOP|Y_BOTTOM|TEXT stdout lines
      5. Applies bimodal furigana filter on bounding-box heights
      6. Cleans up both temp files
    """
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_img = f.name
    with tempfile.NamedTemporaryFile(suffix=".ps1", delete=False,
                                     mode="w", encoding="utf-8") as f:
        tmp_ps = f.name
        f.write(_PS_SCRIPT)

    cv2.imwrite(tmp_img, frame)

    try:
        proc = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy", "Bypass",
                "-File", tmp_ps,
                "-ImagePath", os.path.abspath(tmp_img),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if "NOPACK" in stderr:
                print("⚠️  [Windows OCR] Japanese language pack not installed — "
                      "Settings > Time & Language > Language & Region > Add Japanese")
            else:
                first_err = stderr.splitlines()[0] if stderr else "unknown"
                print(f"⚠️  [Windows OCR] PowerShell error: {first_err}")
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return "", elapsed_ms

        stdout = proc.stdout.strip()
        if not stdout:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return "", elapsed_ms

        # Parse Y_TOP|Y_BOTTOM|TEXT lines
        texts, heights, centres = [], [], []
        for raw in stdout.splitlines():
            parts = raw.split("|", 2)
            if len(parts) != 3:
                continue
            try:
                y_top    = float(parts[0])
                y_bottom = float(parts[1])
                text     = parts[2].strip()
            except ValueError:
                continue
            if not text:
                continue
            texts.append(text)
            heights.append(y_bottom - y_top)
            centres.append((y_top + y_bottom) / 2.0)

        if not texts:
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            return "", elapsed_ms

        # Sort top-to-bottom by vertical centre
        order   = sorted(range(len(centres)), key=lambda i: centres[i])
        texts   = [texts[i]   for i in order]
        heights = [heights[i] for i in order]
        centres = [centres[i] for i in order]

        # Bimodal furigana filter — same as all other backends
        filtered_texts = _bimodal_furigana_filter(texts, heights, centres)
        japanese = " ".join(filtered_texts).strip()

        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return japanese, elapsed_ms

    except subprocess.TimeoutExpired:
        print(f"⚠️  [Windows OCR] PowerShell timed out on region '{region_name}'")
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return "", elapsed_ms

    except Exception as e:
        print(f"⚠️  [Windows OCR] Unexpected error on region '{region_name}': {e}")
        elapsed_ms = round((time.perf_counter() - t0) * 1000)
        return "", elapsed_ms

    finally:
        try: os.unlink(tmp_img)
        except OSError: pass
        try: os.unlink(tmp_ps)
        except OSError: pass


# ── Furigana filter ───────────────────────────────────────────────────────────
# Inlined here so this file has zero dependency on ocr_utils.py —
# same algorithm as zelda_apple_ocr.py and zelda_paddle_ocr.py.

def _bimodal_furigana_filter(texts, heights, centres):
    """Bimodal gap split on bounding-box heights to separate furigana from main text."""
    if not texts:
        return []

    sorted_h    = sorted(heights)
    furi_thresh = sorted_h[0]

    if len(sorted_h) >= 2:
        gaps = [(sorted_h[i + 1] - sorted_h[i], i) for i in range(len(sorted_h) - 1)]
        max_gap, gap_idx = max(gaps)
        if max_gap > sorted_h[-1] * 0.20:
            furi_thresh = sorted_h[gap_idx + 1]

    median_h      = float(np.median(sorted_h))
    large_centres = [c for h, c in zip(heights, centres) if h >= furi_thresh]

    filtered = []
    for t, h, cy in zip(texts, heights, centres):
        if h >= furi_thresh:
            filtered.append(t)
        elif large_centres and any(abs(cy - lc) < median_h * 1.5 for lc in large_centres):
            filtered.append(t)
    return filtered


# ── Preprocessing ─────────────────────────────────────────────────────────────
# Identical to zelda_apple_ocr.py — row-density furigana suppression,
# threshold at 160, 2× Lanczos upscale, 20px black border.

def preprocess_crop(crop):
    """Row-density furigana suppression.
    Threshold → blank sparse furigana rows → 2× Lanczos upscale → border."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    if mask.max() == 0:
        h, w = crop.shape[:2]
        return np.zeros((h * 2 + 40, w * 2 + 40, 3), dtype=np.uint8)

    result = np.zeros_like(crop)
    result[mask == 255] = (255, 255, 255)

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


# ── Entry point ───────────────────────────────────────────────────────────────

do_ocr = windows_ocr

if __name__ == '__main__':
    zelda_core.register_ocr_backend(do_ocr, preprocess_crop)
    zelda_core.main()
