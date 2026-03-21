"""
processor_windows_ocr.py
========================
OCR Processor: Windows.Media.Ocr via PowerShell subprocess
Preprocessing: Row-density furigana suppression (Zelda Classic)

Avoids all Python WinRT bindings (winrt, winsdk) which fail to compile on
Python 3.13. Instead, shells out to a PowerShell script that loads the WinRT
assemblies natively and returns OCR results as plain text.

Requirements:
  • Windows 10/11 only
  • PowerShell 5.1+ (built into Windows, no install needed)
  • Japanese language pack: Settings → Time & Language → Language & Region
    → Add Japanese. Windows 11 includes OCR automatically with the language.

On macOS/Linux returns a clean unavailability message without crashing.

Standalone usage (Windows only):
    python processor_windows_ocr.py /path/to/images/folder
"""

import os
import sys
import glob
import time
import platform
import tempfile
import subprocess
import cv2
import numpy as np

# ── Processor metadata ────────────────────────────────────────────────────────
NAME        = "Windows OCR (Japanese)"
DESCRIPTION = ("Windows.Media.Ocr via PowerShell · On-device · No Python bindings · "
               "Row-density furigana suppression · Windows 10/11 only")

# ── Preprocessing + utils ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_utils import preprocess_row_density, bimodal_furigana_filter, img_to_b64

_IS_WINDOWS = platform.system() == "Windows"

# ── PowerShell OCR script ─────────────────────────────────────────────────────
# Written to a temp .ps1 file at runtime to avoid command-line escaping issues.
# Returns one line per detection in format:  Y_TOP|Y_BOTTOM|TEXT
_PS_SCRIPT = r"""
param([string]$ImagePath)

# Force UTF-8 output encoding for the subprocess pipe
$OutputEncoding             = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding  = [System.Text.Encoding]::UTF8
[Console]::InputEncoding   = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Add-Type -AssemblyName System.Runtime.WindowsRuntime

# Helper: block on WinRT IAsyncOperation<T> from PowerShell
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

# Load WinRT types into PowerShell session
[Windows.Storage.StorageFile,            Windows.Storage,          ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine,            Windows.Media.Ocr,        ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Globalization.Language,         Windows.Globalization,    ContentType=WindowsRuntime] | Out-Null

# Create Japanese OCR engine
$lang   = [Windows.Globalization.Language]::new('ja')
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)

if ($null -eq $engine) {
    Write-Error "NOPACK: Japanese OCR language pack not installed. Go to Settings > Time & Language > Language & Region > Add Japanese."
    exit 1
}

# Load the image file
$file    = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($ImagePath)) ([Windows.Storage.StorageFile])
$stream  = Await ($file.OpenReadAsync()) ([Windows.Storage.Streams.IRandomAccessStreamWithContentType])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap  = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

# Run OCR
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

# Output each line as Y_TOP|Y_BOTTOM|TEXT for Python to parse
foreach ($line in $result.Lines) {
    $r    = $line.BoundingRect
    $text = ($line.Words | ForEach-Object { $_.Text }) -join ''
    Write-Output ("{0}|{1}|{2}" -f [int]$r.Y, [int]($r.Y + $r.Height), $text)
}
"""


# ── OCR engine ────────────────────────────────────────────────────────────────

def _run_ocr(frame: np.ndarray) -> tuple:
    """
    Write frame to a temp PNG, run the PowerShell OCR script on it,
    parse Y_TOP|Y_BOTTOM|TEXT lines, apply bimodal furigana filter.
    Returns (text: str, elapsed_ms: int).
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_img = f.name
    with tempfile.NamedTemporaryFile(suffix=".ps1", delete=False,
                                     mode="w", encoding="utf-8") as f:
        tmp_ps = f.name
        f.write(_PS_SCRIPT)

    cv2.imwrite(tmp_img, frame)

    try:
        t0 = time.perf_counter()
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
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()[0] if proc.stderr.strip() else "unknown error"
            # Surface the NOPACK message cleanly
            if "NOPACK" in err:
                return ("[Windows OCR: Japanese language pack not installed — "
                        "Settings > Time & Language > Language & Region > Add Japanese]", elapsed_ms)
            return f"[Windows OCR PS error: {err}]", elapsed_ms

        stdout = proc.stdout.strip()
        if not stdout:
            return "", elapsed_ms

        # Parse "Y_TOP|Y_BOTTOM|TEXT" output lines
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
            return "", elapsed_ms

        # Sort top-to-bottom by vertical centre
        order   = sorted(range(len(centres)), key=lambda i: centres[i])
        texts   = [texts[i]   for i in order]
        heights = [heights[i] for i in order]
        centres = [centres[i] for i in order]

        filtered = bimodal_furigana_filter(texts, heights, centres)
        return "\n".join(filtered), elapsed_ms

    finally:
        try: os.unlink(tmp_img)
        except OSError: pass
        try: os.unlink(tmp_ps)
        except OSError: pass


# ── Public interface ──────────────────────────────────────────────────────────

def process_image(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        # Unicode path fallback (handles spaces, special chars on Windows)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"text": "[failed to load image]", "elapsed_ms": 0, "preprocessed_b64": ""}

    preprocessed = preprocess_row_density(img)
    b64 = img_to_b64(preprocessed)

    if not _IS_WINDOWS:
        return {
            "text":             "[Windows OCR — runs on Windows 10/11 only]",
            "elapsed_ms":       0,
            "preprocessed_b64": b64,
        }

    try:
        text, elapsed_ms = _run_ocr(preprocessed)
    except subprocess.TimeoutExpired:
        text       = "[Windows OCR error: PowerShell timed out after 30s]"
        elapsed_ms = 0
    except Exception as e:
        text       = f"[Windows OCR error: {e}]"
        elapsed_ms = 0

    return {"text": text, "elapsed_ms": elapsed_ms, "preprocessed_b64": b64}


# ── Standalone CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _IS_WINDOWS:
        print("Windows OCR is only available on Windows 10/11.")
        sys.exit(0)

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    paths  = sorted(
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.jpeg"))
    )
    if not paths:
        print(f"No images found in: {folder}")
        sys.exit(1)

    print(f"[{NAME}] Processing {len(paths)} image(s) in: {folder}\n")
    for p in paths:
        r = process_image(p)
        print(f"{'─'*60}")
        print(f"File  : {os.path.basename(p)}")
        print(f"Time  : {r['elapsed_ms']} ms")
        print(f"Text  : {r['text']}")
    print(f"{'─'*60}")
