"""
ocr_benchmark.py
================
Core OCR Benchmark Runner
=========================
Runs a configurable list of OCR processor modules against a folder of images
and generates a self-contained HTML accuracy report.

Each processor module must expose:
    NAME        (str)  — display name
    DESCRIPTION (str)  — one-line description of the preprocessing approach
    process_image(img_path: str) -> dict
        Returns: { "text": str, "elapsed_ms": int, "preprocessed_b64": str }

Usage:
    # Run all default processors
    python ocr_benchmark.py /path/to/images

    # Run specific processors only
    python ocr_benchmark.py /path/to/images --processors processor_apple processor_rapid_a

    # Run with a custom output path
    python ocr_benchmark.py /path/to/images --output /path/to/report.html

Default processor list (edit DEFAULT_PROCESSORS to change):
    processor_apple     — Apple Vision + row-density furigana suppression
    processor_paddle    — PaddleOCR v5 mobile + row-density furigana suppression
    processor_rapid_a   — RapidOCR + row-density (Approach A: engine comparison)
    processor_rapid_b   — RapidOCR + CLAHE adaptive (Approach B)
    processor_rapid_c   — RapidOCR + morph sharpen (Approach C)

Adding a new processor:
    1. Create processor_myname.py with the required interface above.
    2. Add "processor_myname" to DEFAULT_PROCESSORS, or pass via --processors.
"""

import argparse
import importlib
import sys
import os
import glob
import base64
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_PROCESSORS = [
    "processor_apple",
    "processor_paddle",
    "processor_rapid_a",
    "processor_rapid_b",
    "processor_rapid_c",
    "processor_manga_ocr",
    "processor_manga_ocr_colour",
    "processor_easy_ocr",
    "processor_windows_ocr"
]

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")

# Per-processor accent colours for the HTML report
PROC_COLORS = [
    "#4f8ef7",  # blue
    "#f77f4f",  # orange
    "#4fc97a",  # green
    "#b44ff7",  # purple
    "#f7c84f",  # yellow
    "#f74f9a",  # pink
    "#4ff7e8",  # teal
]


# ── Processor loading ─────────────────────────────────────────────────────────

def load_processors(names: list) -> list:
    """Import processor modules by name. Skips unavailable ones with a warning."""
    processors = []
    for name in names:
        try:
            mod = importlib.import_module(name)
            # Validate interface
            assert hasattr(mod, "NAME"),          f"{name}: missing NAME"
            assert hasattr(mod, "DESCRIPTION"),   f"{name}: missing DESCRIPTION"
            assert callable(getattr(mod, "process_image", None)), f"{name}: missing process_image()"
            processors.append(mod)
            print(f"  ✓  {mod.NAME}")
        except ImportError as e:
            print(f"  ✗  {name} — import failed: {e}")
        except AssertionError as e:
            print(f"  ✗  {e}")
    return processors


# ── Image discovery ───────────────────────────────────────────────────────────

def get_images(folder: str) -> list:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(paths))


def file_to_b64(path: str) -> tuple:
    """Return (base64_string, mime_type) for a given image path."""
    ext  = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "bmp": "bmp", "webp": "webp"}.get(ext, "png")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), mime


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(folder: str, processor_names: list, output_path: str = None) -> str:
    """
    Run all processors against all images in folder.
    Returns the path of the generated HTML report.
    """
    print(f"\n{'═'*60}")
    print(f"  OCR Benchmark")
    print(f"  Folder : {folder}")
    print(f"  Loading processors...")
    print(f"{'─'*60}")

    # Add script dir to sys.path so processor_*.py files are importable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    processors = load_processors(processor_names)
    if not processors:
        print("\nNo processors loaded — nothing to do.")
        return None

    images = get_images(folder)
    if not images:
        print(f"\nNo images found in: {folder}")
        return None

    print(f"{'─'*60}")
    print(f"  {len(images)} image(s) × {len(processors)} processor(s) = "
          f"{len(images) * len(processors)} OCR runs\n")

    # results[img_path][proc_name] = {text, elapsed_ms, preprocessed_b64}
    results = {}

    for img_path in images:
        img_name = os.path.basename(img_path)
        print(f"[{img_name}]")
        results[img_path] = {}

        for proc in processors:
            pad = max(0, 30 - len(proc.NAME))
            print(f"  {proc.NAME}{' '*pad}", end="", flush=True)
            try:
                r = proc.process_image(img_path)
                results[img_path][proc.NAME] = r
                preview = repr(r["text"][:50]) if r["text"] else "(empty)"
                print(f"{r['elapsed_ms']:>6} ms  →  {preview}")
            except Exception as e:
                print(f"   ERROR: {e}")
                results[img_path][proc.NAME] = {
                    "text": f"[ERROR: {e}]",
                    "elapsed_ms": 0,
                    "preprocessed_b64": "",
                }
        print()

    # Generate report
    if output_path is None:
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(folder, f"ocr_benchmark_{ts}.html")

    html = _build_html(folder, processors, images, results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"{'═'*60}")
    print(f"  ✓ Report saved: {output_path}")
    print(f"{'═'*60}\n")
    return output_path


# ── HTML Report ───────────────────────────────────────────────────────────────

def _build_html(folder, processors, images, results) -> str:
    proc_names  = [p.NAME for p in processors]
    proc_color  = {p.NAME: PROC_COLORS[i % len(PROC_COLORS)] for i, p in enumerate(processors)}
    proc_desc   = {p.NAME: getattr(p, "DESCRIPTION", "") for p in processors}

    # ── Timing summary table ──────────────────────────────────────────────────
    th_cells = "".join(f'<th style="color:{proc_color[n]}">{n}</th>' for n in proc_names)
    timing_rows_html = ""
    for img_path in images:
        name = os.path.basename(img_path)
        cells = ""
        for pname in proc_names:
            ms = results[img_path].get(pname, {}).get("elapsed_ms", "–")
            cells += f"<td>{ms}{'ms' if isinstance(ms, int) else ''}</td>"
        timing_rows_html += f"<tr><td class='fn'>{name}</td>{cells}</tr>"

    # ── Per-image sections ────────────────────────────────────────────────────
    sections_html = ""
    for img_path in images:
        img_name      = os.path.basename(img_path)
        orig_b64, mime = file_to_b64(img_path)

        proc_cards_html = ""
        for pname in proc_names:
            r          = results[img_path].get(pname, {})
            text       = r.get("text", "")
            ms         = r.get("elapsed_ms", 0)
            pre_b64    = r.get("preprocessed_b64", "")
            color      = proc_color[pname]
            desc       = proc_desc.get(pname, "")

            prepro_html = (
                f'<img class="pre-img" src="data:image/png;base64,{pre_b64}" alt="preprocessed">'
                if pre_b64 else
                '<div class="no-pre">no preprocessed image</div>'
            )

            text_html = (
                text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("\n", "<br>")
                if text else
                '<span class="empty">(no text detected)</span>'
            )

            proc_cards_html += f'''
            <div class="pcard" style="border-top:3px solid {color}">
              <div class="pcard-hdr">
                <span class="pname" style="color:{color}">{pname}</span>
                <span class="ptime">{ms} ms</span>
              </div>
              <div class="pdesc">{desc}</div>
              {prepro_html}
              <div class="ptext">{text_html}</div>
            </div>'''

        sections_html += f'''
        <section class="img-section">
          <h2 class="img-name">{img_name}</h2>
          <div class="img-row">
            <div class="orig-wrap">
              <div class="orig-label">Original</div>
              <img class="orig-img" src="data:image/{mime};base64,{orig_b64}" alt="{img_name}">
            </div>
            <div class="pcard-grid">
              {proc_cards_html}
            </div>
          </div>
        </section>'''

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = "".join(
        f'<span class="leg-item"><span class="leg-dot" style="background:{proc_color[n]}"></span>{n}</span>'
        for n in proc_names
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Benchmark — {os.path.basename(folder)}</title>
<style>
/* ── Reset & base ─────────────────────────────────────────── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0d0f18;color:#dde0ee;padding:28px 24px;line-height:1.5;
}}
a{{color:#7aadff}}

/* ── Header ───────────────────────────────────────────────── */
.report-hdr{{margin-bottom:32px}}
.report-hdr h1{{font-size:1.55rem;font-weight:700;color:#fff;margin-bottom:4px}}
.meta{{font-size:0.82rem;color:#606480}}
.legend{{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}}
.leg-item{{display:flex;align-items:center;gap:5px;font-size:0.8rem;color:#9098b8}}
.leg-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}

/* ── Summary card ─────────────────────────────────────────── */
.summary{{
  background:#12151f;border:1px solid #1e2235;border-radius:12px;
  padding:20px 22px;margin-bottom:36px;overflow-x:auto
}}
.summary h2{{font-size:1rem;color:#c0c8e8;margin-bottom:14px;font-weight:600}}
table{{border-collapse:collapse;width:100%;font-size:0.82rem;min-width:500px}}
th,td{{padding:7px 14px;border:1px solid #1e2235;text-align:center}}
th{{background:#191c2e;color:#8892b4;font-weight:500;font-size:0.78rem;
    text-transform:uppercase;letter-spacing:.04em}}
td.fn{{text-align:left;color:#7aadff;font-family:monospace;font-size:0.78rem}}
tr:hover td{{background:#171a2a}}

/* ── Image section ────────────────────────────────────────── */
.img-section{{
  background:#12151f;border:1px solid #1e2235;border-radius:12px;
  padding:20px 22px;margin-bottom:28px
}}
.img-name{{
  font-size:0.88rem;color:#7aadff;font-family:monospace;font-weight:500;
  margin-bottom:14px;padding-bottom:10px;border-bottom:1px solid #1e2235
}}
.img-row{{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}}

/* ── Original image card ──────────────────────────────────── */
.orig-wrap{{flex-shrink:0;max-width:340px}}
.orig-label{{font-size:0.7rem;color:#606480;text-transform:uppercase;
             letter-spacing:.06em;margin-bottom:5px}}
.orig-img{{
  max-width:100%;max-height:220px;border-radius:7px;
  border:1px solid #1e2235;display:block;object-fit:contain
}}

/* ── Processor cards ──────────────────────────────────────── */
.pcard-grid{{display:flex;flex-wrap:wrap;gap:12px;flex:1;min-width:0}}
.pcard{{
  background:#0d0f18;border:1px solid #1e2235;border-radius:8px;
  padding:12px 13px;min-width:200px;max-width:260px;flex:1
}}
.pcard-hdr{{
  display:flex;justify-content:space-between;align-items:center;
  margin-bottom:3px
}}
.pname{{font-weight:600;font-size:0.82rem}}
.ptime{{
  font-size:0.72rem;color:#606480;background:#12151f;
  padding:2px 7px;border-radius:4px;white-space:nowrap
}}
.pdesc{{font-size:0.7rem;color:#505878;margin-bottom:8px;line-height:1.45}}
.pre-img{{
  width:100%;border-radius:4px;margin-bottom:8px;display:block;
  border:1px solid #1e2235;image-rendering:pixelated
}}
.no-pre{{
  height:36px;background:#12151f;border-radius:4px;margin-bottom:8px;
  display:flex;align-items:center;justify-content:center;
  font-size:0.7rem;color:#383c52
}}
.ptext{{
  font-size:0.83rem;line-height:1.75;color:#ccd0e8;
  background:#0a0c14;padding:8px 10px;border-radius:5px;
  min-height:38px;word-break:break-all
}}
.ptext .empty{{color:#383c52;font-style:italic}}
</style>
</head>
<body>

<div class="report-hdr">
  <h1>🔬 OCR Benchmark Report</h1>
  <p class="meta">
    Folder: <code>{folder}</code> &nbsp;·&nbsp;
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} &nbsp;·&nbsp;
    {len(images)} image(s) &nbsp;·&nbsp; {len(proc_names)} processor(s)
  </p>
  <div class="legend">{legend_html}</div>
</div>

<div class="summary">
  <h2>⏱ Timing Summary</h2>
  <table>
    <thead><tr><th>Image</th>{th_cells}</tr></thead>
    <tbody>{timing_rows_html}</tbody>
  </table>
</div>

{sections_html}

</body>
</html>'''


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OCR processors against an image folder and generate an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder",
        help="Path to folder containing images to benchmark."
    )
    parser.add_argument(
        "--processors", "-p",
        nargs="+",
        default=DEFAULT_PROCESSORS,
        metavar="MODULE",
        help="Processor module names to include (default: all five)."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output HTML report path (default: <folder>/ocr_benchmark_<timestamp>.html)."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a directory.")
        sys.exit(1)

    run_benchmark(args.folder, args.processors, args.output)
