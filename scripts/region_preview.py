"""
region_preview.py
=================
Real-time preview of auto dialogue/item box detection.

First run (or press C to recalibrate):
  - Grabs a frame from IP Webcam
  - Opens it — drag a rectangle around the TV SCREEN (not bezel/stand)
  - Press ENTER or SPACE to confirm, ESC to redo
  - Saves to tv_bounds.json next to this script

Then runs live detection on the TV region:
  GREEN box = DIALOGUE detected
  RED box   = ITEM_BOX detected
  Grey text = nothing detected

Controls:
  C  — recalibrate TV bounds
  S  — save annotated frame to ~/Downloads/
  P  — pause / unpause
  Q  — quit

Deps:
  pip install opencv-python numpy
"""

import cv2
import numpy as np
import json
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
IP_WEBCAM_URL  = "http://192.168.1.107:8080/video"
TV_BOUNDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tv_bounds.json")
SAVE_DIR       = os.path.expanduser("~/Downloads")
DISPLAY_W      = 1280
DISPLAY_H      = 800

COLOR_DIALOGUE = (0,   255,   0)   # green
COLOR_ITEM_BOX = (0,     0, 255)   # red
COLOR_NONE     = (120, 120, 120)   # grey
COLOR_STATUS   = (200, 200,   0)   # yellow

# ── TV bounds persistence ─────────────────────────────────────────────────────

def load_tv_bounds():
    try:
        with open(TV_BOUNDS_FILE, "r") as f:
            d = json.load(f)
        assert all(k in d for k in ("x", "y", "w", "h"))
        print(f"📦  TV bounds loaded: x={d['x']} y={d['y']} w={d['w']} h={d['h']}")
        return d
    except (FileNotFoundError, AssertionError, json.JSONDecodeError):
        return None

def save_tv_bounds(bounds):
    with open(TV_BOUNDS_FILE, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"✅  TV bounds saved: x={bounds['x']} y={bounds['y']} "
          f"w={bounds['w']} h={bounds['h']}")

def crop_to_tv(frame, bounds):
    h, w = frame.shape[:2]
    x  = max(0, bounds["x"])
    y  = max(0, bounds["y"])
    x2 = min(w, x + bounds["w"])
    y2 = min(h, y + bounds["h"])
    return frame[y:y2, x:x2]

# ── Calibration ───────────────────────────────────────────────────────────────

def grab_frame(cap):
    """Drain a few frames and return a fresh one."""
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌  Failed to grab frame from camera.")
        return None
    return frame

def calibrate(cap):
    """
    Show a camera frame, let user drag a rectangle around the TV screen.
    Returns bounds dict or None if aborted.
    Mirrors the pattern from calibrate.py.
    """
    print("\n📺  CALIBRATION")
    print("   Draw a rectangle around the TV SCREEN (not bezel or stand).")
    print("   Click and drag → ENTER or SPACE to confirm → ESC to redo\n")

    while True:
        frame = grab_frame(cap)
        if frame is None:
            return None

        fh, fw = frame.shape[:2]
        scale   = min(1.0, DISPLAY_W / fw)
        display = cv2.resize(frame, (int(fw * scale), int(fh * scale)))

        cv2.namedWindow("Calibrate — draw TV screen bounds", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI(
            "Calibrate — draw TV screen bounds",
            display,
            fromCenter=False,
            showCrosshair=True
        )
        cv2.destroyAllWindows()

        x, y, rw, rh = roi
        if rw == 0 or rh == 0:
            print("⚠️  No region selected — try again.")
            continue

        # Scale back to original resolution
        if scale < 1.0:
            x  = int(x  / scale)
            y  = int(y  / scale)
            rw = int(rw / scale)
            rh = int(rh / scale)

        bounds = {"x": x, "y": y, "w": rw, "h": rh}

        # Preview the crop so user can confirm
        crop = crop_to_tv(frame, bounds)
        if crop.size == 0:
            print("⚠️  Empty crop — try again.")
            continue

        print("👁   Previewing TV crop — press any key to confirm, ESC to redo.")
        cv2.namedWindow("TV crop preview — ENTER to confirm, ESC to redo", cv2.WINDOW_NORMAL)
        cv2.imshow("TV crop preview — ENTER to confirm, ESC to redo", crop)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key == 27:  # ESC — redo
            print("↩️  Redoing selection...")
            continue

        save_tv_bounds(bounds)
        return bounds

# ── Detection ─────────────────────────────────────────────────────────────────

def detect_ui_regions(frame):
    """
    Find DIALOGUE pill and/or ITEM_BOX in a BotW frame (already cropped to TV).
    Returns [] when nothing found (cutscene, gameplay, no UI).
    """
    h, w = frame.shape[:2]
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    _, white = cv2.threshold(enhanced, 185, 255, cv2.THRESH_BINARY)

    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 30))
    connected = cv2.dilate(white, k_h)
    connected = cv2.dilate(connected, k_v)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        connected, connectivity=8
    )

    dialogue_candidates = []
    item_candidates     = []

    for i in range(1, num):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        cx   = centroids[i][0] / w
        cy   = centroids[i][1] / h

        if area < w * h * 0.003:
            continue

        wf    = bw / w
        hf    = bh / h
        raw_d = white[y:y+bh, x:x+bw].sum() / (255 * max(bw * bh, 1))

        if raw_d < 0.008:
            continue

        # DIALOGUE: bottom half, moderate width, centred
        if (cy > 0.65
                and 0.18 < wf < 0.70
                and 0.06 < hf < 0.32
                and 0.22 < cx < 0.78
                and raw_d > 0.008):
            centre_score = max(0, 1.0 - abs(cx - 0.50) * 3) ** 2
            score = (raw_d ** 2) * wf * cy * centre_score
            dialogue_candidates.append({
                "type": "DIALOGUE", "x": x, "y": y, "w": bw, "h": bh,
                "score": score, "density": raw_d
            })

        # ITEM_BOX: middle of frame, strictly centred, multi-line
        elif (0.35 < cy < 0.72
                and 0.18 < wf < 0.40
                and 0.12 < hf < 0.30
                and 0.35 < cx < 0.65
                and raw_d > 0.08):
            item_candidates.append({
                "type": "ITEM_BOX", "x": x, "y": y, "w": bw, "h": bh,
                "score": hf * raw_d, "density": raw_d
            })

    best_dialogue = sorted(dialogue_candidates, key=lambda r: r["score"], reverse=True)[:1]
    best_item     = sorted(item_candidates,     key=lambda r: r["score"], reverse=True)[:1]

    # Mutual exclusion: keep whichever scores 5x higher
    if best_dialogue and best_item:
        if best_dialogue[0]["score"] > best_item[0]["score"] * 5:
            best_item = []
        else:
            best_dialogue = []

    PAD    = 35
    result = []
    for r in best_dialogue + best_item:
        rx  = max(0, r["x"] - PAD)
        ry  = max(0, r["y"] - PAD)
        rx2 = min(w, r["x"] + r["w"] + PAD)
        ry2 = min(h, r["y"] + r["h"] + PAD)
        result.append({
            "type":    r["type"],
            "x": rx,   "y": ry,
            "w": rx2 - rx,
            "h": ry2 - ry,
            "density": round(r["density"], 3),
            "score":   round(r["score"],   5),
        })
    return result

# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlay(frame, regions, detect_ms, fps, paused=False):
    vis = frame.copy()
    fh, fw = vis.shape[:2]

    if regions:
        for r in regions:
            color = COLOR_DIALOGUE if r["type"] == "DIALOGUE" else COLOR_ITEM_BOX
            cv2.rectangle(vis, (r["x"], r["y"]),
                          (r["x"] + r["w"], r["y"] + r["h"]), color, 3)
            label = f"{r['type']}  d={r['density']}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            lx = r["x"]
            ly = max(0, r["y"] - th - 10)
            cv2.rectangle(vis, (lx, ly), (lx + tw + 8, ly + th + 8), color, -1)
            cv2.putText(vis, label, (lx + 4, ly + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    else:
        msg = "No UI region detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(vis, msg, ((fw - tw) // 2, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NONE, 2)

    # Status bar
    bar_h = 34
    cv2.rectangle(vis, (0, fh - bar_h), (fw, fh), (20, 20, 20), -1)
    status = (f"detect={detect_ms:.0f}ms  fps={fps:.1f}  regions={len(regions)}"
              f"  [C]alibrate  [S]ave  [P]ause  [Q]uit")
    cv2.putText(vis, status, (8, fh - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_STATUS, 1)

    if paused:
        cv2.rectangle(vis, (fw//2 - 85, 8), (fw//2 + 85, 48), (0, 0, 0), -1)
        cv2.putText(vis, "PAUSED", (fw//2 - 55, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    return vis

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("🎮  BotW Region Detection Preview")
    print(f"📡  Connecting to {IP_WEBCAM_URL} ...")

    cap = cv2.VideoCapture(IP_WEBCAM_URL)
    if not cap.isOpened():
        print(f"❌  Cannot open stream: {IP_WEBCAM_URL}")
        print("    Make sure IP Webcam is running on your phone.")
        return

    print("✅  Camera connected.\n")

    # Load or create TV bounds
    tv_bounds = load_tv_bounds()
    if tv_bounds is None:
        print("📺  No tv_bounds.json found — running calibration...")
        tv_bounds = calibrate(cap)
        if tv_bounds is None:
            print("❌  Calibration aborted.")
            cap.release()
            return

    cv2.namedWindow("BotW Region Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BotW Region Detection", DISPLAY_W, DISPLAY_H)

    paused      = False
    last_frame  = None
    last_regions= []
    frame_times = []
    save_count  = 0
    detect_ms   = 0

    print("\n🟢  Running — GREEN=DIALOGUE  RED=ITEM_BOX")
    print("    C=recalibrate  S=save  P=pause  Q=quit\n")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            paused = not paused
            print(f"\n{'⏸  Paused' if paused else '▶️  Resumed'}")
        elif key == ord('c'):
            print("\n📺  Recalibrating...")
            new_bounds = calibrate(cap)
            if new_bounds:
                tv_bounds = new_bounds
                print("✅  TV bounds updated.")
            continue
        elif key == ord('s') and last_frame is not None:
            save_count += 1
            fname = os.path.join(SAVE_DIR, f"botw_region_{save_count:03d}.png")
            annotated = draw_overlay(last_frame, last_regions, detect_ms, 0)
            cv2.imwrite(fname, annotated)
            print(f"\n💾  Saved: {fname}  regions={[r['type'] for r in last_regions]}")

        if not paused:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Crop to TV screen
            tv_crop = crop_to_tv(frame, tv_bounds)
            if tv_crop.size == 0:
                continue

            # Detect
            t0        = time.perf_counter()
            regions   = detect_ui_regions(tv_crop)
            detect_ms = (time.perf_counter() - t0) * 1000

            last_frame   = tv_crop.copy()
            last_regions = regions

            # FPS
            now = time.perf_counter()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            fps = len(frame_times)

            region_types = [r["type"] for r in regions]
            print(f"\r  {str(region_types or 'none'):35s}  "
                  f"detect={detect_ms:4.0f}ms  fps={fps:2d}   ",
                  end="", flush=True)
        else:
            tv_crop  = last_frame if last_frame is not None else \
                       np.zeros((480, 854, 3), dtype=np.uint8)
            regions  = last_regions
            fps      = 0

        # Draw and display
        vis   = draw_overlay(tv_crop, regions, detect_ms, fps, paused)
        th, tw = vis.shape[:2]
        scale = min(DISPLAY_W / tw, DISPLAY_H / th)
        disp  = cv2.resize(vis, (int(tw * scale), int(th * scale)))
        cv2.imshow("BotW Region Detection", disp)

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n👋  Done.")

if __name__ == "__main__":
    main()
