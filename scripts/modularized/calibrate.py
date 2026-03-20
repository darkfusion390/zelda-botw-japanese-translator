"""
calibrate.py
============
One-time tool to set one or more named crop regions.

Run:  python3 calibrate.py
- Grabs a frame from your capture device
- Draw a rectangle around a text region, then press ENTER or SPACE to confirm
- A prompt appears — type a name for that region (e.g. "dialogue", "item_title")
- The frame is shown again — draw another region, or press Q to finish
- All regions saved to bounds.json

Then run zelda_translator.py as normal.
"""

import cv2
import json
import sys
import numpy as np

VIDEO_SOURCE = 0
BOUNDS_FILE  = "bounds.json"


def grab_frame(source):
    import platform
    print(f"📡  Connecting to {source} ...")
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        print("❌  Cannot connect. Check VIDEO_SOURCE.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌  Failed to grab frame.")
        sys.exit(1)
    print(f"✅  Got frame: {frame.shape[1]}w x {frame.shape[0]}h")
    return frame


def scale_frame(frame, max_display=1280):
    h, w = frame.shape[:2]
    scale = min(1.0, max_display / w)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale))), scale
    return frame.copy(), 1.0


def draw_existing_regions(base, regions, scale):
    """Overlay already-saved regions on the display frame as labelled rectangles."""
    overlay = base.copy()
    for name, b in regions.items():
        x  = int(b["x"] * scale)
        y  = int(b["y"] * scale)
        x2 = int((b["x"] + b["w"]) * scale)
        y2 = int((b["y"] + b["h"]) * scale)
        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 255, 80), 2)
        cv2.putText(overlay, name, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
    return overlay


def _type_in_window(crop, prompt_text):
    """Generic keyboard-input widget rendered on top of a crop preview.
    Returns the typed string on ENTER, or None on ESC."""
    label = ""
    win   = prompt_text
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        display = crop.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(display, f"{prompt_text}: {label}_", (4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
        cv2.imshow(win, display)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyWindow(win)
            return None
        elif key in (13, 10):
            cv2.destroyWindow(win)
            return label.strip() or None
        elif key == 8:
            label = label[:-1]
        elif 32 <= key <= 126:
            label += chr(key)


def ask_region_meta(frame, bounds, scale):
    """Ask the user to name the region and optionally assign it to a group.
    Two sequential keyboard-input windows — name first, then group.
    Returns (name, group) where group may be None, or (None, None) on ESC."""
    x, y, w, h = bounds["x"], bounds["y"], bounds["w"], bounds["h"]
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        print("⚠️   Crop is empty — try again.")
        return None, None

    name = _type_in_window(crop, "Region name")
    if name is None:
        return None, None

    group = _type_in_window(crop, "Group name (blank = none)")
    # group is allowed to be None — means ungrouped
    return name, group


def select_region(display_frame, frame, regions, scale):
    """Show the full frame with existing regions overlaid.
    User draws a rectangle. Returns bounds dict, or None if Q was pressed."""
    win = "Draw region — ENTER/SPACE confirm | Q quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    annotated = draw_existing_regions(display_frame, regions, scale)
    cv2.putText(annotated, "Draw box → ENTER/SPACE confirm   Q = done",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 80), 2)

    roi = cv2.selectROI(win, annotated, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        return None

    return {
        "x": int(rx / scale),
        "y": int(ry / scale),
        "w": int(rw / scale),
        "h": int(rh / scale),
    }


def save_bounds(regions):
    with open(BOUNDS_FILE, "w") as f:
        json.dump(regions, f, indent=2)
    print(f"\n✅  Saved {len(regions)} region(s) to {BOUNDS_FILE}:")
    for name, b in regions.items():
        grp = b.get("group") or "none"
        print(f"    {name}: x={b['x']} y={b['y']} w={b['w']} h={b['h']}  group={grp}")
    print(f"\n🎮  You can now run:  python3 zelda_translator.py")


def main():
    print("🎮  Zelda Translator — Calibration Tool")
    print("─" * 40)
    print("  Draw a box around each text region, name it, optionally group it.")
    print("  Grouped regions (e.g. item_title + item_body in group 'item') are")
    print("  OCR'd together and treated as one logical unit.")
    print("  Press Q (or draw nothing + ENTER) when done.\n")

    frame = grab_frame(VIDEO_SOURCE)
    display_frame, scale = scale_frame(frame)

    regions = {}

    while True:
        bounds = select_region(display_frame, frame, regions, scale)

        if bounds is None:
            break

        name, group = ask_region_meta(frame, bounds, scale)
        if name is None:
            print("↩️   Redoing selection...")
            continue

        if name in regions:
            print(f"⚠️   '{name}' already exists — overwriting.")

        regions[name] = {
            "x":     bounds["x"],
            "y":     bounds["y"],
            "w":     bounds["w"],
            "h":     bounds["h"],
            "group": group,   # None = ungrouped, string = group name
        }
        grp_label = f" (group: {group})" if group else ""
        print(f"✔  Region '{name}'{grp_label} added. Draw another or press Q to finish.")

    if not regions:
        print("⚠️   No regions defined. Nothing saved.")
        sys.exit(0)

    save_bounds(regions)


if __name__ == "__main__":
    main()
