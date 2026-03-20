"""
calibrate.py
============
One-time tool to set the dialogue box crop region.

Run:  python3 calibrate.py
- Grabs a frame from your IP webcam
- Opens it in a window — drag a rectangle around the dialogue box
- Press ENTER or SPACE to confirm, ESC to retry
- Saves bounds to bounds.json in the same directory

Then run zelda_translator.py as normal.
"""

import cv2
import json
import sys
import numpy as np
import platform

# VIDEO_SOURCE = "http://192.168.1.107:8080/video"
VIDEO_SOURCE = 0  # OpenCV webcam capture device index (default 0)
BOUNDS_FILE   = "bounds.json"

def grab_frame(url):
    print(f"📡  Connecting to {url} ...")
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_AVFOUNDATION)
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("❌  Cannot connect. Check VIDEO_SOURCE.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Drain a few frames so we get a fresh one
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌  Failed to grab frame.")
        sys.exit(1)

    print(f"✅  Got frame: {frame.shape[1]}w x {frame.shape[0]}h")
    return frame

def select_roi(frame):
    print("\n📦  Draw a rectangle around the dialogue box.")
    print("    Click and drag → ENTER or SPACE to confirm → ESC to cancel and retry\n")

    # Scale down for display if frame is very large
    h, w = frame.shape[:2]
    max_display = 1280
    scale = min(1.0, max_display / w)
    display = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame.copy()

    cv2.namedWindow("Calibrate — draw dialogue box", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Calibrate — draw dialogue box", display, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        print("⚠️   No region selected.")
        return None

    # Scale coords back to original resolution if we downscaled
    if scale < 1.0:
        x  = int(x  / scale)
        y  = int(y  / scale)
        rw = int(rw / scale)
        rh = int(rh / scale)

    return {"x": x, "y": y, "w": rw, "h": rh}

def preview_crop(frame, bounds):
    """Show the selected crop so user can confirm it looks right."""
    x, y, w, h = bounds["x"], bounds["y"], bounds["w"], bounds["h"]
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        print("⚠️   Crop is empty — try again.")
        return False

    print(f"\n👁   Previewing crop — press any key to confirm, ESC to redo")
    cv2.namedWindow("Crop preview — press any key to confirm, ESC to redo", cv2.WINDOW_NORMAL)
    cv2.imshow("Crop preview — press any key to confirm, ESC to redo", crop)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key != 27  # ESC = redo

def save_bounds(bounds):
    with open(BOUNDS_FILE, "w") as f:
        json.dump(bounds, f, indent=2)
    print(f"\n✅  Bounds saved to {BOUNDS_FILE}")
    print(f"    x={bounds['x']}  y={bounds['y']}  w={bounds['w']}  h={bounds['h']}")
    print(f"\n🎮  You can now run:  python3 zelda_translator.py")

def main():
    print("🎮  Zelda Translator — Calibration Tool")
    print("─" * 40)

    frame = grab_frame(VIDEO_SOURCE)

    while True:
        bounds = select_roi(frame)
        if bounds is None:
            retry = input("Retry? (y/n): ").strip().lower()
            if retry != 'y':
                print("Aborted.")
                sys.exit(0)
            frame = grab_frame(VIDEO_SOURCE)
            continue

        confirmed = preview_crop(frame, bounds)
        if confirmed:
            save_bounds(bounds)
            break
        else:
            print("↩️   Redoing selection...")
            # Optionally re-grab a fresh frame
            frame = grab_frame(VIDEO_SOURCE)

if __name__ == "__main__":
    main()
