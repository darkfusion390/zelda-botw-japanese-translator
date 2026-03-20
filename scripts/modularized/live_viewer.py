"""
live_viewer.py
==============
Simple live video viewer using the same capture setup as calibrate.py.
Press Q to quit, S to save a screenshot.
"""
import cv2
import sys
import platform

VIDEO_SOURCE = 0

def main():
    print(f"📡  Connecting to {VIDEO_SOURCE} ...")
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(VIDEO_SOURCE, backend)
    if not cap.isOpened():
        print("❌  Cannot connect.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅  Connected: {w}x{h} @ {fps}fps")
    print("Press Q to quit | S to save screenshot")

    screenshot_count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read failed")
            continue

        frame_count += 1
        if frame_count % 30 == 1:
            print(f"📷  Frame {frame_count} — mean brightness: {frame.mean():.1f}")

        cv2.imshow("Live Viewer [Q=quit | S=screenshot]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(filename, frame)
            print(f"💾  Saved: {filename}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
