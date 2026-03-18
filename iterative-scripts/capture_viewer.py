import cv2
import subprocess


def list_devices():
    """List available AVFoundation devices via ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    print("=== Available Devices ===")
    for line in result.stderr.splitlines():
        if "AVFoundation" in line and ("video" in line.lower() or "] [" in line):
            print(line)
    print("=========================\n")


def find_device():
    """Scan indices 0-5 and return the first working non-default camera."""
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Found working device at index: {i}")
                return i
    return 0


def main(device_index=None):
    list_devices()

    if device_index is None:
        device_index = find_device()

    print(f"Opening device index: {device_index}")
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)

    # Set resolution — adjust to match your capture card output
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {actual_w}x{actual_h} @ {actual_fps}fps\n")
    print("Press 'q' to quit | 's' to save a screenshot")

    if not cap.isOpened():
        print("ERROR: Could not open device. Try a different index.")
        return

    screenshot_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        cv2.imshow("Capture Card Feed  [q = quit | s = screenshot]", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting.")
            break

        elif key == ord("s"):
            filename = f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Set a specific index here if you already know it, e.g. main(1)
    # Otherwise leave as main() to auto-detect
    main()
