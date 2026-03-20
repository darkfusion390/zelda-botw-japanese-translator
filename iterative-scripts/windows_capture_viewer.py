import cv2

def find_devices():
    """Scan indices 0-9 and print all working devices."""
    print("=== Scanning for devices ===")
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"  [OK] Device {i}: working")
                found.append(i)
            else:
                print(f"  [--] Device {i}: opens but no frame")
        else:
            print(f"  [  ] Device {i}: not found")
    print("============================\n")
    return found

def main(device_index=None):
    found = find_devices()

    if not found:
        print("ERROR: No working video devices found.")
        return

    if device_index is None:
        device_index = found[0]
        print(f"Auto-selected device index: {device_index}")
        print(f"To use a different device, edit main(X) at the bottom of the script.\n")
    
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Stream: {actual_w}x{actual_h} @ {actual_fps}fps")
    print("Press 'q' to quit | 's' to save a screenshot\n")

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
