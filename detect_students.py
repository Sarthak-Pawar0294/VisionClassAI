# LEGACY PROTOTYPE — do not use in production.
#
# This script was the original standalone person-detection demo written before
# the src/ package existed.  It is NOT part of the refactored pipeline and is
# NOT imported by any other module.
#
# Superseded by:
#   src/realtime_behavior.py  — webcam inference with central configuration
#   src/classroom_tracking.py — person tracking + behavior association
#
# Kept as a separate reference demo; its camera/model work is guarded below.
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load the same local pretrained model, independent of the shell directory.
    model = YOLO(str(PROJECT_ROOT / "yolo11n.pt"))
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return 1

        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Could not read camera.")
                break

            # Run YOLO
            results = model(
                frame,
                device=device,
                classes=[0],       # class 0 = person
                conf=0.40,
                verbose=False
            )

            # Draw detections
            annotated_frame = results[0].plot()

            # Count detected people
            person_count = len(results[0].boxes)

            cv2.putText(
                annotated_frame,
                f"Students Detected: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("VisionClass - Student Detection", annotated_frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
