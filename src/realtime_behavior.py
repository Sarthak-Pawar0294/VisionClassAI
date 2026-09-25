import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from ultralytics import YOLO

from src.config import get_settings


def main(device_spec: str | None = None) -> None:
    settings = get_settings(device_spec=device_spec)

    model = YOLO(str(settings.behavior_model))

    cap = cv2.VideoCapture(settings.webcam_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open webcam (index {settings.webcam_index}).")
        return

    print("VisionClass behavior detection")
    print(f"Device: {settings.device}")
    print(f"Confidence threshold: {settings.behavior_conf}")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        results = model.predict(
            source=frame,
            imgsz=settings.inference_imgsz,
            conf=settings.behavior_conf,
            device=settings.device,
            verbose=False,
        )

        result = results[0]

        annotated_frame = result.plot()

        if len(result.boxes) > 0:
            print("\n--- Predictions ---")

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                print(f"{class_name:30s} confidence = {confidence:.3f}")

        else:
            print(f"No prediction above {settings.behavior_conf}")

        cv2.imshow(
            "VisionClass - Behavior Detection",
            annotated_frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time webcam behavior detection")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(device_spec=args.device)
