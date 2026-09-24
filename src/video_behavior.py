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

    cap = cv2.VideoCapture(str(settings.video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    settings.video_output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(settings.video_output),
        fourcc,
        fps,
        (width, height),
    )

    print("=" * 60)
    print("VisionClass - Classroom Video Analysis")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print(f"Input : {settings.video_path}")
    print(f"Output: {settings.video_output}")
    print("Processing video...")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            source=frame,
            imgsz=settings.inference_imgsz,
            conf=settings.behavior_conf,
            device=settings.device,
            verbose=False,
        )

        annotated = results[0].plot()

        out.write(annotated)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()

    print("=" * 60)
    print("VIDEO PROCESSING COMPLETE")
    print(f"Saved to: {settings.video_output}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run behavior detection on a video file")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(device_spec=args.device)
