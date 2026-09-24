import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from ultralytics import YOLO

from src.association import assign_behaviors_to_students, build_student_records
from src.config import get_settings


def main(device_spec: str | None = None) -> None:
    settings = get_settings(device_spec=device_spec)

    person_model = YOLO(str(settings.person_model))
    behavior_model = YOLO(str(settings.behavior_model))

    cap = cv2.VideoCapture(str(settings.video_path))

    if not cap.isOpened():
        print("ERROR: Could not open classroom video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(settings.tracking_output_video),
        fourcc,
        fps,
        (width, height),
    )

    print("=" * 60)
    print("VisionClass - Classroom Tracking")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print("Processing video...")
    print("Press Q to stop early.")

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        person_results = person_model.track(
            source=frame,
            classes=[0],
            imgsz=settings.inference_imgsz,
            conf=settings.person_conf,
            device=settings.device,
            persist=True,
            tracker=settings.tracker,
            verbose=False,
        )

        person_result = person_results[0]

        behavior_results = behavior_model.predict(
            source=frame,
            imgsz=settings.inference_imgsz,
            conf=settings.behavior_conf,
            device=settings.device,
            verbose=False,
        )

        behavior_result = behavior_results[0]

        students = build_student_records(person_result, default_behavior=None)

        if behavior_result.boxes is not None:
            assign_behaviors_to_students(
                students,
                behavior_result.boxes.xyxy.cpu().tolist(),
                behavior_result.boxes.cls.int().cpu().tolist(),
                behavior_result.boxes.conf.cpu().tolist(),
                behavior_model.names,
            )

        annotated = frame.copy()

        for student in students:
            x1, y1, x2, y2 = map(int, student["box"])

            student_id = student["id"]
            behavior = student["behavior"]
            confidence = student["confidence"]

            if behavior is not None:
                label = (
                    f"Student {student_id}: "
                    f"{behavior} "
                    f"{confidence:.2f}"
                )
            else:
                label = f"Student {student_id}: unknown"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        out.write(annotated)

        frame_number += 1

        if frame_number % 25 == 0:
            print(f"Processed {frame_number} frames")

        cv2.imshow(
            "VisionClass - Classroom Tracking",
            annotated,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("=" * 60)
    print("CLASSROOM TRACKING COMPLETE")
    print("Output saved to:")
    print(settings.tracking_output_video)
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classroom person tracking + behavior labels")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(device_spec=args.device)
