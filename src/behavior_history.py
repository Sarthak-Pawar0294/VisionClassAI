"""
Temporal behavior analysis pipeline.

Processes a classroom video frame-by-frame, associates behavior detections to
tracked persons, and produces:

  - A per-frame history CSV  (history_csv)   — one row per student per frame
  - A per-student summary CSV (summary_csv)  — total seconds per behavior class
  - An annotated output video (history_output_video)

Memory design
-------------
The per-frame CSV is written **incrementally** in batches of FLUSH_EVERY frames.
This keeps memory usage O(1) in video length regardless of duration.

The ``history`` counter dict (used for the summary) remains in memory but is
bounded: at most (n_unique_student_ids × n_behavior_classes) entries — a small
constant even for long recordings.

Unknown cause tagging
---------------------
The history CSV includes an ``unknown_cause`` column to help diagnose why a
student record has ``behavior='unknown'``.

Two observable failure modes exist:

1. **no_match** (per-student column value) — ``assign_behaviors_to_students()``
   did not write a behavior label to this student this frame.  This happens when:

   a. The behavior model produced no detection above threshold in this frame.
   b. The model fired a detection but a different student had a higher IoU with
      it (the current student was *outcompeted*).

   Cases (a) and (b) are **indistinguishable** from per-student data alone.
   The tag ``no_match`` is therefore the accurate label — it means
   "this student was not assigned a behavior", not "the model was silent".

2. **no_association** (global counter in the summary printout, not per-student)
   — A behavior detection box overlapped *no* person box at all and was
   discarded.  This is a separate, globally-trackable failure mode.

For students with a matched behavior, ``unknown_cause`` is empty ("").
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import csv
from collections import defaultdict

import cv2
from ultralytics import YOLO

from src.association import assign_behaviors_to_students, build_student_records
from src.config import SUMMARY_BEHAVIOR_COLUMNS, get_settings

# Number of frames between CSV flushes.  Lower = less data lost on crash;
# higher = fewer file-write syscalls.
FLUSH_EVERY = 100

HISTORY_FIELDNAMES = [
    "frame",
    "time_seconds",
    "student_id",
    "behavior",
    "confidence",
    "unknown_cause",
]


def _count_unassociated(
    behavior_result: object,
    students: list[dict],
) -> int:
    """
    Return the number of behavior detections that were not associated to any
    person (no_association events).  Used only for the summary print.
    """
    if behavior_result.boxes is None:
        return 0
    count = 0
    for box in behavior_result.boxes.xyxy.cpu().tolist():
        from src.association import find_student_for_behavior
        if find_student_for_behavior(box, students) is None:
            count += 1
    return count


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
        str(settings.history_output_video),
        fourcc,
        fps,
        (width, height),
    )

    # Bounded in-memory aggregation for summary CSV (O(students × classes))
    history: defaultdict[int, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    # Diagnostic counters
    total_no_association = 0

    # Open the per-frame history CSV now so we can stream rows immediately
    settings.history_csv.parent.mkdir(parents=True, exist_ok=True)
    history_file = open(
        settings.history_csv, "w", newline="", encoding="utf-8"
    )
    history_writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDNAMES)
    history_writer.writeheader()

    # Pending rows buffer — flushed every FLUSH_EVERY frames
    pending_rows: list[dict] = []

    frame_number = 0

    print("=" * 60)
    print("VisionClass - Temporal Behavior Analysis")
    print("=" * 60)
    print(f"Device:     {settings.device}")
    print(f"FPS:        {fps:.2f}")
    print(f"CSV flush:  every {FLUSH_EVERY} frames")
    print("Processing classroom video...")
    print("Press Q to stop early.")
    print()

    try:
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

            students = build_student_records(person_result, default_behavior="unknown")

            if behavior_result.boxes is not None:
                assign_behaviors_to_students(
                    students,
                    behavior_result.boxes.xyxy.cpu().tolist(),
                    behavior_result.boxes.cls.int().cpu().tolist(),
                    behavior_result.boxes.conf.cpu().tolist(),
                    behavior_model.names,
                )
                total_no_association += _count_unassociated(behavior_result, students)

            timestamp = frame_number / fps

            for student in students:
                student_id = int(student["id"])
                behavior = student["behavior"]
                confidence = student["confidence"]

                history[student_id][behavior] += 1

                unknown_cause = "no_match" if behavior == "unknown" else ""

                pending_rows.append(
                    {
                        "frame": frame_number,
                        "time_seconds": round(timestamp, 3),
                        "student_id": student_id,
                        "behavior": behavior,
                        "confidence": round(confidence, 4),
                        "unknown_cause": unknown_cause,
                    }
                )

            # Flush to disk every FLUSH_EVERY frames
            if (frame_number + 1) % FLUSH_EVERY == 0:
                history_writer.writerows(pending_rows)
                history_file.flush()
                pending_rows.clear()

            # Draw annotations
            annotated = frame.copy()
            for student in students:
                x1, y1, x2, y2 = map(int, student["box"])
                student_id = student["id"]
                behavior = student["behavior"]
                confidence = student["confidence"]

                label = (
                    f"Student {student_id} | "
                    f"{behavior} | "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            out.write(annotated)
            frame_number += 1

            if frame_number % 25 == 0:
                print(f"Processed {frame_number} frames")

            cv2.imshow("VisionClass - Behavior History", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always flush remaining rows and close resources cleanly
        if pending_rows:
            history_writer.writerows(pending_rows)
            history_file.flush()
            pending_rows.clear()
        history_file.close()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Write summary CSV (still from in-memory history dict — bounded size)
    behaviors = SUMMARY_BEHAVIOR_COLUMNS
    with open(settings.summary_csv, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["student_id"] + behaviors
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for student_id in sorted(history):
            row = {"student_id": student_id}
            for behavior in behaviors:
                frame_count = history[student_id][behavior]
                row[behavior] = round(frame_count / fps, 2)
            writer.writerow(row)

    print()
    print("=" * 60)
    print("TEMPORAL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Frames processed  : {frame_number}")
    print(f"Video             : {settings.history_output_video}")
    print(f"History CSV       : {settings.history_csv}")
    print(f"Summary CSV       : {settings.summary_csv}")
    print(f"Unassociated dets : {total_no_association} (behavior fired but no person overlap)")
    print()

    print("Student behavior duration:")
    print()
    for student_id in sorted(history):
        print(f"Student {student_id}:")
        for behavior in behaviors:
            seconds = history[student_id][behavior] / fps
            if seconds > 0:
                print(f"  {behavior:30s}: {seconds:.2f} sec")
        print()

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Temporal behavior history + CSV export")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(device_spec=args.device)
