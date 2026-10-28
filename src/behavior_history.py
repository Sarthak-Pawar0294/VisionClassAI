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
import math
import re
import shutil
import subprocess
from uuid import uuid4
from collections import defaultdict

import cv2
from ultralytics import YOLO

from src.association import assign_behaviors_to_students, build_student_records, BehaviorCache
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
    "is_cached",
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


def _find_ffmpeg_h264_encoder() -> tuple[str, str] | None:
    """Find an already-installed FFmpeg H.264 encoder without adding a dependency."""
    executable = shutil.which("ffmpeg")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for encoder in ("libx264", "libopenh264"):
        if re.search(rf"^\s*V\S*\s+{re.escape(encoder)}\s", result.stdout, re.MULTILINE):
            return executable, encoder
    return None


def _run_h264_transcode(
    source: Path,
    destination: Path,
    executable: str,
    encoder: str,
    stop_requested,
) -> bool:
    """Create a browser-oriented H.264 MP4; return False if cancellation was requested."""
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.mp4")
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-crf", "23"])
    else:
        command.extend(["-b:v", "2M"])
    command.append(str(temporary))

    process = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        while True:
            try:
                _stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if not stop_requested():
                    continue
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                return False
        if process.returncode != 0:
            detail = (stderr or "").strip().splitlines()
            reason = detail[-1] if detail else f"FFmpeg exited with code {process.returncode}."
            raise RuntimeError(f"Could not encode browser-playable H.264 video: {reason}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg completed without creating an annotated MP4 output.")
        temporary.replace(destination)
        return True
    except OSError as exc:
        raise RuntimeError(f"Could not run the configured FFmpeg encoder: {exc}") from exc
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)


def _probe_video_codec(path: Path) -> str | None:
    """Use an already-installed ffprobe when available to confirm the MP4 stream codec."""
    executable = shutil.which("ffprobe")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Could not verify the annotated MP4 codec with ffprobe: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else "ffprobe returned an error."
        raise RuntimeError(f"Could not verify the annotated MP4 codec: {reason}")
    codec = result.stdout.strip().splitlines()
    return codec[0].strip().lower() if codec else None


class OutputValidationCancelled(RuntimeError):
    """Raised internally when a video job is cancelled during output validation."""


def _validate_annotated_video(
    path: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_fps: float,
    output_root: Path,
    codec: str,
    stop_requested=lambda: False,
) -> dict:
    """Reopen and decode the whole generated file before publishing it as an artifact."""
    root = output_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RuntimeError("Annotated output is missing or outside its run directory.")
    output_bytes = resolved.stat().st_size
    if expected_frames <= 0 or output_bytes < max(128, expected_frames):
        raise RuntimeError("Annotated output is empty or implausibly small for the processed frames.")
    if expected_width <= 0 or expected_height <= 0 or not math.isfinite(expected_fps) or expected_fps <= 0:
        raise RuntimeError("Annotated output has invalid expected dimensions or frame rate.")

    capture = cv2.VideoCapture(str(resolved))
    decoded_frames = 0
    try:
        if not capture.isOpened():
            raise RuntimeError("Annotated MP4 could not be reopened by the installed OpenCV decoder.")
        output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if output_width > 0 and output_height > 0 and (output_width, output_height) != (expected_width, expected_height):
            raise RuntimeError(
                f"Annotated output dimensions are {output_width}x{output_height}; expected {expected_width}x{expected_height}."
            )
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise RuntimeError("Annotated MP4 reports an invalid frame rate.")
        fps_tolerance = max(0.1, expected_fps * 0.02)
        if abs(output_fps - expected_fps) > fps_tolerance:
            raise RuntimeError(
                f"Annotated output frame rate is {output_fps:.3f} FPS; expected {expected_fps:.3f} FPS."
            )
        while True:
            if stop_requested():
                raise OutputValidationCancelled("Annotated output validation was cancelled.")
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim < 2 or frame.shape[1] != expected_width or frame.shape[0] != expected_height:
                raise RuntimeError("Annotated output contains a frame with unexpected dimensions.")
            decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames <= 0:
        raise RuntimeError("Annotated MP4 opened but no output frame could be decoded.")
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"Annotated output contains {decoded_frames} decodable frames; {expected_frames} frames were processed."
        )
    if stop_requested():
        raise OutputValidationCancelled("Annotated output validation was cancelled.")
    probed_codec = _probe_video_codec(resolved)
    if stop_requested():
        raise OutputValidationCancelled("Annotated output validation was cancelled.")
    if probed_codec and probed_codec != "h264":
        raise RuntimeError(
            f"Annotated MP4 contains {probed_codec}, not browser-compatible H.264. "
            "Enable an H.264 encoder and retry the video run."
        )
    return {
        "validated": True,
        "container": "mp4",
        "codec": codec,
        "codec_verified": probed_codec == "h264" if probed_codec else None,
        "width": expected_width,
        "height": expected_height,
        "fps": output_fps,
        "processed_frames": expected_frames,
        "decoded_output_frames": decoded_frames,
        "output_bytes": output_bytes,
    }


def process_video(
    input_path: Path | str,
    history_csv: Path | str,
    summary_csv: Path | str,
    output_video: Path | str,
    *,
    settings,
    person_model=None,
    behavior_model=None,
    stop_requested=lambda: False,
    progress_callback=None,
    output_status_callback=None,
    show_preview: bool = False,
    output_root: Path | str | None = None,
    overwrite_output: bool = False,
) -> dict:
    """Run the existing tracker/history pipeline with injectable job controls.

    CSV schema, cache behavior, association, aggregation and annotation semantics
    remain shared with the standalone history CLI.
    """
    input_path = Path(input_path)
    history_csv = Path(history_csv)
    summary_csv = Path(summary_csv)
    output_video = Path(output_video)
    output_root = Path(output_root) if output_root is not None else output_video.parent
    if output_video.exists() and not overwrite_output:
        raise RuntimeError(f"Refusing to overwrite an existing annotated output: {output_video}")
    person_model = person_model or YOLO(str(settings.person_model))
    behavior_model = behavior_model or YOLO(str(settings.behavior_model))
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    out = None
    history_file = None
    pending_rows: list[dict] = []
    history: defaultdict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_no_association = 0
    behavior_detection_total = 0
    history_record_total = 0
    frame_number = 0
    cancelled = False
    processing_error = None
    encoder = _find_ffmpeg_h264_encoder()
    raw_video_path = output_video.with_name(f".{output_video.stem}.{uuid4().hex}.raw.mp4")
    staged_video_path = (
        output_video.with_name(f".{output_video.stem}.{uuid4().hex}.h264.mp4")
        if encoder
        else raw_video_path
    )
    output_codec = f"H.264 ({encoder[1]})" if encoder else "H.264 (OpenCV avc1)"
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open or decode video: {input_path.name}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) & ~1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) & ~1
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read valid dimensions from the source video.")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        total_frames_value = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_value if total_frames_value > 0 else None
        out = cv2.VideoWriter(
            str(raw_video_path),
            cv2.VideoWriter_fourcc(*("mp4v" if encoder else "avc1")),
            fps,
            (width, height),
        )
        if hasattr(out, "isOpened") and not out.isOpened():
            if encoder:
                raise RuntimeError("Could not initialize the OpenCV intermediate MP4 writer.")
            raise RuntimeError(
                "Could not initialize browser-compatible H.264 output. Enable FFmpeg with libx264/libopenh264, "
                "or use an OpenCV video backend that supports the avc1 H.264 encoder."
            )

        history_file = open(history_csv, "w", newline="", encoding="utf-8")
        history_writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDNAMES)
        history_writer.writeheader()
        behavior_cache = BehaviorCache(max_age=30)

        while not stop_requested():
            try:
                ret, frame = cap.read()
            except Exception as exc:
                raise RuntimeError(f"Could not read video frame {frame_number}: {exc}") from exc
            if not ret:
                break

            person_result = person_model.track(
                source=frame,
                classes=[0],
                imgsz=settings.inference_imgsz,
                conf=settings.person_conf,
                device=settings.device,
                persist=True,
                tracker=settings.tracker,
                verbose=False,
            )[0]
            behavior_result = behavior_model.predict(
                source=frame,
                imgsz=settings.inference_imgsz,
                conf=settings.behavior_conf,
                device=settings.device,
                verbose=False,
            )[0]
            students = build_student_records(person_result, default_behavior=None)
            raw_counts = {name: 0 for name in SUMMARY_BEHAVIOR_COLUMNS}
            if behavior_result.boxes is not None:
                behavior_detection_total += len(behavior_result.boxes)
                class_ids = behavior_result.boxes.cls.int().cpu().tolist()
                for class_id in class_ids:
                    name = behavior_model.names[int(class_id)]
                    if name in raw_counts:
                        raw_counts[name] += 1
                assign_behaviors_to_students(
                    students,
                    behavior_result.boxes.xyxy.cpu().tolist(),
                    class_ids,
                    behavior_result.boxes.conf.cpu().tolist(),
                    behavior_model.names,
                )
                total_no_association += _count_unassociated(behavior_result, students)
            behavior_cache.apply(students)
            behavior_cache.update(students)

            timestamp = frame_number / fps
            for student in students:
                student_id = int(student["id"])
                behavior = student["behavior"] or "unknown"
                confidence = student["confidence"]
                is_cached = bool(student.get("from_cache", False))
                history[student_id][behavior] += 1
                pending_rows.append({
                    "frame": frame_number,
                    "time_seconds": round(timestamp, 3),
                    "student_id": student_id,
                    "behavior": behavior,
                    "confidence": round(confidence, 4),
                    "is_cached": str(is_cached).lower(),
                    "unknown_cause": "no_match" if behavior == "unknown" else "",
                })
            history_record_total += len(students)
            if (frame_number + 1) % FLUSH_EVERY == 0:
                history_writer.writerows(pending_rows)
                history_file.flush()
                pending_rows.clear()

            annotated = frame.copy()
            for student in students:
                x1, y1, x2, y2 = map(int, student["box"])
                behavior = student["behavior"] or "unknown"
                confidence = student["confidence"]
                cached = bool(student.get("from_cache", False))
                label = (
                    f"Tracker {student['id']}: unknown"
                    if behavior == "unknown"
                    else f"Tracker {student['id']}: {behavior} {confidence:.2f}{' [~]' if cached else ''}"
                )
                color = (150, 150, 150) if behavior == "unknown" else ((255, 200, 50) if cached else (50, 255, 50))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), baseline = cv2.getTextSize(label, font, 0.5, 1)
                if y1 - text_h - baseline < 0:
                    cv2.rectangle(annotated, (x1, y1), (x1 + text_w, y1 + text_h + baseline), color, -1)
                    cv2.putText(annotated, label, (x1, y1 + text_h), font, 0.5, (0, 0, 0), 1)
                else:
                    cv2.rectangle(annotated, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - baseline), font, 0.5, (0, 0, 0), 1)
            out.write(annotated[:height, :width])
            frame_number += 1
            if progress_callback:
                progress_callback({
                    "processed_frames": frame_number,
                    "total_frames": total_frames,
                    "progress_percent": min(100.0, round(100 * frame_number / total_frames, 2)) if total_frames else None,
                    "history_record_total": history_record_total,
                    "behavior_detection_total": behavior_detection_total,
                    "frame_detection_counts": raw_counts,
                })
            if show_preview:
                cv2.imshow("VisionClass - Behavior History", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cancelled = True
                    break
        cancelled = cancelled or bool(stop_requested())
        if pending_rows:
            history_writer.writerows(pending_rows)
            history_file.flush()
            pending_rows.clear()
    except BaseException as exc:
        processing_error = exc
        raise
    finally:
        cleanup_errors = []
        for cleanup in (
            history_file.close if history_file is not None else None,
            cap.release,
            out.release if out is not None else None,
            cv2.destroyAllWindows if show_preview else None,
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except BaseException as exc:
                cleanup_errors.append(str(exc) or type(exc).__name__)
        if processing_error is not None or cleanup_errors:
            raw_video_path.unlink(missing_ok=True)
            if staged_video_path != raw_video_path:
                staged_video_path.unlink(missing_ok=True)
        if cleanup_errors and processing_error is None:
            raise RuntimeError(f"Could not finalize annotated video resources: {cleanup_errors[0]}")

    try:
        with open(summary_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["student_id"] + SUMMARY_BEHAVIOR_COLUMNS)
            writer.writeheader()
            for tracker_id in sorted(history):
                writer.writerow({
                    "student_id": tracker_id,
                    **{behavior: round(history[tracker_id][behavior] / fps, 2) for behavior in SUMMARY_BEHAVIOR_COLUMNS},
                })
        output_validation = None
        cancelled = cancelled or bool(stop_requested())
        if not cancelled:
            if output_status_callback:
                output_status_callback("encoding_h264" if encoder else "finalizing_output")
            if encoder:
                converted = _run_h264_transcode(
                    raw_video_path,
                    staged_video_path,
                    encoder[0],
                    encoder[1],
                    stop_requested,
                )
                if not converted:
                    cancelled = True
            if not cancelled:
                if output_status_callback:
                    output_status_callback("validating_output")
                try:
                    output_validation = _validate_annotated_video(
                        staged_video_path,
                        expected_frames=frame_number,
                        expected_width=width,
                        expected_height=height,
                        expected_fps=fps,
                        output_root=output_root,
                        codec=output_codec,
                        stop_requested=stop_requested,
                    )
                except OutputValidationCancelled:
                    cancelled = True
                if not cancelled:
                    if output_video.exists() and not overwrite_output:
                        raise RuntimeError(f"Refusing to overwrite an existing annotated output: {output_video}")
                    staged_video_path.replace(output_video)
                    output_validation["output_bytes"] = output_video.stat().st_size
        return {
            "processed_frames": frame_number,
            "total_frames": total_frames,
            "behavior_detection_total": behavior_detection_total,
            "history_record_total": history_record_total,
            "unassociated_detections": total_no_association,
            "cancelled": cancelled,
            "fps": fps,
            "output_video": output_validation,
        }
    finally:
        raw_video_path.unlink(missing_ok=True)
        if staged_video_path != raw_video_path:
            staged_video_path.unlink(missing_ok=True)


def main(device_spec: str | None = None) -> None:
    settings = get_settings(device_spec=device_spec)
    person_model = YOLO(str(settings.person_model))
    behavior_model = YOLO(str(settings.behavior_model))
    print("=" * 60)
    print("VisionClass - Temporal Behavior Analysis")
    print("=" * 60)
    print(f"Device:     {settings.device}")
    print(f"CSV flush:  every {FLUSH_EVERY} frames")
    print("Processing classroom video... Press Q to stop early.")
    result = process_video(
        settings.video_path,
        settings.history_csv,
        settings.summary_csv,
        settings.history_output_video,
        settings=settings,
        person_model=person_model,
        behavior_model=behavior_model,
        show_preview=True,
        overwrite_output=True,
    )
    print(f"Frames processed  : {result['processed_frames']}")
    if result["cancelled"] or not result["output_video"]:
        print("Video             : Not saved (run cancelled or no output video produced)")
    else:
        print(f"Video             : {settings.history_output_video}")
    print(f"History CSV       : {settings.history_csv}")
    print(f"Summary CSV       : {settings.summary_csv}")
    print(f"Unassociated detections: {result['unassociated_detections']}")


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
