"""Run-scoped upload, processing, and analytics helpers for the local app."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
from ultralytics import YOLO

from src import analyze_behavior
from src import generate_analytics_report
from src.behavior_history import process_video
from src.config import get_settings


ALLOWED_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})
ARTIFACT_FILENAMES = {
    "source_video": "input",
    "annotated_video": "annotated.mp4",
    "history_csv": "behavior_history.csv",
    "summary_csv": "behavior_summary.csv",
    "analytics_report": "behavior_analytics_report.md",
    "analytics_dashboard": "behavior_analytics_dashboard.html",
}
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class UploadValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class RunStore:
    """Own generated run directories and strictly allowlisted artifact paths."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    def new_run_id(self) -> str:
        return uuid.uuid4().hex

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise UploadValidationError("Invalid run identifier.", 404)
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root):
            raise UploadValidationError("Invalid run directory.", 404)
        return path

    def create_upload(self, filename: str, stream, max_bytes: int) -> tuple[str, Path, int]:
        extension = Path(filename.replace("\\", "/")).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise UploadValidationError(
                "Unsupported video extension. Choose MP4, AVI, MOV, or MKV."
            )
        if max_bytes <= 0:
            raise UploadValidationError("The configured upload limit is invalid.", 500)
        self.root.mkdir(parents=True, exist_ok=True)
        run_id = self.new_run_id()
        directory = self.run_dir(run_id)
        directory.mkdir(exist_ok=False)
        input_name = f"input{extension}"
        destination = directory / input_name
        total = 0
        try:
            with destination.open("xb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise UploadValidationError(
                            "Video exceeds the configured payload limit.", 413
                        )
                    output.write(chunk)
            if total == 0:
                raise UploadValidationError("The uploaded video is empty.")
            if not destination.resolve().is_relative_to(directory):
                raise UploadValidationError("Upload path escaped its run directory.", 500)
            return run_id, destination, total
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def validate_decoder(self, run_id: str, video_path: Path) -> dict[str, Any]:
        directory = self.run_dir(run_id)
        resolved = video_path.resolve()
        if not resolved.is_relative_to(directory) or not resolved.is_file():
            raise UploadValidationError("Uploaded video is not inside its run directory.")
        capture = cv2.VideoCapture(str(resolved))
        try:
            if not capture.isOpened():
                raise UploadValidationError(
                    "OpenCV could not open this video. Check the file and installed codec support."
                )
            ok, _frame = capture.read()
            if not ok:
                raise UploadValidationError(
                    "OpenCV opened the file but could not decode a video frame."
                )
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            return {
                "total_frames": count if count > 0 else None,
                "source_fps": fps if fps > 0 else None,
            }
        finally:
            capture.release()

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def write_manifest(self, run_id: str, data: dict[str, Any]) -> None:
        path = self.manifest_path(run_id)
        temporary = path.with_suffix(".json.tmp")
        payload = dict(data)
        payload["run_id"] = run_id
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)

    def read_manifest(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = self.manifest_path(run_id)
        except UploadValidationError:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("run_id") != run_id:
            return None
        return data

    def remove_new_run(self, run_id: str) -> None:
        """Remove a run created by the current request before a job is accepted."""
        path = self.run_dir(run_id)
        if path.exists() and not path.is_symlink():
            shutil.rmtree(path)

    def add_artifact(self, run_id: str, key: str, path: Path) -> bool:
        expected = ARTIFACT_FILENAMES.get(key)
        directory = self.run_dir(run_id)
        resolved = path.resolve()
        if expected is None or not resolved.is_relative_to(directory):
            raise UploadValidationError("Artifact path is outside its run directory.", 500)
        if key == "source_video":
            if not any(resolved.name == f"input{extension}" for extension in ALLOWED_EXTENSIONS):
                raise UploadValidationError("Source artifact has an invalid generated name.", 500)
        elif resolved.name != expected:
            raise UploadValidationError("Artifact has an invalid generated name.", 500)
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            return False
        manifest = self.read_manifest(run_id) or {"run_id": run_id}
        if key == "annotated_video" and not (manifest.get("output_video") or {}).get("validated"):
            raise UploadValidationError(
                "Annotated output was not validated and cannot be published.", 500
            )
        artifacts = dict(manifest.get("artifacts", {}))
        artifacts[key] = resolved.name
        manifest["artifacts"] = artifacts
        self.write_manifest(run_id, manifest)
        return True

    def artifact_path(self, run_id: str, key: str) -> Path:
        if key not in ARTIFACT_FILENAMES:
            raise UploadValidationError("Unknown artifact.", 404)
        manifest = self.read_manifest(run_id)
        if not manifest:
            raise UploadValidationError("Run not found.", 404)
        expected = ARTIFACT_FILENAMES[key]
        actual_name = manifest.get("artifacts", {}).get(key)
        if key == "source_video":
            if not any(actual_name == f"input{extension}" for extension in ALLOWED_EXTENSIONS):
                raise UploadValidationError("This artifact was not generated for the selected run.", 404)
            expected = actual_name
        elif actual_name != expected:
            raise UploadValidationError("This artifact was not generated for the selected run.", 404)
        path = (self.run_dir(run_id) / expected).resolve()
        if not path.is_relative_to(self.run_dir(run_id)) or not path.is_file():
            raise UploadValidationError("Artifact is unavailable.", 404)
        return path

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        runs = []
        for child in self.root.iterdir():
            if child.is_symlink() or not child.is_dir() or not RUN_ID_PATTERN.fullmatch(child.name):
                continue
            try:
                manifest = self.read_manifest(child.name)
            except UploadValidationError:
                manifest = None
            if manifest:
                runs.append(manifest)
        runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return runs


def analytics_bundle(history_path: Path, summary_path: Path) -> dict[str, Any]:
    """Expose the existing analysis functions with honest tracker-ID labels."""
    rows = analyze_behavior.load_history(history_path)
    summary_rows = analyze_behavior.load_summary(summary_path)
    if not rows:
        raise UploadValidationError("No per-frame tracker records are available for analytics.", 422)
    session = analyze_behavior.session_stats(rows)
    n_tracker_ids = session.pop("n_students")
    tracker_ids = session.pop("unique_student_ids")
    session["n_tracker_ids"] = n_tracker_ids
    session["tracker_ids"] = tracker_ids
    per_tracker = analyze_behavior.per_student_stats(rows)
    for value in per_tracker.values():
        value.pop("live_confidences", None)
        value.pop("frames_seen", None)
    temporal = analyze_behavior.temporal_analysis(rows)
    for item in temporal["all_runs"]:
        item["tracker_id"] = item.pop("student_id")
    for item in temporal["longest_run_per_behavior"].values():
        if item is not None and "student_id" in item:
            item["tracker_id"] = item.pop("student_id")
    temporal["transitions_per_tracker_id"] = temporal.pop("transitions_per_student")
    temporal["avg_transitions_per_tracker_id"] = temporal.pop("avg_transitions_per_student")
    temporal["run_count"] = len(temporal.pop("all_runs"))
    coverage = analyze_behavior.coverage_analysis(per_tracker, session["n_frames"])
    coverage["full_coverage_tracker_ids"] = coverage.pop("full_coverage_students")
    coverage["short_track_tracker_ids"] = coverage.pop("short_track_students")
    coverage["high_unknown_tracker_ids"] = coverage.pop("high_unknown_students")
    coverage["strong_coverage_tracker_ids"] = coverage.pop("strong_coverage_students")
    unknown = analyze_behavior.unknown_analysis(rows, session["n_frames"])
    unknown["frames_all_unknown_count"] = len(unknown.pop("frames_all_unknown"))
    unknown["frames_any_known_count"] = len(unknown.pop("frames_any_known"))
    unknown["tracker_ids_always_unknown"] = unknown.pop("students_always_unknown")
    unknown["short_track_tracker_ids"] = unknown.pop("short_track_student_ids")
    unknown["per_tracker_id_unknown_pct"] = unknown.pop("per_student_unknown_pct")
    return {
        "session": session,
        "summary_rows": [
            {"tracker_id": row.pop("student_id"), **row}
            for row in summary_rows
        ],
        "distribution": analyze_behavior.overall_distribution(rows),
        "per_tracker_id": {str(key): value for key, value in per_tracker.items()},
        "temporal": temporal,
        "coverage": coverage,
        "confidence": {
            key: value for key, value in analyze_behavior.confidence_analysis(rows).items()
            if key != "all_live_conf"
        },
        "cache": analyze_behavior.cache_analysis(rows),
        "unknown": unknown,
        "semantics": {
            "frame_detections": "Raw behavior boxes returned by the behavior model per frame; these are not tracker records.",
            "history_records": "One tracker ID's behavior record in one processed frame, including cached rows.",
            "tracker_ids": "Per-video tracking identifiers; IDs are not linked across videos.",
            "unique_real_world_students": None,
        },
    }


def make_video_worker(store: RunStore, run_id: str, input_path: Path) -> Callable:
    """Build a job worker after the upload has been validated and isolated."""
    def worker(context) -> None:
        directory = store.run_dir(run_id)
        history_path = directory / "behavior_history.csv"
        summary_path = directory / "behavior_summary.csv"
        annotated_path = directory / "annotated.mp4"
        manifest = store.read_manifest(run_id) or {"run_id": run_id, "artifacts": {}}
        manifest.update(status="running", error=None)
        store.write_manifest(run_id, manifest)
        context.report(camera_status="not_applicable", inference_status="starting", run_id=run_id, error=None)
        try:
            settings = get_settings()
            if not settings.person_model.is_file():
                raise RuntimeError(
                    f"Person model not found: {settings.person_model}. Set VISIONCLASS_PERSON_MODEL to an existing checkpoint."
                )
            if not settings.behavior_model.is_file():
                raise RuntimeError(
                    f"Behavior model not found: {settings.behavior_model}. Set VISIONCLASS_BEHAVIOR_MODEL to an existing checkpoint."
                )
            try:
                person_model = YOLO(str(settings.person_model))
                behavior_model = YOLO(str(settings.behavior_model))
            except Exception as exc:
                raise RuntimeError(f"Could not load configured detection model: {exc}") from exc

            last_manifest_frame = 0
            last_progress: dict[str, Any] = {}
            def update(progress: dict[str, Any]) -> None:
                nonlocal last_manifest_frame, last_progress
                last_progress = progress
                context.report(
                    inference_status="running",
                    processed_frames=progress["processed_frames"],
                    total_frames=progress["total_frames"],
                    progress_percent=progress["progress_percent"],
                    history_record_total=progress["history_record_total"],
                    behavior_detection_total=progress["behavior_detection_total"],
                    class_counts=progress["frame_detection_counts"],
                    error=None,
                )
                if last_manifest_frame == 0 or progress["processed_frames"] - last_manifest_frame >= 10:
                    last_manifest_frame = progress["processed_frames"]
                    current = store.read_manifest(run_id) or manifest
                    current.update(status="running", **progress)
                    store.write_manifest(run_id, current)

            def update_output_status(stage: str) -> None:
                context.report(inference_status="finalizing_output")
                current = store.read_manifest(run_id) or manifest
                current.update(status="running", output_stage=stage)
                store.write_manifest(run_id, current)

            result = process_video(
                input_path,
                history_path,
                summary_path,
                annotated_path,
                settings=settings,
                person_model=person_model,
                behavior_model=behavior_model,
                stop_requested=lambda: context.stop_requested,
                progress_callback=update,
                output_status_callback=update_output_status,
                output_root=directory,
            )
            if result["cancelled"] or context.stop_requested:
                current = store.read_manifest(run_id) or manifest
                current.update(status="cancelled", output_stage="cancelled", progress=result, artifacts={}, error=None)
                store.write_manifest(run_id, current)
                context.report(camera_status="not_applicable", inference_status="stopped", error=None)
                return

            output_validation = result.get("output_video")
            if not isinstance(output_validation, dict) or output_validation.get("validated") is not True:
                annotated_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Annotated video output was not validated; the run cannot be marked completed."
                )
            current = store.read_manifest(run_id) or manifest
            final_progress_percent = (
                min(100.0, round(100 * result["processed_frames"] / result["total_frames"], 2))
                if result["total_frames"]
                else None
            )
            current.update(
                output_video=output_validation,
                progress=result,
                processed_frames=result["processed_frames"],
                total_frames=result["total_frames"],
                progress_percent=final_progress_percent,
                history_record_total=result["history_record_total"],
                behavior_detection_total=result["behavior_detection_total"],
            )
            if "frame_detection_counts" in last_progress:
                current["frame_detection_counts"] = last_progress["frame_detection_counts"]
            else:
                current.pop("frame_detection_counts", None)
            store.write_manifest(run_id, current)

            artifacts: dict[str, str] = {}
            for key, path in (
                ("source_video", input_path),
                ("annotated_video", annotated_path),
                ("history_csv", history_path),
                ("summary_csv", summary_path),
            ):
                if store.add_artifact(run_id, key, path):
                    artifacts[key] = ARTIFACT_FILENAMES[key]
                elif key == "annotated_video":
                    raise RuntimeError(
                        "Validated annotated video could not be registered as a run artifact."
                    )

            optional_errors = []
            try:
                if not analyze_behavior.load_history(history_path):
                    raise ValueError("No tracker records were written; analytics reports need at least one history row.")
                report_path = directory / ARTIFACT_FILENAMES["analytics_report"]
                generate_analytics_report.generate_run_report(
                    history_path, summary_path, report_path, processing_metrics=result
                )
                if store.add_artifact(run_id, "analytics_report", report_path):
                    artifacts["analytics_report"] = report_path.name
            except Exception as exc:
                optional_errors.append(f"Markdown analytics report unavailable: {exc}")
            try:
                dashboard_path = directory / ARTIFACT_FILENAMES["analytics_dashboard"]
                generate_analytics_report.generate_run_dashboard(
                    history_path, summary_path, dashboard_path
                )
                if store.add_artifact(run_id, "analytics_dashboard", dashboard_path):
                    artifacts["analytics_dashboard"] = dashboard_path.name
            except Exception as exc:
                optional_errors.append(f"HTML analytics report unavailable: {exc}")

            current = store.read_manifest(run_id) or manifest
            detection_total = result["behavior_detection_total"]
            unassociated = result["unassociated_detections"]
            association_rate = (
                max(0.0, (detection_total - unassociated) / detection_total)
                if detection_total
                else None
            )
            current.update(
                status="completed",
                output_stage="complete",
                progress=result,
                quality_status=(
                    "not_assessable" if association_rate is None
                    else "degraded" if association_rate <= 0.05
                    else "pass"
                ),
                association_rate=association_rate,
                artifacts=artifacts,
                optional_errors=optional_errors,
                error=None,
            )
            store.write_manifest(run_id, current)
            context.report(
                inference_status="completed",
                camera_status="not_applicable",
                processed_frames=result["processed_frames"],
                total_frames=result["total_frames"],
                progress_percent=100.0 if result["total_frames"] else None,
                history_record_total=result["history_record_total"],
                behavior_detection_total=result["behavior_detection_total"],
                artifacts=list(artifacts),
                error=None,
            )
        except Exception as exc:
            current = store.read_manifest(run_id) or manifest
            current.update(status="error", artifacts={}, error=str(exc))
            store.write_manifest(run_id, current)
            context.report(inference_status="error", error=str(exc), artifacts=[])
            raise
    return worker
