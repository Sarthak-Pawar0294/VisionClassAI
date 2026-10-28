"""
Central configuration for VisionClass.

Paths default to project-relative locations. Override with environment variables
or CLI flags on scripts that support them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch

# Project root (VisionClass/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DeviceSpec = Union[int, str]


def resolve_device(spec: str | None = None) -> DeviceSpec:
    """
    Resolve Ultralytics/PyTorch device.

    spec / VISIONCLASS_DEVICE:
      - "auto" (default): CUDA GPU 0 if available, else "cpu"
      - "cpu"
      - "0", "1", ...: CUDA device index
    """
    raw = (spec or os.environ.get("VISIONCLASS_DEVICE", "auto")).strip().lower()
    if raw == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if raw == "cpu":
        return "cpu"
    return int(raw)


def _path_from_env(env_key: str, default: Path) -> Path:
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    return default


def _discover_behavior_model(root: Path) -> Path:
    """
    Auto-discover the most recently trained behavior model under runs/.

    Search rule:
      1. Walk every directory matching runs/**/weights/best.pt (any depth).
      2. Keep only those that actually exist on disk.
      3. Sort by directory modification time (newest first).
      4. Return the path of the newest valid best.pt.

    If no valid model is found, return a sentinel Path that does NOT exist so
    callers can detect the missing-model case and print a clear error message.

    Override at any time via the VISIONCLASS_BEHAVIOR_MODEL environment variable.
    """
    runs_dir = root / "runs"
    candidates: list[tuple[float, Path]] = []

    if runs_dir.is_dir():
        for best_pt in runs_dir.rglob("weights/best.pt"):
            if best_pt.is_file():
                mtime = best_pt.stat().st_mtime
                candidates.append((mtime, best_pt))

    if candidates:
        # Newest (highest mtime) first
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    # Sentinel: no trained model found
    return root / "runs" / "detect" / "runs" / "visionclass_behavior" / "weights" / "best.pt"


@dataclass(frozen=True)
class Settings:
    project_root: Path

    behavior_model: Path
    person_model: Path
    pretrained_model: Path

    dataset_yaml: Path

    video_path: Path
    video_output: Path
    tracking_output_video: Path
    history_output_video: Path

    history_csv: Path
    summary_csv: Path

    behavior_conf: float
    person_conf: float
    inference_imgsz: int
    train_imgsz: int
    train_epochs: int
    train_batch: int
    eval_batch: int

    tracker: str
    train_workers: int
    eval_workers: int

    device: DeviceSpec
    webcam_index: int

    train_project: str
    train_name: str
    eval_project: str
    eval_name: str


def get_settings(
    *,
    device_spec: str | None = None,
    dataset_yaml: Path | str | None = None,
    behavior_model: Path | str | None = None,
) -> Settings:
    root = PROJECT_ROOT

    # Resolve behavior_model: explicit arg > env var > auto-discovery
    if behavior_model:
        resolved_behavior_model = Path(behavior_model).resolve()
    else:
        env_model = os.environ.get("VISIONCLASS_BEHAVIOR_MODEL")
        if env_model:
            resolved_behavior_model = Path(env_model).expanduser().resolve()
        else:
            resolved_behavior_model = _discover_behavior_model(root)

    return Settings(
        project_root=root,
        behavior_model=resolved_behavior_model,
        person_model=_path_from_env(
            "VISIONCLASS_PERSON_MODEL",
            root / "yolo11n.pt",
        ),
        pretrained_model=_path_from_env(
            "VISIONCLASS_PRETRAINED_MODEL",
            root / "yolo11n.pt",
        ),
        dataset_yaml=Path(dataset_yaml).resolve()
        if dataset_yaml
        else _path_from_env(
            "VISIONCLASS_DATASET_YAML",
            root / "datasets" / "student_classroom_behavior" / "data.yaml",
        ),
        video_path=_path_from_env(
            "VISIONCLASS_VIDEO",
            root / "data" / "videos" / "classroom.mp4",
        ),
        video_output=_path_from_env(
            "VISIONCLASS_VIDEO_OUTPUT",
            root / "data" / "videos" / "classroom_result.mp4",
        ),
        tracking_output_video=_path_from_env(
            "VISIONCLASS_TRACKING_OUTPUT",
            root / "data" / "videos" / "classroom_tracking.mp4",
        ),
        history_output_video=_path_from_env(
            "VISIONCLASS_HISTORY_VIDEO_OUTPUT",
            root / "data" / "videos" / "classroom_history.mp4",
        ),
        history_csv=_path_from_env(
            "VISIONCLASS_HISTORY_CSV",
            root / "data" / "classroom_behavior_history.csv",
        ),
        summary_csv=_path_from_env(
            "VISIONCLASS_SUMMARY_CSV",
            root / "data" / "classroom_behavior_summary.csv",
        ),
        behavior_conf=float(os.environ.get("VISIONCLASS_BEHAVIOR_CONF", "0.20")),
        person_conf=float(os.environ.get("VISIONCLASS_PERSON_CONF", "0.30")),
        inference_imgsz=int(os.environ.get("VISIONCLASS_INFERENCE_IMGSZ", "960")),
        train_imgsz=int(os.environ.get("VISIONCLASS_TRAIN_IMGSZ", "640")),
        train_epochs=int(os.environ.get("VISIONCLASS_TRAIN_EPOCHS", "10")),
        train_batch=int(os.environ.get("VISIONCLASS_TRAIN_BATCH", "8")),
        eval_batch=int(os.environ.get("VISIONCLASS_EVAL_BATCH", "8")),
        tracker=os.environ.get(
            "VISIONCLASS_TRACKER",
            str(root / "config" / "bytetrack.yaml")
        ),
        train_workers=int(os.environ.get("VISIONCLASS_TRAIN_WORKERS", "2")),
        eval_workers=int(os.environ.get("VISIONCLASS_EVAL_WORKERS", "0")),
        device=resolve_device(device_spec),
        webcam_index=int(os.environ.get("VISIONCLASS_WEBCAM_INDEX", "0")),
        train_project=os.environ.get("VISIONCLASS_TRAIN_PROJECT", "runs"),
        train_name=os.environ.get("VISIONCLASS_TRAIN_NAME", "visionclass_behavior"),
        eval_project=os.environ.get("VISIONCLASS_EVAL_PROJECT", "runs"),
        eval_name=os.environ.get("VISIONCLASS_EVAL_NAME", "visionclass_test"),
    )


# Default behavior labels for summary CSV (matches Roboflow dataset + unknown)
SUMMARY_BEHAVIOR_COLUMNS = [
    "handrise",
    "read",
    "sleep",
    "stand",
    "using_electronic_devices",
    "write",
    "unknown",
]
