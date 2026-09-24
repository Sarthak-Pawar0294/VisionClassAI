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

    return Settings(
        project_root=root,
        behavior_model=Path(behavior_model).resolve()
        if behavior_model
        else _path_from_env(
            "VISIONCLASS_BEHAVIOR_MODEL",
            root
            / "runs"
            / "detect"
            / "runs"
            / "visionclass_behavior-3"
            / "weights"
            / "best.pt",
        ),
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
            root / "data" / "dataset" / "data.yaml",
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
        inference_imgsz=int(os.environ.get("VISIONCLASS_INFERENCE_IMGSZ", "1280")),
        train_imgsz=int(os.environ.get("VISIONCLASS_TRAIN_IMGSZ", "640")),
        train_epochs=int(os.environ.get("VISIONCLASS_TRAIN_EPOCHS", "10")),
        train_batch=int(os.environ.get("VISIONCLASS_TRAIN_BATCH", "8")),
        eval_batch=int(os.environ.get("VISIONCLASS_EVAL_BATCH", "8")),
        tracker=os.environ.get("VISIONCLASS_TRACKER", "bytetrack.yaml"),
        train_workers=int(os.environ.get("VISIONCLASS_TRAIN_WORKERS", "2")),
        eval_workers=int(os.environ.get("VISIONCLASS_EVAL_WORKERS", "0")),
        device=resolve_device(device_spec),
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
