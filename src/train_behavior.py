import sys
from pathlib import Path
from dataclasses import replace

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from ultralytics import YOLO

from src.config import get_settings


def main(
    dataset_yaml: Path | str | None = None,
    device_spec: str | None = None,
    train_imgsz: int | None = None,
    train_epochs: int | None = None,
    train_batch: int | None = None,
    train_workers: int | None = None,
    train_project: str | None = None,
    train_name: str | None = None,
) -> None:
    settings = get_settings(
        dataset_yaml=dataset_yaml,
        device_spec=device_spec,
    )

    # Override configuration values using CLI arguments.
    # Settings is a frozen dataclass, so create a modified copy.
    overrides = {}

    if train_imgsz is not None:
        overrides["train_imgsz"] = train_imgsz

    if train_epochs is not None:
        overrides["train_epochs"] = train_epochs

    if train_batch is not None:
        overrides["train_batch"] = train_batch

    if train_workers is not None:
        overrides["train_workers"] = train_workers

    if train_project is not None:
        overrides["train_project"] = train_project

    if train_name is not None:
        overrides["train_name"] = train_name

    if overrides:
        settings = replace(settings, **overrides)

    print("=" * 60)
    print("VisionClass - Behavior Detection Training")
    print("=" * 60)

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available() and isinstance(settings.device, int):
        print(f"GPU: {torch.cuda.get_device_name(settings.device)}")

    print(f"Device: {settings.device}")
    print(f"Dataset: {settings.dataset_yaml}")

    print(f"Training image size: {settings.train_imgsz}")
    print(f"Training epochs: {settings.train_epochs}")
    print(f"Training batch size: {settings.train_batch}")
    print(f"Training workers: {settings.train_workers}")
    print(f"Training project: {settings.train_project}")
    print(f"Training name: {settings.train_name}")

    if not settings.dataset_yaml.is_file():
        print()
        print(
            "ERROR: Dataset YAML not found. Set VISIONCLASS_DATASET_YAML or pass "
            "--dataset pointing to your Roboflow data.yaml."
        )
        return

    print("=" * 60)

    model = YOLO(str(settings.pretrained_model))

    model.train(
        data=str(settings.dataset_yaml),
        epochs=settings.train_epochs,
        imgsz=settings.train_imgsz,
        batch=settings.train_batch,
        device=settings.device,
        workers=settings.train_workers,
        project=settings.train_project,
        name=settings.train_name,
        save=True,
        val=True,
        seed=42,
        plots=True,
    )

    print()
    print("=" * 60)
    print("Training completed.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train YOLO behavior detector"
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to YOLO data.yaml "
            "(default: datasets/student_classroom_behavior/data.yaml "
            "or VISIONCLASS_DATASET_YAML)"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device: auto, cpu, or GPU index "
            "(default: auto / VISIONCLASS_DEVICE)"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Training image size, e.g. 640 or 960",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Training batch size",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of dataloader workers",
    )

    parser.add_argument(
        "--project",
        default=None,
        help="Ultralytics training project directory",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Ultralytics training run name",
    )

    args = parser.parse_args()

    main(
        dataset_yaml=args.dataset,
        device_spec=args.device,
        train_imgsz=args.imgsz,
        train_epochs=args.epochs,
        train_batch=args.batch,
        train_workers=args.workers,
        train_project=args.project,
        train_name=args.name,
    )