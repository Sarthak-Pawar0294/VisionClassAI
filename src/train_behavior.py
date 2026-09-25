import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from ultralytics import YOLO

from src.config import get_settings


def main(
    dataset_yaml: Path | str | None = None,
    device_spec: str | None = None,
) -> None:
    settings = get_settings(dataset_yaml=dataset_yaml, device_spec=device_spec)

    print("=" * 60)
    print("VisionClass - Behavior Detection Training")
    print("=" * 60)

    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available() and isinstance(settings.device, int):
        print(f"GPU: {torch.cuda.get_device_name(settings.device)}")

    print(f"Device: {settings.device}")
    print(f"Dataset: {settings.dataset_yaml}")

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

    parser = argparse.ArgumentParser(description="Train YOLO behavior detector")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to YOLO data.yaml (default: data/dataset/data.yaml or VISIONCLASS_DATASET_YAML)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(dataset_yaml=args.dataset, device_spec=args.device)
