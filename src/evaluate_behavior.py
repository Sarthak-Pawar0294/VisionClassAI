import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

from src.config import get_settings


def main(
    dataset_yaml: Path | str | None = None,
    device_spec: str | None = None,
    behavior_model: Path | str | None = None,
) -> None:
    settings = get_settings(
        dataset_yaml=dataset_yaml,
        device_spec=device_spec,
        behavior_model=behavior_model,
    )

    print("=" * 60)
    print("VisionClass - Model Evaluation")
    print("=" * 60)
    print(f"Device: {settings.device}")
    print(f"Model: {settings.behavior_model}")
    print(f"Dataset: {settings.dataset_yaml}")

    if not settings.behavior_model.is_file():
        print("ERROR: Behavior model weights not found.")
        return

    if not settings.dataset_yaml.is_file():
        print(
            "ERROR: Dataset YAML not found. Set VISIONCLASS_DATASET_YAML or pass --dataset."
        )
        return

    model = YOLO(str(settings.behavior_model))

    results = model.val(
        data=str(settings.dataset_yaml),
        split="test",
        imgsz=settings.train_imgsz,
        batch=settings.eval_batch,
        device=settings.device,
        workers=settings.eval_workers,
        plots=True,
        project=settings.eval_project,
        name=settings.eval_name,
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION COMPLETE")
    print("=" * 60)

    print(f"mAP50:     {results.box.map50:.4f}")
    print(f"mAP50-95:  {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall:    {results.box.mr:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate behavior model on test split")
    parser.add_argument("--dataset", default=None, help="Path to YOLO data.yaml")
    parser.add_argument("--model", default=None, help="Path to behavior model weights")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    main(
        dataset_yaml=args.dataset,
        device_spec=args.device,
        behavior_model=args.model,
    )
