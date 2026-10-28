import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

from src.config import get_settings


def _resolve_evaluation_imgsz(imgsz: int | None, inference_imgsz: int) -> int:
    return inference_imgsz if imgsz is None else imgsz


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

def _save_per_class_metrics(
    results: object,
    model: object,
    save_dir: Path,
) -> None:
    """
    Extract per-class AP metrics from a YOLO val() result and write to CSV.

    Columns:
        class_id
        class_name
        precision
        recall
        ap50
        ap50_95

    Falls back gracefully if the Ultralytics API does not expose
    per-class data for this version.
    """

    try:
        box = results.box

        # Class indices for which AP data exists
        class_indices = box.ap_class_index.tolist()

        # Per-class AP
        ap50_per_class = box.ap50.tolist()
        ap_per_class = box.ap.tolist()

        # Per-class precision
        p_per_class = (
            box.p.tolist()
            if hasattr(box, "p")
            else [None] * len(class_indices)
        )

        # Per-class recall
        r_per_class = (
            box.r.tolist()
            if hasattr(box, "r")
            else [None] * len(class_indices)
        )

        rows = []

        for i, class_id in enumerate(class_indices):

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": model.names.get(
                        class_id,
                        f"class_{class_id}",
                    ),
                    "precision": (
                        round(p_per_class[i], 6)
                        if p_per_class[i] is not None
                        else ""
                    ),
                    "recall": (
                        round(r_per_class[i], 6)
                        if r_per_class[i] is not None
                        else ""
                    ),
                    "ap50": round(
                        ap50_per_class[i],
                        6,
                    ),
                    "ap50_95": round(
                        ap_per_class[i],
                        6,
                    ),
                }
            )

        csv_path = save_dir / "per_class_metrics.csv"

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class_id",
                    "class_name",
                    "precision",
                    "recall",
                    "ap50",
                    "ap50_95",
                ],
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"\nPer-class metrics saved → {csv_path}"
        )

        print()

        print(
            f"  {'Class':<30} "
            f"{'P':>8} "
            f"{'R':>8} "
            f"{'AP50':>8} "
            f"{'AP50-95':>9}"
        )

        print("  " + "-" * 68)

        for row in rows:

            print(
                f"  {row['class_name']:<30} "
                f"{str(row['precision']):>8} "
                f"{str(row['recall']):>8} "
                f"{str(row['ap50']):>8} "
                f"{str(row['ap50_95']):>9}"
            )

    except AttributeError as e:

        print(
            f"\nWARNING: Could not extract per-class metrics ({e})."
        )

        print(
            "  Ultralytics API may differ for this version."
        )

        print(
            "  Overall metrics are still printed above."
        )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def main(
    dataset_yaml: Path | str | None = None,
    device_spec: str | None = None,
    behavior_model: Path | str | None = None,
    imgsz: int | None = None,
) -> None:

    settings = get_settings(
        dataset_yaml=dataset_yaml,
        device_spec=device_spec,
        behavior_model=behavior_model,
    )

    # -----------------------------------------------------------------------
    # Select evaluation image size
    #
    # If --imgsz is provided, use it.
    # Otherwise use the configured inference image size.
    # -----------------------------------------------------------------------

    evaluation_imgsz = _resolve_evaluation_imgsz(
        imgsz,
        settings.inference_imgsz,
    )

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------

    print("=" * 60)
    print("VisionClass - Model Evaluation")
    print("=" * 60)

    print(f"Device:      {settings.device}")
    print(f"Model:       {settings.behavior_model}")
    print(f"Dataset:     {settings.dataset_yaml}")
    print(f"Image size:  {evaluation_imgsz}")

    # -----------------------------------------------------------------------
    # Check model
    # -----------------------------------------------------------------------

    if not settings.behavior_model.is_file():

        print(
            "ERROR: Behavior model weights not found."
        )

        return

    # -----------------------------------------------------------------------
    # Check dataset
    # -----------------------------------------------------------------------

    if not settings.dataset_yaml.is_file():

        print(
            "ERROR: Dataset YAML not found."
        )

        print(
            f"Expected dataset: {settings.dataset_yaml}"
        )

        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    model = YOLO(
        str(settings.behavior_model)
    )

    # -----------------------------------------------------------------------
    # Run YOLO test evaluation
    # -----------------------------------------------------------------------

    results = model.val(
        data=str(settings.dataset_yaml),
        split="test",
        imgsz=evaluation_imgsz,
        batch=settings.eval_batch,
        device=settings.device,
        workers=settings.eval_workers,
        plots=True,
        project=settings.eval_project,
        name=settings.eval_name,
    )

    # -----------------------------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"mAP50:       {results.box.map50:.4f}"
    )

    print(
        f"mAP50-95:    {results.box.map:.4f}"
    )

    print(
        f"Precision:   {results.box.mp:.4f}"
    )

    print(
        f"Recall:      {results.box.mr:.4f}"
    )

    # -----------------------------------------------------------------------
    # Locate evaluation output directory
    # -----------------------------------------------------------------------

    try:

        save_dir = Path(
            results.save_dir
        )

    except AttributeError:

        save_dir = (
            Path(settings.eval_project)
            / settings.eval_name
        )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Save per-class metrics
    # -----------------------------------------------------------------------

    _save_per_class_metrics(
        results,
        model,
        save_dir,
    )

    print(
        "=" * 60
    )


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VisionClass behavior model "
            "on the test split."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Path to YOLO data.yaml. If omitted, use "
            "datasets/student_classroom_behavior/data.yaml or "
            "VISIONCLASS_DATASET_YAML."
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to behavior model weights. "
            "If omitted, the configured VisionClass model is used."
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device: auto, cpu, or GPU index "
            "(default: auto / VISIONCLASS_DEVICE)."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help=(
            "Evaluation image size, e.g. 640, 960, or 1280. "
            "If omitted, settings.inference_imgsz is used (default: 960)."
        ),
    )

    args = parser.parse_args()

    main(
        dataset_yaml=args.dataset,
        device_spec=args.device,
        behavior_model=args.model,
        imgsz=args.imgsz,
    )