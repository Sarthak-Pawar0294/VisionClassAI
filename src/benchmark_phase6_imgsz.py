from pathlib import Path
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = str(
    PROJECT_ROOT
    / "datasets"
    / "student_classroom_behavior"
    / "data.yaml"
)

MODEL = str(
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "runs"
    / "phase6"
    / "visionclass_960-2"
    / "weights"
    / "best.pt"
)

DEVICE = 0

IMAGE_SIZES = [640, 960, 1280]

PROJECT = str(PROJECT_ROOT / "runs" / "detect" / "runs" / "phase6_imgsz_benchmark")


# ============================================================
# Run benchmark
# ============================================================

def main() -> None:
    print("=" * 70)
    print("VisionClass - Phase 6 Image Size Benchmark")
    print("=" * 70)

    print(f"Model:   {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Device:  {DEVICE}")
    print(f"Sizes:   {IMAGE_SIZES}")
    print()

    model = YOLO(MODEL)

    results_table = []


    for imgsz in IMAGE_SIZES:

        print()
        print("=" * 70)
        print(f"EVALUATING AT {imgsz}x{imgsz}")
        print("=" * 70)

        result = model.val(
            data=DATASET,
            split="test",
            imgsz=imgsz,
            batch=4,
            device=DEVICE,
            workers=0,
            plots=True,
            project=PROJECT,
            name=f"imgsz_{imgsz}",
            exist_ok=True,
            verbose=True,
        )

        # --------------------------------------------------------
        # Overall metrics
        # --------------------------------------------------------

        precision = float(
            result.box.mp
        )

        recall = float(
            result.box.mr
        )

        map50 = float(
            result.box.map50
        )

        map50_95 = float(
            result.box.map
        )

        # --------------------------------------------------------
        # Per-class AP50
        # --------------------------------------------------------

        device_class_id = 4

        device_ap50 = float(
            result.box.ap50[device_class_id]
        )

        device_ap50_95 = float(
            result.box.ap[device_class_id]
        )

        results_table.append({
            "imgsz": imgsz,
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "device_AP50": device_ap50,
            "device_AP50-95": device_ap50_95,
        })


    # ============================================================
    # Print comparison
    # ============================================================

    print()
    print()
    print("=" * 90)
    print("PHASE 6 IMAGE SIZE COMPARISON")
    print("=" * 90)

    print(
        f"{'Size':<10}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'mAP50':<12}"
        f"{'mAP50-95':<14}"
        f"{'Device AP50':<14}"
        f"{'Device AP50-95':<16}"
    )

    print("-" * 90)

    for row in results_table:

        print(
            f"{row['imgsz']:<10}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
            f"{row['mAP50']:<12.4f}"
            f"{row['mAP50-95']:<14.4f}"
            f"{row['device_AP50']:<14.4f}"
            f"{row['device_AP50-95']:<16.4f}"
        )

    print("=" * 90)

    print()
    print("Benchmark complete.")
    print(f"Results saved under: {PROJECT}")

if __name__ == "__main__":
    main()
