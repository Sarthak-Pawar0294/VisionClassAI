import statistics
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "student_classroom_behavior"
)

SPLITS = ["train", "valid", "test"]

TARGET_CLASS = 4
TARGET_NAME = "using_electronic_devices"


# ============================================================
# Helper
# ============================================================

def percentile(values, p):

    if not values:
        return 0.0

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


def print_stats(name, values):

    if not values:
        print(f"{name}: no data")
        return

    print(f"{name}:")
    print(f"  Count : {len(values)}")
    print(
        f"  Min   : {min(values):.6f}"
    )
    print(
        f"  P10   : {percentile(values, 0.10):.6f}"
    )
    print(
        f"  P25   : {percentile(values, 0.25):.6f}"
    )
    print(
        f"  Median: {statistics.median(values):.6f}"
    )
    print(
        f"  P75   : {percentile(values, 0.75):.6f}"
    )
    print(
        f"  P90   : {percentile(values, 0.90):.6f}"
    )
    print(
        f"  Max   : {max(values):.6f}"
    )


def main() -> int:
    all_widths = []
    all_heights = []
    all_areas = []
    split_stats = {}

    for split in SPLITS:
        labels_dir = DATASET_ROOT / split / "labels"
        if not labels_dir.exists():
            print(f"Skipping missing directory: {labels_dir}")
            continue

        widths = []
        heights = []
        areas = []
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as label_file_handle:
                for line in label_file_handle:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        class_id = int(parts[0])
                        _, _, width, height = map(float, parts[1:])
                    except ValueError:
                        continue
                    if class_id != TARGET_CLASS:
                        continue

                    # YOLO normalized dimensions
                    widths.append(width)
                    heights.append(height)
                    areas.append(width * height)
                    all_widths.append(width)
                    all_heights.append(height)
                    all_areas.append(width * height)

        split_stats[split] = {
            "count": len(areas),
            "widths": widths,
            "heights": heights,
            "areas": areas,
        }

    print("=" * 70)
    print("VisionClass - Device Bounding Box Size Analysis")
    print("=" * 70)
    print()
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Target class: {TARGET_CLASS} = {TARGET_NAME}")

    for split, stats in split_stats.items():
        print()
        print("=" * 70)
        print(f"{split.upper()} SPLIT")
        print("=" * 70)
        print(f"Device instances: {stats['count']}")
        print()
        print_stats("Normalized box WIDTH", stats["widths"])
        print()
        print_stats("Normalized box HEIGHT", stats["heights"])
        print()
        print_stats("Normalized box AREA", stats["areas"])

    print()
    print("=" * 70)
    print("OVERALL DEVICE BOX DISTRIBUTION")
    print("=" * 70)
    print()
    print_stats("Width", all_widths)
    print()
    print_stats("Height", all_heights)
    print()
    print_stats("Area", all_areas)

    print()
    print("=" * 70)
    print("SMALL-OBJECT ANALYSIS")
    print("=" * 70)
    if all_areas:
        thresholds = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
        for threshold in thresholds:
            count = sum(area < threshold for area in all_areas)
            percentage = count / len(all_areas) * 100
            print(
                f"Area < {threshold:<7}: {count:5} / "
                f"{len(all_areas)} ({percentage:6.2f}%)"
            )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
