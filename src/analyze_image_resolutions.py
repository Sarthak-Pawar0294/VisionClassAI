from collections import Counter
from pathlib import Path

import cv2


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "student_classroom_behavior"
)
TEST_IMAGES = DATASET_ROOT / "test" / "images"


def main() -> int:
    if not TEST_IMAGES.is_dir():
        print(f"Test image directory not found: {TEST_IMAGES}")
        print("Restore the canonical dataset under datasets/student_classroom_behavior.")
        return 1

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    resolutions = []
    failed = []

    for image_path in sorted(TEST_IMAGES.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in image_extensions
        ):
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            failed.append(image_path.name)
            continue

        height, width = image.shape[:2]
        resolutions.append((width, height))

    print("=" * 70)
    print("VisionClass - Test Image Resolution Analysis")
    print("=" * 70)
    print()
    print(f"Test directory: {TEST_IMAGES}")
    print(f"Images analyzed: {len(resolutions)}")
    print(f"Failed images:   {len(failed)}")
    print()
    print("=" * 70)
    print("RESOLUTION DISTRIBUTION")
    print("=" * 70)

    counts = Counter(resolutions)
    for resolution, count in counts.most_common():
        width, height = resolution
        percentage = count / len(resolutions) * 100
        print(
            f"{width} x {height:<5} : "
            f"{count:4} images "
            f"({percentage:6.2f}%)"
        )

    if resolutions:
        widths = [width for width, _ in resolutions]
        heights = [height for _, height in resolutions]
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"Width  - min: {min(widths)}, "
            f"max: {max(widths)}, "
            f"average: {sum(widths) / len(widths):.1f}"
        )
        print(
            f"Height - min: {min(heights)}, "
            f"max: {max(heights)}, "
            f"average: {sum(heights) / len(heights):.1f}"
        )

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
