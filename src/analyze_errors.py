import argparse
import csv
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


# ============================================================
# Class configuration
# ============================================================

TARGET_CLASSES = {
    0: "handrise",
    1: "read",
    2: "sleep",
    3: "stand",
    4: "using_electronic_devices",
    5: "write",
}

CLASS_NAMES = {
    0: "handrise",
    1: "read",
    2: "sleep",
    3: "stand",
    4: "using_electronic_devices",
    5: "write",
}


# ============================================================
# Utility functions
# ============================================================

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    """Convert normalized YOLO xywh to pixel xyxy."""

    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    return [
        cx - bw / 2,
        cy - bh / 2,
        cx + bw / 2,
        cy + bh / 2,
    ]


def iou(box_a, box_b):
    """Calculate IoU between two xyxy boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def read_labels(label_path, img_w, img_h):
    """Read YOLO labels and convert them to pixel boxes."""

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                continue

            box = xywhn_to_xyxy(
                x,
                y,
                w,
                h,
                img_w,
                img_h,
            )

            labels.append({
                "class_id": cls_id,
                "box": box,
            })

    return labels


def draw_box(image, box, label, color):
    """Draw a labelled bounding box."""

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# Dataset path resolution
# ============================================================

def resolve_test_paths(dataset_yaml):
    """Resolve test images and labels from the dataset root."""

    dataset_yaml = Path(dataset_yaml).resolve()

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}"
        )

    dataset_root = dataset_yaml.parent

    test_images = (
        dataset_root
        / "test"
        / "images"
    ).resolve()

    test_labels = (
        dataset_root
        / "test"
        / "labels"
    ).resolve()

    if not test_images.exists():
        raise FileNotFoundError(
            f"Test images directory not found: "
            f"{test_images}"
        )

    if not test_labels.exists():
        raise FileNotFoundError(
            f"Test labels directory not found: "
            f"{test_labels}"
        )

    return (
        dataset_yaml,
        test_images,
        test_labels,
    )

# ============================================================
# Main
# ============================================================

def main(
    dataset,
    model_path,
    device,
    imgsz,
    conf,
    iou_match,
    output_dir,
):

    # --------------------------------------------------------
    # Resolve dataset
    # --------------------------------------------------------

    (
        dataset_yaml,
        test_images,
        test_labels,
    ) = resolve_test_paths(dataset)

    if not test_images.exists():
        raise FileNotFoundError(
            f"Test images directory not found: "
            f"{test_images}"
        )

    if not test_labels.exists():
        raise FileNotFoundError(
            f"Test labels directory not found: "
            f"{test_labels}"
        )

    # --------------------------------------------------------
    # Resolve model
    # --------------------------------------------------------

    model_path = Path(model_path).resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    for class_id, class_name in TARGET_CLASSES.items():

        class_dir = (
            output_dir
            / f"{class_id}_{class_name}"
        )

        for failure_type in [
            "false_negative",
            "localization",
            "wrong_class",
        ]:
            (
                class_dir / failure_type
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(str(model_path))

    # --------------------------------------------------------
    # Find test images
    # --------------------------------------------------------

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_paths = sorted(
        [
            p
            for p in test_images.iterdir()
            if p.is_file()
            and p.suffix.lower()
            in image_extensions
        ]
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No test images found in: "
            f"{test_images}"
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("=" * 70)
    print("VisionClass - Targeted Error Analysis")
    print("=" * 70)

    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Model:        {model_path}")
    print(f"Test images:  {test_images}")
    print(f"Test labels:  {test_labels}")
    print(f"Images:       {len(image_paths)}")
    print(f"Device:       {device}")
    print(f"Image size:   {imgsz}")
    print(f"Confidence:   {conf}")
    print(f"Match IoU:    {iou_match}")
    print(f"Output:       {output_dir}")
    print()

    # --------------------------------------------------------
    # Summary storage
    # --------------------------------------------------------

    summary = []

    # Keep visual examples manageable.
    saved_examples = {
        class_id: {
            "false_negative": 0,
            "localization": 0,
            "wrong_class": 0,
        }
        for class_id in TARGET_CLASSES
    }

    MAX_EXAMPLES = 25

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    results = model.predict(
        source=str(test_images),
        imgsz=imgsz,
        conf=conf,
        device=device,
        stream=True,
        verbose=False,
        save=False,
    )

    processed = 0

    # ========================================================
    # Process every image
    # ========================================================

    for result in results:

        processed += 1

        image_path = Path(result.path)

        label_path = (
            test_labels
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original = result.orig_img.copy()

        img_h, img_w = original.shape[:2]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        ground_truth = read_labels(
            label_path,
            img_w,
            img_h,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = []

        if result.boxes is not None:

            boxes = (
                result.boxes.xyxy
                .cpu()
                .tolist()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            for (
                box,
                cls_id,
                confidence,
            ) in zip(
                boxes,
                classes,
                confidences,
            ):

                predictions.append({
                    "class_id": int(cls_id),
                    "box": box,
                    "confidence": float(
                        confidence
                    ),
                })

        # ====================================================
        # Analyze target classes
        # ====================================================

        for target_class in TARGET_CLASSES:

            target_gt = [
                gt
                for gt in ground_truth
                if gt["class_id"]
                == target_class
            ]

            for gt_index, gt in enumerate(
                target_gt
            ):

                # ------------------------------------------------
                # Find same-class prediction with highest IoU
                # ------------------------------------------------

                same_class_predictions = [
                    p
                    for p in predictions
                    if p["class_id"]
                    == target_class
                ]

                best_same = None
                best_same_iou = 0.0

                for p in same_class_predictions:

                    current_iou = iou(
                        gt["box"],
                        p["box"],
                    )

                    if current_iou > best_same_iou:

                        best_same_iou = current_iou
                        best_same = p

                # ------------------------------------------------
                # Correct detection
                # ------------------------------------------------

                if (
                    best_same is not None
                    and best_same_iou >= iou_match
                ):
                    continue

                # ------------------------------------------------
                # Find strongest wrong-class prediction
                # ------------------------------------------------

                best_wrong = None
                best_wrong_iou = 0.0

                for p in predictions:

                    if (
                        p["class_id"]
                        == target_class
                    ):
                        continue

                    current_iou = iou(
                        gt["box"],
                        p["box"],
                    )

                    if current_iou > best_wrong_iou:

                        best_wrong_iou = current_iou
                        best_wrong = p

                # ------------------------------------------------
                # Determine failure type
                # ------------------------------------------------

                if (
                    best_same is not None
                    and best_same_iou >= 0.10
                ):

                    failure_type = (
                        "localization"
                    )

                elif (
                    best_wrong is not None
                    and best_wrong_iou >= 0.10
                ):

                    failure_type = (
                        "wrong_class"
                    )

                else:

                    failure_type = (
                        "false_negative"
                    )

                # ------------------------------------------------
                # Save visual example
                # ------------------------------------------------

                if (
                    saved_examples[
                        target_class
                    ][failure_type]
                    < MAX_EXAMPLES
                ):

                    visual = original.copy()

                    # Ground truth = GREEN
                    draw_box(
                        visual,
                        gt["box"],
                        (
                            f"GT: "
                            f"{TARGET_CLASSES[target_class]}"
                        ),
                        (0, 255, 0),
                    )

                    # Localization prediction
                    if (
                        failure_type
                        == "localization"
                        and best_same is not None
                    ):

                        draw_box(
                            visual,
                            best_same["box"],
                            (
                                f"PRED: "
                                f"{TARGET_CLASSES[target_class]} "
                                f"{best_same['confidence']:.2f} "
                                f"IoU={best_same_iou:.2f}"
                            ),
                            (0, 165, 255),
                        )

                    # Wrong-class prediction
                    elif (
                        failure_type
                        == "wrong_class"
                        and best_wrong is not None
                    ):

                        wrong_name = (
                            CLASS_NAMES.get(
                                best_wrong["class_id"],
                                str(
                                    best_wrong[
                                        "class_id"
                                    ]
                                ),
                            )
                        )

                        draw_box(
                            visual,
                            best_wrong["box"],
                            (
                                f"PRED: "
                                f"{wrong_name} "
                                f"{best_wrong['confidence']:.2f} "
                                f"IoU={best_wrong_iou:.2f}"
                            ),
                            (0, 0, 255),
                        )

                    # Output folder
                    output_folder = (
                        output_dir
                        / (
                            f"{target_class}_"
                            f"{TARGET_CLASSES[target_class]}"
                        )
                        / failure_type
                    )

                    output_name = (
                        f"{image_path.stem}"
                        f"_gt{gt_index}"
                        f"_{failure_type}.jpg"
                    )

                    cv2.imwrite(
                        str(
                            output_folder
                            / output_name
                        ),
                        visual,
                    )

                    saved_examples[
                        target_class
                    ][failure_type] += 1

                # ------------------------------------------------
                # Store CSV row
                # ------------------------------------------------

                wrong_class_name = ""

                if best_wrong is not None:

                    wrong_class_name = (
                        CLASS_NAMES.get(
                            best_wrong["class_id"],
                            str(
                                best_wrong[
                                    "class_id"
                                ]
                            ),
                        )
                    )

                summary.append({
                    "image": image_path.name,
                    "class_id": target_class,
                    "class_name": (
                        TARGET_CLASSES[target_class]
                    ),
                    "failure_type": failure_type,
                    "same_class_iou": round(
                        best_same_iou,
                        4,
                    ),
                    "wrong_class": (
                        wrong_class_name
                    ),
                    "wrong_class_iou": round(
                        best_wrong_iou,
                        4,
                    ),
                })

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            processed % 50 == 0
            or processed == len(image_paths)
        ):

            print(
                f"Processed "
                f"{processed}/"
                f"{len(image_paths)} images"
            )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = (
        output_dir
        / "error_analysis.csv"
    )

    fieldnames = [
        "image",
        "class_id",
        "class_name",
        "failure_type",
        "same_class_iou",
        "wrong_class",
        "wrong_class_iou",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary)

    # ========================================================
    # Print summary
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    for class_id, class_name in TARGET_CLASSES.items():

        class_rows = [
            row
            for row in summary
            if row["class_id"] == class_id
        ]

        print()
        print(class_name)
        print("-" * 40)

        for failure_type in [
            "false_negative",
            "wrong_class",
            "localization",
        ]:

            count = sum(
                row["failure_type"]
                == failure_type
                for row in class_rows
            )

            print(
                f"{failure_type:<20}: "
                f"{count}"
            )

        print(
            "Visual examples saved: "
            f"{output_dir / f'{class_id}_{class_name}'}"
        )

    print()
    print(f"CSV report: {csv_path}")
    print("=" * 70)


# ============================================================
# Command-line interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "VisionClass targeted error analysis"
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Path to YOLO dataset data.yaml"
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Path to trained YOLO .pt model"
        ),
    )

    parser.add_argument(
        "--device",
        default="0",
        help=(
            "Inference device, e.g. 0 or cpu"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help=(
            "Inference image size"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "Prediction confidence threshold"
        ),
    )

    parser.add_argument(
        "--iou-match",
        type=float,
        default=0.50,
        help=(
            "IoU threshold for correct matching"
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "runs"
            / "detect"
            / "runs"
            / "error_analysis"
        ),
        help=(
            "Output directory"
        ),
    )

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou_match=args.iou_match,
        output_dir=args.output,
    )
