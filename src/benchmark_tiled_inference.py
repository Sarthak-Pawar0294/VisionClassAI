from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = [
    "handrise",
    "read",
    "sleep",
    "stand",
    "using_electronic_devices",
    "write",
]


def load_yolo_labels(label_path, width, height):
    targets = []

    if not label_path.exists():
        return targets

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        targets.append({
            "cls": int(cls),
            "box": [x1, y1, x2, y2],
        })

    return targets


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def evaluate_predictions(predictions, targets, iou_threshold=0.5):
    """
    Simple detection evaluation.

    A target is considered detected when a prediction of the same
    class overlaps it by at least IoU 0.5.
    """

    matched_predictions = set()

    tp = {i: 0 for i in range(6)}
    fn = {i: 0 for i in range(6)}
    fp = {i: 0 for i in range(6)}

    for target in targets:
        best_idx = None
        best_iou = 0.0

        for idx, pred in enumerate(predictions):
            if idx in matched_predictions:
                continue

            if pred["cls"] != target["cls"]:
                continue

            iou = box_iou(target["box"], pred["box"])

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            tp[target["cls"]] += 1
            matched_predictions.add(best_idx)
        else:
            fn[target["cls"]] += 1

    for idx, pred in enumerate(predictions):
        if idx not in matched_predictions:
            fp[pred["cls"]] += 1

    return tp, fp, fn


def run_standard(model, image, imgsz, conf):
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        device=0,
        verbose=False,
    )

    predictions = []

    for result in results:
        if result.boxes is None:
            continue

        for box, cls, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            predictions.append({
                "cls": int(cls),
                "box": box.tolist(),
                "conf": float(score),
            })

    return predictions


def run_tiled(model, image, tile_size, overlap, conf):
    h, w = image.shape[:2]

    stride = int(tile_size * (1 - overlap))

    predictions = []

    y_positions = list(range(0, max(1, h - tile_size + 1), stride))
    x_positions = list(range(0, max(1, w - tile_size + 1), stride))

    if y_positions[-1] != h - tile_size:
        y_positions.append(max(0, h - tile_size))

    if x_positions[-1] != w - tile_size:
        x_positions.append(max(0, w - tile_size))

    for y in y_positions:
        for x in x_positions:

            tile = image[
                y:min(y + tile_size, h),
                x:min(x + tile_size, w)
            ]

            results = model.predict(
                source=tile,
                imgsz=tile_size,
                conf=conf,
                device=0,
                verbose=False,
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box, cls, score in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                ):
                    bx1, by1, bx2, by2 = box

                    predictions.append({
                        "cls": int(cls),
                        "box": [
                            bx1 + x,
                            by1 + y,
                            bx2 + x,
                            by2 + y,
                        ],
                        "conf": float(score),
                    })

    # NMS across all tiles
    if predictions:
        boxes = torch_boxes = np.array(
            [p["box"] for p in predictions],
            dtype=np.float32
        )

        scores = np.array(
            [p["conf"] for p in predictions],
            dtype=np.float32
        )

        classes = np.array(
            [p["cls"] for p in predictions],
            dtype=np.int32
        )

        keep = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=conf,
        nms_threshold=0.5,
        )

        if len(keep) > 0:
            keep = np.array(keep).reshape(-1)

            predictions = [
                predictions[int(i)]
                for i in keep
            ]

    return predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=320
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    dataset = Path(args.dataset)
    test_images = dataset / "test" / "images"
    test_labels = dataset / "test" / "labels"

    model = YOLO(args.model)

    print("=" * 70)
    print("VisionClass - Tiled Inference Benchmark")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Test set:   {test_images}")
    print(f"Tile size:  {args.tile_size}")
    print(f"Overlap:    {args.overlap}")
    print(f"Confidence: {args.conf}")
    print()

    total_tp = {i: 0 for i in range(6)}
    total_fp = {i: 0 for i in range(6)}
    total_fn = {i: 0 for i in range(6)}

    images = sorted(
        list(test_images.glob("*.jpg")) +
        list(test_images.glob("*.jpeg")) +
        list(test_images.glob("*.png"))
    )

    for count, image_path in enumerate(images, 1):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        label_path = test_labels / f"{image_path.stem}.txt"

        targets = load_yolo_labels(
            label_path,
            image.shape[1],
            image.shape[0]
        )

        predictions = run_tiled(
            model,
            image,
            args.tile_size,
            args.overlap,
            args.conf,
        )

        tp, fp, fn = evaluate_predictions(
            predictions,
            targets,
            iou_threshold=0.5
        )

        for cls in range(6):
            total_tp[cls] += tp[cls]
            total_fp[cls] += fp[cls]
            total_fn[cls] += fn[cls]

        if count % 50 == 0:
            print(f"Processed {count}/{len(images)} images")

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for cls in range(6):

        tp = total_tp[cls]
        fp = total_fp[cls]
        fn = total_fn[cls]

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        print(
            f"{CLASS_NAMES[cls]:28s} "
            f"TP={tp:4d} "
            f"FP={fp:4d} "
            f"FN={fn:4d} "
            f"P={precision:.4f} "
            f"R={recall:.4f}"
        )

    print()
    print("=" * 70)
    print("DEVICE RESULT")
    print("=" * 70)

    cls = 4

    tp = total_tp[cls]
    fp = total_fp[cls]
    fn = total_fn[cls]

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"TP:        {tp}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print()
    print("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()