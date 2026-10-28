"""Offline, non-destructive confidence-sweep diagnostics for VisionClass.

The confidence sweep uses one low-confidence Ultralytics prediction pass per
image set. Predictions are filtered after that pass, so every requested
threshold shares the same NMS result. A second pass at the configured live
confidence compares this diagnostic path with the legacy webcam call settings.
This tool never changes production configuration, model weights, or source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import yaml


EXPECTED_CLASS_NAMES = (
    "handrise",
    "read",
    "sleep",
    "stand",
    "using_electronic_devices",
    "write",
)
DEFAULT_THRESHOLDS = (0.001, 0.05, 0.10, 0.20, 0.30)
DEFAULT_IMAGE_LIMIT = 100
NMS_IOU_DEFAULT = 0.7
MAX_DETECTIONS_PER_IMAGE = 300
MATCH_IOU = 0.5
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_COLORS = (
    (40, 220, 255),
    (80, 220, 100),
    (255, 180, 40),
    (220, 80, 220),
    (40, 150, 255),
    (255, 100, 100),
)

PREDICTION_CSV_FIELDS = (
    "image_id",
    "threshold",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "normalized_area",
    "image_width",
    "image_height",
)
IMAGE_CSV_FIELDS = (
    "image_id",
    "status",
    "width",
    "height",
    "threshold",
    "detection_count",
    "classes_with_detections",
    "ground_truth_instances",
    "true_positives",
    "false_positives",
    "false_negatives",
    "inference_latency_ms",
    "message",
)
CLASS_CSV_FIELDS = (
    "threshold",
    "class_id",
    "class_name",
    "ground_truth_instances",
    "detections",
    "detections_lost_from_previous_threshold",
    "images_with_detections",
    "true_positives",
    "false_positives",
    "false_negatives",
    "precision",
    "recall",
    "f1",
    "confidence_count",
    "confidence_min",
    "confidence_median",
    "confidence_mean",
    "confidence_max",
    "normalized_area_median",
    "ground_truth_area_median",
)


@dataclass(frozen=True)
class ImageInput:
    path: Path
    image_id: str
    label_path: Path | None


def parse_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    """Validate and return a sorted, unique confidence-threshold sequence."""
    if not values:
        raise ValueError("At least one confidence threshold is required.")
    thresholds = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Confidence thresholds must be finite values from 0 through 1.")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Confidence thresholds must not contain duplicates.")
    return tuple(sorted(thresholds))


def normalize_class_names(names: Mapping[Any, Any] | Sequence[Any]) -> tuple[str, ...]:
    """Normalize Ultralytics/YAML class names to an ordered tuple."""
    if isinstance(names, Mapping):
        try:
            ordered = [names[index] if index in names else names[str(index)] for index in range(len(names))]
        except (KeyError, TypeError) as exc:
            raise ValueError("Class names must use contiguous IDs starting at zero.") from exc
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        ordered = list(names)
    else:
        raise ValueError("Class names must be a list or an ID-to-name mapping.")
    result = tuple(str(name) for name in ordered)
    if not result or any(not name for name in result):
        raise ValueError("Class names must be non-empty.")
    return result


def verify_class_names(model_names: Any, dataset_names: Any) -> tuple[str, ...]:
    model_order = normalize_class_names(model_names)
    dataset_order = normalize_class_names(dataset_names)
    expected = EXPECTED_CLASS_NAMES
    if model_order != dataset_order:
        raise ValueError(
            "Checkpoint and dataset class names/order differ: "
            f"checkpoint={model_order!r}, dataset={dataset_order!r}."
        )
    if model_order != expected:
        raise ValueError(
            "Checkpoint and dataset do not match the six VisionClass classes: "
            f"found {model_order!r}, expected {expected!r}."
        )
    return model_order


def normalized_box_area(box: Sequence[float], width: int, height: int) -> float:
    """Return the xyxy box area divided by the full image area."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)


def xywhn_to_xyxy(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    center_x, center_y, box_width, box_height = (float(value) for value in box)
    pixel_center_x = center_x * width
    pixel_center_y = center_y * height
    pixel_width = box_width * width
    pixel_height = box_height * height
    return (
        pixel_center_x - pixel_width / 2.0,
        pixel_center_y - pixel_height / 2.0,
        pixel_center_x + pixel_width / 2.0,
        pixel_center_y + pixel_height / 2.0,
    )


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def class_aware_one_to_one_match(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    iou_threshold: float = MATCH_IOU,
) -> tuple[set[int], set[int]]:
    """Match confidence-ranked predictions to same-class GT boxes once each."""
    matched_prediction_indices: set[int] = set()
    matched_ground_truth_indices: set[int] = set()
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: (-float(predictions[index]["confidence"]), index),
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        candidates = [
            (box_iou(prediction["box"], target["box"]), target_index)
            for target_index, target in enumerate(ground_truth)
            if target_index not in matched_ground_truth_indices
            and int(target["class_id"]) == int(prediction["class_id"])
        ]
        if not candidates:
            continue
        best_iou, best_target_index = max(candidates, key=lambda item: (item[0], -item[1]))
        if best_iou >= iou_threshold:
            matched_prediction_indices.add(prediction_index)
            matched_ground_truth_indices.add(best_target_index)
    return matched_prediction_indices, matched_ground_truth_indices


def safe_image_stem(image_id: str) -> str:
    normalized = image_id.replace("\\", "/")
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("._") or "image"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:100]}-{digest}"


def create_unique_output_directory(
    explicit_path: str | Path | None,
    default_parent: str | Path,
) -> Path:
    """Create a fresh output root atomically; never reuse or overwrite one."""
    if explicit_path is not None:
        destination = Path(explicit_path).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=False)
        return destination

    parent = Path(default_parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = parent / f"{stamp}-{uuid.uuid4().hex[:8]}"
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return destination
    raise FileExistsError(f"Could not reserve a unique diagnostics directory under {parent}.")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_dataset_yaml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    try:
        with yaml_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse dataset YAML {yaml_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {yaml_path}")
    if "names" not in config:
        raise ValueError(f"Dataset YAML has no 'names' class map: {yaml_path}")
    normalize_class_names(config["names"])
    if int(config.get("nc", len(normalize_class_names(config["names"])))) != len(
        normalize_class_names(config["names"])
    ):
        raise ValueError("Dataset YAML 'nc' does not match the number of class names.")
    return yaml_path, config


def _dataset_validation_roots(yaml_path: Path, config: Mapping[str, Any]) -> list[Path]:
    if "val" not in config or config["val"] in (None, ""):
        raise ValueError("Dataset YAML has no validation split ('val'); refusing to select another split.")
    configured_root = config.get("path", yaml_path.parent)
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        root = yaml_path.parent / root
    root = root.resolve()
    raw_val = config["val"]
    values = raw_val if isinstance(raw_val, list) else [raw_val]
    roots: list[Path] = []
    for value in values:
        val_path = Path(str(value)).expanduser()
        if not val_path.is_absolute():
            val_path = root / val_path
        val_path = val_path.resolve()
        if not val_path.is_dir():
            raise FileNotFoundError(
                f"Validation image directory is missing or not a directory: {val_path}. "
                "This diagnostic currently expects validation image directories."
            )
        roots.append(val_path)
    return roots


def _label_root_for_image_dir(image_dir: Path) -> Path | None:
    if image_dir.name.lower() != "images":
        return None
    candidate = image_dir.parent / "labels"
    return candidate if candidate.is_dir() else None


def _collect_image_inputs(
    source: str | Path | None,
    yaml_path: Path,
    dataset_config: Mapping[str, Any],
) -> tuple[list[ImageInput], list[Path], str]:
    if source is None:
        image_dirs = _dataset_validation_roots(yaml_path, dataset_config)
        source_description = "dataset validation split"
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Image source does not exist: {source_path}")
        if source_path.is_file():
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image extension: {source_path.suffix or '(none)'}")
            image_dirs = [source_path.parent]
            source_description = "single image"
        elif source_path.is_dir():
            image_dirs = [source_path]
            source_description = "user-supplied image directory"
        else:
            raise ValueError(f"Image source is neither a file nor directory: {source_path}")

    inputs: list[ImageInput] = []
    source_roots: list[Path] = []
    single_file = source is not None and Path(source).expanduser().resolve().is_file()
    for directory_index, image_dir in enumerate(image_dirs):
        source_roots.append(image_dir.resolve())
        label_root = _label_root_for_image_dir(image_dir)
        image_paths = (
            [Path(source).expanduser().resolve()]
            if single_file
            else sorted(
                (path.resolve() for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: path.as_posix().casefold(),
            )
        )
        for image_path in image_paths:
            try:
                relative = image_path.relative_to(image_dir.resolve())
            except ValueError:
                relative = Path(image_path.name)
            image_id = relative.as_posix()
            if len(image_dirs) > 1:
                image_id = f"{directory_index + 1}/{image_id}"
            label_path = None
            if label_root is not None:
                label_path = (label_root / relative).with_suffix(".txt").resolve()
            inputs.append(ImageInput(image_path, image_id, label_path))

    if not inputs:
        raise ValueError("No supported image files were found in the selected source.")
    identifiers = [image.image_id for image in inputs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Image identifiers are not unique; use one source directory at a time.")
    return inputs, source_roots, source_description


def _select_images(images: Sequence[ImageInput], limit: int, seed: int) -> tuple[list[ImageInput], bool]:
    if limit < 0:
        raise ValueError("Image limit must be 0 (all images) or a positive integer.")
    if limit == 0 or len(images) <= limit:
        return list(images), False
    selected = random.Random(seed).sample(list(images), limit)
    selected.sort(key=lambda image: image.image_id.casefold())
    return selected, True


def _parse_label_file(path: Path, class_count: int) -> list[dict[str, Any]]:
    labels = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read label file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} must contain class_id and four normalized values.")
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains a non-numeric label value.") from exc
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"{path}:{line_number} has out-of-range class ID {class_id}.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number} contains a non-finite box value.")
        center_x, center_y, box_width, box_height = values
        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            raise ValueError(f"{path}:{line_number} has a center outside normalized image bounds.")
        if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
            raise ValueError(f"{path}:{line_number} has a non-positive or out-of-range box size.")
        labels.append({"class_id": class_id, "normalized_box": tuple(values)})
    return labels


def _load_ground_truth(
    images: Sequence[ImageInput],
    class_count: int,
) -> tuple[bool, dict[str, list[dict[str, Any]]], str | None]:
    if any(image.label_path is None for image in images):
        return False, {}, "No associated YOLO labels were found; only predictions can be reported."
    missing = [image.label_path for image in images if image.label_path and not image.label_path.is_file()]
    if missing:
        return False, {}, f"{len(missing)} selected image(s) do not have an associated label file."
    labels_by_id: dict[str, list[dict[str, Any]]] = {}
    try:
        for image in images:
            assert image.label_path is not None
            labels_by_id[image.image_id] = _parse_label_file(image.label_path, class_count)
    except ValueError as exc:
        return False, {}, f"Invalid YOLO labels; ground-truth metrics are disabled: {exc}"
    return True, labels_by_id, None


def _tensor_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _extract_predictions(result: Any, class_names: Sequence[str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tensor_list(boxes.xyxy)
    class_ids = _tensor_list(boxes.cls)
    confidences = _tensor_list(boxes.conf)
    if not (len(xyxy) == len(class_ids) == len(confidences)):
        raise ValueError("Model result box, class, and confidence arrays have different lengths.")
    predictions = []
    for box, raw_class_id, raw_confidence in zip(xyxy, class_ids, confidences):
        class_id = int(raw_class_id)
        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"Checkpoint returned out-of-range class ID {class_id}.")
        confidence = float(raw_confidence)
        if not math.isfinite(confidence):
            raise ValueError("Checkpoint returned a non-finite confidence value.")
        coordinates = tuple(float(value) for value in box)
        if len(coordinates) != 4:
            raise ValueError("Checkpoint returned a bounding box that is not xyxy.")
        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": confidence,
                "box": coordinates,
            }
        )
    return predictions


def _readable_images(images: Sequence[ImageInput]) -> tuple[list[ImageInput], dict[str, dict[str, int]], list[dict[str, str]]]:
    readable = []
    dimensions: dict[str, dict[str, int]] = {}
    errors: list[dict[str, str]] = []
    for image in images:
        frame = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
        if frame is None:
            errors.append({"image_id": image.image_id, "status": "unreadable", "message": "OpenCV could not decode this image."})
            continue
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            errors.append({"image_id": image.image_id, "status": "unreadable", "message": "Decoded image has invalid dimensions."})
            continue
        readable.append(image)
        dimensions[image.image_id] = {"width": int(width), "height": int(height)}
    return readable, dimensions, errors


def _result_path(result: Any) -> Path:
    path = getattr(result, "path", None)
    if path is None:
        raise ValueError("Inference result has no source path.")
    return Path(str(path)).expanduser().resolve()


def _predict_set(
    model: Any,
    images: Sequence[ImageInput],
    *,
    conf: float,
    imgsz: int,
    iou: float,
    device: int | str,
    class_names: Sequence[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float | None]]:
    if not images:
        return {}, {}
    image_by_path = {image.path.resolve(): image for image in images}
    predictions_by_id: dict[str, list[dict[str, Any]]] = {}
    inference_ms_by_id: dict[str, float | None] = {}
    results = model.predict(
        source=[str(image.path) for image in images],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=MAX_DETECTIONS_PER_IMAGE,
        batch=1,
        device=device,
        save=False,
        show=False,
        verbose=False,
        stream=True,
    )
    for result in results:
        result_path = _result_path(result)
        image = image_by_path.get(result_path)
        if image is None:
            raise ValueError(f"Inference returned an unexpected source path: {result_path}")
        if image.image_id in predictions_by_id:
            raise ValueError(f"Inference returned the same image more than once: {image.image_id}")
        predictions_by_id[image.image_id] = _extract_predictions(result, class_names)
        speed = getattr(result, "speed", {}) or {}
        raw_latency = speed.get("inference") if isinstance(speed, Mapping) else None
        try:
            inference_ms_by_id[image.image_id] = float(raw_latency) if raw_latency is not None else None
        except (TypeError, ValueError):
            inference_ms_by_id[image.image_id] = None
    return predictions_by_id, inference_ms_by_id


def _clip_ground_truth(
    normalized_labels: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    return [
        {
            "class_id": int(label["class_id"]),
            "box": xywhn_to_xyxy(label["normalized_box"], width, height),
        }
        for label in normalized_labels
    ]


def _confidence_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _median_or_none(values: Sequence[float]) -> float | None:
    return _confidence_stats(values)["median"]


def _draw_annotation(
    original: Any,
    image_id: str,
    threshold: float,
    predictions: Sequence[Mapping[str, Any]],
) -> Any:
    annotated = original.copy()
    height, width = annotated.shape[:2]
    title = f"{image_id} | threshold={threshold:g}"
    title = title[: max(24, min(110, width // 7))]
    cv2.rectangle(annotated, (0, 0), (width, min(30, height - 1)), (24, 24, 24), -1)
    cv2.putText(annotated, title, (6, min(21, height - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    for prediction in predictions:
        class_id = int(prediction["class_id"])
        color = CLASS_COLORS[class_id]
        x1, y1, x2, y2 = (int(round(value)) for value in prediction["box"])
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{prediction['class_name']} {float(prediction['confidence']):.3f}"
        baseline_y = max(10, y1 - 5)
        cv2.putText(annotated, label, (x1, baseline_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return annotated


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _create_annotation(
    output_dir: Path,
    image: ImageInput,
    threshold: float,
    original: Any,
    predictions: Sequence[Mapping[str, Any]],
) -> Path:
    threshold_key = f"{threshold:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    destination = output_dir / "annotated" / f"conf_{threshold_key}" / f"{safe_image_stem(image.image_id)}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite annotation: {destination}")
    annotated = _draw_annotation(original, image.image_id, threshold, predictions)
    if not cv2.imwrite(str(destination), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Could not write annotated image: {destination}")
    return destination


def _safe_f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else None


def _class_metrics_for_threshold(
    threshold: float,
    class_names: Sequence[str],
    images: Sequence[ImageInput],
    predictions_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    dimensions: Mapping[str, Mapping[str, int]],
    labels_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth_available: bool,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    class_data = {
        class_id: {
            "detections": 0,
            "images_with_detections": 0,
            "confidence_values": [],
            "area_values": [],
            "ground_truth_instances": 0,
            "ground_truth_areas": [],
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
        for class_id in range(len(class_names))
    }
    zero_detection_images = 0
    image_rows = []
    for image in images:
        size = dimensions[image.image_id]
        width, height = size["width"], size["height"]
        retained = [
            dict(prediction)
            for prediction in predictions_by_id.get(image.image_id, ())
            if float(prediction["confidence"]) >= threshold
        ]
        if not retained:
            zero_detection_images += 1
        for prediction in retained:
            prediction["normalized_area"] = normalized_box_area(prediction["box"], width, height)
            class_entry = class_data[int(prediction["class_id"])]
            class_entry["detections"] += 1
            class_entry["confidence_values"].append(float(prediction["confidence"]))
            class_entry["area_values"].append(float(prediction["normalized_area"]))
        for class_id in class_data:
            if any(int(prediction["class_id"]) == class_id for prediction in retained):
                class_data[class_id]["images_with_detections"] += 1

        ground_truth = []
        if ground_truth_available:
            ground_truth = _clip_ground_truth(labels_by_id.get(image.image_id, ()), width, height)
            for label in ground_truth:
                class_data[int(label["class_id"])]["ground_truth_instances"] += 1
                class_data[int(label["class_id"])]["ground_truth_areas"].append(
                    normalized_box_area(label["box"], width, height)
                )
            matched_predictions, matched_ground_truth = class_aware_one_to_one_match(
                retained, ground_truth, MATCH_IOU
            )
            for prediction_index, prediction in enumerate(retained):
                class_entry = class_data[int(prediction["class_id"])]
                if prediction_index in matched_predictions:
                    class_entry["true_positives"] += 1
                else:
                    class_entry["false_positives"] += 1
            for ground_truth_index, label in enumerate(ground_truth):
                if ground_truth_index not in matched_ground_truth:
                    class_data[int(label["class_id"])]["false_negatives"] += 1

        image_tp = sum(entry["true_positives"] for entry in class_data.values())
        image_fp = sum(entry["false_positives"] for entry in class_data.values())
        image_fn = sum(entry["false_negatives"] for entry in class_data.values())
        image_rows.append(
            {
                "image_id": image.image_id,
                "status": "processed",
                "width": width,
                "height": height,
                "threshold": threshold,
                "detection_count": len(retained),
                "classes_with_detections": sum(
                    any(int(prediction["class_id"]) == class_id for prediction in retained)
                    for class_id in class_data
                ),
                "ground_truth_instances": len(ground_truth) if ground_truth_available else "",
                "true_positives": "",
                "false_positives": "",
                "false_negatives": "",
                "inference_latency_ms": "",
                "message": "",
            }
        )

    rows = []
    for class_id, name in enumerate(class_names):
        entry = class_data[class_id]
        confidences = _confidence_stats(entry["confidence_values"])
        tp = entry["true_positives"] if ground_truth_available else None
        fp = entry["false_positives"] if ground_truth_available else None
        fn = entry["false_negatives"] if ground_truth_available else None
        precision = tp / (tp + fp) if ground_truth_available and tp + fp else None
        recall = tp / (tp + fn) if ground_truth_available and tp + fn else None
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "ground_truth_instances": entry["ground_truth_instances"] if ground_truth_available else None,
                "detections": entry["detections"],
                "images_with_detections": entry["images_with_detections"],
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": _safe_f1(tp, fp, fn) if ground_truth_available else None,
                "confidence": confidences,
                "normalized_area_median": _median_or_none(entry["area_values"]),
                "ground_truth_area_median": _median_or_none(entry["ground_truth_areas"]) if ground_truth_available else None,
            }
        )
    return rows, zero_detection_images, image_rows


def _compare_detection_sets(
    sweep: Sequence[Mapping[str, Any]],
    live: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.999,
    confidence_tolerance: float = 1e-5,
) -> dict[str, int]:
    unmatched_live = set(range(len(live)))
    matched_sweep: set[int] = set()
    for sweep_index, sweep_item in sorted(
        enumerate(sweep), key=lambda item: (-float(item[1]["confidence"]), item[0])
    ):
        candidates = []
        for live_index in unmatched_live:
            live_item = live[live_index]
            if int(sweep_item["class_id"]) != int(live_item["class_id"]):
                continue
            confidence_delta = abs(float(sweep_item["confidence"]) - float(live_item["confidence"]))
            overlap = box_iou(sweep_item["box"], live_item["box"])
            if confidence_delta <= confidence_tolerance and overlap >= iou_threshold:
                candidates.append((overlap, -live_index, live_index))
        if candidates:
            _, _, live_index = max(candidates)
            unmatched_live.remove(live_index)
            matched_sweep.add(sweep_index)
    return {
        "matched_detections": len(matched_sweep),
        "sweep_only_detections": len(sweep) - len(matched_sweep),
        "live_only_detections": len(unmatched_live),
    }


def _format_csv_float(value: Any) -> Any:
    return "" if value is None else value


def _write_outputs(
    output_dir: Path,
    *,
    class_names: Sequence[str],
    images: Sequence[ImageInput],
    all_inputs: Sequence[ImageInput],
    readable_images: Sequence[ImageInput],
    dimensions: Mapping[str, Mapping[str, int]],
    unreadable: Sequence[Mapping[str, str]],
    predictions_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    inference_ms_by_id: Mapping[str, float | None],
    labels_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth_available: bool,
    ground_truth_reason: str | None,
    thresholds: Sequence[float],
    model_path: Path,
    dataset_yaml: Path,
    source_description: str,
    source_count: int,
    sampled: bool,
    sample_seed: int,
    limit: int,
    imgsz: int,
    nms_iou: float,
    device: int | str,
    live_conf: float,
    live_imgsz: int,
    live_predictions_by_id: Mapping[str, Sequence[Mapping[str, Any]]] | None,
    live_error: str | None,
    total_runtime_seconds: float,
    run_started_utc: str,
) -> dict[str, Any]:
    prediction_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    class_csv_rows: list[dict[str, Any]] = []
    threshold_summaries = []
    previous_counts: dict[int, int] = {class_id: 0 for class_id in range(len(class_names))}
    readable_by_id = {image.image_id: image for image in readable_images}
    unreadable_by_id = {item["image_id"]: item for item in unreadable}

    for threshold in thresholds:
        class_rows, zero_count, threshold_image_rows = _class_metrics_for_threshold(
            threshold,
            class_names,
            readable_images,
            predictions_by_id,
            dimensions,
            labels_by_id,
            ground_truth_available,
        )
        current_counts = {int(row["class_id"]): int(row["detections"]) for row in class_rows}
        for row in class_rows:
            class_id = int(row["class_id"])
            row["detections_lost_from_previous_threshold"] = max(
                0, previous_counts[class_id] - current_counts[class_id]
            )
            row["previous_threshold"] = thresholds[thresholds.index(threshold) - 1] if thresholds.index(threshold) else None
        previous_counts = current_counts

        for image_row in threshold_image_rows:
            image = readable_by_id[image_row["image_id"]]
            image_row["inference_latency_ms"] = inference_ms_by_id.get(image.image_id)
            original = cv2.imread(str(image.path), cv2.IMREAD_COLOR)
            if original is None:
                image_row["status"] = "unreadable_during_annotation"
                image_row["message"] = "OpenCV could not reopen this image for annotation."
                image_row["detection_count"] = ""
                image_row["classes_with_detections"] = ""
                image_row["ground_truth_instances"] = ""
                image_row["inference_latency_ms"] = ""
                image_rows.append(image_row)
                continue
            retained = [
                dict(prediction)
                for prediction in predictions_by_id.get(image.image_id, ())
                if float(prediction["confidence"]) >= threshold
            ]
            annotation = _create_annotation(output_dir, image, threshold, original, retained)
            for prediction in retained:
                x1, y1, x2, y2 = (float(value) for value in prediction["box"])
                prediction_rows.append(
                    {
                        "image_id": image.image_id,
                        "threshold": threshold,
                        "class_id": int(prediction["class_id"]),
                        "class_name": str(prediction["class_name"]),
                        "confidence": float(prediction["confidence"]),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "normalized_area": normalized_box_area(prediction["box"], image_row["width"], image_row["height"]),
                        "image_width": image_row["width"],
                        "image_height": image_row["height"],
                    }
                )
            image_row["true_positives"] = ""
            image_row["false_positives"] = ""
            image_row["false_negatives"] = ""
            image_row["message"] = f"Annotated output: {annotation.relative_to(output_dir).as_posix()}"
            image_rows.append(image_row)

        for image in all_inputs:
            unreadable_row = unreadable_by_id.get(image.image_id)
            if unreadable_row:
                image_rows.append(
                    {
                        "image_id": image.image_id,
                        "status": "unreadable",
                        "width": "",
                        "height": "",
                        "threshold": threshold,
                        "detection_count": "",
                        "classes_with_detections": "",
                        "ground_truth_instances": "",
                        "true_positives": "",
                        "false_positives": "",
                        "false_negatives": "",
                        "inference_latency_ms": "",
                        "message": unreadable_row["message"],
                    }
                )

        for row in class_rows:
            confidence = row["confidence"]
            class_csv_rows.append(
                {
                    "threshold": threshold,
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "ground_truth_instances": _format_csv_float(row["ground_truth_instances"]),
                    "detections": row["detections"],
                    "detections_lost_from_previous_threshold": row["detections_lost_from_previous_threshold"],
                    "images_with_detections": row["images_with_detections"],
                    "true_positives": _format_csv_float(row["true_positives"]),
                    "false_positives": _format_csv_float(row["false_positives"]),
                    "false_negatives": _format_csv_float(row["false_negatives"]),
                    "precision": _format_csv_float(row["precision"]),
                    "recall": _format_csv_float(row["recall"]),
                    "f1": _format_csv_float(row["f1"]),
                    "confidence_count": confidence["count"],
                    "confidence_min": _format_csv_float(confidence["min"]),
                    "confidence_median": _format_csv_float(confidence["median"]),
                    "confidence_mean": _format_csv_float(confidence["mean"]),
                    "confidence_max": _format_csv_float(confidence["max"]),
                    "normalized_area_median": _format_csv_float(row["normalized_area_median"]),
                    "ground_truth_area_median": _format_csv_float(row["ground_truth_area_median"]),
                }
            )

        threshold_summaries.append(
            {
                "threshold": threshold,
                "processed_images_with_zero_detections": zero_count,
                "total_retained_detections": sum(current_counts.values()),
                "per_class": class_rows,
            }
        )

    live_comparison: dict[str, Any] = {
        "configured_live_confidence": live_conf,
        "configured_live_image_size": live_imgsz,
        "sweep_image_size": imgsz,
        "same_image_size_as_live_configuration": imgsz == live_imgsz,
        "nms_iou": nms_iou,
        "method": "Separate inference pass using the configured live confidence and the same NMS settings.",
        "available": live_predictions_by_id is not None,
    }
    if live_error:
        live_comparison["error"] = live_error
    elif live_predictions_by_id is not None:
        per_image_comparison = []
        totals = {"matched_detections": 0, "sweep_only_detections": 0, "live_only_detections": 0}
        changed_images = 0
        for image in readable_images:
            sweep_predictions = [
                prediction
                for prediction in predictions_by_id.get(image.image_id, ())
                if float(prediction["confidence"]) >= live_conf
            ]
            live_predictions = list(live_predictions_by_id.get(image.image_id, ()))
            comparison = _compare_detection_sets(sweep_predictions, live_predictions)
            for key in totals:
                totals[key] += comparison[key]
            changed = bool(comparison["sweep_only_detections"] or comparison["live_only_detections"])
            changed_images += int(changed)
            per_image_comparison.append({"image_id": image.image_id, **comparison})
        live_comparison.update(totals)
        live_comparison["images_with_differences"] = changed_images
        live_comparison["per_image"] = per_image_comparison

    summary = {
        "schema_version": 1,
        "tool": "VisionClass offline detection diagnostic",
        "run_started_utc": run_started_utc,
        "model": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "source": source_description,
        "images_available": source_count,
        "images_selected": len(all_inputs),
        "image_limit": limit,
        "sampled": sampled,
        "sample_seed": sample_seed if sampled else None,
        "images_processed": len(readable_images),
        "unreadable_images": len(unreadable),
        "inference_results_returned": len(predictions_by_id),
        "inference_results_missing": len(readable_images) - len(predictions_by_id),
        "ground_truth": {
            "available": ground_truth_available,
            "reason": ground_truth_reason,
            "matching_iou": MATCH_IOU,
            "method": (
                "Predictions are sorted by descending confidence (original order breaks ties); "
                "each prediction is matched to the highest-IoU unmatched ground-truth box of the same class at IoU >= 0.5. "
                "A ground-truth box and a prediction can each be matched once."
            ),
            "scope": "Metrics use only images decoded and returned by inference; they are threshold-specific fixed-IoU metrics, not Ultralytics mAP.",
        },
        "inference": {
            "device": str(device),
            "image_size": imgsz,
            "batch_size": 1,
            "confidence_floor": min(thresholds),
            "nms_iou": nms_iou,
            "max_detections_per_image": MAX_DETECTIONS_PER_IMAGE,
            "threshold_method": "One shared post-NMS candidate set per image at the lowest requested confidence; higher thresholds filter this set without recomputing NMS.",
            "timing_note": "Recorded per-image Ultralytics inference latency and total diagnostic wall time are offline measurements, not sustained webcam FPS.",
        },
        "thresholds": threshold_summaries,
        "live_pipeline_comparison": live_comparison,
        "runtime_seconds": total_runtime_seconds,
        "generated_artifacts": {
            "predictions_csv": "predictions.csv",
            "per_image_summary_csv": "per_image_summary.csv",
            "class_metrics_csv": "class_metrics.csv",
            "annotations_directory": "annotated/",
            "annotation_count": len(prediction_rows) and len(readable_images) * len(thresholds) or len(readable_images) * len(thresholds),
        },
        "warnings": [ground_truth_reason] if ground_truth_reason else [],
    }
    _write_csv(output_dir / "predictions.csv", PREDICTION_CSV_FIELDS, prediction_rows)
    _write_csv(output_dir / "per_image_summary.csv", IMAGE_CSV_FIELDS, image_rows)
    _write_csv(output_dir / "class_metrics.csv", CLASS_CSV_FIELDS, class_csv_rows)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return summary


def run_diagnostic(
    *,
    model_path: str | Path,
    dataset_yaml: str | Path,
    source: str | Path | None,
    output: str | Path | None,
    imgsz: int,
    thresholds: Sequence[float],
    nms_iou: float,
    device: int | str,
    image_limit: int,
    seed: int,
    live_confidence: float,
    live_imgsz: int,
    project_root: str | Path,
    model_loader: Callable[[Path], Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()
    threshold_values = parse_thresholds(thresholds)
    if imgsz <= 0:
        raise ValueError("Image size must be a positive integer.")
    if not math.isfinite(nms_iou) or not (0.0 < nms_iou <= 1.0):
        raise ValueError("NMS IoU must be a finite value greater than 0 and at most 1.")
    if not math.isfinite(live_confidence) or not (0.0 <= live_confidence <= 1.0):
        raise ValueError("Configured live confidence must be between 0 and 1.")
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Behavior checkpoint not found: {model_path}")
    yaml_path, dataset_config = _read_dataset_yaml(dataset_yaml)
    class_names = normalize_class_names(dataset_config["names"])
    images, source_roots, source_description = _collect_image_inputs(source, yaml_path, dataset_config)
    selected_images, sampled = _select_images(images, image_limit, seed)
    output_candidate = (
        Path(output).expanduser().resolve()
        if output is not None
        else (Path(project_root).resolve() / "data" / "diagnostics")
    )
    for source_root in source_roots:
        if _path_is_within(output_candidate, source_root.resolve()):
            raise ValueError(f"Output directory must be outside the input image source: {source_root}")

    ground_truth_available, labels_by_id, ground_truth_reason = _load_ground_truth(
        selected_images, len(class_names)
    )
    if not ground_truth_available:
        print(f"WARNING: {ground_truth_reason}", file=sys.stderr)

    if model_loader is None:
        from ultralytics import YOLO

        model_loader = lambda path: YOLO(str(path))
    model = model_loader(model_path)
    class_names = verify_class_names(model.names, dataset_config["names"])

    readable_images, dimensions, unreadable = _readable_images(selected_images)
    for item in unreadable:
        print(f"WARNING: {item['image_id']}: {item['message']}", file=sys.stderr)
    output_dir = create_unique_output_directory(output, Path(project_root) / "data" / "diagnostics")
    low_predictions, inference_ms_by_id = _predict_set(
        model,
        readable_images,
        conf=min(threshold_values),
        imgsz=imgsz,
        iou=nms_iou,
        device=device,
        class_names=class_names,
    )
    missing_results = [image for image in readable_images if image.image_id not in low_predictions]
    for image in missing_results:
        print(
            f"WARNING: {image.image_id}: the inference stream returned no result for this readable image.",
            file=sys.stderr,
        )
    inference_images = [image for image in readable_images if image.image_id in low_predictions]
    live_predictions: dict[str, list[dict[str, Any]]] | None = None
    live_error = None
    try:
        live_predictions, _ = _predict_set(
            model,
            inference_images,
            conf=live_confidence,
            imgsz=live_imgsz,
            iou=NMS_IOU_DEFAULT,
            device=device,
            class_names=class_names,
        )
    except Exception as exc:  # the main diagnostic remains useful if this reference pass fails
        live_predictions = None
        live_error = f"Configured-live reference pass failed: {type(exc).__name__}: {exc}"
        print(f"WARNING: {live_error}", file=sys.stderr)

    summary = _write_outputs(
        output_dir,
        class_names=class_names,
        images=inference_images,
        all_inputs=selected_images,
        readable_images=inference_images,
        dimensions=dimensions,
        unreadable=unreadable + [
            {"image_id": image.image_id, "status": "inference_missing", "message": "Inference returned no result for this readable image."}
            for image in missing_results
        ],
        predictions_by_id=low_predictions,
        inference_ms_by_id=inference_ms_by_id,
        labels_by_id=labels_by_id,
        ground_truth_available=ground_truth_available,
        ground_truth_reason=ground_truth_reason,
        thresholds=threshold_values,
        model_path=model_path,
        dataset_yaml=yaml_path,
        source_description=source_description,
        source_count=len(images),
        sampled=sampled,
        sample_seed=seed,
        limit=image_limit,
        imgsz=imgsz,
        nms_iou=nms_iou,
        device=device,
        live_conf=live_confidence,
        live_imgsz=live_imgsz,
        live_predictions_by_id=live_predictions,
        live_error=live_error,
        total_runtime_seconds=time.perf_counter() - started,
        run_started_utc=run_started_utc,
    )
    return output_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare VisionClass behavior detections across confidence thresholds on images. "
            "The default source is the dataset validation split; no webcam is opened."
        )
    )
    parser.add_argument("--model", help="Behavior checkpoint path; defaults to VisionClass configuration.")
    parser.add_argument("--source", help="One image or image directory. Omit to use the dataset validation split.")
    parser.add_argument("--output", help="New output directory. Existing directories are never reused.")
    parser.add_argument("--data", help="YOLO dataset YAML; defaults to VisionClass configuration.")
    parser.add_argument("--imgsz", type=int, help="Sweep inference image size; defaults to configured live image size.")
    parser.add_argument(
        "--conf",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        metavar="THRESHOLD",
        help="Confidence thresholds to compare (default: 0.001 0.05 0.10 0.20 0.30).",
    )
    parser.add_argument("--iou", type=float, default=NMS_IOU_DEFAULT, help="Fixed NMS IoU for every pass (default: 0.7).")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_IMAGE_LIMIT,
        help="Maximum images (default: 100); use 0 to process the full selected source.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sample seed when --limit is applied (default: 42).")
    parser.add_argument("--device", help="Ultralytics device: auto, cpu, or CUDA index; defaults to VisionClass configuration.")
    return parser


def _print_summary(output_dir: Path, summary: Mapping[str, Any]) -> None:
    print(f"Diagnostics saved to: {output_dir}")
    print(
        f"Images: selected={summary['images_selected']}, processed={summary['images_processed']}, "
        f"unreadable={summary['unreadable_images']}"
    )
    print(
        f"Device={summary['inference']['device']}, imgsz={summary['inference']['image_size']}, "
        f"NMS IoU={summary['inference']['nms_iou']}, batch=1, "
        f"total runtime={summary['runtime_seconds']:.2f}s"
    )
    print("Per-class fixed-IoU metrics by confidence threshold:")
    for threshold_data in summary["thresholds"]:
        threshold = threshold_data["threshold"]
        print(f"  conf={threshold:g}  detections={threshold_data['total_retained_detections']}")
        for row in threshold_data["per_class"]:
            if row["true_positives"] is None:
                print(f"    {row['class_name']}: detections={row['detections']} (ground truth unavailable)")
            else:
                print(
                    f"    {row['class_name']}: det={row['detections']} "
                    f"TP={row['true_positives']} FP={row['false_positives']} FN={row['false_negatives']} "
                    f"P={row['precision']} R={row['recall']} F1={row['f1']}"
                )
    comparison = summary["live_pipeline_comparison"]
    if comparison.get("available"):
        print(
            "Live-pipeline comparison: "
            f"matched={comparison['matched_detections']}, "
            f"sweep-only={comparison['sweep_only_detections']}, "
            f"live-only={comparison['live_only_detections']}"
        )
    elif comparison.get("error"):
        print(f"Live-pipeline comparison unavailable: {comparison['error']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from src.config import get_settings

        settings = get_settings(
            device_spec=args.device,
            dataset_yaml=args.data,
            behavior_model=args.model,
        )
        output_dir, summary = run_diagnostic(
            model_path=settings.behavior_model,
            dataset_yaml=settings.dataset_yaml,
            source=args.source,
            output=args.output,
            imgsz=args.imgsz if args.imgsz is not None else settings.inference_imgsz,
            thresholds=args.conf,
            nms_iou=args.iou,
            device=settings.device,
            image_limit=args.limit,
            seed=args.seed,
            live_confidence=settings.behavior_conf,
            live_imgsz=settings.inference_imgsz,
            project_root=settings.project_root,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_summary(output_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
