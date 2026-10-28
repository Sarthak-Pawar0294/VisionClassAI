import math
from pathlib import Path

import cv2

DATASET_ROOT = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "student_classroom_behavior"
)
TEST_IMAGES = DATASET_ROOT / "test" / "images"
TEST_LABELS = DATASET_ROOT / "test" / "labels"

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "detect"
    / "runs"
    / "phase5_error_analysis"
    / "device_false_negative_contact_sheet.jpg"
)

IMAGE_NAMES = [
    "frame_0001_jpg.rf.a94b5c2f54578c57dc29573d8d3e1ec3.jpg",
    "frame_0012_jpg.rf.178dddf15a953a27e17c06d243667299.jpg",
    "frame_0019_jpg.rf.045c52efc8f8c1c3d84276afaeb38268.jpg",
    "frame_0023_jpg.rf.d08108dd498efed247932857f5ba7de6.jpg",
    "frame_0034_jpg.rf.ac4c91bdc097fd61ef6b0d9a3900a825.jpg",
    "frame_0040_jpg.rf.1043b6003d208bac05cd47ef17a4a322.jpg",
    "frame_0040_jpg.rf.332554331ece9bfcc6a303e072c3300c.jpg",
    "frame_0060_jpg.rf.52da6d5429e3fa57779ac46f43553194.jpg",
    "frame_0062_jpg.rf.364ef0192a77c9c53e8f222facec8979.jpg",
    "frame_0086_jpg.rf.f80774b9281c1ba5d56e1593585518f3.jpg",
    "frame_0111_jpg.rf.aa453d4534641013c7265e88777c1aef.jpg",
    "frame_0116_jpg.rf.0350c7af4831208603b65560986faf95.jpg",
    "frame_0125_jpg.rf.3bf7c3042981ebbf4df833816b64f572.jpg",
    "frame_0141_jpg.rf.1e728c80a9a1efc4638cadc9c35a0370.jpg",
    "frame_0142_jpg.rf.5303b43fc7b773cc95f694854ceb537d.jpg",
    "frame_0143_jpg.rf.c83d0c844d59bee837861a3cf64a0814.jpg",
    "frame_0145_jpg.rf.60999d80fad9f21c367118b324e4af29.jpg",
    "frame_0151_jpg.rf.9e81a32dc4093b065514645be9ecd622.jpg",
    "frame_0160_jpg.rf.2a000a2780e149b965f61f1c60a52f7e.jpg",
    "frame_0165_jpg.rf.62500dc8c75aa868d4d157f0eee77066.jpg",
]

COLS = 4
CELL_W = 500
CELL_H = 400


def draw_device_boxes(image, label_path):
    h, w = image.shape[:2]

    if not label_path.exists():
        return image

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])

            # Only class 4 = using_electronic_devices
            if cls_id != 4:
                continue

            x, y, bw, bh = map(float, parts[1:])

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                image,
                "GT: device",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return image


def main():
    rows = math.ceil(len(IMAGE_NAMES) / COLS)

    sheet = 255 * (
        __import__("numpy").ones(
            (rows * CELL_H, COLS * CELL_W, 3),
            dtype="uint8",
        )
    )

    for i, name in enumerate(IMAGE_NAMES):
        image_path = TEST_IMAGES / name
        label_path = TEST_LABELS / (Path(name).stem + ".txt")

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = draw_device_boxes(image, label_path)

        image = cv2.resize(
            image,
            (CELL_W, CELL_H - 35),
        )

        # Add filename/index area
        cell = 255 * (
            __import__("numpy").ones(
                (CELL_H, CELL_W, 3),
                dtype="uint8",
            )
        )

        cell[0:CELL_H - 35, 0:CELL_W] = image

        cv2.putText(
            cell,
            f"{i + 1}: {name[:42]}",
            (8, CELL_H - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        row = i // COLS
        col = i % COLS

        y1 = row * CELL_H
        x1 = col * CELL_W

        sheet[
            y1:y1 + CELL_H,
            x1:x1 + CELL_W
        ] = cell

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(OUTPUT),
        sheet,
    )

    print(f"Saved contact sheet:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
