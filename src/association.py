"""
Geometry helpers and behavior-to-person association.

Association rule (single shared implementation)
-----------------------------------------------
For each behavior detection bounding box:

1. Compute the behavior box center.
2. **Candidates** are tracked persons whose box contains that center point.
3. If there are no candidates, the detection is not assigned to any person.
4. If there is one candidate, that person is selected.
5. If there are multiple candidates (overlapping person boxes), select the
   person with the **highest IoU** between the behavior box and the person box.
   This prefers the strongest spatial overlap rather than list order.
6. Tie-break: smaller person box area, then lower tracker id (deterministic).

After a person is chosen for a detection, the person's label is updated only
if the detection confidence is **greater than** the confidence already stored
for that person in the current frame (same as the original per-frame merge rule).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_inside_box(point: Sequence[float], box: Sequence[float]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter = intersection_area(box_a, box_b)
    if inter <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def find_student_for_behavior(
    behavior_box: Sequence[float],
    students: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """
    Return the student dict that best matches this behavior detection, or None.
    """
    center = box_center(behavior_box)
    candidates = [
        student
        for student in students
        if point_inside_box(center, student["box"])
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def sort_key(student: Mapping[str, Any]) -> tuple[float, float, int]:
        overlap = iou(behavior_box, student["box"])
        area = box_area(student["box"])
        sid = int(student["id"])
        return (overlap, -area, -sid)

    return max(candidates, key=sort_key)


def build_student_records(
    person_result: Any,
    *,
    default_behavior: str | None = None,
) -> list[dict[str, Any]]:
    """Build per-frame student dicts from a YOLO person tracking result."""
    students: list[dict[str, Any]] = []
    if person_result.boxes is None or person_result.boxes.id is None:
        return students

    person_boxes = person_result.boxes.xyxy.cpu().tolist()
    person_ids = person_result.boxes.id.int().cpu().tolist()

    for box, student_id in zip(person_boxes, person_ids):
        students.append(
            {
                "id": student_id,
                "box": box,
                "behavior": default_behavior,
                "confidence": 0.0,
            }
        )
    return students


def assign_behaviors_to_students(
    students: list[dict[str, Any]],
    behavior_boxes: Iterable[Sequence[float]],
    class_ids: Iterable[int],
    confidences: Iterable[float],
    class_names: Mapping[int, str],
) -> None:
    """
    Apply behavior detections to in-frame student records (mutates students).
    """
    for behavior_box, class_id, confidence in zip(
        behavior_boxes,
        class_ids,
        confidences,
    ):
        student = find_student_for_behavior(behavior_box, students)
        if student is None:
            continue
        if confidence > student["confidence"]:
            student["behavior"] = class_names[int(class_id)]
            student["confidence"] = float(confidence)
