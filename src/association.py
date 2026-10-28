"""
Geometry helpers and behavior-to-person association.

Association rule (single shared implementation)
-----------------------------------------------
For each behavior detection bounding box:

1. **Candidates** are tracked persons whose bounding box has any pixel-level
   overlap with the behavior box (``intersection_area > 0``).

   *Why overlap instead of center-point containment?*

   The previous rule required the behavior-box center to lie inside the person
   box.  This silently discards detections for behaviors whose bounding box
   extends outside the person box — most critically:

   - ``handrise``: the raised-arm box extends **above** the student's head, so
     its center is above the person box and containment returns False.
   - Any behavior annotated on a body part at the frame edge or occluded area.

   Using intersection area (any overlap > 0) captures these cases while the
   IoU-based sorting step still selects the *best* matching person.

2. If there are no candidates, the detection is not assigned to any person.
3. If there is one candidate, that person is selected.
4. If there are multiple candidates (overlapping person boxes), select the
   person with the **highest IoU** between the behavior box and the person box.
   This prefers the strongest spatial overlap rather than list order.
5. Tie-break: smaller person box area, then lower tracker id (deterministic).

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

    A student is a candidate if their person bounding box overlaps with the
    behavior box by any non-zero area.  The best candidate is chosen by maximum
    IoU; ties broken by smaller person-box area, then lower tracker ID.
    """
    candidates = [
        student
        for student in students
        if intersection_area(behavior_box, student["box"]) > 0
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


class BehaviorCache:
    """
    Per-student temporal behavior smoother.

    The behavior model fires a detection on only a fraction of tracked
    students each frame (typically 3–6 detections for 10–15 tracked
    persons).  Without smoothing, students with no detection in the
    current frame display "unknown" even if they had a clear label
    one or two frames ago.

    BehaviorCache keeps the **most-recent** behavior label and
    confidence for each tracker ID and replays it into the per-frame
    student records for up to ``max_age`` frames after the last live
    detection.  Confidence is linearly decayed toward ``decay_floor``
    as the cached entry ages, signalling that the label is stale.

    Usage (inside the per-frame loop)::

        cache = BehaviorCache(max_age=30)   # once, before the loop

        # ... per frame:
        students = build_student_records(person_result)
        assign_behaviors_to_students(students, ...)   # live detections
        cache.apply(students)                         # fill gaps from cache
        cache.update(students)                        # store new live hits

    Design constraints
    ------------------
    * ``max_age`` is in **frames**, not seconds.  At 25 fps, 30 frames
      ≈ 1.2 seconds — long enough to bridge typical inter-detection
      gaps without masking real behavior changes.
    * Only entries that received a **live** detection this frame are
      written back to the cache (``update`` step).  Stale entries
      continue to age and expire on their own.
    * The cache is unbounded in the number of student IDs it stores,
      but this is safe because the number of unique tracker IDs in a
      single classroom video is small (tens to low hundreds).
    """

    # Sentinel confidence written when replaying a cached label.
    # Kept above zero so the label is visible but below typical live
    # detection confidence values (0.2–0.9).
    _CACHED_CONF: float = 0.10

    def __init__(self, max_age: int = 30, decay_floor: float = 0.05) -> None:
        """
        Parameters
        ----------
        max_age:
            Number of frames a cached entry is replayed before expiring.
            Must be a positive integer.
        decay_floor:
            Minimum confidence value produced by the linear decay.
            The confidence of a cached entry decays from the original
            detection confidence toward this value over ``max_age`` frames.
        """
        if max_age < 1:
            raise ValueError(f"max_age must be >= 1, got {max_age!r}")
        self._max_age    = max_age
        self._decay_floor = float(decay_floor)
        # {student_id: {"behavior": str, "confidence": float, "age": int}}
        self._entries: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, students: list[dict[str, Any]]) -> None:
        """
        Fill missing behaviors from the cache (mutates student records).

        Students that already have a live behavior (``behavior is not None``
        and ``confidence > 0``) are **not** overwritten.
        """
        for student in students:
            if student["behavior"] is not None:
                continue  # live detection takes priority

            sid = int(student["id"])
            entry = self._entries.get(sid)
            if entry is None or entry["age"] >= self._max_age:
                continue

            # Linear decay: confidence drops from original → decay_floor
            # over max_age frames.
            age_fraction = entry["age"] / self._max_age
            decayed_conf = (
                entry["confidence"] * (1.0 - age_fraction)
                + self._decay_floor * age_fraction
            )

            student["behavior"]   = entry["behavior"]
            student["confidence"] = decayed_conf
            student["from_cache"] = True  # optional diagnostic key

    def update(self, students: list[dict[str, Any]]) -> None:
        """
        Write live detections back to the cache and age existing entries.

        Must be called **after** ``apply`` and after the frame's student
        records have been fully populated.
        """
        live_ids: set[int] = set()

        for student in students:
            if student["behavior"] is None:
                continue
            if student.get("from_cache"):
                continue  # don't update the cache from its own replayed label

            sid = int(student["id"])
            live_ids.add(sid)
            self._entries[sid] = {
                "behavior":   student["behavior"],
                "confidence": student["confidence"],
                "age":        0,
            }

        # Age non-live entries; entries from this frame stay at age=0
        for sid, entry in list(self._entries.items()):
            if sid not in live_ids:
                entry["age"] += 1
                if entry["age"] >= self._max_age:
                    del self._entries[sid]

    def clear(self) -> None:
        """Reset the cache (e.g. when re-opening a video)."""
        self._entries.clear()
