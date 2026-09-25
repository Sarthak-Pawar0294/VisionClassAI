"""Unit tests for geometry and behavior-person association (no YOLO)."""

from unittest.mock import MagicMock

import pytest

from src.association import (
    assign_behaviors_to_students,
    box_area,
    box_center,
    build_student_records,
    find_student_for_behavior,
    intersection_area,
    iou,
    point_inside_box,
)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_box_center():
    assert box_center([0, 0, 10, 20]) == (5.0, 10.0)


def test_point_inside_box():
    assert point_inside_box((5, 5), [0, 0, 10, 10]) is True
    assert point_inside_box((15, 5), [0, 0, 10, 10]) is False


def test_point_on_box_boundary():
    assert point_inside_box((0, 0), [0, 0, 10, 10]) is True
    assert point_inside_box((10, 10), [0, 0, 10, 10]) is True


def test_intersection_area_no_overlap():
    assert intersection_area([0, 0, 5, 5], [10, 10, 20, 20]) == 0.0


def test_intersection_area_partial():
    # [0,0,10,10] ∩ [5,5,15,15] = [5,5,10,10] = 25
    assert intersection_area([0, 0, 10, 10], [5, 5, 15, 15]) == 25.0


def test_intersection_area_full_containment():
    # inner box [2,2,5,5] fully inside outer [0,0,10,10] → area of inner = 9
    assert intersection_area([2, 2, 5, 5], [0, 0, 10, 10]) == 9.0


def test_intersection_area_touching_edge():
    # Boxes touch at one edge but do not overlap
    assert intersection_area([0, 0, 5, 5], [5, 0, 10, 5]) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# find_student_for_behavior — overlap-based candidate selection
# ──────────────────────────────────────────────────────────────────────────────

def test_find_student_single_candidate():
    behavior = [40, 40, 60, 60]
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0}]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 1


def test_find_student_no_candidate_no_overlap():
    """Behavior box far outside every person box → no match."""
    behavior = [200, 200, 220, 220]
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0}]
    assert find_student_for_behavior(behavior, students) is None


def test_find_student_no_candidate_touching_only():
    """Boxes that merely touch at one edge (zero-area intersection) → no match."""
    # behavior touches person box at right edge (x2=100 = person x1=100)
    behavior = [50, 50, 100, 80]
    students = [{"id": 1, "box": [100, 50, 200, 80], "behavior": None, "confidence": 0.0}]
    assert find_student_for_behavior(behavior, students) is None


def test_find_student_handrise_box_above_person_box():
    """
    Regression test for the handrise geometry fix.

    Scenario: a student is seated; their body bounding box starts at y=100.
    The behaviour model detects a raised arm in a box mostly above the head:
    behavior = [150, 20, 250, 120]  (center y = 70, which is ABOVE the person box)

    Under the old center-point rule:  center=(200, 70), person box y1=100 → FAILS
    Under the new overlap rule:       boxes overlap in the region [150,100,250,120] → PASSES
    """
    # Raised-arm behavior box: mostly above the person, but bottom edge overlaps the shoulder
    behavior = [150, 20, 250, 120]
    students = [
        # Seated person: body from y=100 to y=350
        {"id": 5, "box": [120, 100, 280, 350], "behavior": None, "confidence": 0.0}
    ]

    # Verify the center is indeed outside the person box (old rule would fail)
    cx, cy = box_center(behavior)
    x1, y1, x2, y2 = 120, 100, 280, 350
    assert not (x1 <= cx <= x2 and y1 <= cy <= y2), \
        "Precondition: center must be OUTSIDE person box for this test to be meaningful"

    # But there IS pixel overlap (bottom 20px of behavior box)
    assert intersection_area(behavior, [120, 100, 280, 350]) > 0, \
        "Precondition: overlap must exist"

    match = find_student_for_behavior(behavior, students)
    assert match is not None, \
        "Handrise box should be associated despite center being above the person box"
    assert match["id"] == 5


def test_find_student_prefers_higher_iou_over_list_order():
    """When two candidates overlap, the one with higher IoU wins — not list position."""
    behavior = [45, 45, 55, 55]
    students = [
        {"id": 1, "box": [0, 0, 100, 100],  "behavior": None, "confidence": 0.0},
        {"id": 2, "box": [40, 40, 60, 60],  "behavior": None, "confidence": 0.0},
    ]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 2
    assert iou(behavior, students[1]["box"]) > iou(behavior, students[0]["box"])


def test_find_student_tie_break_lower_id_when_iou_and_area_equal():
    """Equal IoU AND equal area → lower tracker ID wins (deterministic)."""
    behavior = [0, 0, 6, 6]
    p7_box = [0, 0, 9, 6]   # area=54, IoU=2/3
    p3_box = [0, 0, 6, 9]   # area=54, IoU=2/3

    assert abs(iou(behavior, p7_box) - iou(behavior, p3_box)) < 1e-9
    assert abs(box_area(p7_box) - box_area(p3_box)) < 1e-9

    students = [
        {"id": 7, "box": p7_box, "behavior": None, "confidence": 0.0},
        {"id": 3, "box": p3_box, "behavior": None, "confidence": 0.0},
    ]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 3, f"Expected id=3 (lower), got id={match['id']}"


def test_find_student_tie_break_note_area_path_untested():
    """
    Document the known gap: equal-IoU / different-area branch of the sort key
    is not exercised.  See the Phase 2 analysis report for the geometric proof
    of why this case is hard to construct with integer-coordinate rectangles.
    """
    pass  # living documentation — always passes


# ──────────────────────────────────────────────────────────────────────────────
# assign_behaviors_to_students
# ──────────────────────────────────────────────────────────────────────────────

def test_assign_behaviors_keeps_highest_confidence_per_student():
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0}]
    names = {0: "write", 1: "sleep"}
    assign_behaviors_to_students(
        students,
        [[10, 10, 30, 30], [12, 12, 28, 28]],
        [0, 1],
        [0.4, 0.9],
        names,
    )
    assert students[0]["behavior"] == "sleep"
    assert students[0]["confidence"] == 0.9


def test_assign_behaviors_does_not_overwrite_on_equal_confidence():
    """Strictly-greater-than rule: equal confidence must NOT overwrite."""
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": "write", "confidence": 0.5}]
    names = {0: "sleep"}
    assign_behaviors_to_students(
        students,
        [[10, 10, 30, 30]],
        [0],
        [0.5],
        names,
    )
    assert students[0]["behavior"] == "write"
    assert students[0]["confidence"] == 0.5


def test_assign_behaviors_skips_unmatched():
    students = [{"id": 1, "box": [0, 0, 10, 10], "behavior": "unknown", "confidence": 0.0}]
    names = {0: "write"}
    assign_behaviors_to_students(
        students,
        [[500, 500, 520, 520]],  # far outside person box
        [0],
        [0.99],
        names,
    )
    assert students[0]["behavior"] == "unknown"
    assert students[0]["confidence"] == 0.0


def test_assign_behaviors_handrise_box_overlapping_edge():
    """
    Behavior box that only partially overlaps the person box (handrise case)
    should still be successfully assigned.
    """
    # Behavior box mostly above person box, but bottom edge overlaps top of person
    behavior_box = [10, 0, 90, 30]   # center y=15, person starts at y=20
    students = [
        {"id": 1, "box": [0, 20, 100, 200], "behavior": "unknown", "confidence": 0.0}
    ]
    names = {0: "handrise"}
    assign_behaviors_to_students(
        students,
        [behavior_box],
        [0],
        [0.55],
        names,
    )
    assert students[0]["behavior"] == "handrise"
    assert students[0]["confidence"] == pytest.approx(0.55)


# ──────────────────────────────────────────────────────────────────────────────
# build_student_records
# ──────────────────────────────────────────────────────────────────────────────

def _make_person_result(boxes_xyxy: list, ids: list):
    """Build a minimal mock of a YOLO tracking result."""
    import torch

    mock_boxes = MagicMock()
    mock_boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    mock_boxes.id = torch.tensor(ids, dtype=torch.float32)

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    return mock_result


def test_build_student_records_returns_correct_structure():
    result = _make_person_result(
        boxes_xyxy=[[10.0, 20.0, 50.0, 80.0], [60.0, 5.0, 100.0, 90.0]],
        ids=[1, 2],
    )
    records = build_student_records(result, default_behavior="unknown")
    assert len(records) == 2

    first = records[0]
    assert first["id"] == 1
    assert first["box"] == [10.0, 20.0, 50.0, 80.0]
    assert first["behavior"] == "unknown"
    assert first["confidence"] == 0.0


def test_build_student_records_empty_when_no_ids():
    """If tracker has not assigned IDs yet (id tensor is None), return empty list."""
    mock_result = MagicMock()
    mock_result.boxes.id = None
    records = build_student_records(mock_result)
    assert records == []
