"""Unit tests for geometry and behavior-person association (no YOLO)."""

from src.association import (
    assign_behaviors_to_students,
    box_center,
    find_student_for_behavior,
    iou,
    point_inside_box,
)


def test_box_center():
    assert box_center([0, 0, 10, 20]) == (5.0, 10.0)


def test_point_inside_box():
    assert point_inside_box((5, 5), [0, 0, 10, 10]) is True
    assert point_inside_box((15, 5), [0, 0, 10, 10]) is False


def test_point_on_box_boundary():
    assert point_inside_box((0, 0), [0, 0, 10, 10]) is True
    assert point_inside_box((10, 10), [0, 0, 10, 10]) is True


def test_find_student_single_candidate():
    behavior = [40, 40, 60, 60]
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0}]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 1


def test_find_student_no_candidate():
    behavior = [200, 200, 220, 220]
    students = [{"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0}]
    assert find_student_for_behavior(behavior, students) is None


def test_find_student_prefers_higher_iou_over_list_order():
    # Behavior center at (50, 50) — inside both persons.
    behavior = [45, 45, 55, 55]
    students = [
        {"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0},
        {"id": 2, "box": [40, 40, 60, 60], "behavior": None, "confidence": 0.0},
    ]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 2
    assert iou(behavior, students[1]["box"]) > iou(behavior, students[0]["box"])


def test_find_student_tie_break_smaller_area():
    behavior = [45, 45, 55, 55]
    # Same IoU with behavior box for two identical-size nested boxes — pick smaller area.
    students = [
        {"id": 10, "box": [0, 0, 200, 200], "behavior": None, "confidence": 0.0},
        {"id": 5, "box": [30, 30, 70, 70], "behavior": None, "confidence": 0.0},
    ]
    match = find_student_for_behavior(behavior, students)
    assert match is not None
    assert match["id"] == 5


def test_assign_behaviors_keeps_highest_confidence_per_student():
    students = [
        {"id": 1, "box": [0, 0, 100, 100], "behavior": None, "confidence": 0.0},
    ]
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


def test_assign_behaviors_skips_unmatched():
    students = [
        {"id": 1, "box": [0, 0, 10, 10], "behavior": "unknown", "confidence": 0.0},
    ]
    names = {0: "write"}
    assign_behaviors_to_students(
        students,
        [[500, 500, 520, 520]],
        [0],
        [0.99],
        names,
    )
    assert students[0]["behavior"] == "unknown"
    assert students[0]["confidence"] == 0.0
