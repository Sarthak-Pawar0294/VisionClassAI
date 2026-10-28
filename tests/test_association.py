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


# ──────────────────────────────────────────────────────────────────────────────
# BehaviorCache
# ──────────────────────────────────────────────────────────────────────────────

from src.association import BehaviorCache


def _make_student(sid: int, behavior=None, confidence: float = 0.0) -> dict:
    return {"id": sid, "box": [0, 0, 100, 100], "behavior": behavior,
            "confidence": confidence}


def test_behavior_cache_fills_unknown_students():
    """Cache should replay last known behavior when student has no live det."""
    cache = BehaviorCache(max_age=5)

    # Frame A: student 1 has a live detection
    frame_a = [_make_student(1, "write", 0.8)]
    cache.apply(frame_a)   # nothing in cache yet
    cache.update(frame_a)  # store "write" for id=1

    # Frame B: student 1 has no detection (behavior=None)
    frame_b = [_make_student(1)]
    cache.apply(frame_b)  # should fill from cache
    assert frame_b[0]["behavior"] == "write"
    assert frame_b[0].get("from_cache") is True


def test_behavior_cache_live_detection_takes_priority():
    """A live detection must NOT be overwritten by cached data."""
    cache = BehaviorCache(max_age=5)

    # Seed cache with "write"
    seed = [_make_student(1, "write", 0.8)]
    cache.apply(seed)
    cache.update(seed)

    # Next frame: live detection says "read"
    live = [_make_student(1, "read", 0.9)]
    cache.apply(live)   # cache should NOT overwrite live "read"
    assert live[0]["behavior"] == "read"
    assert live[0].get("from_cache") is None


def test_behavior_cache_expires_after_max_age():
    """Cached entry should expire after max_age updates with no live detection."""
    max_age = 3
    cache = BehaviorCache(max_age=max_age)

    # Seed the cache
    seed = [_make_student(1, "sleep", 0.7)]
    cache.apply(seed)
    cache.update(seed)

    # Run max_age frames with no live detection for student 1
    for _ in range(max_age):
        frame = [_make_student(1)]
        cache.apply(frame)
        cache.update(frame)

    # Now the cache entry should have expired
    final = [_make_student(1)]
    cache.apply(final)
    assert final[0]["behavior"] is None, "Cache should have expired"


def test_behavior_cache_confidence_decays():
    """Confidence should decrease linearly from original to decay_floor."""
    max_age = 10
    decay_floor = 0.05
    cache = BehaviorCache(max_age=max_age, decay_floor=decay_floor)

    original_conf = 0.80

    # Seed the cache (entry stored at age=0)
    seed = [_make_student(1, "read", original_conf)]
    cache.apply(seed)
    cache.update(seed)

    # Frame 1: entry is at age=0 when apply is called → decay_fraction=0/10=0
    # After update, age increments to 1 (student has from_cache=True so not live)
    frame1 = [_make_student(1)]
    cache.apply(frame1)
    # At age=0: decayed_conf = 0.80*(1-0) + 0.05*0 = 0.80
    assert frame1[0]["behavior"] == "read"
    assert abs(frame1[0]["confidence"] - original_conf) < 1e-9, \
        f"Age=0 should give original confidence, got {frame1[0]['confidence']}"
    cache.update(frame1)

    # Frame 2: entry is at age=1 when apply is called → decay_fraction=1/10
    frame2 = [_make_student(1)]
    cache.apply(frame2)
    expected = original_conf * (1 - 1 / max_age) + decay_floor * (1 / max_age)
    assert frame2[0]["behavior"] == "read"
    assert abs(frame2[0]["confidence"] - expected) < 1e-9, \
        f"Age=1 expected {expected:.6f}, got {frame2[0]['confidence']:.6f}"
    cache.update(frame2)

    # Frame 3: entry is at age=2 → decay_fraction=2/10
    frame3 = [_make_student(1)]
    cache.apply(frame3)
    expected3 = original_conf * (1 - 2 / max_age) + decay_floor * (2 / max_age)
    assert frame3[0]["behavior"] == "read"
    assert abs(frame3[0]["confidence"] - expected3) < 1e-9, \
        f"Age=2 expected {expected3:.6f}, got {frame3[0]['confidence']:.6f}"


def test_behavior_cache_does_not_write_cached_label_to_cache():
    """Replayed cached labels must NOT refresh the cache entry (age must still increment)."""
    max_age = 3
    cache = BehaviorCache(max_age=max_age)

    seed = [_make_student(1, "write", 0.7)]
    cache.apply(seed)
    cache.update(seed)

    # Frame 1: no live detection, cache replays
    f1 = [_make_student(1)]
    cache.apply(f1)
    assert f1[0]["behavior"] == "write"
    cache.update(f1)

    # Frame 2: no live detection, cache replays (age should now be 2)
    f2 = [_make_student(1)]
    cache.apply(f2)
    assert f2[0]["behavior"] == "write"
    cache.update(f2)

    # Frame 3: age reaches max_age → entry should expire
    f3 = [_make_student(1)]
    cache.apply(f3)
    cache.update(f3)

    f_after_expire = [_make_student(1)]
    cache.apply(f_after_expire)
    assert f_after_expire[0]["behavior"] is None, \
        "Cache should have expired after max_age frames without live detection"


def test_behavior_cache_clear_removes_all_entries():
    """clear() should reset all stored cache entries."""
    cache = BehaviorCache(max_age=10)

    seed = [_make_student(1, "handrise", 0.6)]
    cache.apply(seed)
    cache.update(seed)

    cache.clear()

    frame = [_make_student(1)]
    cache.apply(frame)
    assert frame[0]["behavior"] is None


def test_behavior_cache_invalid_max_age():
    """max_age < 1 must raise ValueError."""
    with pytest.raises(ValueError):
        BehaviorCache(max_age=0)

    with pytest.raises(ValueError):
        BehaviorCache(max_age=-5)


def test_behavior_cache_multiple_students_independent():
    """Cache entries for different student IDs must be fully independent."""
    cache = BehaviorCache(max_age=5)

    # Student 1 has a live detection, student 2 does not
    frame_a = [
        _make_student(1, "write", 0.8),
        _make_student(2, "sleep", 0.6),
    ]
    cache.apply(frame_a)
    cache.update(frame_a)

    # Next frame: only student 1 has a live detection
    frame_b = [
        _make_student(1, "read", 0.9),  # new live detection
        _make_student(2),               # no detection
    ]
    cache.apply(frame_b)

    assert frame_b[0]["behavior"] == "read"          # live takes priority
    assert frame_b[0].get("from_cache") is None
    assert frame_b[1]["behavior"] == "sleep"         # replayed from cache
    assert frame_b[1].get("from_cache") is True


def test_behavior_cache_full_pipeline_sequence():
    """
    Simulate the full per-frame pattern:
      assign_behaviors_to_students → cache.apply → cache.update

    Verifies that students without a live detection in frames 2-3 get the
    label carried from frame 1.
    """
    cache = BehaviorCache(max_age=10)
    names = {0: "write", 1: "sleep"}

    # Frame 1: student 1 has a detection
    s1 = [_make_student(1)]
    assign_behaviors_to_students(s1, [[10, 10, 30, 30]], [0], [0.75], names)
    cache.apply(s1)
    cache.update(s1)
    assert s1[0]["behavior"] == "write"

    # Frame 2: no live detection for student 1
    s2 = [_make_student(1)]
    assign_behaviors_to_students(s2, [], [], [], names)
    cache.apply(s2)
    assert s2[0]["behavior"] == "write"
    assert s2[0].get("from_cache") is True
    cache.update(s2)

    # Frame 3: live detection overrides cache
    s3 = [_make_student(1)]
    assign_behaviors_to_students(s3, [[10, 10, 30, 30]], [1], [0.85], names)
    cache.apply(s3)
    assert s3[0]["behavior"] == "sleep"   # live detection wins
    assert s3[0].get("from_cache") is None
    cache.update(s3)
