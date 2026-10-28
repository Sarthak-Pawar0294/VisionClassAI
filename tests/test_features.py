from types import SimpleNamespace

import pytest

from src.features import calculate_ear, extract_features


def test_eye_aspect_ratio_uses_shared_feature_helper():
    points = [(0, 0), (1, 2), (3, 2), (4, 0), (3, -2), (1, -2)]
    landmarks = [SimpleNamespace(x=x / 4, y=y / 4) for x, y in points]

    assert calculate_ear(landmarks, range(6), width=4, height=4) == pytest.approx(1.0)


def test_missing_landmarks_produce_explicit_empty_feature_values():
    features, current_nose = extract_features(None, None, width=640, height=480)

    assert current_nose is None
    assert features == {
        "face_detected": 0,
        "pose_detected": 0,
        "ear": 0.0,
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "shoulder_angle": 0.0,
        "body_lean": 0.0,
        "movement": 0.0,
    }
