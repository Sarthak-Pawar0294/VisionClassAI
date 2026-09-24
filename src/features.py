import cv2
import math
import numpy as np


# ============================================================
# FACIAL LANDMARK INDICES
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# BASIC GEOMETRY
# ============================================================

def distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(landmarks, eye_indices, width, height):
    """
    Calculate Eye Aspect Ratio (EAR).

    Higher EAR -> eye more open
    Lower EAR  -> eye more closed
    """

    points = []

    for index in eye_indices:
        x = landmarks[index].x * width
        y = landmarks[index].y * height
        points.append((x, y))

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])

    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_average_ear(landmarks, width, height):
    """Calculate average EAR of both eyes."""

    left_ear = calculate_ear(
        landmarks,
        LEFT_EYE,
        width,
        height
    )

    right_ear = calculate_ear(
        landmarks,
        RIGHT_EYE,
        width,
        height
    )

    average_ear = (left_ear + right_ear) / 2.0

    return average_ear


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(landmarks, width, height):
    """
    Estimate head orientation using facial landmarks.

    Returns:
        yaw
        pitch
        roll
    """

    # 2D image points
    image_points = np.array([
        (landmarks[1].x * width,
         landmarks[1].y * height),       # Nose

        (landmarks[152].x * width,
         landmarks[152].y * height),     # Chin

        (landmarks[33].x * width,
         landmarks[33].y * height),      # Left eye

        (landmarks[263].x * width,
         landmarks[263].y * height),     # Right eye

        (landmarks[61].x * width,
         landmarks[61].y * height),      # Left mouth

        (landmarks[291].x * width,
         landmarks[291].y * height)      # Right mouth
    ], dtype=np.float64)

    # Approximate 3D facial model
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye
        (43.3, 32.7, -26.0),      # Right eye
        (-28.9, -28.9, -24.1),    # Left mouth
        (28.9, -28.9, -24.1)      # Right mouth
    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = float(angles[0])
    yaw = float(angles[1])
    roll = float(angles[2])

    return yaw, pitch, roll


# ============================================================
# BODY POSTURE
# ============================================================

def calculate_body_features(pose_landmarks, width, height):
    """
    Calculate simple body posture features.

    Uses shoulders and hips.
    """

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    left_hip = pose_landmarks[23]
    right_hip = pose_landmarks[24]

    ls = np.array([
        left_shoulder.x * width,
        left_shoulder.y * height
    ])

    rs = np.array([
        right_shoulder.x * width,
        right_shoulder.y * height
    ])

    lh = np.array([
        left_hip.x * width,
        left_hip.y * height
    ])

    rh = np.array([
        right_hip.x * width,
        right_hip.y * height
    ])

    # Shoulder angle
    shoulder_angle = math.degrees(
        math.atan2(
            rs[1] - ls[1],
            rs[0] - ls[0]
        )
    )

    # Center of shoulders
    shoulder_center = (ls + rs) / 2

    # Center of hips
    hip_center = (lh + rh) / 2

    # Body lean
    body_lean = math.degrees(
        math.atan2(
            shoulder_center[0] - hip_center[0],
            hip_center[1] - shoulder_center[1]
        )
    )

    return float(shoulder_angle), float(body_lean)


# ============================================================
# NOSE MOVEMENT
# ============================================================

def calculate_movement(
    landmarks,
    previous_nose,
    width,
    height
):
    """
    Calculate frame-to-frame nose movement.

    Returns:
        movement
        current_nose
    """

    nose = np.array([
        landmarks[1].x * width,
        landmarks[1].y * height
    ])

    if previous_nose is None:
        return 0.0, nose

    movement = np.linalg.norm(nose - previous_nose)

    # Normalize approximately by frame diagonal
    diagonal = math.sqrt(width ** 2 + height ** 2)

    if diagonal > 0:
        movement = movement / diagonal

    return float(movement), nose


# ============================================================
# COMPLETE FEATURE EXTRACTION
# ============================================================

def extract_features(
    face_landmarks,
    pose_landmarks,
    width,
    height,
    previous_nose=None
):
    """
    Extract all available behavioral features.

    Returns:
        feature dictionary
        current nose position
    """

    features = {
        "face_detected": 0,
        "pose_detected": 0,

        "ear": 0.0,

        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,

        "shoulder_angle": 0.0,
        "body_lean": 0.0,

        "movement": 0.0
    }

    current_nose = previous_nose

    # --------------------------------------------------------
    # FACE
    # --------------------------------------------------------

    if face_landmarks is not None:

        landmarks = face_landmarks.landmark

        features["face_detected"] = 1

        features["ear"] = calculate_average_ear(
            landmarks,
            width,
            height
        )

        yaw, pitch, roll = calculate_head_pose(
            landmarks,
            width,
            height
        )

        features["yaw"] = yaw
        features["pitch"] = pitch
        features["roll"] = roll

        movement, current_nose = calculate_movement(
            landmarks,
            previous_nose,
            width,
            height
        )

        features["movement"] = movement

    # --------------------------------------------------------
    # BODY
    # --------------------------------------------------------

    if pose_landmarks is not None:

        landmarks = pose_landmarks.landmark

        features["pose_detected"] = 1

        shoulder_angle, body_lean = calculate_body_features(
            landmarks,
            width,
            height
        )

        features["shoulder_angle"] = shoulder_angle
        features["body_lean"] = body_lean

    return features, current_nose