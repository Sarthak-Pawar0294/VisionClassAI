import cv2
import mediapipe as mp
import numpy as np
import math


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def eye_aspect_ratio(landmarks, eye_indices, width, height):

    points = []

    for idx in eye_indices:
        x = landmarks[idx].x * width
        y = landmarks[idx].y * height
        points.append(np.array([x, y]))

    # Eye points:
    # 0 = outer/inner corner
    # 1,2 = upper points
    # 3 = opposite corner
    # 4,5 = lower points

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])

    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)

    return ear


# MediaPipe Face Mesh eye landmark indices

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ============================================================
# HEAD POSE
# ============================================================

def calculate_head_pose(landmarks, width, height):

    # 2D facial landmarks
    image_points = np.array([
        (landmarks[1].x * width, landmarks[1].y * height),       # Nose
        (landmarks[152].x * width, landmarks[152].y * height),   # Chin
        (landmarks[33].x * width, landmarks[33].y * height),     # Left eye
        (landmarks[263].x * width, landmarks[263].y * height),   # Right eye
        (landmarks[61].x * width, landmarks[61].y * height),     # Left mouth
        (landmarks[291].x * width, landmarks[291].y * height)    # Right mouth
    ], dtype=np.float64)

    # Approximate 3D facial model
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose
        (0.0, -63.6, -12.5),      # Chin
        (-43.3, 32.7, -26.0),     # Left eye
        (43.3, 32.7, -26.0),      # Right eye
        (-28.9, -28.9, -24.1),    # Left mouth
        (28.9, -28.9, -24.1)      # Right mouth
    ])

    focal_length = width

    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll


# ============================================================
# BODY POSTURE
# ============================================================

def calculate_posture(pose_landmarks, width, height):

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

    shoulder_center = (ls + rs) / 2
    hip_center = (lh + rh) / 2

    body_lean = math.degrees(
        math.atan2(
            shoulder_center[0] - hip_center[0],
            hip_center[1] - shoulder_center[1]
        )
    )

    return shoulder_angle, body_lean


# ============================================================
# MAIN CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

previous_nose = None

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    pose_results = pose.process(rgb)


    # ========================================================
    # FACE FEATURES
    # ========================================================

    if face_results.multi_face_landmarks:

        face_landmarks = face_results.multi_face_landmarks[0].landmark

        left_ear = eye_aspect_ratio(
            face_landmarks,
            LEFT_EYE,
            width,
            height
        )

        right_ear = eye_aspect_ratio(
            face_landmarks,
            RIGHT_EYE,
            width,
            height
        )

        ear = (left_ear + right_ear) / 2

        pitch, yaw, roll = calculate_head_pose(
            face_landmarks,
            width,
            height
        )

        # Draw face landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            face_results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_CONTOURS
        )


        # ====================================================
        # DISPLAY FACE FEATURES
        # ====================================================

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Roll: {roll:.1f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # BODY FEATURES
    # ========================================================

    if pose_results.pose_landmarks:

        pose_landmarks = pose_results.pose_landmarks.landmark

        shoulder_angle, body_lean = calculate_posture(
            pose_landmarks,
            width,
            height
        )

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.putText(
            frame,
            f"Shoulder Angle: {shoulder_angle:.1f}",
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Body Lean: {body_lean:.1f}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "VisionClass - Behavioral Features",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "VisionClass - Feature Extraction",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

face_mesh.close()
pose.close()