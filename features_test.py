# LEGACY DEMO — MediaPipe face + pose landmark visualisation.
#
# Standalone webcam script that draws FaceMesh and Pose landmarks on live video.
# NOT a pytest test file despite the name.  Not part of the src/ pipeline.
#
# Superseded by: feature_extraction.py (more complete feature extraction demo).
# Kept for reference.  Do not import from other modules.
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_drawing = mp.solutions.drawing_utils
pose_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection / landmarks
    face_results = face_mesh.process(rgb)

    # Pose detection
    pose_results = pose.process(rgb)

    # Draw face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            face_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        pose_drawing.draw_landmarks(
            frame,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("VisionClass - Face + Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
pose.close()