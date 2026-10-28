# EXPERIMENTAL — MediaPipe branch (not connected to the YOLO pipeline).
#
# Standalone webcam demo that extracts facial and body posture features using
# Google MediaPipe (FaceMesh + Pose). Runs independently from the YOLO app and
# reuses the calculations in src/features.py.
#
# Status: this file pre-dates the src/ refactor and has not been integrated.
# The extracted features (EAR, head pose, body lean) are not consumed by any
# YOLO-based inference script.
#
# Kept as a standalone manual demo; it is not imported by the application.
import cv2
import mediapipe as mp
from src.features import extract_features


# ============================================================
# MAIN CAMERA
# ============================================================

def main() -> None:
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose = None
    cap = None
    try:
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(0)
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)
            face_landmarks = (
                face_results.multi_face_landmarks[0]
                if face_results.multi_face_landmarks
                else None
            )
            features, _ = extract_features(
                face_landmarks,
                pose_results.pose_landmarks,
                width,
                height,
            )


            # ========================================================
            # FACE FEATURES
            # ========================================================

            if face_results.multi_face_landmarks:

                ear = features["ear"]
                yaw = features["yaw"]
                pitch = features["pitch"]
                roll = features["roll"]

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

                shoulder_angle = features["shoulder_angle"]
                body_lean = features["body_lean"]

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


    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if pose is not None:
            pose.close()
        face_mesh.close()


if __name__ == "__main__":
    main()
