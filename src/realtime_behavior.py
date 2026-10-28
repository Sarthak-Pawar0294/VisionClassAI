import sys
import time
import webbrowser
from collections import deque
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
from ultralytics import YOLO

from src.config import get_settings
from src.live_dashboard import BEHAVIOR_CLASSES, LiveDashboardServer, LiveDashboardState


def _predict_frame(model, settings, frame):
    """Run the shared behavior inference and return its annotation and boxes."""
    results = model.predict(
        source=frame,
        imgsz=settings.inference_imgsz,
        conf=settings.behavior_conf,
        device=settings.device,
        save=False,
        show=False,
        verbose=False,
    )
    result = results[0]
    annotated_frame = result.plot()

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            if class_name not in BEHAVIOR_CLASSES:
                continue
            detections.append({"class_name": class_name, "confidence": confidence})
    return annotated_frame, detections


def _open_webcam_capture(index: int):
    """Open the configured camera with a reliable Windows capture backend."""
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def webcam_session_worker(context) -> None:
    """Headless webcam worker used by the Flask session manager."""
    settings = get_settings()
    if not settings.behavior_model.is_file():
        raise RuntimeError(
            f"Behavior model not found: {settings.behavior_model}. "
            "Set VISIONCLASS_BEHAVIOR_MODEL to an existing checkpoint."
        )

    try:
        model = YOLO(str(settings.behavior_model))
    except Exception as exc:
        raise RuntimeError(
            f"Could not load behavior model {settings.behavior_model}: {exc}"
        ) from exc

    if context.stop_requested:
        return

    cap = None
    try:
        try:
            cap = _open_webcam_capture(settings.webcam_index)
        except Exception as exc:
            message = (
                f"Could not open webcam (index {settings.webcam_index}): {exc}. "
                "Check the camera connection, permissions, and VISIONCLASS_WEBCAM_INDEX."
            )
            context.report(camera_status="error", inference_status="error", error=message)
            raise RuntimeError(message) from exc
        context.register_cleanup(cap.release)
        try:
            camera_open = cap.isOpened()
        except Exception as exc:
            message = (
                f"Could not open webcam (index {settings.webcam_index}): {exc}. "
                "Check the camera connection, permissions, and VISIONCLASS_WEBCAM_INDEX."
            )
            context.report(camera_status="error", inference_status="error", error=message)
            raise RuntimeError(message) from exc
        if not camera_open:
            message = (
                f"Could not open webcam (index {settings.webcam_index}). "
                "Check the camera connection, permissions, and "
                "VISIONCLASS_WEBCAM_INDEX."
            )
            context.report(camera_status="error", inference_status="error", error=message)
            raise RuntimeError(message)

        context.report(camera_status="open", inference_status="starting", error=None)
        frame_durations: deque[float] = deque(maxlen=30)
        frame_number = 0
        while not context.stop_requested:
            frame_started = time.perf_counter()
            try:
                ret, frame = cap.read()
            except Exception as exc:
                message = f"Could not read a webcam frame: {exc}"
                context.report(camera_status="error", inference_status="error", error=message)
                raise RuntimeError(message) from exc
            if not ret:
                message = "Could not read a webcam frame. Check that the camera remains connected."
                context.report(camera_status="error", inference_status="error", error=message)
                raise RuntimeError(message)
            observed_at = time.time()

            try:
                annotated_frame, detections = _predict_frame(model, settings, frame)
            except Exception as exc:
                message = f"Inference failed: {exc}"
                context.report(inference_status="error", error=message)
                raise RuntimeError(message) from exc

            frame_number += 1
            frame_durations.append(time.perf_counter() - frame_started)
            processing_fps = len(frame_durations) / sum(frame_durations)
            class_counts = {name: 0 for name in BEHAVIOR_CLASSES}
            for detection in detections:
                class_counts[detection["class_name"]] += 1
            encoded, jpeg = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not encoded:
                message = "Could not encode the annotated webcam frame for the dashboard."
                context.report(inference_status="error", error=message)
                raise RuntimeError(message)
            context.report(
                camera_status="open",
                inference_status="running",
                frame_number=frame_number,
                observed_at=observed_at,
                processing_fps=processing_fps,
                total_detections=len(detections),
                class_counts=class_counts,
                detections=detections,
                stream_jpeg=jpeg.tobytes(),
                error=None,
            )
    except Exception:
        raise


def main(device_spec: str | None = None) -> int:
    settings = get_settings(device_spec=device_spec)

    if not settings.behavior_model.is_file():
        print(
            f"ERROR: Behavior model not found: {settings.behavior_model}. "
            "Set VISIONCLASS_BEHAVIOR_MODEL to an existing checkpoint.",
            file=sys.stderr,
        )
        return 1

    try:
        model = YOLO(str(settings.behavior_model))
    except Exception as exc:
        print(
            f"ERROR: Could not load behavior model {settings.behavior_model}: {exc}",
            file=sys.stderr,
        )
        return 1

    cap = None
    try:
        cap = cv2.VideoCapture(settings.webcam_index)
        camera_open = cap.isOpened()
    except Exception as exc:
        print(
            f"ERROR: Could not open webcam (index {settings.webcam_index}): {exc}",
            file=sys.stderr,
        )
        if cap is not None:
            try:
                cap.release()
            except Exception as cleanup_error:
                print(f"ERROR: Could not release webcam: {cleanup_error}", file=sys.stderr)
        return 1

    if not camera_open:
        print(
            f"ERROR: Could not open webcam (index {settings.webcam_index}). "
            "Check the camera connection, permissions, and VISIONCLASS_WEBCAM_INDEX.",
            file=sys.stderr,
        )
        try:
            cap.release()
        except Exception as exc:
            print(f"ERROR: Could not release webcam: {exc}", file=sys.stderr)
        return 1

    dashboard_state = LiveDashboardState()
    dashboard_state.set_status(camera_status="open", inference_status="starting")
    dashboard = None

    try:
        dashboard = LiveDashboardServer(dashboard_state)
        dashboard_url = dashboard.start()
    except Exception as exc:
        print(
            f"WARNING: Could not start the local live dashboard at "
            f"http://127.0.0.1:8765/: {exc}",
            file=sys.stderr,
        )
        print("Continuing webcam inference and preview without the dashboard.", file=sys.stderr)
        if dashboard is not None:
            try:
                dashboard.stop()
            except Exception as cleanup_error:
                print(f"ERROR: Could not clean up dashboard startup: {cleanup_error}", file=sys.stderr)
        dashboard_state.set_status(
            inference_status="error",
            error=f"Dashboard server unavailable: {exc}",
        )
        dashboard = None
        dashboard_state.close()
    else:
        try:
            if webbrowser.open(dashboard_url, new=0):
                print(f"Live dashboard: {dashboard_url}")
            else:
                print(f"WARNING: Could not open a browser automatically. Open {dashboard_url}")
        except Exception as exc:
            print(f"WARNING: Could not open a browser automatically: {exc}")
            print(f"Open the live dashboard manually: {dashboard_url}")

    print("VisionClass behavior detection")
    print(f"Device: {settings.device}")
    print(f"Confidence threshold: {settings.behavior_conf}")
    print("Press Q to quit the webcam session and close the dashboard.")

    frame_durations: deque[float] = deque(maxlen=30)
    frame_number = 0

    exit_code = 0
    try:
        while True:
            frame_started = time.perf_counter()
            try:
                ret, frame = cap.read()
            except Exception as exc:
                message = f"Could not read a webcam frame: {exc}"
                print(f"ERROR: {message}", file=sys.stderr)
                if dashboard is not None:
                    dashboard_state.set_status(
                        camera_status="error",
                        inference_status="error",
                        error=message,
                    )
                exit_code = 1
                break

            if not ret:
                message = "Could not read a webcam frame. Check that the camera remains connected."
                print(f"ERROR: {message}", file=sys.stderr)
                if dashboard is not None:
                    dashboard_state.set_status(
                        camera_status="error",
                        inference_status="error",
                        error=message,
                    )
                exit_code = 1
                break

            try:
                annotated_frame, detections = _predict_frame(model, settings, frame)

                frame_number += 1

                if detections:
                    print("\n--- Predictions ---")
                    for detection in detections:
                        print(
                            f"{detection['class_name']:30s} "
                            f"confidence = {detection['confidence']:.3f}"
                        )
                else:
                    print(f"No prediction above {settings.behavior_conf}")

                cv2.imshow(
                    "VisionClass - Behavior Detection",
                    annotated_frame,
                )

                quit_requested = cv2.waitKey(1) & 0xFF == ord("q")

                jpeg_bytes = None
                if dashboard is not None:
                    encoded, jpeg = cv2.imencode(
                        ".jpg",
                        annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 80],
                    )
                    if not encoded:
                        raise RuntimeError("Could not encode the annotated webcam frame as JPEG.")
                    jpeg_bytes = jpeg.tobytes()

                frame_durations.append(time.perf_counter() - frame_started)
                processing_fps = len(frame_durations) / sum(frame_durations)
                if dashboard is not None and jpeg_bytes is not None:
                    dashboard_state.publish(
                        jpeg_bytes,
                        detections,
                        processing_fps,
                        frame_number,
                    )

                if quit_requested or (
                    dashboard is not None and dashboard_state.stop_requested
                ):
                    break

            except Exception as exc:
                message = f"Inference or dashboard frame update failed: {exc}"
                print(f"ERROR: {message}", file=sys.stderr)
                if dashboard is not None:
                    dashboard_state.set_status(
                        inference_status="error",
                        error=message,
                    )
                exit_code = 1
                break

    finally:
        for cleanup_name, cleanup in (
            ("webcam", cap.release),
            ("OpenCV windows", cv2.destroyAllWindows),
            ("dashboard", dashboard.stop if dashboard is not None else None),
        ):
            if cleanup is None:
                continue
            try:
                cleanup()
            except Exception as exc:
                print(f"ERROR: Could not clean up {cleanup_name}: {exc}", file=sys.stderr)
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time webcam behavior detection")
    parser.add_argument(
        "--device",
        default=None,
        help='Device: auto, cpu, or GPU index (default: auto / VISIONCLASS_DEVICE)',
    )
    args = parser.parse_args()
    raise SystemExit(main(device_spec=args.device))
