"""Deterministic lifecycle tests for the single-active-job manager."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import src.realtime_behavior as realtime_behavior
import src.session_manager as session_manager_module
from src.session_manager import (
    ActiveJobConflict,
    JobStatus,
    SessionManager,
    WorkerStartError,
    transition_status,
)


def test_job_state_transitions_accept_valid_and_reject_invalid_edges():
    assert transition_status(JobStatus.STARTING, JobStatus.RUNNING) is JobStatus.RUNNING
    assert transition_status(JobStatus.RUNNING, JobStatus.STOP_REQUESTED) is JobStatus.STOP_REQUESTED
    assert transition_status(JobStatus.STOP_REQUESTED, JobStatus.CANCELLED) is JobStatus.CANCELLED
    with pytest.raises(ValueError, match="Invalid job state transition"):
        transition_status(JobStatus.IDLE, JobStatus.RUNNING)
    with pytest.raises(ValueError, match="Invalid job state transition"):
        transition_status(JobStatus.COMPLETED, JobStatus.RUNNING)


def test_concurrent_starts_claim_one_slot_and_invoke_one_worker():
    gate = threading.Barrier(3)
    worker_started = threading.Event()
    release_worker = threading.Event()
    calls = []
    manager = None

    def worker(context):
        calls.append(context)
        worker_started.set()
        release_worker.wait(3)

    manager = SessionManager({"webcam": worker})
    results = []

    def start_together():
        gate.wait(timeout=2)
        try:
            results.append(("accepted", manager.start("webcam")))
        except ActiveJobConflict as exc:
            results.append(("conflict", exc.snapshot))

    threads = [threading.Thread(target=start_together) for _ in range(2)]
    for thread in threads:
        thread.start()
    gate.wait(timeout=2)
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert worker_started.wait(timeout=2)
    assert [kind for kind, _ in results].count("accepted") == 1
    assert [kind for kind, _ in results].count("conflict") == 1
    assert len(calls) == 1
    _, stop_snapshot = manager.request_stop()
    assert stop_snapshot["status"] == "stop_requested"
    release_worker.set()
    finished, final = manager.wait(stop_snapshot["job_id"], timeout=2)
    assert finished
    assert final["status"] == "cancelled"


def test_stop_cleanup_terminal_order_and_restart():
    cleanup_observations = []
    entered = threading.Event()
    manager = None

    def worker(context):
        context.register_cleanup(
            lambda: cleanup_observations.append(manager.snapshot()["status"])
        )
        entered.set()
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    initial = manager.start()
    assert entered.wait(timeout=2)
    requested, stopping = manager.request_stop()
    assert requested
    assert stopping["status"] == "stop_requested"
    assert stopping["stop_requested"] is True

    finished, terminal = manager.wait(initial["job_id"], timeout=2)
    assert finished
    assert cleanup_observations == ["stop_requested"]
    assert terminal["status"] == "cancelled"
    assert terminal["worker_alive"] is False

    second = manager.start()
    requested, _ = manager.request_stop()
    assert requested
    finished, second_terminal = manager.wait(second["job_id"], timeout=2)
    assert finished
    assert second_terminal["status"] == "cancelled"


def test_worker_exception_is_reported_and_cleanup_runs_once(capsys):
    cleanup = []

    def worker(context):
        context.register_cleanup(lambda: cleanup.append("released"))
        raise RuntimeError("primary processing failure")

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished
    assert final["status"] == "error"
    assert final["error"] == "primary processing failure"
    assert cleanup == ["released"]
    assert "primary processing failure" in capsys.readouterr().err


def test_cleanup_failure_is_reported_without_hiding_primary_worker_error():
    def worker(context):
        def fail_cleanup():
            raise OSError("release failed")

        context.register_cleanup(fail_cleanup)
        raise RuntimeError("inference failed")

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished
    assert final["status"] == "error"
    assert final["error"] == "inference failed"
    assert final["cleanup_errors"] == ["release failed"]


def test_cleanup_failure_without_worker_error_sets_error_state():
    def worker(context):
        context.register_cleanup(lambda: (_ for _ in ()).throw(OSError("close failed")))

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished
    assert final["status"] == "error"
    assert final["error"] == "Cleanup failed: close failed"
    assert final["cleanup_errors"] == ["close failed"]


def test_worker_start_failure_does_not_invoke_worker(monkeypatch):
    calls = []

    class BrokenThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread unavailable")

        def is_alive(self):
            return False

    monkeypatch.setattr("src.session_manager.threading.Thread", BrokenThread)
    manager = SessionManager({"webcam": lambda context: calls.append(context)})

    with pytest.raises(WorkerStartError, match="thread unavailable") as error:
        manager.start()

    assert calls == []
    assert error.value.snapshot["status"] == "error"
    assert error.value.snapshot["worker_alive"] is False


def test_five_cooperative_lifecycle_cycles_are_bounded_and_release_resources():
    released = []
    cycle_numbers = iter(range(5))

    def worker(context):
        cycle = next(cycle_numbers)
        context.register_cleanup(lambda: released.append(cycle))
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    for _ in range(5):
        started = manager.start()
        requested, snapshot = manager.request_stop()
        assert requested
        assert snapshot["status"] == "stop_requested"
        finished, terminal = manager.wait(started["job_id"], timeout=2)
        assert finished
        assert terminal["status"] == "cancelled"
        assert terminal["worker_alive"] is False

    assert len(released) == 5
    assert sorted(released) == list(range(5))


def test_camera_mode_uses_directshow_on_windows_and_default_elsewhere(monkeypatch):
    capture = Mock()
    video_capture = Mock(return_value=capture)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", video_capture)

    monkeypatch.setattr(realtime_behavior.sys, "platform", "win32")
    assert realtime_behavior._open_webcam_capture(3) is capture
    video_capture.assert_called_once_with(3, realtime_behavior.cv2.CAP_DSHOW)

    video_capture.reset_mock()
    monkeypatch.setattr(realtime_behavior.sys, "platform", "linux")
    assert realtime_behavior._open_webcam_capture(3) is capture
    video_capture.assert_called_once_with(3)


def test_live_history_samples_timestamped_frames_once_per_interval(monkeypatch):
    wall_clock = [100.0]
    monotonic_clock = [0.0]
    fake_time = SimpleNamespace(
        time=lambda: wall_clock[0],
        monotonic=lambda: monotonic_clock[0],
    )
    monkeypatch.setattr(session_manager_module, "time", fake_time)

    def worker(context):
        for frame_number, wall, monotonic, count in (
            (1, 100.0, 0.0, 1),
            (2, 100.5, 0.5, 3),
            (3, 101.1, 1.1, 2),
        ):
            wall_clock[0] = wall
            monotonic_clock[0] = monotonic
            context.report(
                frame_number=frame_number,
                observed_at=wall,
                total_detections=count,
                class_counts={"handrise": count},
                processing_fps=12.0,
            )

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    finished, terminal = manager.wait(started["job_id"], timeout=2)
    assert finished and terminal["status"] == "completed"

    history = manager.live_history(1, now=102.0)
    assert [point["frame_number"] for point in history["points"]] == [1, 3]
    assert [point["observed_at"] for point in history["points"]] == [100.0, 101.1]
    assert history["points"][0]["class_counts"] == {"handrise": 1}
    assert history["latest_observation_at"] == 101.1
    assert history["metric_semantics"].endswith("not unique people.")
    assert manager.snapshot()["live_history_sample_count"] == 2


def test_zero_detection_webcam_frames_update_live_history_and_status(monkeypatch):
    monkeypatch.setattr(session_manager_module, "LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS", 0.0)
    reported = threading.Event()
    continue_worker = threading.Event()
    classes = ("handrise", "read", "sleep", "stand", "using_electronic_devices", "write")
    zero_counts = {name: 0 for name in classes}

    def worker(context):
        context.report(
            camera_status="open",
            inference_status="running",
            frame_number=1,
            observed_at=100.0,
            total_detections=0,
            class_counts=zero_counts,
            detections=[],
        )
        context.report(
            camera_status="open",
            inference_status="running",
            frame_number=2,
            observed_at=101.0,
            total_detections=0,
            class_counts=zero_counts,
            detections=[],
        )
        reported.set()
        continue_worker.wait(2)

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    try:
        assert reported.wait(2)
        snapshot = manager.snapshot()
        history = manager.live_history(1, now=102.0)

        assert snapshot["status"] == "running"
        assert snapshot["camera_status"] == "open"
        assert snapshot["inference_status"] == "running"
        assert snapshot["frame_number"] == 2
        assert snapshot["total_detections"] == 0
        assert snapshot["session_detection_total"] == 0
        assert snapshot["class_counts"] == zero_counts
        assert [point["frame_number"] for point in history["points"]] == [1, 2]
        assert [point["observed_at"] for point in history["points"]] == [100.0, 101.0]
        assert all(point["total_detections"] == 0 for point in history["points"])
        assert all(point["class_counts"] == zero_counts for point in history["points"])
    finally:
        continue_worker.set()
        manager.request_stop()
        assert manager.wait(started["job_id"], timeout=2)[0]


def test_live_history_is_bounded_to_six_hundred_samples(monkeypatch):
    wall_clock = [1000.0]
    monotonic_clock = [0.0]
    fake_time = SimpleNamespace(
        time=lambda: wall_clock[0],
        monotonic=lambda: monotonic_clock[0],
    )
    monkeypatch.setattr(session_manager_module, "time", fake_time)
    monkeypatch.setattr(session_manager_module, "LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS", 0.0)

    def worker(context):
        for frame_number in range(1, 651):
            observed_at = 1000.0 + frame_number
            wall_clock[0] = observed_at
            monotonic_clock[0] = float(frame_number)
            context.report(
                frame_number=frame_number,
                observed_at=observed_at,
                class_counts={"read": frame_number % 4},
                total_detections=frame_number % 4,
            )

    manager = SessionManager({"webcam": worker})
    started = manager.start()
    finished, _terminal = manager.wait(started["job_id"], timeout=2)
    assert finished
    history = manager.live_history(10, now=1650.0)
    assert len(history["points"]) == 600
    assert history["points"][0]["frame_number"] == 51
    assert history["points"][-1]["frame_number"] == 650


def test_headless_webcam_worker_publishes_frame_and_releases_camera(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    settings = SimpleNamespace(
        behavior_model=Path(__file__).resolve(),
        device="cpu",
        webcam_index=0,
        behavior_conf=0.2,
        inference_imgsz=960,
    )
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})

    class FakeBox:
        cls = [0]
        conf = [0.91]

    class FakeResult:
        boxes = [FakeBox()]

        def plot(self):
            return frame

    class FakeModel:
        names = {0: "handrise"}

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            manager.request_stop()
            return [FakeResult()]

    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index, backend=None: capture)
    monkeypatch.setattr(
        realtime_behavior.cv2,
        "imshow",
        lambda *args: pytest.fail("headless worker must not open a preview window"),
    )
    started = manager.start()
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished
    assert final["status"] == "cancelled"
    assert final["camera_status"] == "closed"
    assert final["inference_status"] == "stopped"
    assert final["frame_number"] == 1
    assert final["total_detections"] == 1
    assert final["class_counts"]["handrise"] == 1
    capture.release.assert_called_once()


def test_headless_webcam_worker_publishes_genuine_zero_detection_frame(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    classes = ("handrise", "read", "sleep", "stand", "using_electronic_devices", "write")
    settings = SimpleNamespace(
        behavior_model=Path(__file__).resolve(),
        device="cpu",
        webcam_index=0,
        behavior_conf=0.2,
        inference_imgsz=960,
    )
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)
    manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})

    class EmptyResult:
        boxes = None

        def plot(self):
            return frame

    class FakeModel:
        names = dict(enumerate(classes))

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            manager.request_stop()
            return [EmptyResult()]

    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index, backend=None: capture)
    started = manager.start()
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished and final["status"] == "cancelled"
    assert final["frame_number"] == 1
    assert final["camera_status"] == "closed"
    assert final["inference_status"] == "stopped"
    assert final["total_detections"] == 0
    assert final["session_detection_total"] == 0
    assert final["class_counts"] == {name: 0 for name in classes}
    assert final["detections"] == []
    history = manager.live_history(1)
    assert len(history["points"]) == 1
    assert history["points"][0]["total_detections"] == 0
    assert history["points"][0]["class_counts"] == {name: 0 for name in classes}
    capture.release.assert_called_once()


def test_headless_webcam_worker_model_and_camera_startup_errors(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing.pt"
    settings = SimpleNamespace(
        behavior_model=missing_model,
        device="cpu",
        webcam_index=2,
        behavior_conf=0.2,
        inference_imgsz=960,
    )
    capture = Mock()
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index, backend=None: capture)
    model_manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})
    model_job = model_manager.start()
    finished, model_final = model_manager.wait(model_job["job_id"], timeout=2)
    assert finished
    assert model_final["status"] == "error"
    assert model_final["camera_status"] == "closed"
    assert model_final["inference_status"] == "error"
    assert "VISIONCLASS_BEHAVIOR_MODEL" in model_final["error"]
    capture.assert_not_called()

    settings.behavior_model = Path(__file__).resolve()
    capture_instance = Mock()
    capture_instance.isOpened.return_value = False
    monkeypatch.setattr(realtime_behavior, "YOLO", lambda path: object())
    monkeypatch.setattr(
        realtime_behavior.cv2,
        "VideoCapture",
        lambda index, backend=None: capture_instance,
    )
    camera_manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})
    camera_job = camera_manager.start()
    finished, camera_final = camera_manager.wait(camera_job["job_id"], timeout=2)
    assert finished
    assert camera_final["status"] == "error"
    assert "webcam (index 2)" in camera_final["error"]
    capture_instance.release.assert_called_once()


def test_headless_worker_camera_open_exception_and_frame_read_failure(monkeypatch):
    settings = SimpleNamespace(
        behavior_model=Path(__file__).resolve(),
        device="cpu",
        webcam_index=2,
        behavior_conf=0.2,
        inference_imgsz=960,
    )
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", lambda path: object())
    monkeypatch.setattr(
        realtime_behavior.cv2,
        "VideoCapture",
        Mock(side_effect=OSError("driver unavailable")),
    )
    open_manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})
    open_job = open_manager.start()
    finished, open_final = open_manager.wait(open_job["job_id"], timeout=2)
    assert finished
    assert open_final["status"] == "error"
    assert "webcam (index 2)" in open_final["error"]
    assert "VISIONCLASS_WEBCAM_INDEX" in open_final["error"]

    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (False, None)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index, backend=None: capture)
    read_manager = SessionManager({"webcam": realtime_behavior.webcam_session_worker})
    read_job = read_manager.start()
    finished, read_final = read_manager.wait(read_job["job_id"], timeout=2)
    assert finished
    assert read_final["status"] == "error"
    assert "remains connected" in read_final["error"]
    capture.read.assert_called_once()
    capture.release.assert_called_once()
