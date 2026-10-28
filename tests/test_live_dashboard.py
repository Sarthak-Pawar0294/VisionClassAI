"""Tests for live dashboard state and local HTTP server."""

import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytest

import src.realtime_behavior as realtime_behavior
from src.live_dashboard import (
    BEHAVIOR_CLASSES,
    LiveDashboardServer,
    LiveDashboardState,
)


def test_live_dashboard_state_reports_current_frame_metrics():
    state = LiveDashboardState()
    state.publish(
        b"jpeg-data",
        [
            {"class_name": "read", "confidence": 0.83},
            {"class_name": "read", "confidence": 0.72},
            {"class_name": "sleep", "confidence": 0.91},
        ],
        processing_fps=12.5,
        frame_number=7,
    )

    snapshot = state.snapshot()
    assert snapshot["camera_status"] == "open"
    assert snapshot["inference_status"] == "running"
    assert snapshot["processing_fps"] == 12.5
    assert snapshot["frame_number"] == 7
    assert snapshot["total_detections"] == 3
    assert snapshot["class_counts"] == {
        name: count
        for name, count in zip(BEHAVIOR_CLASSES, (0, 2, 1, 0, 0, 0))
    }
    assert snapshot["detections"] == [
        {"class_name": "read", "confidence": 0.83},
        {"class_name": "read", "confidence": 0.72},
        {"class_name": "sleep", "confidence": 0.91},
    ]


def test_live_dashboard_metrics_are_replaced_for_each_frame():
    state = LiveDashboardState()
    state.publish(
        b"frame-one",
        [{"class_name": "write", "confidence": 0.8}],
        processing_fps=10.0,
        frame_number=1,
    )
    state.publish(
        b"frame-two",
        [],
        processing_fps=11.0,
        frame_number=2,
    )

    snapshot = state.snapshot()
    assert snapshot["frame_number"] == 2
    assert snapshot["total_detections"] == 0
    assert snapshot["detections"] == []
    assert snapshot["class_counts"] == {name: 0 for name in BEHAVIOR_CLASSES}


def test_live_dashboard_keeps_error_status_until_a_frame_succeeds():
    state = LiveDashboardState()
    state.set_status(
        camera_status="error",
        inference_status="error",
        error="Camera disconnected",
    )
    state.set_status(inference_status="stopped")

    snapshot = state.snapshot()
    assert snapshot["camera_status"] == "error"
    assert snapshot["inference_status"] == "stopped"
    assert snapshot["error"] == "Camera disconnected"


def test_live_dashboard_serves_html_state_and_mjpeg_then_stops():
    state = LiveDashboardState()
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    state.publish(
        encoded.tobytes(),
        [{"class_name": "handrise", "confidence": 0.95}],
        processing_fps=15.0,
        frame_number=1,
    )
    server = LiveDashboardServer(state, port=0)

    try:
        url = server.start()
        assert server.start() == url
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.status == 200
            assert b"VisionClass Live" in response.read()

        with urllib.request.urlopen(url + "api/state", timeout=3) as response:
            payload = json.loads(response.read())
        assert payload["frame_number"] == 1
        assert payload["class_counts"]["handrise"] == 1
        assert payload["processing_fps"] == 15.0

        request = urllib.request.Request(url + "api/stop", method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 202
            assert json.loads(response.read()) == {"stop_requested": True}
        assert state.stop_requested

        request = urllib.request.Request(url + "stream.mjpg")
        with urllib.request.urlopen(request, timeout=3) as response:
            assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
            assert response.readline() == b"--frame\r\n"
            assert response.readline() == b"Content-Type: image/jpeg\r\n"
            content_length = int(response.readline().split(b":", 1)[1])
            assert response.readline() == b"\r\n"
            frame = response.read(content_length)
        assert frame.startswith(b"\xff\xd8")
    finally:
        server.stop()

    assert not server.is_running


def test_realtime_starts_dashboard_after_camera_and_reuses_inference(monkeypatch):
    events = []
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    fake_settings = SimpleNamespace(
        behavior_model=Path(__file__).resolve(),
        device=0,
        webcam_index=0,
        behavior_conf=0.2,
        inference_imgsz=960,
    )

    class FakeCapture:
        def isOpened(self):
            events.append("camera_open")
            return True

        def read(self):
            return True, frame

        def release(self):
            events.append("camera_release")

    class FakeBox:
        cls = [0]
        conf = [0.91]

    class FakeResult:
        boxes = [FakeBox()]

        @staticmethod
        def plot():
            return frame

    class FakeModel:
        names = {0: "handrise"}

        def __init__(self, path):
            events.append("model_load")

        def predict(self, **kwargs):
            events.append("inference")
            return [FakeResult()]

    class FakeDashboard:
        def __init__(self, state):
            self.state = state
            events.append("dashboard_construct")

        def start(self):
            events.append("dashboard_start")
            return "http://127.0.0.1:8765/"

        def stop(self):
            events.append("dashboard_stop")

    browser_open = Mock(return_value=True)
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: fake_settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index: FakeCapture())
    monkeypatch.setattr(realtime_behavior.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(realtime_behavior.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(realtime_behavior.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardServer", FakeDashboard)
    monkeypatch.setattr(realtime_behavior.webbrowser, "open", browser_open)

    realtime_behavior.main(device_spec="0")

    assert events.index("camera_open") < events.index("dashboard_start")
    assert events.index("dashboard_start") < events.index("inference")
    assert events.count("inference") == 1
    assert events.count("dashboard_start") == 1
    assert events[-2:] == ["camera_release", "dashboard_stop"]
    browser_open.assert_called_once_with("http://127.0.0.1:8765/", new=0)


def _settings_with_model(model_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        behavior_model=model_path,
        device="cpu",
        webcam_index=0,
        behavior_conf=0.2,
        inference_imgsz=960,
    )


def _install_minimal_realtime_mocks(monkeypatch, *, settings, model, capture, dashboard):
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", model)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(realtime_behavior.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(realtime_behavior.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(realtime_behavior.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardServer", dashboard)
    monkeypatch.setattr(realtime_behavior.webbrowser, "open", Mock(return_value=True))


def test_realtime_missing_model_reports_stderr_and_stops_before_camera(monkeypatch, tmp_path, capsys):
    missing_model = tmp_path / "missing.pt"
    settings = _settings_with_model(missing_model)
    video_capture = Mock()
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", video_capture)

    assert realtime_behavior.main() == 1

    error = capsys.readouterr().err
    assert "missing.pt" in error
    assert "VISIONCLASS_BEHAVIOR_MODEL" in error
    video_capture.assert_not_called()


def test_realtime_model_load_failure_reports_stderr_and_stops_before_camera(monkeypatch, capsys):
    settings = _settings_with_model(Path(__file__).resolve())
    video_capture = Mock()
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", Mock(side_effect=ValueError("bad checkpoint")))
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", video_capture)

    assert realtime_behavior.main() == 1

    error = capsys.readouterr().err
    assert "Could not load behavior model" in error
    assert "bad checkpoint" in error
    video_capture.assert_not_called()


def test_realtime_unavailable_camera_reports_error_and_releases_capture(monkeypatch, capsys):
    settings = _settings_with_model(Path(__file__).resolve())

    class FakeModel:
        def __init__(self, path):
            pass

    capture = Mock()
    capture.isOpened.return_value = False
    dashboard = Mock()
    _install_minimal_realtime_mocks(monkeypatch, settings=settings, model=FakeModel,
                                    capture=capture, dashboard=dashboard)

    assert realtime_behavior.main() == 1

    error = capsys.readouterr().err
    assert "webcam (index 0)" in error
    assert "VISIONCLASS_WEBCAM_INDEX" in error
    capture.release.assert_called_once()
    dashboard.assert_not_called()


def test_realtime_dashboard_start_failure_continues_without_dashboard(monkeypatch, capsys):
    events = []
    settings = _settings_with_model(Path(__file__).resolve())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            events.append("read")
            return True, frame

        def release(self):
            events.append("release")

    class FakeResult:
        boxes = None

        def plot(self):
            return frame

    class FakeModel:
        names = {}

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            events.append("inference")
            return [FakeResult()]

    dashboard_instance = Mock()
    dashboard_instance.start.side_effect = OSError("port busy")

    def dashboard_factory(state):
        events.append("dashboard")
        return dashboard_instance

    browser_open = Mock(return_value=True)
    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index: FakeCapture())
    monkeypatch.setattr(realtime_behavior.cv2, "imshow", lambda *args: events.append("preview"))
    monkeypatch.setattr(realtime_behavior.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(realtime_behavior.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardServer", dashboard_factory)
    monkeypatch.setattr(realtime_behavior.webbrowser, "open", browser_open)

    assert realtime_behavior.main() == 0

    error = capsys.readouterr().err
    assert "127.0.0.1:8765" in error
    assert "port busy" in error
    assert "without the dashboard" in error
    assert "inference" in events
    assert "preview" in events
    assert events[-1] == "release"
    dashboard_instance.stop.assert_called_once()
    browser_open.assert_not_called()


@pytest.mark.parametrize(
    "read_failure",
    [False, RuntimeError("camera read exploded")],
    ids=["false-return", "exception"],
)
def test_realtime_frame_read_failure_exits_nonzero_and_cleans_up(
    monkeypatch, capsys, read_failure
):
    settings = _settings_with_model(Path(__file__).resolve())

    class FakeModel:
        def __init__(self, path):
            pass

    capture = Mock()
    capture.isOpened.return_value = True
    if isinstance(read_failure, Exception):
        capture.read.side_effect = read_failure
    else:
        capture.read.return_value = (read_failure, None)
    dashboard_states = []
    dashboard_instance = Mock()
    dashboard_instance.start.return_value = "http://127.0.0.1:8765/"

    def dashboard_factory(state):
        dashboard_states.append(state)
        return dashboard_instance

    dashboard_instance.stop.side_effect = lambda: dashboard_states[0].set_status(
        inference_status="stopped"
    )

    _install_minimal_realtime_mocks(
        monkeypatch,
        settings=settings,
        model=FakeModel,
        capture=capture,
        dashboard=dashboard_factory,
    )

    assert realtime_behavior.main() == 1

    error = capsys.readouterr().err
    assert "Could not read a webcam frame" in error
    if isinstance(read_failure, Exception):
        assert "camera read exploded" in error
    capture.read.assert_called_once()
    assert dashboard_states[0].snapshot()["camera_status"] == "error"
    assert dashboard_states[0].snapshot()["inference_status"] == "stopped"
    capture.release.assert_called_once()
    dashboard_instance.stop.assert_called_once()


def test_realtime_inference_exception_reports_error_and_cleans_up(monkeypatch, capsys):
    settings = _settings_with_model(Path(__file__).resolve())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, frame)

    class FakeModel:
        names = {}

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            raise RuntimeError("inference failed")

    dashboard_instance = Mock()
    dashboard_instance.start.return_value = "http://127.0.0.1:8765/"
    _install_minimal_realtime_mocks(
        monkeypatch,
        settings=settings,
        model=FakeModel,
        capture=capture,
        dashboard=lambda state_arg: dashboard_instance,
    )

    assert realtime_behavior.main() == 1

    assert "inference failed" in capsys.readouterr().err
    capture.release.assert_called_once()
    dashboard_instance.stop.assert_called_once()


def test_live_dashboard_five_start_stop_cycles_rebind_same_port():
    first = LiveDashboardServer(LiveDashboardState(), port=0)
    first_url = first.start()
    port = urlsplit(first_url).port
    first.stop()
    first.stop()
    assert not first.is_running
    with pytest.raises(RuntimeError, match="already stopped"):
        first.start()

    for _ in range(5):
        state = LiveDashboardState()
        server = LiveDashboardServer(state, port=port)
        stream_opened = threading.Event()
        stream_errors = []
        stream_thread = None
        try:
            url = server.start()
            assert server.is_running
            with urllib.request.urlopen(url, timeout=3) as response:
                assert response.status == 200
                assert b"VisionClass Live" in response.read()

            def consume_stream():
                try:
                    with urllib.request.urlopen(url + "stream.mjpg", timeout=3) as response:
                        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
                        stream_opened.set()
                        response.read()
                except Exception as exc:
                    stream_errors.append(exc)

            stream_thread = threading.Thread(target=consume_stream)
            stream_thread.start()
            assert stream_opened.wait(timeout=3)
        finally:
            server.stop()
            if stream_thread is not None and stream_thread.is_alive():
                stream_thread.join(timeout=3)
        assert not server.is_running
        assert stream_thread is not None
        assert not stream_thread.is_alive()
        assert not stream_errors


def test_live_dashboard_thread_start_failure_closes_socket_and_allows_recovery(monkeypatch):
    failed_state = LiveDashboardState()
    failed_server = LiveDashboardServer(failed_state, port=0)
    port = urlsplit(failed_server.url).port
    monkeypatch.setattr(
        failed_server._thread,
        "start",
        Mock(side_effect=RuntimeError("thread unavailable")),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        failed_server.start()

    assert not failed_server.is_running
    assert failed_state.next_frame(0, timeout=0.01) is None

    recovered_server = LiveDashboardServer(LiveDashboardState(), port=port)
    try:
        recovered_url = recovered_server.start()
        with urllib.request.urlopen(recovered_url, timeout=3) as response:
            assert response.status == 200
    finally:
        recovered_server.stop()
    assert not recovered_server.is_running


def test_realtime_dashboard_stop_request_exits_after_mocked_iteration(monkeypatch):
    settings = _settings_with_model(Path(__file__).resolve())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    state = LiveDashboardState()
    events = []

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            events.append("camera_release")

    class FakeResult:
        boxes = None

        def plot(self):
            return frame

    class FakeModel:
        names = {}

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            events.append("inference")
            state.request_stop()
            return [FakeResult()]

    class FakeDashboard:
        def __init__(self, dashboard_state):
            pass

        def start(self):
            return "http://127.0.0.1:8765/"

        def stop(self):
            events.append("dashboard_stop")

    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index: FakeCapture())
    monkeypatch.setattr(realtime_behavior.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(realtime_behavior.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(realtime_behavior.cv2, "imencode", lambda *args: (True, SimpleNamespace(tobytes=lambda: b"jpeg")))
    monkeypatch.setattr(realtime_behavior.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardServer", FakeDashboard)
    monkeypatch.setattr(realtime_behavior.webbrowser, "open", lambda *args, **kwargs: True)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardState", lambda: state)

    result = []
    worker = threading.Thread(target=lambda: result.append(realtime_behavior.main()))
    worker.start()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result == [0]
    assert events.count("inference") == 1
    assert events[-2:] == ["camera_release", "dashboard_stop"]


def test_http_dashboard_stop_exits_mocked_cli_worker_and_cleans_up(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    settings = _settings_with_model(Path(__file__).resolve())
    state_holder = {}
    dashboard_holder = {}
    inference_started = threading.Event()
    events = []

    class FakeCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            events.append("camera_release")

    class FakeResult:
        boxes = None

        def plot(self):
            return frame

    class FakeModel:
        names = {}

        def __init__(self, path):
            pass

        def predict(self, **kwargs):
            inference_started.set()
            assert state_holder["state"]._stop_event.wait(timeout=2)
            return [FakeResult()]

    def dashboard_factory(state):
        state_holder["state"] = state
        dashboard = LiveDashboardServer(state, port=0)
        dashboard_holder["server"] = dashboard
        return dashboard

    monkeypatch.setattr(realtime_behavior, "get_settings", lambda **kwargs: settings)
    monkeypatch.setattr(realtime_behavior, "YOLO", FakeModel)
    monkeypatch.setattr(realtime_behavior.cv2, "VideoCapture", lambda index: FakeCapture())
    monkeypatch.setattr(realtime_behavior.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(realtime_behavior.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(realtime_behavior.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(realtime_behavior, "LiveDashboardServer", dashboard_factory)
    monkeypatch.setattr(realtime_behavior.webbrowser, "open", lambda *args, **kwargs: True)

    result = []
    worker = threading.Thread(target=lambda: result.append(realtime_behavior.main()))
    worker.start()
    assert inference_started.wait(timeout=2)
    url = dashboard_holder["server"].url
    request = urllib.request.Request(url + "api/stop", method="POST")
    with urllib.request.urlopen(request, timeout=2) as response:
        assert response.status == 202
        assert json.loads(response.read()) == {"stop_requested": True}

    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result == [0]
    assert events == ["camera_release"]
    assert not dashboard_holder["server"].is_running
