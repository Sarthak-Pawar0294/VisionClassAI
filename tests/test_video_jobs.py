"""Upload, decoder, video-pipeline, analytics, and artifact-boundary tests."""

import csv
import io
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.behavior_history as behavior_history
import src.video_jobs as video_jobs
from src.session_manager import SessionManager


class _Value:
    def __init__(self, value):
        self.value = value

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    def __init__(self, *, xyxy=None, ids=None, classes=None, confidences=None):
        self.xyxy = _Value(xyxy) if xyxy is not None else None
        self.id = _Value(ids) if ids is not None else None
        self.cls = _Value(classes) if classes is not None else None
        self.conf = _Value(confidences) if confidences is not None else None

    def __len__(self):
        return len(self.cls.value) if self.cls is not None else 0


def _settings():
    return SimpleNamespace(
        person_model=Path("person.pt"),
        behavior_model=Path("behavior.pt"),
        device="cpu",
        inference_imgsz=960,
        person_conf=0.3,
        behavior_conf=0.2,
        tracker="tracker.yaml",
    )


def _valid_mock_output(path, **expected):
    assert Path(path).is_file()
    return {
        "validated": True,
        "container": "mp4",
        "codec": expected["codec"],
        "width": expected["expected_width"],
        "height": expected["expected_height"],
        "fps": expected["expected_fps"],
        "processed_frames": expected["expected_frames"],
        "decoded_output_frames": expected["expected_frames"],
        "output_bytes": Path(path).stat().st_size,
    }


def test_upload_store_generates_unique_safe_paths_and_enforces_extension(tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    first_id, first_path, first_size = store.create_upload("..\\private\\lesson.MP4", io.BytesIO(b"abc"), 10)
    second_id, second_path, _ = store.create_upload("clip.mkv", io.BytesIO(b"xy"), 10)

    assert first_id != second_id
    assert first_path.name == "input.mp4"
    assert first_path.parent.name == first_id
    assert first_size == 3
    assert second_path.name == "input.mkv"
    assert first_path.read_bytes() == b"abc"
    with pytest.raises(video_jobs.UploadValidationError, match="Unsupported video extension"):
        store.create_upload("lesson.exe", io.BytesIO(b"data"), 10)


def test_upload_store_removes_partial_file_when_payload_limit_is_exceeded(tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    with pytest.raises(video_jobs.UploadValidationError) as error:
        store.create_upload("lesson.mp4", io.BytesIO(b"123456"), 5)

    assert error.value.status_code == 413
    assert list((tmp_path / "runs").iterdir()) == []


def test_decoder_validation_requires_open_and_decoded_frame_and_releases_capture(monkeypatch, tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    run_id, video, _ = store.create_upload("lesson.avi", io.BytesIO(b"video"), 20)
    calls = []

    class Capture:
        def isOpened(self): return True
        def read(self): return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def get(self, key):
            return {video_jobs.cv2.CAP_PROP_FRAME_COUNT: 17, video_jobs.cv2.CAP_PROP_FPS: 25.0}.get(key, 0)
        def release(self): calls.append("release")

    monkeypatch.setattr(video_jobs.cv2, "VideoCapture", lambda path: Capture())
    assert store.validate_decoder(run_id, video) == {"total_frames": 17, "source_fps": 25.0}
    assert calls == ["release"]


def test_decoder_rejects_open_but_undecodable_file(monkeypatch, tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    run_id, video, _ = store.create_upload("lesson.mov", io.BytesIO(b"video"), 20)
    capture = SimpleNamespace(isOpened=lambda: True, read=lambda: (False, None), release=lambda: None)
    monkeypatch.setattr(video_jobs.cv2, "VideoCapture", lambda path: capture)
    with pytest.raises(video_jobs.UploadValidationError, match="could not decode"):
        store.validate_decoder(run_id, video)


def test_run_ids_and_artifact_names_cannot_escape_run_directory(tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    with pytest.raises(video_jobs.UploadValidationError, match="Run not found"):
        store.artifact_path("..", "history_csv")
    run_id, input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"video"), 20)
    store.write_manifest(run_id, {"status": "completed", "artifacts": {"history_csv": "../outside.csv"}})
    with pytest.raises(video_jobs.UploadValidationError, match="not generated"):
        store.artifact_path(run_id, "history_csv")
    outside = tmp_path / "outside.csv"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(video_jobs.UploadValidationError, match="outside its run"):
        store.add_artifact(run_id, "history_csv", outside)
    assert input_path.exists()


def test_analytics_bundle_uses_existing_csv_semantics_and_tracker_labels(tmp_path):
    history = tmp_path / "behavior_history.csv"
    summary = tmp_path / "behavior_summary.csv"
    rows = [
        {"frame": 0, "time_seconds": 0.0, "student_id": 7, "behavior": "write", "confidence": 0.8, "is_cached": "false", "unknown_cause": ""},
        {"frame": 1, "time_seconds": 0.04, "student_id": 7, "behavior": "write", "confidence": 0.7, "is_cached": "true", "unknown_cause": ""},
        {"frame": 1, "time_seconds": 0.04, "student_id": 9, "behavior": "unknown", "confidence": 0.0, "is_cached": "false", "unknown_cause": "no_match"},
    ]
    with history.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame", "time_seconds", "student_id", "behavior", "confidence", "is_cached", "unknown_cause"])
        writer.writeheader()
        writer.writerows(rows)
    with summary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["student_id", "handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown"])
        writer.writeheader()
        writer.writerow({"student_id": 7, "write": 0.08})
        writer.writerow({"student_id": 9, "unknown": 0.04})

    bundle = video_jobs.analytics_bundle(history, summary)
    assert bundle["session"]["n_tracker_ids"] == 2
    assert bundle["session"]["total_rows"] == 3
    assert bundle["distribution"]["counts"]["write"] == 2
    assert bundle["cache"]["cached_rows"] == 1
    assert bundle["per_tracker_id"]["7"]["total_records"] == 2
    assert bundle["summary_rows"][0]["tracker_id"] == 7
    assert "student_id" not in bundle["summary_rows"][0]
    assert bundle["temporal"]["avg_transitions_per_tracker_id"] is not None
    assert "avg_transitions_per_student" not in bundle["temporal"]
    assert "student_id" not in bundle["temporal"]["longest_run_per_behavior"]["write"]
    assert "tracker_id" in bundle["temporal"]["longest_run_per_behavior"]["write"]
    assert bundle["semantics"]["unique_real_world_students"] is None
    assert "all_live_conf" not in bundle["confidence"]
    assert "live_confidences" not in bundle["per_tracker_id"]["7"]


def test_run_report_and_static_dashboard_are_truthful_and_self_contained(tmp_path):
    history = tmp_path / "behavior_history.csv"
    summary = tmp_path / "behavior_summary.csv"
    report = tmp_path / "behavior_analytics_report.md"
    dashboard = tmp_path / "behavior_analytics_dashboard.html"
    history.write_text(
        "frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause\n"
        "0,0.000,4,read,0.8000,false,\n1,0.040,4,read,0.7000,true,\n",
        encoding="utf-8",
    )
    summary.write_text("student_id,handrise,read,sleep,stand,using_electronic_devices,write,unknown\n4,0,0.08,0,0,0,0,0\n", encoding="utf-8")
    video_jobs.generate_analytics_report.generate_run_report(history, summary, report, processing_metrics={"behavior_detection_total": 1})
    video_jobs.generate_analytics_report.generate_run_dashboard(history, summary, dashboard)
    text = report.read_text(encoding="utf-8")
    page = dashboard.read_text(encoding="utf-8")
    assert "Raw behavior boxes processed: 1" in text
    assert "Tracker IDs" in text
    assert "not verified real-world student identities" in text
    assert "10.9 seconds" not in text
    assert "32 students" not in text
    assert "Tracker IDs are video-scoped" in page
    assert "fonts.googleapis.com" not in page


def test_legacy_dashboard_generator_name_remains_an_alias():
    from src.build_dashboard import generate_dashboard
    from src.generate_analytics_report import generate_run_dashboard

    assert generate_dashboard is generate_run_dashboard


def test_process_video_reuses_tracker_history_and_reports_cleanup(monkeypatch, tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    events = []

    class Capture:
        def __init__(self, _path): self.index = 0
        def isOpened(self): return True
        def get(self, key):
            return {behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8, behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 8, behavior_history.cv2.CAP_PROP_FPS: 25, behavior_history.cv2.CAP_PROP_FRAME_COUNT: 2}.get(key, 0)
        def read(self):
            self.index += 1
            return (True, frame) if self.index <= 2 else (False, None)
        def release(self): events.append("capture_release")

    class Writer:
        def __init__(self, path, *_args):
            self.file = open(path, "wb")
        def isOpened(self): return True
        def write(self, _frame): self.file.write(b"frame")
        def release(self): self.file.close(); events.append("writer_release")

    person_result = SimpleNamespace(boxes=_Boxes(xyxy=[[0, 0, 7, 7]], ids=[4]))
    behavior_result = SimpleNamespace(boxes=_Boxes(xyxy=[[0, 0, 7, 7]], classes=[5], confidences=[0.9]))
    person_model = SimpleNamespace(track=lambda **_kwargs: [person_result])
    behavior_model = SimpleNamespace(predict=lambda **_kwargs: [behavior_result], names={5: "write"})
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(behavior_history, "_find_ffmpeg_h264_encoder", lambda: None)
    monkeypatch.setattr(behavior_history, "_validate_annotated_video", _valid_mock_output)
    progress = []
    settings = _settings()
    result = behavior_history.process_video(
        tmp_path / "input.mp4", tmp_path / "history.csv", tmp_path / "summary.csv", tmp_path / "annotated.mp4",
        settings=settings, person_model=person_model, behavior_model=behavior_model,
        progress_callback=progress.append,
    )

    assert result["processed_frames"] == 2
    assert result["behavior_detection_total"] == 2
    assert result["history_record_total"] == 2
    assert result["output_video"]["validated"] is True
    assert result["output_video"]["decoded_output_frames"] == 2
    assert (tmp_path / "annotated.mp4").is_file()
    assert progress[-1]["progress_percent"] == 100.0
    assert sorted(events) == ["capture_release", "writer_release"]
    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert written[0]["student_id"] == "4"
    assert written[0]["behavior"] == "write"
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    assert summary_rows[0]["write"] == "0.08"


def test_process_video_cancellation_finishes_current_frame_then_releases(monkeypatch, tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []

    class Capture:
        def __init__(self, _path): self.index = 0
        def isOpened(self): return True
        def get(self, _key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 8,
                behavior_history.cv2.CAP_PROP_FPS: 25,
                behavior_history.cv2.CAP_PROP_FRAME_COUNT: 50,
            }.get(_key, 0)
        def read(self): self.index += 1; return True, frame
        def release(self): calls.append("capture_release")

    class Writer:
        def __init__(self, path, *_args): self.file = open(path, "wb")
        def isOpened(self): return True
        def write(self, _frame): calls.append("write")
        def release(self): self.file.close(); calls.append("writer_release")

    person_result = SimpleNamespace(boxes=_Boxes(xyxy=[[0, 0, 7, 7]], ids=[1]))
    behavior_result = SimpleNamespace(boxes=None)
    person_model = SimpleNamespace(track=lambda **_kwargs: [person_result])
    behavior_model = SimpleNamespace(predict=lambda **_kwargs: [behavior_result], names={})
    stop = {"requested": False}
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(behavior_history, "_find_ffmpeg_h264_encoder", lambda: None)
    result = behavior_history.process_video(
        tmp_path / "input.mp4", tmp_path / "history.csv", tmp_path / "summary.csv", tmp_path / "annotated.mp4",
        settings=_settings(), person_model=person_model, behavior_model=behavior_model,
        stop_requested=lambda: stop["requested"],
        progress_callback=lambda _p: stop.update(requested=True),
    )
    assert result["cancelled"] is True
    assert result["processed_frames"] == 1
    assert calls.count("write") == 1
    assert calls.count("writer_release") == 1
    assert calls.count("capture_release") == 1
    assert not (tmp_path / "annotated.mp4").exists()
    assert not list(tmp_path.glob("*.raw.mp4"))


def test_output_validation_rejects_missing_empty_undecodable_and_incomplete_files(monkeypatch, tmp_path):
    expected = {
        "expected_frames": 2,
        "expected_width": 8,
        "expected_height": 6,
        "expected_fps": 25.0,
        "output_root": tmp_path,
        "codec": "H.264 (test)",
    }
    missing = tmp_path / "missing.mp4"
    with pytest.raises(RuntimeError, match="missing or outside"):
        behavior_history._validate_annotated_video(missing, **expected)

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(RuntimeError, match="empty or implausibly small"):
        behavior_history._validate_annotated_video(empty, **expected)

    output = tmp_path / "invalid.mp4"
    output.write_bytes(b"x" * 256)
    with pytest.raises(RuntimeError, match="outside its run directory"):
        behavior_history._validate_annotated_video(
            output,
            **{**expected, "output_root": tmp_path / "isolated-run"},
        )

    class Capture:
        def __init__(self, *, opened=True, frames=1):
            self.opened = opened
            self.frames = frames
            self.released = False

        def isOpened(self): return self.opened
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 6,
                behavior_history.cv2.CAP_PROP_FPS: 25.0,
            }.get(key, 0)
        def read(self):
            if self.frames <= 0:
                return False, None
            self.frames -= 1
            return True, np.zeros((6, 8, 3), dtype=np.uint8)
        def release(self): self.released = True

    bad_decoder = Capture(opened=False)
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", lambda _path: bad_decoder)
    with pytest.raises(RuntimeError, match="could not be reopened"):
        behavior_history._validate_annotated_video(output, **expected)
    assert bad_decoder.released

    incomplete_decoder = Capture(frames=1)
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", lambda _path: incomplete_decoder)
    with pytest.raises(RuntimeError, match="contains 1 decodable frames; 2 frames were processed"):
        behavior_history._validate_annotated_video(output, **expected)
    assert incomplete_decoder.released


def test_output_validation_requires_h264_when_ffprobe_reports_an_incompatible_codec(monkeypatch, tmp_path):
    output = tmp_path / "wrong-codec.mp4"
    output.write_bytes(b"x" * 256)

    class Capture:
        released = False
        def isOpened(self): return True
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 6,
                behavior_history.cv2.CAP_PROP_FPS: 25.0,
            }.get(key, 0)
        def read(self):
            if getattr(self, "read_once", False): return False, None
            self.read_once = True
            return True, np.zeros((6, 8, 3), dtype=np.uint8)
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(behavior_history, "_probe_video_codec", lambda _path: "mpeg4")
    with pytest.raises(RuntimeError, match="not browser-compatible H.264"):
        behavior_history._validate_annotated_video(
            output,
            expected_frames=1,
            expected_width=8,
            expected_height=6,
            expected_fps=25,
            output_root=tmp_path,
            codec="H.264 (test)",
        )
    assert capture.released


def test_output_validation_observes_cancellation_and_releases_decoder(monkeypatch, tmp_path):
    output = tmp_path / "cancelled-check.mp4"
    output.write_bytes(b"x" * 256)

    class Capture:
        released = False
        def isOpened(self): return True
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 6,
                behavior_history.cv2.CAP_PROP_FPS: 25.0,
            }.get(key, 0)
        def read(self): pytest.fail("validation should observe cancellation before reading")
        def release(self): self.released = True

    capture = Capture()
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", lambda _path: capture)
    with pytest.raises(behavior_history.OutputValidationCancelled):
        behavior_history._validate_annotated_video(
            output,
            expected_frames=1,
            expected_width=8,
            expected_height=6,
            expected_fps=25,
            output_root=tmp_path,
            codec="H.264 (test)",
            stop_requested=lambda: True,
        )
    assert capture.released


def test_writer_initialization_and_finalization_failures_release_video_resources(monkeypatch, tmp_path):
    events = []

    class Capture:
        def __init__(self, _path): pass
        def isOpened(self): return True
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 6,
                behavior_history.cv2.CAP_PROP_FPS: 25.0,
                behavior_history.cv2.CAP_PROP_FRAME_COUNT: 0,
            }.get(key, 0)
        def read(self): return False, None
        def release(self): events.append("capture_release")

    class InitializationFailure:
        def __init__(self, _path, *_args): pass
        def isOpened(self): return False
        def release(self): events.append("writer_release")

    monkeypatch.setattr(behavior_history, "_find_ffmpeg_h264_encoder", lambda: None)
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", InitializationFailure)
    with pytest.raises(RuntimeError, match="browser-compatible H.264"):
        behavior_history.process_video(
            tmp_path / "input.mp4", tmp_path / "history.csv", tmp_path / "summary.csv", tmp_path / "annotated.mp4",
            settings=_settings(), person_model=object(), behavior_model=object(),
        )
    assert events == ["capture_release", "writer_release"]
    assert not (tmp_path / "annotated.mp4").exists()

    events.clear()

    class FinalizationFailure:
        def __init__(self, path, *_args): self.file = open(path, "wb")
        def isOpened(self): return True
        def write(self, _frame): pass
        def release(self):
            self.file.close()
            events.append("writer_release")
            raise OSError("writer finalization failed")

    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", FinalizationFailure)
    with pytest.raises(RuntimeError, match="Could not finalize annotated video resources: writer finalization failed"):
        behavior_history.process_video(
            tmp_path / "input.mp4", tmp_path / "history.csv", tmp_path / "summary.csv", tmp_path / "annotated.mp4",
            settings=_settings(), person_model=object(), behavior_model=object(),
        )
    assert events == ["capture_release", "writer_release"]
    assert not (tmp_path / "annotated.mp4").exists()
    assert not list(tmp_path.glob("*.raw.mp4"))


def test_processing_exception_releases_writer_and_does_not_publish_partial_output(monkeypatch, tmp_path):
    events = []
    frame = np.zeros((6, 8, 3), dtype=np.uint8)

    class Capture:
        def __init__(self, _path): self.used = False
        def isOpened(self): return True
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 8,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 6,
                behavior_history.cv2.CAP_PROP_FPS: 25.0,
                behavior_history.cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(key, 0)
        def read(self): self.used = True; return True, frame
        def release(self): events.append("capture_release")

    class Writer:
        def __init__(self, path, *_args): self.file = open(path, "wb")
        def isOpened(self): return True
        def write(self, _frame): pass
        def release(self): self.file.close(); events.append("writer_release")

    monkeypatch.setattr(behavior_history, "_find_ffmpeg_h264_encoder", lambda: None)
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", Writer)
    bad_model = SimpleNamespace(track=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("mock inference failure")))
    with pytest.raises(RuntimeError, match="mock inference failure"):
        behavior_history.process_video(
            tmp_path / "input.mp4", tmp_path / "history.csv", tmp_path / "summary.csv", tmp_path / "annotated.mp4",
            settings=_settings(), person_model=bad_model, behavior_model=object(),
        )
    assert events == ["capture_release", "writer_release"]
    assert not (tmp_path / "annotated.mp4").exists()
    assert not list(tmp_path.glob("*.raw.mp4"))


def test_odd_source_dimensions_are_cropped_to_even_output_dimensions(monkeypatch, tmp_path):
    frame = np.zeros((7, 9, 3), dtype=np.uint8)
    writer_size = []
    written_shapes = []
    validation_dimensions = []

    class Capture:
        def __init__(self, _path): self.read_count = 0
        def isOpened(self): return True
        def get(self, key):
            return {
                behavior_history.cv2.CAP_PROP_FRAME_WIDTH: 9,
                behavior_history.cv2.CAP_PROP_FRAME_HEIGHT: 7,
                behavior_history.cv2.CAP_PROP_FPS: 12.0,
                behavior_history.cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(key, 0)
        def read(self):
            self.read_count += 1
            return (True, frame) if self.read_count == 1 else (False, None)
        def release(self): pass

    class Writer:
        def __init__(self, path, _fourcc, _fps, size):
            writer_size.append(size)
            self.file = open(path, "wb")
        def isOpened(self): return True
        def write(self, image): written_shapes.append(image.shape)
        def release(self): self.file.close()

    def validate(_path, **kwargs):
        validation_dimensions.append((kwargs["expected_width"], kwargs["expected_height"]))
        return {"validated": True, "width": kwargs["expected_width"], "height": kwargs["expected_height"]}

    monkeypatch.setattr(behavior_history, "_find_ffmpeg_h264_encoder", lambda: None)
    monkeypatch.setattr(behavior_history.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(behavior_history.cv2, "VideoWriter", Writer)
    monkeypatch.setattr(behavior_history, "_validate_annotated_video", validate)
    person_model = SimpleNamespace(track=lambda **_kwargs: [SimpleNamespace(boxes=SimpleNamespace(id=None))])
    behavior_model = SimpleNamespace(predict=lambda **_kwargs: [SimpleNamespace(boxes=None)], names={})

    result = behavior_history.process_video(
        tmp_path / "input.mp4",
        tmp_path / "history.csv",
        tmp_path / "summary.csv",
        tmp_path / "annotated.mp4",
        settings=_settings(),
        person_model=person_model,
        behavior_model=behavior_model,
        output_root=tmp_path,
    )

    assert writer_size == [(8, 6)]
    assert written_shapes == [(6, 8, 3)]
    assert validation_dimensions == [(8, 6)]
    assert (result["output_video"]["width"], result["output_video"]["height"]) == (8, 6)


def test_tiny_synthetic_video_pipeline_produces_decodable_h264_when_available(tmp_path):
    input_path = tmp_path / "synthetic_input.mp4"
    fourcc = behavior_history.cv2.VideoWriter_fourcc(*"mp4v")
    writer = behavior_history.cv2.VideoWriter(str(input_path), fourcc, 12.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("Installed OpenCV build cannot create the small synthetic MP4 input.")
    for index in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 1] = index * 24
        writer.write(frame)
    writer.release()

    probe = behavior_history.cv2.VideoCapture(str(input_path))
    input_decodes = probe.isOpened() and probe.read()[0]
    probe.release()
    if not input_decodes:
        pytest.skip("Installed OpenCV build cannot decode the small synthetic MP4 input.")

    person_model = SimpleNamespace(
        track=lambda **_kwargs: [SimpleNamespace(boxes=SimpleNamespace(id=None))]
    )
    behavior_model = SimpleNamespace(
        predict=lambda **_kwargs: [SimpleNamespace(boxes=None)],
        names={},
    )
    output_path = tmp_path / "annotated.mp4"
    result = behavior_history.process_video(
        input_path,
        tmp_path / "history.csv",
        tmp_path / "summary.csv",
        output_path,
        settings=_settings(),
        person_model=person_model,
        behavior_model=behavior_model,
        output_root=tmp_path,
    )
    assert result["processed_frames"] == 8
    assert result["output_video"]["validated"] is True
    assert result["output_video"]["codec"].startswith("H.264")
    if shutil.which("ffprobe"):
        assert result["output_video"]["codec_verified"] is True
    assert result["output_video"]["decoded_output_frames"] == 8
    assert result["output_video"]["width"] == 64
    assert result["output_video"]["height"] == 48
    assert result["output_video"]["fps"] == pytest.approx(12.0, abs=0.1)
    assert output_path.stat().st_size >= 128


@pytest.mark.parametrize(
    ("unassociated_detections", "quality_status"),
    [(0, "pass"), (1, "degraded")],
)
def test_video_worker_generates_only_real_artifacts_and_reaches_completed_state(
    monkeypatch, tmp_path, unassociated_detections, quality_status
):
    store = video_jobs.RunStore(tmp_path / "runs")
    run_id, input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"video"), 20)
    store.write_manifest(run_id, {"status": "queued", "artifacts": {}})
    person_model_path = tmp_path / "person.pt"
    behavior_model_path = tmp_path / "behavior.pt"
    person_model_path.write_bytes(b"mock")
    behavior_model_path.write_bytes(b"mock")
    settings = SimpleNamespace(person_model=person_model_path, behavior_model=behavior_model_path)
    monkeypatch.setattr(video_jobs, "get_settings", lambda: settings)
    monkeypatch.setattr(video_jobs, "YOLO", lambda path: object())

    def process(input_file, history_file, summary_file, output_file, **kwargs):
        output_file.write_bytes(b"annotated video")
        history_file.write_text(
            "frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause\n"
            "0,0.000,2,read,0.8,false,\n1,0.040,2,read,0.7,true,\n",
            encoding="utf-8",
        )
        summary_file.write_text(
            "student_id,handrise,read,sleep,stand,using_electronic_devices,write,unknown\n2,0,0.08,0,0,0,0,0\n",
            encoding="utf-8",
        )
        kwargs["progress_callback"]({
            "processed_frames": 2,
            "total_frames": 2,
            "progress_percent": 100.0,
            "history_record_total": 2,
            "behavior_detection_total": 1,
            "frame_detection_counts": {"read": 1},
        })
        return {
            "processed_frames": 2,
            "total_frames": 2,
            "history_record_total": 2,
            "behavior_detection_total": 1,
            "unassociated_detections": unassociated_detections,
            "cancelled": False,
            "output_video": {"validated": True, "decoded_output_frames": 2, "codec": "H.264 (test)"},
        }

    monkeypatch.setattr(video_jobs, "process_video", process)
    manager = SessionManager()
    started = manager.start("video", worker=video_jobs.make_video_worker(store, run_id, input_path), run_id=run_id)
    finished, final = manager.wait(started["job_id"], timeout=2)

    assert finished
    assert final["status"] == "completed"
    assert final["camera_status"] == "not_applicable"
    assert final["inference_status"] == "completed"
    manifest = store.read_manifest(run_id)
    assert manifest["status"] == "completed"
    assert manifest["processed_frames"] == manifest["progress"]["processed_frames"] == 2
    assert manifest["history_record_total"] == manifest["progress"]["history_record_total"] == 2
    assert manifest["behavior_detection_total"] == manifest["progress"]["behavior_detection_total"] == 1
    assert manifest["frame_detection_counts"] == {"read": 1}
    assert manifest["quality_status"] == quality_status
    assert manifest["association_rate"] == 1.0 - unassociated_detections
    assert set(manifest["artifacts"]) == {
        "source_video", "annotated_video", "history_csv", "summary_csv", "analytics_report", "analytics_dashboard"
    }
    assert "Tracker IDs" in (store.run_dir(run_id) / "behavior_analytics_report.md").read_text(encoding="utf-8")
    assert store.artifact_path(run_id, "analytics_dashboard").is_file()
    assert final["progress_percent"] == 100.0


def test_video_worker_keeps_core_processing_error_and_reports_optional_artifact_failure(monkeypatch, tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    run_id, input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"video"), 20)
    store.write_manifest(run_id, {"status": "queued", "artifacts": {}})
    person_model_path = tmp_path / "person.pt"
    behavior_model_path = tmp_path / "behavior.pt"
    person_model_path.touch()
    behavior_model_path.touch()
    monkeypatch.setattr(video_jobs, "get_settings", lambda: SimpleNamespace(person_model=person_model_path, behavior_model=behavior_model_path))
    monkeypatch.setattr(video_jobs, "YOLO", lambda path: object())

    def process(_input, history, summary, annotated, **kwargs):
        annotated.write_bytes(b"video")
        history.write_text(
            "frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause\n0,0.000,2,read,0.8,false,\n",
            encoding="utf-8",
        )
        summary.write_text("student_id,handrise,read,sleep,stand,using_electronic_devices,write,unknown\n2,0,0.04,0,0,0,0,0\n", encoding="utf-8")
        return {
            "processed_frames": 1,
            "total_frames": 1,
            "history_record_total": 1,
            "behavior_detection_total": 1,
            "unassociated_detections": 0,
            "cancelled": False,
            "output_video": {"validated": True, "decoded_output_frames": 1, "codec": "H.264 (test)"},
        }

    monkeypatch.setattr(video_jobs, "process_video", process)
    monkeypatch.setattr(video_jobs.generate_analytics_report, "generate_run_report", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("report disk error")))
    manager = SessionManager()
    started = manager.start("video", worker=video_jobs.make_video_worker(store, run_id, input_path), run_id=run_id)
    finished, final = manager.wait(started["job_id"], timeout=2)

    manifest = store.read_manifest(run_id)
    assert finished and final["status"] == "completed"
    assert manifest["status"] == "completed"
    assert "history_csv" in manifest["artifacts"]
    assert "analytics_report" not in manifest["artifacts"]
    assert "analytics_dashboard" in manifest["artifacts"]
    assert "Markdown analytics report unavailable: report disk error" in manifest["optional_errors"]


def test_video_worker_marks_processing_error_when_output_validation_is_missing(monkeypatch, tmp_path):
    store = video_jobs.RunStore(tmp_path / "runs")
    run_id, input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"video"), 20)
    store.write_manifest(run_id, {"status": "queued", "artifacts": {}})
    person_model_path = tmp_path / "person.pt"
    behavior_model_path = tmp_path / "behavior.pt"
    person_model_path.touch()
    behavior_model_path.touch()
    monkeypatch.setattr(video_jobs, "get_settings", lambda: SimpleNamespace(person_model=person_model_path, behavior_model=behavior_model_path))
    monkeypatch.setattr(video_jobs, "YOLO", lambda _path: object())

    def process(_input, history, summary, annotated, **_kwargs):
        annotated.write_bytes(b"unvalidated partial")
        history.write_text(
            "frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause\n",
            encoding="utf-8",
        )
        summary.write_text(
            "student_id,handrise,read,sleep,stand,using_electronic_devices,write,unknown\n",
            encoding="utf-8",
        )
        return {"processed_frames": 1, "cancelled": False, "output_video": {"validated": False}}

    monkeypatch.setattr(video_jobs, "process_video", process)
    manager = SessionManager()
    started = manager.start("video", worker=video_jobs.make_video_worker(store, run_id, input_path), run_id=run_id)
    finished, final = manager.wait(started["job_id"], timeout=2)
    manifest = store.read_manifest(run_id)

    assert finished and final["status"] == "error"
    assert "output was not validated" in final["error"].lower()
    assert manifest["status"] == "error"
    assert manifest["artifacts"] == {}
    assert not (store.run_dir(run_id) / "annotated.mp4").exists()
