"""HTTP upload limits, job arbitration, cancellation, and safe run downloads."""

import io
import threading

import numpy as np

import src.web_app as web_app
import src.video_jobs as video_jobs
from src.session_manager import SessionManager
from src.video_jobs import RunStore
from src.web_app import DEFAULT_PORT, create_app


def _token(client):
    return client.get("/api/csrf", headers={"Host": f"127.0.0.1:{DEFAULT_PORT}"}).get_json()["csrf_token"]


def _headers(token):
    host = f"127.0.0.1:{DEFAULT_PORT}"
    return {"Host": host, "Origin": f"http://{host}", "X-CSRF-Token": token}


class _Decoder:
    def __init__(self, _path, *, decodes=True): self.decodes = decodes; self.released = False
    def isOpened(self): return True
    def read(self): return (self.decodes, np.zeros((2, 2, 3), dtype=np.uint8) if self.decodes else None)
    def get(self, key): return {video_jobs.cv2.CAP_PROP_FRAME_COUNT: 4, video_jobs.cv2.CAP_PROP_FPS: 25.0}.get(key, 0)
    def release(self): self.released = True


def _app(tmp_path, monkeypatch, *, max_upload=100, overhead=512, manager=None, decodes=True):
    decoder = _Decoder(None, decodes=decodes)
    monkeypatch.setattr(video_jobs.cv2, "VideoCapture", lambda path: decoder)
    entered = threading.Event()

    def worker_factory(store, run_id, input_path):
        def worker(context):
            manifest = store.read_manifest(run_id)
            manifest["status"] = "running"
            store.write_manifest(run_id, manifest)
            entered.set()
            context.stop_event.wait(2)
            manifest = store.read_manifest(run_id)
            manifest.update(status="cancelled", progress={"processed_frames": 0}, artifacts={})
            store.write_manifest(run_id, manifest)
        return worker

    monkeypatch.setattr(web_app, "make_video_worker", worker_factory)
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "RUNS_DIR": tmp_path / "runs",
            "MAX_UPLOAD_BYTES": max_upload,
            "MAX_MULTIPART_OVERHEAD_BYTES": overhead,
        },
        session_manager=manager,
    )
    return app, entered, decoder


def test_video_upload_uses_unique_run_storage_and_same_job_manager(tmp_path, monkeypatch):
    app, entered, decoder = _app(tmp_path, monkeypatch)
    client = app.test_client()
    token = _token(client)
    response = client.post(
        "/api/videos",
        headers=_headers(token),
        data={"video": (io.BytesIO(b"safe video bytes"), "..\\room\\period one.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202, response.get_json()
    run_id = response.get_json()["run_id"]
    assert entered.wait(2)
    assert decoder.released is True
    manifest = app.extensions["visionclass_run_store"].read_manifest(run_id)
    assert manifest["status"] == "running"
    assert manifest["display_name"] == "roomperiod_one.mp4"
    input_path = app.extensions["visionclass_run_store"].run_dir(run_id) / "input.mp4"
    assert input_path.read_bytes() == b"safe video bytes"
    assert input_path.parent.parent == (tmp_path / "runs").resolve()
    assert response.get_json()["job"]["job_type"] == "video"

    cancel = client.post(f"/api/runs/{run_id}/cancel", headers=_headers(token))
    assert cancel.status_code == 202
    assert cancel.get_json()["stop_requested"] is True
    manager = app.extensions["visionclass_session_manager"]
    finished, terminal = manager.wait(response.get_json()["job"]["job_id"], timeout=2)
    assert finished and terminal["status"] == "cancelled"
    assert app.extensions["visionclass_run_store"].read_manifest(run_id)["status"] == "cancelled"


def test_upload_extension_payload_and_decoder_errors_are_safe(tmp_path, monkeypatch):
    app, _entered, _decoder = _app(tmp_path, monkeypatch, max_upload=8)
    client = app.test_client()
    token = _token(client)
    unsupported = client.post("/api/videos", headers=_headers(token), data={"video": (io.BytesIO(b"abc"), "file.exe")})
    assert unsupported.status_code == 400
    too_large = client.post("/api/videos", headers=_headers(token), data={"video": (io.BytesIO(b"123456789"), "file.mp4")})
    assert too_large.status_code == 413
    assert app.extensions["visionclass_run_store"].list_runs() == []

    app2, _entered, decoder = _app(tmp_path / "bad", monkeypatch, decodes=False)
    client2 = app2.test_client()
    invalid = client2.post("/api/videos", headers=_headers(_token(client2)), data={"video": (io.BytesIO(b"abc"), "bad.mov")})
    assert invalid.status_code == 400
    assert "could not decode" in invalid.get_json()["error"]
    assert decoder.released is True
    assert app2.extensions["visionclass_run_store"].list_runs() == []


def test_upload_accepts_exact_configured_video_payload_limit(tmp_path, monkeypatch):
    app, entered, _decoder = _app(tmp_path, monkeypatch, max_upload=4)
    client = app.test_client()
    response = client.post(
        "/api/videos",
        headers=_headers(_token(client)),
        data={"video": (io.BytesIO(b"1234"), "exact.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202, response.get_json()
    assert entered.wait(2)
    run_id = response.get_json()["run_id"]
    store = app.extensions["visionclass_run_store"]
    assert (store.run_dir(run_id) / "input.mp4").read_bytes() == b"1234"
    requested, _job = app.extensions["visionclass_session_manager"].request_stop()
    assert requested
    assert app.extensions["visionclass_session_manager"].wait(response.get_json()["job"]["job_id"], timeout=2)[0]


def test_multipart_overhead_cap_rejects_before_run_storage(tmp_path, monkeypatch):
    app, _entered, _decoder = _app(tmp_path, monkeypatch, max_upload=10, overhead=24)
    client = app.test_client()
    response = client.post(
        "/api/videos",
        headers=_headers(_token(client)),
        data={"video": (io.BytesIO(b"x" * 200), "too-big.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "multipart overhead" in response.get_json()["error"]
    assert not (tmp_path / "runs").exists()


def test_second_video_upload_cannot_start_while_webcam_owns_job(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    def webcam(context):
        entered.set()
        release.wait(2)
    manager = SessionManager({"webcam": webcam})
    app, _video_entered, _decoder = _app(tmp_path, monkeypatch, manager=manager)
    first = manager.start("webcam")
    assert entered.wait(2)
    client = app.test_client()
    response = client.post("/api/videos", headers=_headers(_token(client)), data={"video": (io.BytesIO(b"small"), "clip.mp4")})
    assert response.status_code == 409
    assert "already active" in response.get_json()["error"]
    assert app.extensions["visionclass_run_store"].list_runs() == []
    assert manager.snapshot()["job_id"] == first["job_id"]
    release.set()
    assert manager.wait(first["job_id"], timeout=2)[0]


def test_artifact_route_only_downloads_manifested_run_files(tmp_path):
    store = RunStore(tmp_path / "runs")
    run_id, input_path, _ = store.create_upload("clip.mp4", io.BytesIO(b"video bytes"), 20)
    store.write_manifest(run_id, {"status": "completed", "artifacts": {"source_video": input_path.name}})
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": store.root})
    client = app.test_client()
    response = client.get(f"/api/runs/{run_id}/artifacts/source_video")
    assert response.status_code == 200
    assert response.data == b"video bytes"
    assert response.headers["Content-Disposition"].startswith("attachment;")
    assert client.get(f"/api/runs/{run_id}/artifacts/README.md").status_code == 404
    assert client.get(f"/api/runs/../artifacts/source_video").status_code in (404, 308)


def test_completed_run_analytics_timeline_and_inline_playback_are_run_scoped(tmp_path):
    store = RunStore(tmp_path / "runs")
    run_id, input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"source"), 100)
    history = store.run_dir(run_id) / "behavior_history.csv"
    summary = store.run_dir(run_id) / "behavior_summary.csv"
    annotated = store.run_dir(run_id) / "annotated.mp4"
    report = store.run_dir(run_id) / "behavior_analytics_report.md"
    history.write_text(
        "frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause\n"
        "0,0.000,12,read,0.8000,false,\n",
        encoding="utf-8",
    )
    summary.write_text(
        "student_id,handrise,read,sleep,stand,using_electronic_devices,write,unknown\n"
        "12,0,0.04,0,0,0,0,0\n",
        encoding="utf-8",
    )
    annotated.write_bytes(b"annotated-video")
    report.write_text("report", encoding="utf-8")
    manifest = {
        "status": "completed",
        "output_video": {
            "validated": True,
            "container": "mp4",
            "codec": "H.264 (test)",
            "decoded_output_frames": 1,
            "output_bytes": len(b"annotated-video"),
        },
        "progress": {
            "processed_frames": 1,
            "total_frames": 1,
            "behavior_detection_total": 1,
            "history_record_total": 1,
        },
        "artifacts": {
            "source_video": input_path.name,
            "history_csv": history.name,
            "summary_csv": summary.name,
            "annotated_video": annotated.name,
            "analytics_report": report.name,
        },
    }
    store.write_manifest(run_id, manifest)
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": store.root})
    client = app.test_client()

    analytics = client.get(f"/api/runs/{run_id}/analytics")
    assert analytics.status_code == 200, analytics.get_json()
    bundle = analytics.get_json()
    assert bundle["processing"]["raw_behavior_box_detections"] == 1
    assert bundle["session"]["n_tracker_ids"] == 1
    assert bundle["summary_rows"][0]["tracker_id"] == 12
    assert "student_id" not in bundle["summary_rows"][0]
    timeline = client.get(f"/api/runs/{run_id}/timeline?tracker_id=12")
    assert timeline.status_code == 200
    assert timeline.get_json()["rows"][0]["behavior"] == "read"
    playback = client.get(f"/api/runs/{run_id}/playback")
    assert playback.status_code == 200
    assert playback.data == b"annotated-video"
    assert playback.mimetype == "video/mp4"
    assert playback.headers.get("Accept-Ranges") == "bytes"
    assert not playback.headers.get("Content-Disposition", "").startswith("attachment;")
    partial = client.get(f"/api/runs/{run_id}/playback", headers={"Range": "bytes=0-3"})
    assert partial.status_code == 206
    assert partial.data == b"anno"
    assert partial.headers["Content-Range"].startswith("bytes 0-3/")
    assert client.get(f"/api/runs/{run_id}/artifacts/analytics_report").headers["Content-Disposition"].startswith("attachment;")


def test_playback_rejects_incomplete_or_unvalidated_outputs(tmp_path):
    store = RunStore(tmp_path / "runs")
    run_id, _input_path, _ = store.create_upload("lesson.mp4", io.BytesIO(b"source"), 20)
    annotated = store.run_dir(run_id) / "annotated.mp4"
    annotated.write_bytes(b"not a video")
    store.write_manifest(run_id, {
        "status": "running",
        "output_video": {"validated": True},
        "artifacts": {"annotated_video": "annotated.mp4"},
    })
    client = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": store.root}).test_client()
    assert client.get(f"/api/runs/{run_id}/playback").status_code == 409

    store.write_manifest(run_id, {
        "status": "completed",
        "output_video": {"validated": False},
        "artifacts": {"annotated_video": "annotated.mp4"},
    })
    response = client.get(f"/api/runs/{run_id}/playback")
    assert response.status_code == 409
    assert "no validated annotated video" in response.get_json()["error"].lower()
