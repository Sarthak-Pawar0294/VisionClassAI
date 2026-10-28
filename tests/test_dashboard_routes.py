"""Template, browser-reconnect, and live stream route checks."""

import threading
import time

from src.session_manager import SessionManager
from src.video_jobs import RunStore
from src.web_app import DEFAULT_PORT, create_app


def _csrf(client):
    return client.get("/api/csrf", headers={"Host": f"127.0.0.1:{DEFAULT_PORT}"}).get_json()["csrf_token"]


def _mutation_headers(token):
    host = f"127.0.0.1:{DEFAULT_PORT}"
    return {"Host": host, "Origin": f"http://{host}", "X-CSRF-Token": token}


def test_dashboard_pages_render_real_navigation_and_accessible_controls(tmp_path):
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": tmp_path / "runs"})
    client = app.test_client()
    for path in ("/", "/live", "/video", "/analytics"):
        response = client.get(path)
        assert response.status_code in (200, 302)
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    live = client.get("/live").get_data(as_text=True)
    video = client.get("/video").get_data(as_text=True)
    analytics = client.get("/analytics").get_data(as_text=True)
    assert 'aria-label="Main navigation"' in live
    assert 'alt="Live camera frame annotated with behavior detections"' in live
    assert 'id="start-button"' in live and 'id="stop-button"' in live
    assert 'id="trend-window"' in live and 'id="chart-pause"' in live
    assert 'id="series-toggle-list"' in live and 'id="trend-chart"' in live
    assert "not a probability of understanding" in live.lower()
    assert "not ground truth" in live.lower()
    assert 'accept=".mp4,.avi,.mov,.mkv' in video
    assert 'id="cancel-button"' in video
    assert 'id="playback-status"' in video
    assert 'Tracker IDs are generated per video and do not identify unique real-world students.' in analytics
    assert "fonts.googleapis.com" not in live + video + analytics
    live_script = client.get("/static/js/live.js").get_data(as_text=True)
    video_script = client.get("/static/js/video.js").get_data(as_text=True)
    assert "/api/live/history?window=" in live_script
    assert 'svg.removeAttribute("hidden")' in live_script
    assert "graph is showing genuine zero counts" in live_script
    assert "Camera connected; waiting for its first frame and inference result." in live_script
    assert "this frame contains zero behavior detections" in live_script
    assert "Number(job.total_detections || 0).toLocaleString()" in live_script
    assert "Chart updates are paused. Webcam inference" in live_script
    assert "No new webcam frame has arrived for more than five seconds" in live_script
    assert 'async function getToken()' in live_script
    assert 'fetch("/api/csrf", {cache: "no-store"})' in live_script
    assert "if (csrfToken)" not in live_script
    assert live_script.count("await getToken()") == 2
    assert "run.output_video.validated === true" in video_script
    assert 'video.addEventListener("error"' in video_script
    assert "open it in a local media player" in video_script


def test_video_progress_supports_active_flat_and_completed_nested_manifests(tmp_path):
    run_store = RunStore(tmp_path / "runs")
    run_id = run_store.new_run_id()
    run_store.run_dir(run_id).mkdir(parents=True)
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "RUNS_DIR": run_store.root,
    })
    client = app.test_client()

    active_manifest = {
        "status": "running",
        "decoder": {"total_frames": 10_691},
        "processed_frames": 351,
        "total_frames": 10_691,
        "progress_percent": 3.28,
    }
    run_store.write_manifest(run_id, active_manifest)
    active_run = client.get(f"/api/runs/{run_id}").get_json()["run"]
    assert active_run["processed_frames"] == 351
    assert active_run["total_frames"] == 10_691
    assert "progress" not in active_run

    video_script = client.get("/static/js/video.js").get_data(as_text=True)
    assert "const progress = run.progress || run;" in video_script
    assert 'const percent = total && progress.processed_frames ?' in video_script
    assert '`${(progress.processed_frames || 0).toLocaleString()} of ${total.toLocaleString()} frames`' in video_script

    completed_manifest = {
        "status": "completed",
        "progress": {"processed_frames": 10_691, "total_frames": 10_691},
        "output_video": {"validated": True, "decoded_output_frames": 10_691},
        "artifacts": {"annotated_video": "annotated.mp4"},
    }
    run_store.write_manifest(run_id, completed_manifest)
    completed_run = client.get(f"/api/runs/{run_id}").get_json()["run"]
    assert completed_run["progress"] == {"processed_frames": 10_691, "total_frames": 10_691}
    assert "const completePercent = progress.total_frames ? 100 : null;" in video_script


def test_live_history_endpoint_returns_sampled_frame_metrics_and_validates_window():
    observed_at = time.time()
    published = threading.Event()

    def worker(context):
        context.report(
            camera_status="open",
            inference_status="running",
            frame_number=9,
            observed_at=observed_at,
            total_detections=2,
            class_counts={"handrise": 1, "read": 1},
            processing_fps=18.0,
            stream_jpeg=b"jpeg",
        )
        published.set()
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"}, session_manager=manager)
    client = app.test_client()
    started = manager.start()
    assert published.wait(2)
    history_response = client.get("/api/live/history?window=5")
    assert history_response.status_code == 200
    history = history_response.get_json()
    assert history_response.headers["Cache-Control"] == "no-store"
    assert history["window_minutes"] == 5
    assert history["sample_interval_seconds"] == 1.0
    assert history["points"][0]["frame_number"] == 9
    assert history["points"][0]["class_counts"] == {"handrise": 1, "read": 1}
    assert "not unique people" in history["metric_semantics"]
    stale = manager.live_history(5, now=observed_at + 6)
    assert stale["stale"] is True
    assert stale["data_freshness_seconds"] == 6
    assert client.get("/api/live/history?window=2").status_code == 400
    manager.request_stop()
    assert manager.wait(started["job_id"], timeout=2)[0]


def test_browser_refresh_reconnects_to_existing_job_and_stream_without_duplicate_worker(tmp_path):
    entered = threading.Event()
    worker_count = []

    def worker(context):
        worker_count.append(1)
        context.report(camera_status="open", inference_status="running", frame_number=3, stream_jpeg=b"jpeg")
        entered.set()
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": tmp_path / "runs"}, session_manager=manager)
    client = app.test_client()
    client.get("/live")
    token = _csrf(client)
    started = client.post("/api/session/start", headers=_mutation_headers(token))
    assert started.status_code == 202
    job_id = started.get_json()["job"]["job_id"]
    assert entered.wait(2)

    # A page refresh reads the existing state. It does not start another worker.
    client.get("/live")
    refreshed = client.get("/api/state").get_json()["job"]
    assert refreshed["job_id"] == job_id
    duplicate = client.post("/api/session/start", headers=_mutation_headers(token))
    assert duplicate.status_code == 409
    assert len(worker_count) == 1

    stream = client.get(f"/api/live/stream?job_id={job_id}", buffered=False)
    first_chunk = next(stream.response)
    assert b"Content-Type: image/jpeg" in first_chunk
    assert first_chunk.endswith(b"jpeg\r\n")
    stream.close()

    stopped = client.post("/api/session/stop", headers=_mutation_headers(token))
    assert stopped.status_code == 202
    assert stopped.get_json()["stop_requested"] is True
    done, terminal = manager.wait(job_id, timeout=2)
    assert done
    assert terminal["status"] == "cancelled"
    assert terminal["worker_alive"] is False
    assert len(worker_count) == 1


def test_runs_endpoint_is_empty_without_creating_storage(tmp_path):
    run_root = tmp_path / "runs"
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "RUNS_DIR": run_root})
    response = app.test_client().get("/api/runs")
    assert response.status_code == 200
    assert response.get_json() == {"runs": []}
    assert not run_root.exists()
