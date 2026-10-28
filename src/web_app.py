"""Flask localhost foundation for the expanded VisionClass application."""

from __future__ import annotations

import os
import secrets
import sys
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.serving import make_server
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from src.realtime_behavior import webcam_session_worker
from src.config import PROJECT_ROOT
from src.session_manager import (
    ActiveJobConflict,
    SessionManager,
    WorkerStartError,
    WorkerUnavailable,
)
from src.video_jobs import RunStore, UploadValidationError, analytics_bundle, make_video_worker


LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_MAX_UPLOAD_BYTES = 1024 ** 3
DEFAULT_MULTIPART_OVERHEAD_BYTES = 1024 ** 2
CSRF_SESSION_KEY = "_visionclass_csrf_token"
CSRF_HEADER = "X-CSRF-Token"


def _environment_port() -> int:
    value = os.environ.get("VISIONCLASS_APP_PORT", str(DEFAULT_PORT))
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("VISIONCLASS_APP_PORT must be an integer from 1 to 65535.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("VISIONCLASS_APP_PORT must be an integer from 1 to 65535.")
    return port


def create_app(
    config: dict[str, Any] | None = None,
    *,
    session_manager: SessionManager | None = None,
    webcam_worker: Callable | None = None,
) -> Flask:
    """Create a testable application without binding a socket or starting a job."""
    app = Flask(__name__)
    app.config.from_mapping(
        APP_PORT=_environment_port(),
        DEBUG=False,
        TESTING=False,
        SECRET_KEY=secrets.token_urlsafe(48),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        SESSION_COOKIE_SECURE=False,
        MAX_UPLOAD_BYTES=int(os.environ.get("VISIONCLASS_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
        MAX_MULTIPART_OVERHEAD_BYTES=int(
            os.environ.get("VISIONCLASS_MAX_MULTIPART_OVERHEAD_BYTES", DEFAULT_MULTIPART_OVERHEAD_BYTES)
        ),
        RUNS_DIR=Path(
            os.environ.get("VISIONCLASS_RUNS_DIR", PROJECT_ROOT / "data" / "visionclass_runs")
        ),
    )
    if config:
        app.config.update(config)
    try:
        app.config["APP_PORT"] = int(app.config["APP_PORT"])
    except (TypeError, ValueError) as exc:
        raise ValueError("APP_PORT must be an integer from 1 to 65535.") from exc
    if not 1 <= app.config["APP_PORT"] <= 65535:
        raise ValueError("APP_PORT must be an integer from 1 to 65535.")
    if not app.config.get("SECRET_KEY"):
        raise ValueError("A non-empty Flask SECRET_KEY is required.")
    for limit_key in ("MAX_UPLOAD_BYTES", "MAX_MULTIPART_OVERHEAD_BYTES"):
        try:
            app.config[limit_key] = int(app.config[limit_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{limit_key} must be a positive integer.") from exc
        if app.config[limit_key] <= 0:
            raise ValueError(f"{limit_key} must be a positive integer.")
    app.config["RUNS_DIR"] = Path(app.config["RUNS_DIR"]).expanduser().resolve()
    app.config["MAX_CONTENT_LENGTH"] = (
        app.config["MAX_UPLOAD_BYTES"] + app.config["MAX_MULTIPART_OVERHEAD_BYTES"]
    )

    if session_manager is None:
        session_manager = SessionManager(
            {"webcam": webcam_worker or webcam_session_worker}
        )
    app.extensions["visionclass_session_manager"] = session_manager
    run_store = RunStore(app.config["RUNS_DIR"])
    app.extensions["visionclass_run_store"] = run_store
    expected_host = f"{LOOPBACK_HOST}:{app.config['APP_PORT']}"
    expected_origin = f"http://{expected_host}"

    @app.before_request
    def validate_browser_mutation():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        if request.host != expected_host:
            return jsonify(error="Request Host does not match the local VisionClass address."), 403
        if request.headers.get("Origin") != expected_origin:
            return jsonify(error="Request Origin does not match the local VisionClass address."), 403
        expected_token = session.get(CSRF_SESSION_KEY)
        supplied_token = request.headers.get(CSRF_HEADER, "")
        if not expected_token or not secrets.compare_digest(expected_token, supplied_token):
            return jsonify(error="Missing or invalid session security token."), 403
        return None

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", host=LOOPBACK_HOST, port=app.config["APP_PORT"])

    @app.get("/")
    def index():
        return redirect(url_for("live_page"))

    @app.get("/live")
    def live_page():
        return render_template("live.html")

    @app.get("/video")
    def video_page():
        return render_template("video.html")

    @app.get("/analytics")
    def analytics_page():
        return render_template("analytics.html")

    @app.get("/api/csrf")
    def csrf_token():
        token = session.get(CSRF_SESSION_KEY)
        if not token:
            token = secrets.token_urlsafe(32)
            session[CSRF_SESSION_KEY] = token
        response = jsonify(csrf_token=token)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/state")
    def state():
        response = jsonify(server_status="available", job=session_manager.snapshot())
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/live/history")
    def live_history():
        try:
            window_minutes = int(request.args.get("window", "1"))
            history = session_manager.live_history(window_minutes)
        except (TypeError, ValueError):
            return jsonify(error="Choose a live history window of 1, 5, or 10 minutes."), 400
        response = jsonify(history)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/live/stream")
    def live_stream():
        job_id = request.args.get("job_id", "")
        initial = session_manager.snapshot()
        if not job_id or initial.get("job_id") != job_id or initial.get("job_type") != "webcam":
            return jsonify(error="No matching live webcam stream is active."), 404

        def frames():
            sequence = 0
            while True:
                item = session_manager.next_frame(sequence, timeout=2.0, job_id=job_id)
                if item is None:
                    current = session_manager.snapshot()
                    if current.get("job_id") != job_id or not current.get("worker_alive"):
                        break
                    continue
                sequence, jpeg = item
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                    + jpeg
                    + b"\r\n"
                )

        return Response(
            frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    def has_request_body() -> bool:
        return bool(request.content_length) or bool(request.headers.get("Transfer-Encoding"))

    @app.post("/api/session/start")
    def start_session():
        if has_request_body():
            return jsonify(error="Session start does not accept a request body."), 400
        try:
            snapshot = session_manager.start("webcam")
        except ActiveJobConflict as exc:
            return jsonify(error=str(exc), job=exc.snapshot), 409
        except WorkerUnavailable as exc:
            return jsonify(error=str(exc)), 503
        except WorkerStartError as exc:
            return jsonify(error=str(exc), job=exc.snapshot), 503
        return jsonify(job=snapshot), 202

    @app.post("/api/session/stop")
    def stop_session():
        if has_request_body():
            return jsonify(error="Session stop does not accept a request body."), 400
        requested, snapshot = session_manager.request_stop()
        if not requested:
            return jsonify(error="There is no active VisionClass job to stop.", job=snapshot), 409
        response = jsonify(stop_requested=True, job=snapshot)
        response.status_code = 202
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/api/videos")
    def upload_video():
        if request.content_length is not None and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            return jsonify(error="Upload exceeds the configured video payload plus multipart overhead limit."), 413
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify(error="Choose a video file to upload."), 400
        display_name = secure_filename(upload.filename)[:160] or "video"
        try:
            run_id, input_path, upload_bytes = run_store.create_upload(
                display_name,
                upload.stream,
                app.config["MAX_UPLOAD_BYTES"],
            )
            try:
                decoder_info = run_store.validate_decoder(run_id, input_path)
            except Exception:
                run_store.remove_new_run(run_id)
                raise
            run_store.write_manifest(run_id, {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "display_name": display_name,
                "status": "queued",
                "upload_bytes": upload_bytes,
                "decoder": decoder_info,
                "artifacts": {},
                "optional_errors": [],
                "error": None,
            })
            try:
                snapshot = session_manager.start(
                    "video",
                    worker=make_video_worker(run_store, run_id, input_path),
                    run_id=run_id,
                )
            except ActiveJobConflict as exc:
                run_store.remove_new_run(run_id)
                return jsonify(error=str(exc), job=exc.snapshot), 409
            except (WorkerUnavailable, WorkerStartError) as exc:
                manifest = run_store.read_manifest(run_id) or {"run_id": run_id}
                manifest.update(status="error", error=str(exc), artifacts={})
                run_store.write_manifest(run_id, manifest)
                return jsonify(error=str(exc), run_id=run_id), 503
        except UploadValidationError as exc:
            return jsonify(error=str(exc)), exc.status_code
        except OSError as exc:
            print(f"ERROR: Could not store uploaded video: {exc}", file=sys.stderr)
            return jsonify(error="Could not store the uploaded video. Check available disk space and folder permissions."), 507
        return jsonify(run_id=run_id, job=snapshot), 202

    @app.post("/api/runs/<run_id>/cancel")
    def cancel_video(run_id: str):
        if not run_store.read_manifest(run_id):
            return jsonify(error="Run not found."), 404
        snapshot = session_manager.snapshot()
        if snapshot.get("run_id") != run_id or snapshot.get("job_type") != "video":
            return jsonify(error="This video run is not the active job.", job=snapshot), 409
        requested, job = session_manager.request_stop()
        if not requested:
            return jsonify(error="Video job is no longer active.", job=job), 409
        return jsonify(stop_requested=True, job=job), 202

    @app.get("/api/runs")
    def list_runs():
        fields = ("run_id", "created_at", "display_name", "status", "progress", "optional_errors", "error", "artifacts")
        return jsonify(runs=[{key: item.get(key) for key in fields} for item in run_store.list_runs()])

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str):
        manifest = run_store.read_manifest(run_id)
        if not manifest:
            return jsonify(error="Run not found."), 404
        return jsonify(run=manifest)

    @app.get("/api/runs/<run_id>/analytics")
    def run_analytics(run_id: str):
        manifest = run_store.read_manifest(run_id)
        if not manifest:
            return jsonify(error="Run not found."), 404
        if manifest.get("status") != "completed":
            return jsonify(error="Analytics are available after a video run completes."), 409
        try:
            history = run_store.artifact_path(run_id, "history_csv")
            summary = run_store.artifact_path(run_id, "summary_csv")
            bundle = analytics_bundle(history, summary)
            progress = manifest.get("progress") or {}
            bundle["processing"] = {
                "decoded_frames_processed": progress.get("processed_frames"),
                "decoded_frame_total": progress.get("total_frames"),
                "raw_behavior_box_detections": progress.get("behavior_detection_total"),
                "per_frame_tracker_records": progress.get("history_record_total"),
            }
            return jsonify(bundle)
        except UploadValidationError as exc:
            return jsonify(error=str(exc)), exc.status_code
        except (OSError, ValueError, KeyError) as exc:
            return jsonify(error=f"Could not read this run's analytics: {exc}"), 422

    @app.get("/api/runs/<run_id>/timeline")
    def run_timeline(run_id: str):
        try:
            history = run_store.artifact_path(run_id, "history_csv")
            tracker_id = int(request.args.get("tracker_id", ""))
            limit = min(500, max(1, int(request.args.get("limit", "100"))))
            offset = max(0, int(request.args.get("offset", "0")))
        except (UploadValidationError, ValueError) as exc:
            if isinstance(exc, UploadValidationError):
                return jsonify(error=str(exc)), exc.status_code
            return jsonify(error="Provide a valid tracker_id, limit, and offset."), 400
        rows = []
        matched = 0
        with history.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if int(row["student_id"]) != tracker_id:
                    continue
                if matched >= offset and len(rows) < limit:
                    rows.append({
                        "frame": int(row["frame"]),
                        "time_seconds": float(row["time_seconds"]),
                        "behavior": row["behavior"],
                        "confidence": float(row["confidence"]),
                        "is_cached": row["is_cached"].lower() == "true",
                    })
                matched += 1
        return jsonify(tracker_id=tracker_id, offset=offset, limit=limit, rows=rows, total_records=matched)

    @app.get("/api/runs/<run_id>/artifacts/<artifact_name>")
    def download_artifact(run_id: str, artifact_name: str):
        try:
            path = run_store.artifact_path(run_id, artifact_name)
        except UploadValidationError as exc:
            return jsonify(error=str(exc)), exc.status_code
        return send_file(path, as_attachment=True, download_name=path.name, conditional=True)

    @app.get("/api/runs/<run_id>/playback")
    def playback_video(run_id: str):
        manifest = run_store.read_manifest(run_id)
        if not manifest:
            return jsonify(error="Run not found."), 404
        if manifest.get("status") != "completed":
            return jsonify(error="Annotated playback is available only after output finalization."), 409
        validation = manifest.get("output_video") or {}
        if validation.get("validated") is not True:
            return jsonify(error="This run has no validated annotated video output. Reprocess the video and check its output status."), 409
        try:
            path = run_store.artifact_path(run_id, "annotated_video")
        except UploadValidationError as exc:
            return jsonify(error=str(exc)), exc.status_code
        response = send_file(path, mimetype="video/mp4", as_attachment=False, conditional=True)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return jsonify(error="Upload exceeds the configured video payload plus multipart overhead limit."), 413

    @app.after_request
    def local_security_headers(response):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; media-src 'self'; "
            "script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; "
            "base-uri 'self'; form-action 'self'; frame-ancestors 'none'; object-src 'none'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response

    return app


def serve_app(
    app: Flask,
    *,
    server_factory: Callable[..., Any] = make_server,
    manager: SessionManager | None = None,
) -> int:
    """Bind only to loopback, then serve until interrupted or the server fails."""
    if manager is None:
        manager = app.extensions["visionclass_session_manager"]
    port = int(app.config["APP_PORT"])
    try:
        server = server_factory(LOOPBACK_HOST, port, app, threaded=True)
    except OSError as exc:
        print(
            f"ERROR: Could not start VisionClass at http://{LOOPBACK_HOST}:{port}/: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"ERROR: Could not configure VisionClass at http://{LOOPBACK_HOST}:{port}/: {exc}",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    print(f"VisionClass is available at http://{LOOPBACK_HOST}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: VisionClass web server stopped unexpectedly: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        try:
            server.server_close()
        except Exception as exc:
            print(f"ERROR: Could not close VisionClass web server: {exc}", file=sys.stderr)
            exit_code = 1
        if not manager.shutdown(timeout=5.0):
            print("ERROR: VisionClass worker did not stop within 5 seconds.", file=sys.stderr)
            exit_code = 1
    return exit_code


def main() -> int:
    try:
        app = create_app()
    except ValueError as exc:
        print(f"ERROR: Could not configure VisionClass: {exc}", file=sys.stderr)
        return 1
    return serve_app(app)


if __name__ == "__main__":
    raise SystemExit(main())
