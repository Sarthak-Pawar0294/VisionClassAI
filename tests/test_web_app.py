"""API security, configuration, and worker integration tests for Flask."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import pytest
from werkzeug.serving import make_server

from src.session_manager import SessionManager
from src.web_app import (
    DEFAULT_MULTIPART_OVERHEAD_BYTES,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_PORT,
    LOOPBACK_HOST,
    create_app,
    serve_app,
)


def _headers(token, *, host=f"127.0.0.1:{DEFAULT_PORT}", origin=None):
    return {
        "Host": host,
        "Origin": origin or f"http://{host}",
        "X-CSRF-Token": token,
    }


def _token(client, *, host=f"127.0.0.1:{DEFAULT_PORT}"):
    response = client.get("/api/csrf", headers={"Host": host})
    assert response.status_code == 200
    return response.get_json()["csrf_token"]


def test_app_defaults_to_loopback_port_and_allows_environment_override(monkeypatch):
    monkeypatch.delenv("VISIONCLASS_APP_PORT", raising=False)
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    assert app.config["APP_PORT"] == 8766
    assert app.config["DEBUG"] is False
    assert app.config["MAX_UPLOAD_BYTES"] == 1024 ** 3 == DEFAULT_MAX_UPLOAD_BYTES
    assert app.config["MAX_MULTIPART_OVERHEAD_BYTES"] == 1024 ** 2 == DEFAULT_MULTIPART_OVERHEAD_BYTES
    assert app.config["MAX_CONTENT_LENGTH"] == 1024 ** 3 + 1024 ** 2

    monkeypatch.setenv("VISIONCLASS_APP_PORT", "8877")
    overridden = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    assert overridden.config["APP_PORT"] == 8877


@pytest.mark.parametrize("value", ["bad", "0", "65536"])
def test_invalid_port_configuration_fails_clearly(monkeypatch, value):
    monkeypatch.setenv("VISIONCLASS_APP_PORT", value)
    if value == "bad":
        with pytest.raises(ValueError, match="VISIONCLASS_APP_PORT"):
            create_app()
    else:
        with pytest.raises(ValueError, match="APP_PORT"):
            create_app({"APP_PORT": int(value)})


def test_server_socket_factory_is_called_with_loopback_only_and_debug_disabled():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    server = make_server(LOOPBACK_HOST, 0, app, threaded=True)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert 0 < server.server_address[1] < 65536
        assert app.config["DEBUG"] is False
    finally:
        server.server_close()


def test_serve_app_reaches_health_route_on_actual_ephemeral_loopback_socket():
    manager = SessionManager({"webcam": lambda context: pytest.fail("health must not start a worker")})
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"}, session_manager=manager)
    server_ready = threading.Event()
    servers = []
    results = []

    def actual_ephemeral_server(host, configured_port, app_object, **kwargs):
        assert host == "127.0.0.1"
        assert configured_port == DEFAULT_PORT
        server = make_server(host, 0, app_object, **kwargs)
        servers.append(server)
        server_ready.set()
        return server

    thread = threading.Thread(
        target=lambda: results.append(
            serve_app(app, server_factory=actual_ephemeral_server, manager=manager)
        ),
        daemon=True,
    )
    thread.start()
    assert server_ready.wait(timeout=2)
    server = servers[0]
    try:
        address, port = server.server_address
        assert address == "127.0.0.1"
        with urlopen(f"http://{address}:{port}/api/health", timeout=2) as response:
            assert response.status == 200
            assert response.read()
    finally:
        server.shutdown()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert results == [0]
    assert manager.snapshot()["status"] == "idle"


def test_csrf_bootstrap_sets_session_bound_strict_cookie():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"})
    client = app.test_client()
    response = client.get("/api/csrf", headers={"Host": "127.0.0.1:8766"})
    cookie = response.headers["Set-Cookie"]

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert response.get_json()["csrf_token"]


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "127.0.0.1:8766", "Origin": "http://127.0.0.1:8766"},
        _headers("wrong-token"),
        _headers("token", host="localhost:8766"),
        _headers("token", origin="http://localhost:8766"),
    ],
    ids=["missing-token", "invalid-token", "foreign-host", "foreign-origin"],
)
def test_invalid_mutation_security_headers_do_not_change_state(headers):
    manager = SessionManager({"webcam": lambda context: None})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    client = app.test_client()
    _token(client)
    response = client.post("/api/session/start", headers=headers)

    assert response.status_code == 403
    assert manager.snapshot()["status"] == "idle"


def test_csrf_token_from_another_browser_session_is_rejected():
    manager = SessionManager({"webcam": lambda context: None})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    client_one = app.test_client()
    client_two = app.test_client()
    token_one = _token(client_one)
    _token(client_two)

    response = client_two.post(
        "/api/session/start",
        headers=_headers(token_one),
    )
    assert response.status_code == 403
    assert manager.snapshot()["status"] == "idle"


def test_csrf_bootstrap_recovers_after_application_secret_rotates():
    entered = threading.Event()

    def worker(context):
        entered.set()
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "first-process-secret"},
        session_manager=manager,
    )
    client = app.test_client()
    stale_token = _token(client)

    # The default application key is process-local, so a still-open page can
    # retain a token whose signed browser session is invalid after a restart.
    app.config["SECRET_KEY"] = "restarted-process-secret"
    rejected = client.post("/api/session/start", headers=_headers(stale_token))
    assert rejected.status_code == 403
    assert manager.snapshot()["status"] == "idle"

    fresh_token = _token(client)
    assert fresh_token != stale_token
    started = client.post("/api/session/start", headers=_headers(fresh_token))
    assert started.status_code == 202
    job_id = started.get_json()["job"]["job_id"]
    assert entered.wait(timeout=2)

    requested, _snapshot = manager.request_stop()
    assert requested
    finished, terminal = manager.wait(job_id, timeout=2)
    assert finished
    assert terminal["status"] == "cancelled"


def test_start_and_stop_routes_keep_slot_until_cleanup_finishes():
    entered = threading.Event()
    finish_worker = threading.Event()
    cleanup_done = threading.Event()

    def worker(context):
        context.register_cleanup(cleanup_done.set)
        entered.set()
        finish_worker.wait(2)

    manager = SessionManager({"webcam": worker})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    client = app.test_client()
    token = _token(client)
    headers = _headers(token)

    started = client.post("/api/session/start", headers=headers)
    assert started.status_code == 202
    job_id = started.get_json()["job"]["job_id"]
    assert entered.wait(timeout=2)

    began = time.perf_counter()
    stopping = client.post("/api/session/stop", headers=headers)
    elapsed = time.perf_counter() - began
    assert stopping.status_code == 202
    assert stopping.get_json()["stop_requested"] is True
    assert stopping.get_json()["job"]["status"] == "stop_requested"
    assert elapsed < 1.0
    assert cleanup_done.is_set() is False

    duplicate = client.post("/api/session/start", headers=headers)
    assert duplicate.status_code == 409
    assert manager.snapshot()["job_id"] == job_id

    finish_worker.set()
    finished, terminal = manager.wait(job_id, timeout=2)
    assert finished
    assert cleanup_done.is_set()
    assert terminal["status"] == "cancelled"
    assert client.get("/api/health").status_code == 200


def test_competing_http_start_requests_launch_exactly_one_worker():
    entered = threading.Event()
    release_worker = threading.Event()
    calls = []

    def worker(context):
        calls.append(context)
        entered.set()
        release_worker.wait(3)

    manager = SessionManager({"webcam": worker})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    clients = [app.test_client(), app.test_client()]
    tokens = [_token(client) for client in clients]
    barrier = threading.Barrier(3)

    def start(client, token):
        barrier.wait(timeout=2)
        return client.post("/api/session/start", headers=_headers(token)).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(start, client, token) for client, token in zip(clients, tokens)]
        barrier.wait(timeout=2)
        statuses = [future.result(timeout=3) for future in futures]

    assert sorted(statuses) == [202, 409]
    assert entered.wait(timeout=2)
    assert len(calls) == 1
    requested, snapshot = manager.request_stop()
    assert requested
    release_worker.set()
    finished, terminal = manager.wait(snapshot["job_id"], timeout=2)
    assert finished
    assert terminal["status"] == "cancelled"


def test_worker_failure_is_visible_while_flask_remains_available():
    calls = []
    retry_started = threading.Event()

    def worker(context):
        calls.append(context)
        if len(calls) == 1:
            raise RuntimeError("mock camera startup failed")
        retry_started.set()
        context.stop_event.wait(2)

    manager = SessionManager({"webcam": worker})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    client = app.test_client()
    token = _token(client)
    started = client.post("/api/session/start", headers=_headers(token))
    job_id = started.get_json()["job"]["job_id"]

    finished, terminal = manager.wait(job_id, timeout=2)
    assert finished
    assert terminal["status"] == "error"
    assert "mock camera startup failed" in terminal["error"]
    assert client.get("/api/health").get_json()["status"] == "ok"
    assert client.get("/api/state").get_json()["job"]["status"] == "error"

    retry = client.post("/api/session/start", headers=_headers(token))
    assert retry.status_code == 202
    assert retry_started.wait(timeout=2)
    assert len(calls) == 2
    retry_job_id = retry.get_json()["job"]["job_id"]
    client.post("/api/session/stop", headers=_headers(token))
    finished, retry_terminal = manager.wait(retry_job_id, timeout=2)
    assert finished
    assert retry_terminal["status"] == "cancelled"


def test_start_route_rejects_bodies_and_does_not_accept_local_paths():
    calls = []
    manager = SessionManager({"webcam": lambda context: calls.append(context)})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )
    client = app.test_client()
    token = _token(client)
    response = client.post(
        "/api/session/start",
        json={"path": "C:/private/video.mp4"},
        headers=_headers(token),
    )
    assert response.status_code == 400
    assert calls == []
    assert client.get("/private/file").status_code == 404


def test_server_bind_failure_is_clear_and_does_not_start_a_worker(capsys):
    worker_calls = []
    manager = SessionManager({"webcam": lambda context: worker_calls.append(context)})
    app = create_app(
        {"TESTING": True, "SECRET_KEY": "test-secret"},
        session_manager=manager,
    )

    def fail_bind(host, port, app_object, **kwargs):
        assert host == "127.0.0.1"
        assert port == DEFAULT_PORT
        assert kwargs == {"threaded": True}
        raise OSError("port already in use")

    assert serve_app(app, server_factory=fail_bind, manager=manager) == 1
    assert "127.0.0.1:8766" in capsys.readouterr().err
    assert worker_calls == []
    assert manager.snapshot()["status"] == "idle"


def test_startup_disables_reloader_and_closes_server_on_interrupt():
    manager = SessionManager({"webcam": lambda context: None})
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret"}, session_manager=manager)
    calls = []

    class FakeServer:
        def serve_forever(self):
            raise KeyboardInterrupt()

        def server_close(self):
            calls.append("closed")

    def server_factory(host, port, app_object, **kwargs):
        calls.append((host, port, kwargs, app_object.config["DEBUG"]))
        return FakeServer()

    assert serve_app(app, server_factory=server_factory, manager=manager) == 0
    assert calls == [("127.0.0.1", DEFAULT_PORT, {"threaded": True}, False), "closed"]
