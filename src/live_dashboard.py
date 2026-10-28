"""Local live dashboard server for webcam inference."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


BEHAVIOR_CLASSES = (
    "handrise",
    "read",
    "sleep",
    "stand",
    "using_electronic_devices",
    "write",
)

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VisionClass Live</title>
  <style>
    :root { color-scheme: dark; font: 15px/1.5 system-ui, sans-serif; background: #0b1020; color: #e5e7eb; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; }
    main { max-width: 1440px; margin: auto; }
    h1 { margin: 0; font-size: 1.7rem; }
    .subtle { color: #9ca3af; }
    .top { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 20px; }
    .pill { border-radius: 999px; padding: 5px 12px; background: #263244; color: #cbd5e1; }
    .pill.ok { background: #064e3b; color: #a7f3d0; }
    .pill.error { background: #7f1d1d; color: #fecaca; }
    .layout { display: grid; grid-template-columns: minmax(0, 2fr) minmax(300px, 1fr); gap: 18px; }
    section { background: #151c2e; border: 1px solid #29344a; border-radius: 14px; padding: 16px; }
    #feed { display: block; width: 100%; min-height: 240px; max-height: 75vh; object-fit: contain; background: #050811; border-radius: 8px; }
    #feed-status { margin: 8px 0 0; min-height: 24px; }
    .metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
    .metric { background: #0d1424; border-radius: 10px; padding: 12px; }
    .metric span { display: block; color: #9ca3af; font-size: .82rem; }
    .metric strong { font-size: 1.25rem; }
    .counts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .count { display: flex; justify-content: space-between; gap: 8px; padding: 8px 10px; border-radius: 8px; background: #0d1424; }
    .count span:last-child { font-weight: 700; color: #93c5fd; }
    #detections { list-style: none; padding: 0; margin: 12px 0 0; }
    #detections li { padding: 8px 0; border-bottom: 1px solid #29344a; }
    #error { color: #fca5a5; min-height: 1.5em; white-space: pre-wrap; }
    @media (max-width: 850px) { body { padding: 12px; } .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <header class="top">
    <div><h1>VisionClass Live</h1><div class="subtle">Live webcam inference · Phase 6 behavior model</div></div>
    <div id="connection" class="pill">Connecting…</div>
  </header>
  <div class="layout">
    <section>
      <h2>Annotated camera feed</h2>
      <img id="feed" src="/stream.mjpg" alt="Waiting for live webcam frame">
      <div id="feed-status" class="subtle">Waiting for the camera and first inference frame…</div>
      <div id="error" role="alert"></div>
    <button id="stop" type="button">Stop webcam session</button>
  </section>
    <section>
      <h2>Current frame</h2>
      <div class="metrics">
        <div class="metric"><span>Processing rate</span><strong id="fps">—</strong></div>
        <div class="metric"><span>Detections this frame</span><strong id="total">0</strong></div>
        <div class="metric"><span>Camera</span><strong id="camera">Starting</strong></div>
        <div class="metric"><span>Inference</span><strong id="inference">Starting</strong></div>
      </div>
      <h3>Behavior counts in this frame</h3>
      <div id="counts" class="counts"></div>
      <h3>Current labels and confidence</h3>
      <ul id="detections"><li class="subtle">Waiting for detections…</li></ul>
    </section>
  </div>
</main>
<script>
const classes = ["handrise", "read", "sleep", "stand", "using_electronic_devices", "write"];
const counts = document.getElementById("counts");
counts.innerHTML = classes.map(name => `<div class="count"><span>${name}</span><span id="count-${name}">0</span></div>`).join("");
const connection = document.getElementById("connection");
const feed = document.getElementById("feed");
document.getElementById("stop").addEventListener("click", async () => {
  const button = document.getElementById("stop");
  button.disabled = true;
  button.textContent = "Stopping…";
  try {
    await fetch("/api/stop", {method: "POST"});
  } catch (error) {
    button.disabled = false;
    button.textContent = "Stop webcam session";
    document.getElementById("error").textContent = `Could not stop session: ${error}`;
  }
});
feed.onerror = () => {
  document.getElementById("feed-status").textContent = "Live frame stream disconnected; reconnecting…";
  setTimeout(() => { feed.src = "/stream.mjpg?retry=" + Date.now(); }, 1500);
};
async function refreshState() {
  try {
    const response = await fetch("/api/state", {cache: "no-store"});
    if (!response.ok) throw new Error(`Dashboard returned HTTP ${response.status}`);
    const state = await response.json();
    connection.textContent = "Dashboard connected";
    connection.className = "pill ok";
    document.getElementById("camera").textContent = state.camera_status;
    document.getElementById("inference").textContent = state.inference_status;
    document.getElementById("fps").textContent = state.processing_fps == null ? "—" : `${state.processing_fps.toFixed(1)} FPS`;
    document.getElementById("total").textContent = state.total_detections;
    for (const name of classes) document.getElementById(`count-${name}`).textContent = state.class_counts[name] ?? 0;
    const list = document.getElementById("detections");
    list.replaceChildren();
    if (!state.detections.length) {
      const item = document.createElement("li");
      item.className = "subtle";
      item.textContent = "No detections in the current frame.";
      list.appendChild(item);
    } else {
      for (const detection of state.detections) {
        const item = document.createElement("li");
        item.textContent = `${detection.class_name} — ${(detection.confidence * 100).toFixed(1)}%`;
        list.appendChild(item);
      }
    }
    document.getElementById("error").textContent = state.error || "";
    document.getElementById("feed-status").textContent =
      state.frame_number ? `Live · frame ${state.frame_number}` : "Waiting for the first inference frame…";
  } catch (error) {
    connection.textContent = "Dashboard unavailable";
    connection.className = "pill error";
    document.getElementById("feed-status").textContent = "Cannot reach the local dashboard server.";
    document.getElementById("error").textContent = String(error);
  } finally {
    setTimeout(refreshState, 1000);
  }
}
refreshState();
</script>
</body>
</html>
"""


class LiveDashboardState:
    """Thread-safe latest-frame and current-frame metrics shared with HTTP clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._frame_jpeg: bytes | None = None
        self._sequence = 0
        self._closed = False
        self._payload: dict[str, Any] = {
            "camera_status": "starting",
            "inference_status": "starting",
            "processing_fps": None,
            "frame_number": 0,
            "total_detections": 0,
            "class_counts": {name: 0 for name in BEHAVIOR_CLASSES},
            "detections": [],
            "error": None,
        }

    def set_status(
        self,
        *,
        camera_status: str | None = None,
        inference_status: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._condition:
            if camera_status is not None:
                self._payload["camera_status"] = camera_status
            if inference_status is not None:
                self._payload["inference_status"] = inference_status
            if error is not None:
                self._payload["error"] = error
            self._condition.notify_all()

    def publish(
        self,
        frame_jpeg: bytes,
        detections: list[dict[str, Any]],
        processing_fps: float,
        frame_number: int,
    ) -> None:
        class_counts = {name: 0 for name in BEHAVIOR_CLASSES}
        safe_detections = []
        for detection in detections:
            class_name = detection["class_name"]
            if class_name not in class_counts:
                continue
            confidence = float(detection["confidence"])
            class_counts[class_name] += 1
            safe_detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                }
            )

        with self._condition:
            if self._closed:
                return
            self._frame_jpeg = frame_jpeg
            self._sequence += 1
            self._payload.update(
                camera_status="open",
                inference_status="running",
                processing_fps=float(processing_fps),
                frame_number=int(frame_number),
                total_detections=len(safe_detections),
                class_counts=class_counts,
                detections=safe_detections,
                error=None,
            )
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._payload,
                "class_counts": dict(self._payload["class_counts"]),
                "detections": list(self._payload["detections"]),
            }

    def next_frame(self, after_sequence: int, timeout: float = 10.0) -> tuple[int, bytes] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence or self._closed,
                timeout=timeout,
            )
            if self._closed or self._frame_jpeg is None or self._sequence <= after_sequence:
                return None
            return self._sequence, self._frame_jpeg

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()


class _LocalThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class LiveDashboardServer:
    """Local-only dashboard server with a single threaded inference producer."""

    def __init__(self, state: LiveDashboardState, port: int = 8765) -> None:
        self.state = state
        self._server = _LocalThreadingHTTPServer(
            ("127.0.0.1", port),
            self._handler_type(state),
        )
        try:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="visionclass-live-dashboard",
                daemon=True,
            )
        except Exception:
            self.state.close()
            self._server.server_close()
            raise
        self._started = False
        self._stopped = False
        self._lifecycle_lock = threading.Lock()

    @staticmethod
    def _handler_type(state: LiveDashboardState):

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path.split("?", 1)[0] != "/api/stop":
                    self.send_error(404)
                    return
                state.request_stop()
                body = b'{"stop_requested":true}'
                self.send_response(202)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.split("?", 1)[0] == "/":
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.split("?", 1)[0] == "/api/state":
                    body = json.dumps(state.snapshot()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.split("?", 1)[0] == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    sequence = 0
                    try:
                        while True:
                            next_frame = state.next_frame(sequence)
                            if next_frame is None:
                                if state.snapshot()["inference_status"] == "stopped":
                                    break
                                continue
                            sequence, frame = next_frame
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                                + frame
                                + b"\r\n"
                            )
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    return

                self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> str:
        with self._lifecycle_lock:
            if self._stopped:
                raise RuntimeError("VisionClass live dashboard server has already stopped")
            if self._started:
                if self._thread.is_alive():
                    return self.url
                raise RuntimeError("VisionClass live dashboard server thread is not running")
            try:
                self._thread.start()
            except Exception:
                self._stopped = True
                self.state.close()
                self._server.server_close()
                raise
            self._started = True
            return self.url

    def stop(self) -> None:
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._stopped = True
            self.state.set_status(inference_status="stopped")
            self.state.close()
            try:
                if self._thread.is_alive():
                    self._server.shutdown()
                    self._thread.join(timeout=5)
                    if self._thread.is_alive():
                        raise RuntimeError("VisionClass live dashboard server did not stop within 5 seconds")
            finally:
                self._server.server_close()
