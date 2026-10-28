"""Thread-safe ownership and lifecycle management for local analysis jobs."""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping
from uuid import uuid4


class JobStatus(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


TERMINAL_STATUSES = frozenset(
    {JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.ERROR}
)

LIVE_HISTORY_MAX_SAMPLES = 600
LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS = 1.0
LIVE_HISTORY_STALE_AFTER_SECONDS = 5.0

_ALLOWED_TRANSITIONS = {
    JobStatus.STARTING: frozenset(
        {
            JobStatus.RUNNING,
            JobStatus.STOP_REQUESTED,
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.ERROR,
        }
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.STOP_REQUESTED,
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.ERROR,
        }
    ),
    JobStatus.STOP_REQUESTED: frozenset({JobStatus.CANCELLED, JobStatus.ERROR}),
}

_REPORT_FIELDS = frozenset(
    {
        "camera_status",
        "inference_status",
        "frame_number",
        "observed_at",
        "processing_fps",
        "total_detections",
        "class_counts",
        "detections",
        "error",
        "stream_jpeg",
        "processed_frames",
        "total_frames",
        "progress_percent",
        "history_record_total",
        "behavior_detection_total",
        "run_id",
        "artifacts",
    }
)


def transition_status(current: JobStatus | str, target: JobStatus | str) -> JobStatus:
    """Validate and return a job-state transition."""
    current_status = JobStatus(current)
    target_status = JobStatus(target)
    if target_status not in _ALLOWED_TRANSITIONS.get(current_status, frozenset()):
        raise ValueError(
            f"Invalid job state transition: {current_status.value} -> {target_status.value}"
        )
    return target_status


class ActiveJobConflict(RuntimeError):
    """Raised when a second job is requested while one still owns the slot."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        super().__init__("Another VisionClass job is already active.")
        self.snapshot = snapshot


class WorkerUnavailable(RuntimeError):
    """Raised when the requested job type has no configured worker."""


class WorkerStartError(RuntimeError):
    """Raised when Python cannot start the background worker thread."""

    def __init__(self, message: str, snapshot: dict[str, Any]) -> None:
        super().__init__(message)
        self.snapshot = snapshot


Worker = Callable[["JobContext"], None]


@dataclass
class _Job:
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.STARTING
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    cleanups: list[Callable[[], None]] = field(default_factory=list)
    cleanup_started: bool = False
    cleanup_errors: list[str] = field(default_factory=list)
    camera_status: str = "starting"
    inference_status: str = "starting"
    frame_number: int = 0
    processing_fps: float | None = None
    total_detections: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    detections: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    session_detection_total: int = 0
    session_class_counts: dict[str, int] = field(default_factory=dict)
    processed_frames: int = 0
    total_frames: int | None = None
    progress_percent: float | None = None
    history_record_total: int = 0
    behavior_detection_total: int = 0
    run_id: str | None = None
    artifacts: list[str] = field(default_factory=list)
    stream_jpeg: bytes | None = None
    stream_sequence: int = 0
    latest_observation_at: float | None = None
    live_samples: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=LIVE_HISTORY_MAX_SAMPLES)
    )
    last_sample_monotonic: float | None = None


class JobContext:
    """Worker-facing stop, progress, and cleanup interface."""

    def __init__(self, manager: "SessionManager", job: _Job) -> None:
        self._manager = manager
        self._job = job
        self.stop_event = job.stop_event

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def report(self, **values: Any) -> None:
        self._manager._report(self._job, values)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        self._manager._register_cleanup(self._job, callback)


class SessionManager:
    """Own one webcam/video worker until it exits and cleanup finishes."""

    def __init__(self, workers: Mapping[str, Worker] | None = None) -> None:
        self._workers = dict(workers or {})
        self._condition = threading.Condition(threading.RLock())
        self._active: _Job | None = None
        self._latest: _Job | None = None

    def start(
        self,
        job_type: str = "webcam",
        *,
        worker: Worker | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        worker = worker or self._workers.get(job_type)
        if worker is None:
            raise WorkerUnavailable(f"No worker is configured for job type '{job_type}'.")

        with self._condition:
            self._release_if_terminated_locked()
            if self._active is not None:
                raise ActiveJobConflict(self._snapshot_locked(self._active))

            job = _Job(job_id=uuid4().hex, job_type=job_type, run_id=run_id)
            context = JobContext(self, job)
            thread = threading.Thread(
                target=self._run_worker,
                args=(job, worker, context),
                name=f"visionclass-{job_type}-{job.job_id[:8]}",
                daemon=True,
            )
            job.thread = thread
            self._active = job
            self._latest = job
            try:
                thread.start()
            except Exception as exc:
                job.error = f"Could not start the {job_type} worker: {exc}"
                job.inference_status = "error"
                job.status = transition_status(job.status, JobStatus.ERROR)
                self._active = None
                self._condition.notify_all()
                raise WorkerStartError(job.error, self._snapshot_locked(job)) from exc
            self._condition.notify_all()
            return self._snapshot_locked(job)

    def request_stop(self) -> tuple[bool, dict[str, Any]]:
        with self._condition:
            self._release_if_terminated_locked()
            job = self._active
            if job is None or job.status in TERMINAL_STATUSES:
                return False, self._snapshot_locked(job or self._latest)
            if job.status is not JobStatus.STOP_REQUESTED:
                job.status = transition_status(job.status, JobStatus.STOP_REQUESTED)
                job.stop_event.set()
                self._condition.notify_all()
            return True, self._snapshot_locked(job)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            self._release_if_terminated_locked()
            return self._snapshot_locked(self._latest)

    def live_history(self, window_minutes: int, *, now: float | None = None) -> dict[str, Any]:
        """Return a bounded, copied time window of sampled webcam frame metrics."""
        if window_minutes not in {1, 5, 10}:
            raise ValueError("Live history window must be 1, 5, or 10 minutes.")
        current_time = time.time() if now is None else now
        with self._condition:
            self._release_if_terminated_locked()
            job = self._latest
            if job is None or job.job_type != "webcam":
                return {
                    "job_id": None,
                    "status": "idle",
                    "window_minutes": window_minutes,
                    "sample_interval_seconds": LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS,
                    "points": [],
                    "current_time": current_time,
                    "latest_observation_at": None,
                    "data_freshness_seconds": None,
                    "stale": False,
                    "metric_semantics": (
                        "Each point is a webcam frame observation sampled at most once per second. "
                        "Class counts are behavior bounding boxes in that sampled frame, not unique people."
                    ),
                }

            cutoff = current_time - window_minutes * 60
            points = [
                {
                    "observed_at": sample["observed_at"],
                    "frame_number": sample["frame_number"],
                    "total_detections": sample["total_detections"],
                    "class_counts": dict(sample["class_counts"]),
                    "processing_fps": sample["processing_fps"],
                }
                for sample in job.live_samples
                if sample["observed_at"] >= cutoff
            ]
            latest = job.latest_observation_at
            age = max(0.0, current_time - latest) if latest is not None else None
            stale = bool(
                job.status in {JobStatus.STARTING, JobStatus.RUNNING}
                and age is not None
                and age > LIVE_HISTORY_STALE_AFTER_SECONDS
            )
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "window_minutes": window_minutes,
                "sample_interval_seconds": LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS,
                "points": points,
                "current_time": current_time,
                "latest_observation_at": latest,
                "data_freshness_seconds": age,
                "stale": stale,
                "metric_semantics": (
                    "Each point is a webcam frame observation sampled at most once per second. "
                    "Class counts are behavior bounding boxes in that sampled frame, not unique people."
                ),
            }

    def wait(self, job_id: str, timeout: float) -> tuple[bool, dict[str, Any]]:
        """Join a worker for tests and shutdown; HTTP handlers never call this."""
        with self._condition:
            job = self._latest
            if job is None or job.job_id != job_id:
                return True, self._snapshot_locked(job)
            thread = job.thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))
        with self._condition:
            self._release_if_terminated_locked()
            return bool(thread is None or not thread.is_alive()), self._snapshot_locked(job)

    def next_frame(
        self, after_sequence: int, timeout: float, *, job_id: str | None = None
    ) -> tuple[int, bytes] | None:
        """Wait briefly for a newer webcam frame without holding an HTTP request open indefinitely."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                job = self._latest
                if job is None or (job_id is not None and job.job_id != job_id):
                    return None
                if job.stream_jpeg is not None and job.stream_sequence > after_sequence:
                    return job.stream_sequence, job.stream_jpeg
                if job.status in TERMINAL_STATUSES or not job.thread or not job.thread.is_alive():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Request stop and wait only for a bounded process-shutdown interval."""
        _, snapshot = self.request_stop()
        job_id = snapshot.get("job_id")
        if not job_id:
            return True
        finished, _ = self.wait(job_id, timeout)
        return finished

    def _report(self, job: _Job, values: dict[str, Any]) -> None:
        unknown = set(values) - _REPORT_FIELDS
        if unknown:
            raise ValueError(f"Unsupported worker status fields: {', '.join(sorted(unknown))}")
        with self._condition:
            if job.job_type == "webcam" and "class_counts" in values and "frame_number" in values:
                observed_at = float(values.get("observed_at") or time.time())
                job.latest_observation_at = observed_at
                sample_clock = time.monotonic()
                if (
                    job.last_sample_monotonic is None
                    or sample_clock - job.last_sample_monotonic >= LIVE_HISTORY_SAMPLE_INTERVAL_SECONDS
                ):
                    class_counts = {
                        str(name): max(0, int(count))
                        for name, count in values["class_counts"].items()
                    }
                    job.live_samples.append({
                        "observed_at": observed_at,
                        "frame_number": int(values["frame_number"]),
                        "total_detections": max(
                            0,
                            int(values.get("total_detections", sum(class_counts.values()))),
                        ),
                        "class_counts": class_counts,
                        "processing_fps": values.get("processing_fps"),
                    })
                    job.last_sample_monotonic = sample_clock
            for key, value in values.items():
                if key == "observed_at":
                    continue
                if key == "stream_jpeg":
                    if value is not None and not isinstance(value, bytes):
                        raise TypeError("stream_jpeg must be bytes or None.")
                    job.stream_jpeg = value
                    if value is not None:
                        job.stream_sequence += 1
                    continue
                setattr(job, key, value)
            if "total_detections" in values and job.job_type == "webcam":
                job.session_detection_total += int(values["total_detections"])
            if "class_counts" in values and job.job_type == "webcam":
                for name, count in values["class_counts"].items():
                    job.session_class_counts[name] = job.session_class_counts.get(name, 0) + int(count)
            self._condition.notify_all()

    def _register_cleanup(self, job: _Job, callback: Callable[[], None]) -> None:
        if not callable(callback):
            raise TypeError("Cleanup callback must be callable.")
        with self._condition:
            if job.cleanup_started:
                raise RuntimeError("Worker cleanup has already started.")
            job.cleanups.append(callback)

    def _run_worker(self, job: _Job, worker: Worker, context: JobContext) -> None:
        worker_error: BaseException | None = None
        with self._condition:
            if job.status is JobStatus.STARTING:
                job.status = transition_status(job.status, JobStatus.RUNNING)
            self._condition.notify_all()
        try:
            worker(context)
        except BaseException as exc:
            worker_error = exc
        finally:
            with self._condition:
                job.cleanup_started = True
                cleanups = tuple(reversed(job.cleanups))
            for cleanup in cleanups:
                try:
                    cleanup()
                except BaseException as exc:
                    with self._condition:
                        job.cleanup_errors.append(str(exc))
                    print(f"ERROR: VisionClass worker cleanup failed: {exc}", file=sys.stderr)

            with self._condition:
                # A worker may fail before opening the camera (for example, while
                # loading the model) or after the camera was released. Keep the
                # published camera state from implying that it is still open.
                if job.camera_status in {"starting", "open"}:
                    job.camera_status = "closed"
                if worker_error is not None:
                    job.error = str(worker_error) or type(worker_error).__name__
                    job.inference_status = "error"
                    print(
                        f"ERROR: VisionClass {job.job_type} job failed: {job.error}",
                        file=sys.stderr,
                    )
                elif job.cleanup_errors:
                    job.error = f"Cleanup failed: {job.cleanup_errors[0]}"
                    job.camera_status = "error"
                    job.inference_status = "error"
                else:
                    if job.camera_status != "not_applicable":
                        job.camera_status = "closed"
                    if job.inference_status not in {"completed", "stopped"}:
                        job.inference_status = "stopped"

                target = (
                    JobStatus.ERROR
                    if worker_error is not None or job.cleanup_errors
                    else JobStatus.CANCELLED
                    if job.stop_event.is_set()
                    else JobStatus.COMPLETED
                )
                job.status = transition_status(job.status, target)
                job.finished_at = time.monotonic()
                self._condition.notify_all()

    def _release_if_terminated_locked(self) -> None:
        job = self._active
        if (
            job is not None
            and job.status in TERMINAL_STATUSES
            and job.thread is not None
            and not job.thread.is_alive()
        ):
            self._active = None
            self._condition.notify_all()

    @staticmethod
    def _snapshot_locked(job: _Job | None) -> dict[str, Any]:
        if job is None:
            return {
                "job_id": None,
                "job_type": None,
                "status": JobStatus.IDLE.value,
                "camera_status": "idle",
                "inference_status": "idle",
                "frame_number": 0,
                "processing_fps": None,
                "total_detections": 0,
                "session_detection_total": 0,
                "session_class_counts": {},
                "class_counts": {},
                "detections": [],
                "processed_frames": 0,
                "total_frames": None,
                "progress_percent": None,
                "history_record_total": 0,
                "behavior_detection_total": 0,
                "run_id": None,
                "artifacts": [],
                "stream_sequence": 0,
                "latest_observation_at": None,
                "live_history_sample_count": 0,
                "duration_seconds": 0.0,
                "error": None,
                "cleanup_errors": [],
                "stop_requested": False,
                "worker_alive": False,
            }
        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status.value,
            "camera_status": job.camera_status,
            "inference_status": job.inference_status,
            "frame_number": job.frame_number,
            "processing_fps": job.processing_fps,
            "total_detections": job.total_detections,
            "session_detection_total": job.session_detection_total,
            "session_class_counts": dict(job.session_class_counts),
            "class_counts": dict(job.class_counts),
            "detections": [dict(item) for item in job.detections],
            "processed_frames": job.processed_frames,
            "total_frames": job.total_frames,
            "progress_percent": job.progress_percent,
            "history_record_total": job.history_record_total,
            "behavior_detection_total": job.behavior_detection_total,
            "run_id": job.run_id,
            "artifacts": list(job.artifacts),
            "stream_sequence": job.stream_sequence,
            "latest_observation_at": job.latest_observation_at,
            "live_history_sample_count": len(job.live_samples),
            "duration_seconds": max(
                0.0,
                (job.finished_at or time.monotonic()) - job.started_at,
            ),
            "error": job.error,
            "cleanup_errors": list(job.cleanup_errors),
            "stop_requested": job.stop_event.is_set(),
            "worker_alive": bool(job.thread and job.thread.is_alive()),
        }
