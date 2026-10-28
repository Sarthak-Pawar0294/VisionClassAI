# How VisionClass Works

VisionClass runs locally and analyzes frames from a webcam or a selected video. It reports model detections and tracker records; it does not identify a person's real-world identity.

## Live camera path

1. Start the Flask app from the repository root with `.\venv\Scripts\python.exe -B -m src.web_app` and open `http://127.0.0.1:8766/live`.
2. The browser gets a session-bound CSRF token. Start is sent to the Flask API, which reserves the single active-job slot in `src/session_manager.py`.
3. The webcam worker in `src/realtime_behavior.py` resolves settings from `src/config.py`, loads the configured behavior checkpoint, and opens the configured camera. On Windows it selects the DirectShow capture backend; other platforms use OpenCV's default backend.
4. Each readable frame is passed to the existing YOLO model using configured inference size, device, and confidence. The returned behavior boxes are annotated and reported. A valid frame with no boxes still advances frame/status data and is represented honestly as zero detections.
5. Flask state and MJPEG endpoints deliver updates to the page. Stop asks the worker to terminate; resource cleanup completes after its current processing iteration.

The standalone command `.\venv\Scripts\python.exe -B .\src\realtime_behavior.py --device 0` keeps the OpenCV preview and the legacy dashboard on port 8765.

## Video path

1. The Video Analysis page accepts `.mp4`, `.avi`, `.mov`, or `.mkv` payloads up to the configured 1 GiB payload limit plus a separate multipart overhead allowance.
2. The server assigns an opaque run ID and generated input name in a unique directory under `data/visionclass_runs/` by default. It does not use the submitted filename as a path. The decoder is checked before a worker starts.
3. A single worker processes frames through `src/behavior_history.py`. The existing person and behavior checkpoints, association logic, cache behavior, and output semantics are retained.
4. Progress is stored in the run manifest and reported to the UI. A stop request is cooperative and processing completes or cancels after the current processing unit and cleanup.
5. Core output includes annotated video, history CSV, and summary CSV when processing succeeds. Analytics and optional Markdown/HTML reports are produced through existing modules when inputs are valid. Only completed, available artifacts are linked.

The annotated MP4 is video-only in the current pipeline; source audio is not preserved. H.264 output depends on the available FFmpeg/OpenCV encoder and decoder support.

## What the data means

- **Behavior detection:** a bounding box and class returned for one frame. The same visible person may produce boxes across many frames.
- **History/tracker record:** one tracker's assigned behavior for one processed frame. Cached rows are marked separately in the CSV.
- **Tracker ID:** a temporary per-video association label (`student_id` in the existing CSV schema). It is not linked across runs.
- **Unique student:** the system has no verified real-world identity or cross-frame ground-truth identity metric. Reports must not call tracker IDs unique students.

The history schema is `frame,time_seconds,student_id,behavior,confidence,is_cached,unknown_cause`. The summary CSV uses `student_id` followed by the six behavior classes and `unknown`; values retain the existing history-row-count divided by source-FPS aggregation. The source modules define exact fallback, rounding, and analytics semantics.

## Outputs and privacy

Video uploads, annotated video, CSVs, and optional reports are retained under the run directory until manually removed by the operator. Live camera frames are streamed for the current session and are not saved as a live image archive by the current worker. Treat all classroom media and generated reports as sensitive local data. No real-time accuracy, unique-student count, or guaranteed model performance is implied.
