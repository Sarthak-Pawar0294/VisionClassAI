# VisionClass

Local computer-vision pipeline for **classroom student behavior detection** using a custom-trained **YOLO11** model, with optional **person tracking** (ByteTrack) and **CSV time-on-behavior** summaries.

VisionClass includes a loopback-only Flask application for live webcam sessions, local video analysis, and run-scoped analytics. The original standalone webcam CLI and its standard-library dashboard on port 8765 remain available.

## Architecture

```text
Optional training/evaluation tools ── dataset + checkpoint ──┐
                                                           │
Browser ── Flask app (127.0.0.1:8766) ── SessionManager ────┼─ live webcam worker
                                      └── video RunStore ──┼─ video/history worker
                                                           ↓
                                  existing YOLO, association, CSV, analytics
                                                           ↓
                                 dashboard, playback, reports, downloads

Standalone webcam CLI ── standard-library dashboard (127.0.0.1:8765)
MediaPipe demo ── shared src/features.py helpers; separate from the YOLO pipeline
```

The implementation map and data-flow details are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Installation

Python **3.12** is the project's target environment (the existing test environment is Python 3.12.7). A clean-clone install has not been verified. The requirements pin PyTorch to the CUDA 12.8 build used by the current Windows setup; CPU-only and other platform users need platform-appropriate PyTorch wheels and should adapt the three Torch pins before installing.

From PowerShell at the repository root, create a virtual environment and install the declared requirements:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first install command selects the CUDA 12.8 PyTorch wheels matching the exact versions in `requirements.txt`; the second installs the remaining project dependencies and test tools. For CPU-only or non-Windows systems, select the appropriate PyTorch wheel index and adjust the three local Torch pins before installing. This alternative environment has not been validated.

`requirements.txt` contains the Flask/YOLO application and pytest dependencies. It uses `opencv-python`, as required by Ultralytics. Flask is pinned to `3.1.3`; most application dependencies are pinned, while pytest uses a minimum version. The project has not been tested from a clean checkout or across every supported operating system.

The optional MediaPipe webcam demo has its own dependency file because MediaPipe requires `opencv-contrib-python`, while Ultralytics requires `opencv-python`. Those distributions install overlapping `cv2` files and should not be installed together in one environment. To use the demo, create a separate environment (outside the repository) and install only its dependencies:

```powershell
$mpVenv = Join-Path $env:TEMP 'visionclass-mediapipe-venv'
py -3.12 -m venv $mpVenv
& "$mpVenv\Scripts\python.exe" -m pip install -r requirements-mediapipe.txt
& "$mpVenv\Scripts\python.exe" .\feature_extraction.py
```

The application requires a behavior checkpoint, which is not included in this repository. Obtain an authorized `best.pt` from the project owner or an approved model release, then set `VISIONCLASS_BEHAVIOR_MODEL` to its local path. Alternatively, place it under `runs/<experiment>/weights/best.pt`; without an override, VisionClass selects the newest matching checkpoint. The checkpoint used for local validation was at `runs/detect/runs/phase6/visionclass_960-2/weights/best.pt`, but model weights and training outputs are excluded from Git. Video/person-tracking workflows also use `VISIONCLASS_PERSON_MODEL` (default `yolo11n.pt`); Ultralytics may download that pretrained weight if it is absent.

The dataset is needed for training/evaluation, not ordinary webcam or uploaded-video inference. The configured YAML is `datasets/student_classroom_behavior/data.yaml`; the YAML expects `train/images`, `valid/images`, and `test/images` beneath its dataset root. Dataset image and label payloads and ZIP archives are ignored by Git; the YAML, provenance notes, and utility scripts can be published separately. `data.yaml` identifies the Roboflow export as CC BY 4.0; obtain the dataset from its source and follow that attribution/license. A sample video at `data/videos/classroom.mp4` is optional for the Flask app; provide another local video through the Video Analysis page or `VISIONCLASS_VIDEO` for the standalone CLI.

## Configuration

Model, dataset, inference, and training defaults live in `src/config.py`; Flask port, upload limits, run storage, and loopback security are configured by `src/web_app.py`. Override environment-supported values in the shell. `.env.example` is a reference file only; VisionClass does not load `.env` automatically.

| Variable | Purpose |
|----------|---------|
| `VISIONCLASS_DATASET_YAML` | Path to YOLO `data.yaml` |
| `VISIONCLASS_BEHAVIOR_MODEL` | Path to `best.pt` |
| `VISIONCLASS_PERSON_MODEL` | Path to person `yolo11n.pt` |
| `VISIONCLASS_VIDEO` | Input classroom video |
| `VISIONCLASS_DEVICE` | `auto`, `cpu`, or GPU index `0`, `1`, … |
| `VISIONCLASS_WEBCAM_INDEX` | Camera index; default `0` |
| `VISIONCLASS_APP_PORT` | Flask application port; default `8766` |
| `VISIONCLASS_RUNS_DIR` | Video run storage root; default `data/visionclass_runs/` |
| `VISIONCLASS_MAX_UPLOAD_BYTES` | Maximum video payload bytes; default `1073741824` (1 GiB) |
| `VISIONCLASS_MAX_MULTIPART_OVERHEAD_BYTES` | Multipart request allowance; default `1048576` (1 MiB) |

Run application, test, and standalone commands from the **project root** (`VisionClass/`). Path overrides are resolved relative to the current working directory unless absolute. `.env.example` documents variable names but is not automatically loaded; set values in the process environment. See [Configuration](docs/CONFIGURATION.md).

## Usage

### Train

```powershell
.\venv\Scripts\python.exe -B -m src.train_behavior
```

### Evaluate on test split

```powershell
.\venv\Scripts\python.exe -B -m src.evaluate_behavior
```

### Video inference (behavior boxes only)

```powershell
.\venv\Scripts\python.exe -B -m src.video_behavior
```

### Webcam inference

```bash
.\venv\Scripts\python.exe -B .\src\realtime_behavior.py --device 0
```

When the webcam opens, VisionClass starts a local dashboard at
`http://127.0.0.1:8765/` and opens it in the default browser. The page shows the
annotated live feed, current-frame class counts, detections, confidence scores,
and processing rate. Press **Q** in the webcam preview to end the session and
stop the local dashboard server.

### Expanded local dashboard

Run from the project root with the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -B -m src.web_app
```

Open [http://127.0.0.1:8766/](http://127.0.0.1:8766/). The server binds only to
`127.0.0.1`; it does not expose the app to the LAN. The default web port is
`8766` and can be changed with `VISIONCLASS_APP_PORT`. The legacy webcam CLI
continues to use its own dashboard on `http://127.0.0.1:8765/`.

The Live page starts and stops a webcam job through the app. Video Analysis
accepts `.mp4`, `.avi`, `.mov`, and `.mkv` files up to 1 GiB, plus the separate
1 MiB multipart allowance. Files are checked by OpenCV before processing. A
single webcam or video job may run at a time. Completed runs and generated
reports are stored in a unique folder under `data/visionclass_runs/` by default
(or `VISIONCLASS_RUNS_DIR`). Runs are retained until manually removed; the app
does not automatically purge them. Analytics are computed from each run's
existing history and summary CSVs. Tracker IDs are scoped to a video and do
not represent unique real-world student identities. Optional Markdown/HTML
reports are linked only when generation succeeded.

The Live overview records a bounded in-memory trend of timestamped webcam
frames: at most one sampled frame per second and at most 600 samples. Chart
counts are behavior boxes in those sampled frames; a box may appear in many
frames and is not a unique person or a separate event. The current frame,
sampled-frame trends, and cumulative per-frame box observations are labeled
separately. Pausing chart updates does not pause webcam inference. Live images
are streamed to the browser and are not retained to build these charts.

Video output is finalized to an MP4 with H.264 when an existing FFmpeg
installation provides `libx264` or `libopenh264`; otherwise the app requests
OpenCV's `avc1` H.264 encoder. FFmpeg is optional and is not installed by
VisionClass. The app reopens and decodes the completed output before publishing
it, checks dimensions, frame rate, and decoded frame count, and exposes playback
only after validation. If neither available encoder can create a valid H.264
MP4, the run reports an output error with a recovery hint rather than claiming
playable completion. Annotated output contains video only; source audio is not
preserved by the existing processing pipeline. The browser player reports a
decode failure and keeps the annotated download available when one was
validated.

The expanded app requires the configured person and behavior model files to
exist before a job starts. Use `VISIONCLASS_BEHAVIOR_MODEL` and
`VISIONCLASS_PERSON_MODEL` to select existing checkpoints. Missing/unloadable
models, camera failures, unsupported/corrupt videos, occupied ports, and disk
write errors are reported in the dashboard or startup diagnostics. For a camera
error, check OS camera permission and `VISIONCLASS_WEBCAM_INDEX`; for video
decode errors, try a supported container/codec. To stop a web session, use the
dashboard Stop control; it requests termination and cleanup completes after
the current processing iteration. The OpenCV **Q** key remains available in
the standalone CLI.

### Person tracking + behavior labels

```powershell
.\venv\Scripts\python.exe -B -m src.classroom_tracking
```

Output: `data/videos/classroom_tracking.mp4`

### Temporal history + CSV

```powershell
.\venv\Scripts\python.exe -B -m src.behavior_history
```

Outputs:

- `data/videos/classroom_history.mp4`
- `data/classroom_behavior_history.csv`
- `data/classroom_behavior_summary.csv`

### Tests

```bash
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
```

`-B` and `-p no:cacheprovider` avoid writing Python bytecode and pytest's
`.pytest_cache` during the run.

## Directory layout

```text
VisionClass/
├── src/
│   ├── web_app.py          # Flask localhost UI and API
│   ├── session_manager.py  # one active webcam/video worker lifecycle
│   ├── video_jobs.py       # isolated video runs and artifact handling
│   ├── templates/          # live, video, and analytics pages
│   ├── static/             # local dashboard CSS and JavaScript
│   ├── config.py           # paths, thresholds, device
│   ├── association.py      # geometry + behavior-person matching
│   ├── train_behavior.py
│   ├── evaluate_behavior.py
│   ├── video_behavior.py
│   ├── realtime_behavior.py
│   ├── classroom_tracking.py
│   ├── behavior_history.py
│   └── features.py         # Shared MediaPipe helpers; separate from YOLO inference
├── feature_extraction.py   # Standalone MediaPipe webcam demo using src/features.py
├── tests/
├── data/
├── datasets/
│   └── student_classroom_behavior/  # canonical six-class dataset
├── runs/                   # training artifacts (gitignored)
├── requirements.txt
└── README.md
```

## What to commit vs keep local

**Commit:** reviewed source code, tests, dependency declarations, and documentation.

**Keep local unless deliberately reviewed for sharing:** `venv/`, `runs/`, `*.pt`, datasets, classroom videos, run directories, reports, generated CSVs, screenshots, and validation evidence. `.gitignore` excludes common generated and large local artifacts, but review both `git status --short` and the complete files selected for publication. Do not publish identifiable classroom media, secrets, or credentials.

## Dataset

Behavior classes match the Roboflow **student_classroom_behavior** export: `handrise`, `read`, `sleep`, `stand`, `using_electronic_devices`, `write`.

## Project documentation

- [Architecture and data flow](docs/ARCHITECTURE.md)
- [How processing works](docs/HOW_IT_WORKS.md)
- [Development workflow and tests](docs/DEVELOPMENT.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [Known limitations and evidence boundaries](docs/KNOWN_LIMITATIONS.md)

The root-level scripts and dataset utilities are standalone maintenance tools. Importing them is intended to be safe; running a cleaner, benchmark, diagnostic, or camera demo may read or write data and should be done only when that operation is intended.
