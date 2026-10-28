# Development Guide

## Supported local setup

The repository's pinned dependency set targets Python 3.12 and the current Windows CUDA 12.8 environment. Create a project-local virtual environment and install the existing requirements only when setting up a new environment:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The main requirements use `opencv-python`, as required by Ultralytics. MediaPipe is optional and isolated in `requirements-mediapipe.txt`: MediaPipe requires `opencv-contrib-python`, and the two wheel distributions overlap in the `cv2` namespace, so install the demo in a separate environment. A fresh Python 3.12 environment was verified with these requirements, `pip check`, imports, and the full test suite; a clean Git clone has not been tested. CPU-only or non-Windows environments must select appropriate PyTorch wheels and adapt the three Torch pins.

The current project environment was used for repository tests, but a clean-clone installation is not claimed as verified. CPU-only and non-Windows installations may require the appropriate PyTorch and OpenCV packages; see `requirements.txt` and [Configuration](CONFIGURATION.md).

## Run and test

Run the main local app from the repository root:

```powershell
.\venv\Scripts\python.exe -B -m src.web_app
```

Run the full deterministic suite:

```powershell
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
```

`-B` avoids Python bytecode files and `-p no:cacheprovider` disables pytest's cache plugin. The unit suite mocks cameras and models where needed; it does not replace the separate physical webcam or long-video manual validations.

Safe parser checks that do not open a camera include:

```powershell
.\venv\Scripts\python.exe -B -m src.realtime_behavior --help
.\venv\Scripts\python.exe -B -m src.video_behavior --help
.\venv\Scripts\python.exe -B -m src.train_behavior --help
.\venv\Scripts\python.exe -B -m src.evaluate_behavior --help
```

## Repository map

| Path | Role |
|---|---|
| `src/web_app.py` | Flask localhost app, routes, startup, HTTP validation. |
| `src/session_manager.py` | One active worker, lifecycle states, cancellation and cleanup. |
| `src/video_jobs.py` | Upload validation, isolated run storage, video worker and report artifacts. |
| `src/realtime_behavior.py` | Shared webcam worker and legacy CLI entry point. |
| `src/live_dashboard.py` | Legacy standard-library dashboard. |
| `src/config.py` | Shared model, dataset, device, inference, and output defaults. |
| `src/behavior_history.py` | Video processing, associations, CSV output, annotated video finalization. |
| `src/association.py` | Geometry and behavior-to-person association helpers. |
| `src/analyze_behavior.py` | Existing CSV aggregation semantics. |
| `src/generate_analytics_report.py` | Markdown and run-scoped compact HTML reports, using the shared aggregation functions. |
| `src/build_dashboard.py` | Separate detailed interactive dashboard CLI; its `generate_dashboard` name remains an alias to the compact report generator for existing command compatibility. |
| `src/features.py`, `feature_extraction.py` | Reusable experimental MediaPipe feature calculations and a standalone webcam visualization that consumes those helpers; separate from the YOLO application. |
| `tests/` | Unit, route, worker, upload, analytics, and configuration tests. |
| `datasets/` | Dataset snapshots and explicit audit/cleaning utilities; do not run a cleaner casually. |
| `scripts/` | Repository workflow validators and search helpers. |

## Change conventions

- Keep HTTP concerns in `web_app.py`, job ownership in `session_manager.py`, video run storage in `video_jobs.py`, and inference/analytics in their existing modules.
- Avoid import-time hardware access, model construction, dataset scans, or output writes. Put executable behavior in `main()` behind `if __name__ == "__main__":`.
- Keep paths rooted in `Path(__file__)`/`PROJECT_ROOT` or explicit settings. Never embed a developer home directory.
- Preserve history CSV schema, class order, confidence thresholds, checkpoint selection, and model inference settings unless a separately approved specification change says otherwise.
- Tests should use synthetic inputs, mocks, and pytest temporary paths. Do not weaken tests to make a command pass.
- Review `git status --short` and the diff before editing pre-modified files. Do not stage/commit until explicitly authorized.

## Manual tools that can do work

| Command/tool | Effect when directly executed |
|---|---|
| `python -m src.analyze_image_resolutions` | Reads the canonical test image split and prints dimensions/statistics. |
| `python -m src.analyze_device_box_sizes` | Reads canonical YOLO label files and prints normalized device-box statistics. |
| `python -m src.analyze_errors --help` | Displays CLI options; a full invocation runs model inference and writes an error-analysis folder. |
| `python -m src.diagnose_detections --help` | Displays CLI options; a full invocation runs model inference on image inputs and creates a new diagnostic output directory. |
| `python -m src.benchmark_phase6_imgsz` | Runs model evaluation at its configured sizes and writes Ultralytics validation artifacts. It has no help-only mode. |
| `python -m src.make_error_contact_sheet` | Reads selected dataset images and writes a contact sheet under `runs/`. Review the fixed output path before running. |
| Dataset `audit_scb.py` | Scans the corresponding extracted or cleaned dataset snapshot and prints an audit. |
| Dataset `inspect_bad.py` | Opens selected dataset images and writes preview images beside that utility. |
| Dataset `clean_scb.py` | Copies the extracted snapshot and removes listed invalid annotation lines from the copy. It refuses to overwrite an existing destination. |

Diagnostic/evaluation scripts may load a model, scan a dataset, run inference, or create output. Importing these tools should not run those actions, but direct execution remains explicit and potentially data-generating. Use a separate, reviewed output path when the tool supports one; never remove an existing destination just to make a utility run.
