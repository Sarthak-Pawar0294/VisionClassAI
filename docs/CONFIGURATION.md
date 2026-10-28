# Configuration Reference

VisionClass reads configuration from process environment variables and CLI arguments where supported. The project does not automatically load `.env`; `.env.example` is a comment/documentation aid. Default paths are based on `PROJECT_ROOT` in `src/config.py`. Environment path overrides are expanded and resolved relative to the current working directory unless they are absolute.

## Application settings

| Variable | Default | Meaning |
|---|---:|---|
| `VISIONCLASS_APP_PORT` | `8766` | Flask port; validated as 1–65535. The host is fixed at `127.0.0.1`. |
| `VISIONCLASS_RUNS_DIR` | `data/visionclass_runs` | Root for isolated video runs; resolved to an absolute local path. Completed runs are not automatically deleted. |
| `VISIONCLASS_MAX_UPLOAD_BYTES` | `1073741824` | Maximum video payload: 1 GiB. Must be a positive integer. |
| `VISIONCLASS_MAX_MULTIPART_OVERHEAD_BYTES` | `1048576` | Additional multipart request allowance: 1 MiB. Must be a positive integer. |

The app's Flask session secret is generated in memory by `create_app`; there is no persistent secret or authentication system. The legacy standard-library dashboard uses port 8765 and its own configuration in `src/live_dashboard.py`.

## Model, data, and output paths

| Variable | Default | Meaning |
|---|---|---|
| `VISIONCLASS_BEHAVIOR_MODEL` | Newest existing `runs/**/weights/best.pt`; otherwise a missing-path sentinel | Behavior checkpoint. An explicit value disables auto-discovery. |
| `VISIONCLASS_PERSON_MODEL` | `yolo11n.pt` in the project root | Person checkpoint used by tracking/video workflows. |
| `VISIONCLASS_PRETRAINED_MODEL` | `yolo11n.pt` in the project root | Pretrained model path used by training defaults. |
| `VISIONCLASS_DATASET_YAML` | `datasets/student_classroom_behavior/data.yaml` | YOLO dataset configuration. |
| `VISIONCLASS_VIDEO` | `data/videos/classroom.mp4` | Default standalone video input. |
| `VISIONCLASS_VIDEO_OUTPUT` | `data/videos/classroom_result.mp4` | Behavior-only standalone output. |
| `VISIONCLASS_TRACKING_OUTPUT` | `data/videos/classroom_tracking.mp4` | Standalone tracking output. |
| `VISIONCLASS_HISTORY_VIDEO_OUTPUT` | `data/videos/classroom_history.mp4` | Annotated history-pipeline output. |
| `VISIONCLASS_HISTORY_CSV` | `data/classroom_behavior_history.csv` | Standalone history records. |
| `VISIONCLASS_SUMMARY_CSV` | `data/classroom_behavior_summary.csv` | Standalone per-tracker summary. |

## Inference and training/evaluation

| Variable | Default | Meaning |
|---|---:|---|
| `VISIONCLASS_DEVICE` | `auto` | `auto`, `cpu`, or a CUDA device index. `auto` selects CUDA 0 when available, else CPU. |
| `VISIONCLASS_WEBCAM_INDEX` | `0` | Camera device index for webcam workers. |
| `VISIONCLASS_BEHAVIOR_CONF` | `0.20` | Behavior detector confidence threshold. |
| `VISIONCLASS_PERSON_CONF` | `0.30` | Person detector confidence threshold. |
| `VISIONCLASS_INFERENCE_IMGSZ` | `960` | Production inference image size. |
| `VISIONCLASS_TRACKER` | Project `config/bytetrack.yaml` | Tracker configuration path. |
| `VISIONCLASS_TRAIN_IMGSZ` | `640` | Training image size. |
| `VISIONCLASS_TRAIN_EPOCHS` | `10` | Training epochs. |
| `VISIONCLASS_TRAIN_BATCH` | `8` | Training batch size. |
| `VISIONCLASS_EVAL_BATCH` | `8` | Evaluation batch size. |
| `VISIONCLASS_TRAIN_WORKERS` | `2` | Training data-loader workers. |
| `VISIONCLASS_EVAL_WORKERS` | `0` | Evaluation data-loader workers. |
| `VISIONCLASS_TRAIN_PROJECT` | `runs` | Ultralytics training output project. |
| `VISIONCLASS_TRAIN_NAME` | `visionclass_behavior` | Ultralytics training run name. |
| `VISIONCLASS_EVAL_PROJECT` | `runs` | Ultralytics evaluation output project. |
| `VISIONCLASS_EVAL_NAME` | `visionclass_test` | Ultralytics evaluation run name. |

The `.env.example` file lists shell environment names without credentials. Set an override explicitly in PowerShell, for example `$env:VISIONCLASS_DEVICE = "cpu"`, or use the supported CLI option. Model weights and dataset assets are not included by this configuration guide.
