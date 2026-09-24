# VisionClass

Local computer-vision pipeline for **classroom student behavior detection** using a custom-trained **YOLO11** model, with optional **person tracking** (ByteTrack) and **CSV time-on-behavior** summaries.

There is no web UI or API in this repository—scripts run on your machine with OpenCV windows and file outputs.

## Architecture

```text
Roboflow YOLO dataset  →  train_behavior.py  →  runs/.../weights/best.pt
best.pt + yolo11n.pt   →  video / webcam / tracking / history scripts
MediaPipe prototypes   →  feature_extraction.py, src/features.py (not wired to YOLO yet)
```

## Installation

1. Python **3.12** recommended (matches existing venv).
2. Create a virtual environment at the project root (do **not** commit `venv/`):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Place assets locally (not in git by default):
   - Behavior weights: `runs/detect/runs/visionclass_behavior-3/weights/best.pt`
   - Person detector: `yolo11n.pt` (project root; Ultralytics will download if missing)
   - Sample video: `data/videos/classroom.mp4`
   - Dataset YAML for train/eval: `data/dataset/data.yaml` (export from Roboflow) **or** set `VISIONCLASS_DATASET_YAML`

## Configuration

Shared settings live in `src/config.py`. Override with environment variables or CLI flags where supported.

| Variable | Purpose |
|----------|---------|
| `VISIONCLASS_DATASET_YAML` | Path to YOLO `data.yaml` |
| `VISIONCLASS_BEHAVIOR_MODEL` | Path to `best.pt` |
| `VISIONCLASS_PERSON_MODEL` | Path to person `yolo11n.pt` |
| `VISIONCLASS_VIDEO` | Input classroom video |
| `VISIONCLASS_DEVICE` | `auto`, `cpu`, or GPU index `0`, `1`, … |

Run all commands from the **project root** (`VisionClass/`).

## Usage

### Train (requires dataset YAML)

```bash
python src/train_behavior.py --dataset path/to/data.yaml
```

### Evaluate on test split

```bash
python src/evaluate_behavior.py --dataset path/to/data.yaml
```

### Video inference (behavior boxes only)

```bash
python src/video_behavior.py
```

### Webcam inference

```bash
python src/realtime_behavior.py
```

### Person tracking + behavior labels

```bash
python src/classroom_tracking.py
```

Output: `data/videos/classroom_tracking.mp4`

### Temporal history + CSV

```bash
python src/behavior_history.py
```

Outputs:

- `data/videos/classroom_history.mp4`
- `data/classroom_behavior_history.csv`
- `data/classroom_behavior_summary.csv`

### Tests

```bash
pytest
```

## Directory layout

```text
VisionClass/
├── src/
│   ├── config.py           # paths, thresholds, device
│   ├── association.py      # geometry + behavior-person matching
│   ├── train_behavior.py
│   ├── evaluate_behavior.py
│   ├── video_behavior.py
│   ├── realtime_behavior.py
│   ├── classroom_tracking.py
│   ├── behavior_history.py
│   └── features.py         # MediaPipe helpers (unused by YOLO path)
├── feature_extraction.py   # MediaPipe webcam demo (standalone)
├── tests/
├── data/
├── runs/                   # training artifacts (gitignored)
├── requirements.txt
└── README.md
```

## What to commit vs keep local

**Commit:** source code, `tests/`, `requirements.txt`, `README.md`, `.gitignore`, small `data/dataset/.gitkeep` if added.

**Keep local (gitignored by default):** `venv/`, `runs/`, `*.pt`, videos, generated CSVs under `data/`.

## Dataset

Behavior classes match the Roboflow **student_classroom_behavior** export: `handrise`, `read`, `sleep`, `stand`, `using_electronic_devices`, `write`.
