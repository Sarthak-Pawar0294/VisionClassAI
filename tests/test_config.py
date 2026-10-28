"""Tests for configuration helpers (no GPU required)."""

import os
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings, resolve_device


def test_project_root_is_visionclass():
    assert PROJECT_ROOT.name == "VisionClass"
    assert (PROJECT_ROOT / "src" / "config.py").is_file()


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_cpu_or_int():
    device = resolve_device("auto")
    assert device == "cpu" or isinstance(device, int)


def test_resolve_device_explicit_index():
    assert resolve_device("0") == 0
    assert resolve_device("1") == 1


def test_webcam_index_defaults_to_zero():
    settings = get_settings()
    assert settings.webcam_index == 0


def test_webcam_index_from_env(monkeypatch):
    monkeypatch.setenv("VISIONCLASS_WEBCAM_INDEX", "2")
    settings = get_settings()
    assert settings.webcam_index == 2


def test_dataset_yaml_defaults_to_canonical_dataset(monkeypatch):
    monkeypatch.delenv("VISIONCLASS_DATASET_YAML", raising=False)

    settings = get_settings()
    canonical_path = PROJECT_ROOT / "datasets" / "student_classroom_behavior" / "data.yaml"

    assert settings.dataset_yaml == canonical_path
    assert settings.dataset_yaml.is_absolute()
    assert settings.dataset_yaml == canonical_path.resolve()


def test_dataset_yaml_from_env(monkeypatch, tmp_path):
    dataset_yaml = tmp_path / "custom-data.yaml"
    monkeypatch.setenv("VISIONCLASS_DATASET_YAML", str(dataset_yaml))
    settings = get_settings()
    assert settings.dataset_yaml == dataset_yaml.resolve()


def test_behavior_model_discovery_finds_newest_existing_best_pt(monkeypatch, tmp_path):
    """Auto-discovery selects the newest existing runs/**/weights/best.pt."""
    import src.config as config

    monkeypatch.delenv("VISIONCLASS_BEHAVIOR_MODEL", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)

    older = tmp_path / "runs" / "phase-a" / "weights" / "best.pt"
    newer = tmp_path / "runs" / "nested" / "phase-b" / "weights" / "best.pt"
    ignored = tmp_path / "runs" / "phase-c" / "best.pt"
    for candidate in (older, newer, ignored):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(b"temporary test fixture; not model weights")

    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))
    os.utime(ignored, (300, 300))

    settings = get_settings()

    assert settings.behavior_model == newer.resolve()
    assert settings.behavior_model.is_file()


def test_explicit_behavior_model_overrides_discovery(tmp_path):
    """An explicit path passed to get_settings() bypasses auto-discovery."""
    fake_model = tmp_path / "fake_best.pt"
    fake_model.write_bytes(b"")  # create empty file so Path.resolve() works
    settings = get_settings(behavior_model=fake_model)
    assert settings.behavior_model == fake_model.resolve()


def test_env_behavior_model_overrides_discovery(monkeypatch, tmp_path):
    """VISIONCLASS_BEHAVIOR_MODEL env var bypasses auto-discovery."""
    fake_model = tmp_path / "env_best.pt"
    fake_model.write_bytes(b"")
    monkeypatch.setenv("VISIONCLASS_BEHAVIOR_MODEL", str(fake_model))
    settings = get_settings()
    assert settings.behavior_model == fake_model.resolve()


def test_no_hardcoded_windows_paths_in_settings():
    """All default paths in Settings must resolve under PROJECT_ROOT.

    This verifies that no absolute Windows paths from a specific developer's
    machine are baked into config.py defaults — all paths are derived from the
    dynamically-computed PROJECT_ROOT.
    """
    import inspect
    import src.config as cfg_module

    # Read the raw source of config.py and verify no absolute user paths appear
    source = inspect.getsource(cfg_module)
    assert "C:\\\\Users\\\\" not in source, \
        "Absolute Windows path found in config.py source"
    assert "C:/Users/" not in source, \
        "Absolute Windows path found in config.py source"

    # Also verify that all resolved default paths sit under PROJECT_ROOT
    settings = get_settings()
    root = settings.project_root
    path_fields = [
        settings.person_model,
        settings.pretrained_model,
        settings.video_path,
        settings.video_output,
        settings.tracking_output_video,
        settings.history_output_video,
        settings.history_csv,
        settings.summary_csv,
    ]
    for p in path_fields:
        try:
            p.relative_to(root)
        except ValueError:
            raise AssertionError(
                f"Default path is not under PROJECT_ROOT: {p}\n"
                f"PROJECT_ROOT: {root}"
            )


def test_inference_imgsz_default_is_960():
    """Verify that inference_imgsz is set to the validated 960 resolution."""
    settings = get_settings()
    assert settings.inference_imgsz == 960


def test_evaluation_imgsz_defaults_to_configured_inference_size():
    from src.evaluate_behavior import _resolve_evaluation_imgsz

    settings = get_settings()
    assert _resolve_evaluation_imgsz(None, settings.inference_imgsz) == 960


def test_evaluation_imgsz_explicit_override_is_preserved():
    from src.evaluate_behavior import _resolve_evaluation_imgsz

    settings = get_settings()
    assert _resolve_evaluation_imgsz(640, settings.inference_imgsz) == 640


def test_project_local_tracker_config():
    """Verify that the tracker defaults to the project-local config/bytetrack.yaml."""
    settings = get_settings()
    tracker_path = Path(settings.tracker)
    assert tracker_path.is_file(), f"Tracker config missing: {tracker_path}"
    assert tracker_path.parent.name == "config"
    assert tracker_path.name == "bytetrack.yaml"


def test_tracker_config_has_buffer_60():
    """Verify that the local tracker config specifies track_buffer: 60."""
    import yaml
    settings = get_settings()
    with open(settings.tracker, "r", encoding="utf-8") as f:
        tracker_cfg = yaml.safe_load(f)
    
    assert tracker_cfg.get("track_buffer") == 60, "track_buffer must be exactly 60"
    assert tracker_cfg.get("tracker_type") == "bytetrack"
