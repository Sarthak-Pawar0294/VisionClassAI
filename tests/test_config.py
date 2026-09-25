"""Tests for configuration helpers (no GPU required)."""

import os

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


def test_behavior_model_discovery_finds_existing_best_pt():
    """Auto-discovery must resolve to a best.pt that actually exists on disk."""
    settings = get_settings()
    assert settings.behavior_model.is_file(), (
        f"Auto-discovered behavior_model does not exist: {settings.behavior_model}\n"
        "Set VISIONCLASS_BEHAVIOR_MODEL=<path/to/best.pt> or re-train the model."
    )


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

