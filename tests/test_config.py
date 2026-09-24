"""Tests for configuration helpers (no GPU required)."""

from src.config import PROJECT_ROOT, resolve_device


def test_project_root_is_visionclass():
    assert PROJECT_ROOT.name == "VisionClass"
    assert (PROJECT_ROOT / "src" / "config.py").is_file()


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_cpu_or_int():
    device = resolve_device("auto")
    assert device == "cpu" or isinstance(device, int)
