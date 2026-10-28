"""Standalone maintenance tools must not do work merely because they are imported."""

from __future__ import annotations

import runpy
import shutil
import sys
import types
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STANDALONE_TOOLS = [
    "detect_students.py",
    "feature_extraction.py",
    "features_test.py",
    "src/analyze_device_box_sizes.py",
    "src/analyze_image_resolutions.py",
    "src/benchmark_phase6_imgsz.py",
    "src/analyze_errors.py",
    "src/make_error_contact_sheet.py",
]


def _forbidden(*args, **kwargs):
    raise AssertionError("A standalone tool performed work during import")


@pytest.mark.parametrize("relative_path", STANDALONE_TOOLS)
def test_standalone_tools_have_no_import_time_side_effects(
    relative_path, monkeypatch, capsys
):
    """Importing a tool must not touch devices, models, datasets, or outputs."""
    cv2 = types.ModuleType("cv2")
    for name in (
        "VideoCapture",
        "imread",
        "imwrite",
        "imshow",
        "destroyAllWindows",
    ):
        setattr(cv2, name, _forbidden)

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=_forbidden,
        get_device_name=_forbidden,
    )

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = _forbidden

    mediapipe = types.ModuleType("mediapipe")
    yaml = types.ModuleType("yaml")
    yaml.safe_load = _forbidden

    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.open = _forbidden
    image_draw = types.ModuleType("PIL.ImageDraw")
    image_draw.Draw = _forbidden
    pil.Image = image
    pil.ImageDraw = image_draw

    for name, module in {
        "cv2": cv2,
        "torch": torch,
        "ultralytics": ultralytics,
        "mediapipe": mediapipe,
        "yaml": yaml,
        "PIL": pil,
        "PIL.Image": image,
        "PIL.ImageDraw": image_draw,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    for method in ("glob", "rglob", "iterdir", "mkdir", "write_text", "write_bytes"):
        monkeypatch.setattr(Path, method, _forbidden)
    monkeypatch.setattr(shutil, "copytree", _forbidden)

    script_path = PROJECT_ROOT / relative_path
    runpy.run_path(str(script_path), run_name="_visionclass_import_probe")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
