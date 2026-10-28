"""VisionClass — classroom behavior detection pipeline."""

from pathlib import Path


_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
try:
    __version__ = _VERSION_FILE.read_text(encoding="utf-8").strip()
except OSError:
    # Keep source checkouts importable if the optional root version file is absent.
    __version__ = "0+unknown"
