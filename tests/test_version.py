from pathlib import Path

import src


def test_package_version_matches_repository_version_file():
    version_file = Path(src.__file__).resolve().parent.parent / "VERSION"

    assert src.__version__ == version_file.read_text(encoding="utf-8").strip()
    assert src.__version__
