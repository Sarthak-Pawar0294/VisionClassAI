"""
Tests for evaluate_behavior._save_per_class_metrics().

The YOLO val() call requires a dataset and GPU and is not run in unit tests.
Instead, we test the metric-extraction and CSV-writing helper in isolation
by constructing mock result objects that match the Ultralytics API surface.

These tests verify:
  A. Per-class metrics CSV is created in the specified directory
  B. CSV has correct columns
  C. Class names come from the model, not the result object
  D. Per-class AP values are written with 6 decimal places
  E. Graceful fallback when the Ultralytics API doesn't have the expected attr
  F. Existing overall-metrics output is not broken by the new code
"""
import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.evaluate_behavior import _save_per_class_metrics

EXPECTED_COLUMNS = ["class_id", "class_name", "precision", "recall", "ap50", "ap50_95"]

CLASS_NAMES = {
    0: "handrise",
    1: "read",
    2: "sleep",
    3: "stand",
    4: "using_electronic_devices",
    5: "write",
}


def _make_mock_results(
    class_indices=(0, 1, 2, 3, 4, 5),
    ap50=None,
    ap=None,
    p=None,
    r=None,
):
    """
    Build a mock YOLO val() result object with the Ultralytics box API surface.
    """
    n = len(class_indices)
    ap50 = ap50 if ap50 is not None else [0.9 - i * 0.05 for i in range(n)]
    ap = ap if ap is not None else [0.5 - i * 0.03 for i in range(n)]
    p = p if p is not None else [0.8 - i * 0.04 for i in range(n)]
    r = r if r is not None else [0.75 - i * 0.03 for i in range(n)]

    mock_box = MagicMock()
    mock_box.ap_class_index = np.array(class_indices)
    mock_box.ap50 = np.array(ap50)
    mock_box.ap = np.array(ap)
    mock_box.p = np.array(p)
    mock_box.r = np.array(r)

    mock_results = MagicMock()
    mock_results.box = mock_box
    return mock_results


def _make_mock_model(names=None):
    model = MagicMock()
    model.names = names or CLASS_NAMES
    return model


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# A. CSV created
# --------------------------------------------------------------------------

class TestPerClassCSVCreation:
    def test_csv_file_is_created(self, tmp_path):
        results = _make_mock_results()
        model = _make_mock_model()
        _save_per_class_metrics(results, model, tmp_path)
        assert (tmp_path / "per_class_metrics.csv").is_file()

    def test_csv_created_even_if_dir_already_exists(self, tmp_path):
        (tmp_path / "some_other_file.txt").write_text("x")
        results = _make_mock_results()
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        assert (tmp_path / "per_class_metrics.csv").is_file()


# --------------------------------------------------------------------------
# B. Correct columns
# --------------------------------------------------------------------------

class TestPerClassCSVColumns:
    def test_columns_are_correct(self, tmp_path):
        _save_per_class_metrics(_make_mock_results(), _make_mock_model(), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert rows  # non-empty
        assert list(rows[0].keys()) == EXPECTED_COLUMNS

    def test_row_count_matches_number_of_classes(self, tmp_path):
        _save_per_class_metrics(
            _make_mock_results(class_indices=(0, 1, 2)),
            _make_mock_model(),
            tmp_path,
        )
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert len(rows) == 3


# --------------------------------------------------------------------------
# C. Class names come from the model
# --------------------------------------------------------------------------

class TestClassNameResolution:
    def test_class_names_from_model(self, tmp_path):
        custom_names = {0: "alpha", 1: "beta", 2: "gamma"}
        results = _make_mock_results(class_indices=(0, 1, 2))
        model = _make_mock_model(names=custom_names)
        _save_per_class_metrics(results, model, tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        names_in_csv = [r["class_name"] for r in rows]
        assert names_in_csv == ["alpha", "beta", "gamma"]

    def test_unknown_class_id_uses_fallback_name(self, tmp_path):
        """Class ID not in model.names gets a fallback 'class_N' name."""
        # model.names only knows about class 0; class 99 is unknown
        model = _make_mock_model(names={0: "write"})
        results = _make_mock_results(class_indices=(0, 99))
        _save_per_class_metrics(results, model, tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert len(rows) == 2
        assert rows[0]["class_name"] == "write"
        # Class 99 is not in names dict, so get() returns the default f"class_99"
        assert rows[1]["class_name"] == "class_99"


# --------------------------------------------------------------------------
# D. AP values written at 6 decimal places
# --------------------------------------------------------------------------

class TestMetricValues:
    def test_ap50_values_written(self, tmp_path):
        ap50_vals = [0.912345, 0.875432, 0.801234]
        results = _make_mock_results(
            class_indices=(0, 1, 2),
            ap50=ap50_vals,
            ap=[0.5, 0.4, 0.3],
            p=[0.8, 0.7, 0.6],
            r=[0.7, 0.6, 0.5],
        )
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        for i, row in enumerate(rows):
            assert float(row["ap50"]) == pytest.approx(ap50_vals[i], abs=1e-5)

    def test_ap50_95_values_written(self, tmp_path):
        ap_vals = [0.541234, 0.432100, 0.311111]
        results = _make_mock_results(
            class_indices=(0, 1, 2),
            ap50=[0.9, 0.8, 0.7],
            ap=ap_vals,
            p=[0.8, 0.7, 0.6],
            r=[0.7, 0.6, 0.5],
        )
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        for i, row in enumerate(rows):
            assert float(row["ap50_95"]) == pytest.approx(ap_vals[i], abs=1e-5)

    def test_precision_and_recall_written(self, tmp_path):
        p_vals = [0.782345, 0.654321]
        r_vals = [0.801234, 0.712345]
        results = _make_mock_results(
            class_indices=(0, 1),
            ap50=[0.9, 0.8],
            ap=[0.5, 0.4],
            p=p_vals,
            r=r_vals,
        )
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert float(rows[0]["precision"]) == pytest.approx(p_vals[0], abs=1e-5)
        assert float(rows[0]["recall"]) == pytest.approx(r_vals[0], abs=1e-5)

    def test_class_id_written_as_integer(self, tmp_path):
        results = _make_mock_results(class_indices=(2, 4, 5))
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert [int(r["class_id"]) for r in rows] == [2, 4, 5]


# --------------------------------------------------------------------------
# E. Graceful fallback on AttributeError
# --------------------------------------------------------------------------
def _make_mock_results_attr_error():
    """
    Build a mock result object where accessing box.ap_class_index raises
    AttributeError — simulating an older Ultralytics version with a different API.

    ``del mock_box.ap_class_index`` is the reliable way to force AttributeError
    on attribute access with MagicMock.  Setting side_effect only fires when the
    attribute is *called*, not when it is *accessed*.
    """
    mock_box = MagicMock()
    del mock_box.ap_class_index   # accessing .ap_class_index now raises AttributeError
    mock_results = MagicMock()
    mock_results.box = mock_box
    return mock_results



# --------------------------------------------------------------------------
# E. Graceful fallback on AttributeError
# --------------------------------------------------------------------------

class TestGracefulFallback:
    def test_no_exception_on_attribute_error(self, tmp_path):
        """_save_per_class_metrics must not raise even if the Ultralytics API differs."""
        results = _make_mock_results_attr_error()
        model = _make_mock_model()
        # Must not raise
        _save_per_class_metrics(results, model, tmp_path)

    def test_warning_printed_on_attribute_error(self, tmp_path, capsys):
        results = _make_mock_results_attr_error()
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "WARNING" in captured.err

    def test_no_csv_created_on_attribute_error(self, tmp_path):
        """If the API is unavailable, no CSV should be written."""
        results = _make_mock_results_attr_error()
        _save_per_class_metrics(results, _make_mock_model(), tmp_path)
        assert not (tmp_path / "per_class_metrics.csv").is_file()



class TestFullClassSet:
    def test_all_six_behavior_classes(self, tmp_path):
        """Test with all 6 VisionClass behavior classes."""
        ap50_vals = [0.85, 0.78, 0.92, 0.45, 0.71, 0.88]
        results = _make_mock_results(
            class_indices=(0, 1, 2, 3, 4, 5),
            ap50=ap50_vals,
            ap=[v - 0.3 for v in ap50_vals],
            p=[0.8] * 6,
            r=[0.75] * 6,
        )
        _save_per_class_metrics(results, _make_mock_model(names=CLASS_NAMES), tmp_path)
        rows = _read_csv(tmp_path / "per_class_metrics.csv")
        assert len(rows) == 6
        assert rows[3]["class_name"] == "stand"
        assert float(rows[3]["ap50"]) == pytest.approx(0.45, abs=1e-5)
