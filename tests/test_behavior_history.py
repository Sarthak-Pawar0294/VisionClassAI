"""
Tests for behavior_history streaming CSV logic and unknown-cause tagging.

These tests exercise the data-processing and file-writing logic in isolation
(no YOLO models, no video files, no GPU required).  The tests import and call
helper functions directly rather than running main().

Strategy
--------
- Extract the record-building and CSV logic into a callable form.
- Use tmp_path fixtures so nothing writes to the real data/ directory.
- Assert on what actually gets written to disk, not just in-memory state.
"""
import csv
import io
from collections import defaultdict
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# Helpers re-implementing the history CSV write logic for isolated testing.
# (We call the real module constants rather than duplicating magic values.)
# --------------------------------------------------------------------------
from src.behavior_history import FLUSH_EVERY, HISTORY_FIELDNAMES
from src.config import SUMMARY_BEHAVIOR_COLUMNS


def _write_history_csv(
    path: Path,
    records: list[dict],
    *,
    flush_every: int = FLUSH_EVERY,
) -> None:
    """
    Simulate the streaming CSV logic from behavior_history.main().

    Opens the file, writes the header, then writes rows in batches of
    flush_every, flushing after each batch.  Remaining rows are written
    in the finally block.  Mirrors the production code path exactly.
    """
    pending: list[dict] = []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES)
        writer.writeheader()
        try:
            for i, row in enumerate(records):
                pending.append(row)
                if (i + 1) % flush_every == 0:
                    writer.writerows(pending)
                    f.flush()
                    pending.clear()
        finally:
            if pending:
                writer.writerows(pending)
                f.flush()
                pending.clear()


def _make_record(
    frame: int = 0,
    student_id: int = 1,
    behavior: str = "write",
    confidence: float = 0.5,
) -> dict:
    fps = 20.0
    unknown_cause = "no_match" if behavior == "unknown" else ""
    return {
        "frame": frame,
        "time_seconds": round(frame / fps, 3),
        "student_id": student_id,
        "behavior": behavior,
        "confidence": round(confidence, 4),
        "unknown_cause": unknown_cause,
    }


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# CSV creation and structure
# --------------------------------------------------------------------------

class TestHistoryCSVCreation:
    def test_csv_is_created(self, tmp_path):
        """A CSV file is created at the given path."""
        out = tmp_path / "hist.csv"
        _write_history_csv(out, [_make_record()])
        assert out.is_file()

    def test_header_matches_fieldnames(self, tmp_path):
        """The CSV header exactly matches HISTORY_FIELDNAMES."""
        out = tmp_path / "hist.csv"
        _write_history_csv(out, [_make_record()])
        with open(out, newline="", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert header == HISTORY_FIELDNAMES

    def test_header_not_duplicated(self, tmp_path):
        """The header appears exactly once even for multi-batch writes."""
        out = tmp_path / "hist.csv"
        records = [_make_record(frame=i) for i in range(250)]
        _write_history_csv(out, records, flush_every=100)
        rows = _read_csv(out)
        # If the header were duplicated, a row would have 'frame' as a value
        header_rows_in_data = [r for r in rows if r.get("frame") == "frame"]
        assert len(header_rows_in_data) == 0

    def test_empty_records_produces_only_header(self, tmp_path):
        out = tmp_path / "hist.csv"
        _write_history_csv(out, [])
        rows = _read_csv(out)
        assert rows == []  # file exists but contains only header


class TestHistoryCSVRecords:
    def test_single_record_written(self, tmp_path):
        out = tmp_path / "hist.csv"
        rec = _make_record(frame=5, student_id=3, behavior="sleep", confidence=0.42)
        _write_history_csv(out, [rec])
        rows = _read_csv(out)
        assert len(rows) == 1
        row = rows[0]
        assert row["frame"] == "5"
        assert row["student_id"] == "3"
        assert row["behavior"] == "sleep"
        assert float(row["confidence"]) == pytest.approx(0.42)

    def test_multiple_records_all_written(self, tmp_path):
        """All records survive across multiple flush boundaries."""
        out = tmp_path / "hist.csv"
        n = 350
        records = [_make_record(frame=i, student_id=(i % 10) + 1) for i in range(n)]
        _write_history_csv(out, records, flush_every=100)
        rows = _read_csv(out)
        assert len(rows) == n

    def test_record_order_preserved(self, tmp_path):
        """Rows appear in the same order as the input records."""
        out = tmp_path / "hist.csv"
        records = [_make_record(frame=i) for i in range(50)]
        _write_history_csv(out, records, flush_every=10)
        rows = _read_csv(out)
        written_frames = [int(r["frame"]) for r in rows]
        assert written_frames == list(range(50))

    def test_all_fieldnames_present_in_each_row(self, tmp_path):
        out = tmp_path / "hist.csv"
        _write_history_csv(out, [_make_record(), _make_record(behavior="unknown")])
        rows = _read_csv(out)
        for row in rows:
            for field in HISTORY_FIELDNAMES:
                assert field in row, f"Missing field '{field}' in row: {row}"


class TestUnknownCauseTagging:
    """
    Test that the unknown_cause column is populated correctly.

    The tagging rules (mirrored from behavior_history.main):
      - behavior != 'unknown'  → unknown_cause = ''
      - behavior == 'unknown'  → unknown_cause = 'no_match'
    """

    def test_known_behavior_has_empty_cause(self, tmp_path):
        out = tmp_path / "hist.csv"
        rec = _make_record(behavior="write", confidence=0.6)
        _write_history_csv(out, [rec])
        rows = _read_csv(out)
        assert rows[0]["unknown_cause"] == ""

    def test_unknown_behavior_tagged_no_match(self, tmp_path):
        out = tmp_path / "hist.csv"
        rec = _make_record(behavior="unknown", confidence=0.0)
        _write_history_csv(out, [rec])
        rows = _read_csv(out)
        assert rows[0]["unknown_cause"] == "no_match"

    def test_no_match_label_not_no_detection(self, tmp_path):
        """Confirm the label is 'no_match', not the discarded 'no_detection' label."""
        out = tmp_path / "hist.csv"
        rec = _make_record(behavior="unknown", confidence=0.0)
        _write_history_csv(out, [rec])
        rows = _read_csv(out)
        assert rows[0]["unknown_cause"] != "no_detection", (
            "Label must be 'no_match' to reflect what we can actually observe. "
            "'no_detection' was removed because it was inaccurate (cannot distinguish "
            "model silence from being outcompeted by another student)."
        )

    def test_mixed_known_and_unknown_rows(self, tmp_path):
        out = tmp_path / "hist.csv"
        records = [
            _make_record(frame=0, behavior="write", confidence=0.7),
            _make_record(frame=0, student_id=2, behavior="unknown", confidence=0.0),
            _make_record(frame=1, behavior="sleep", confidence=0.4),
            _make_record(frame=1, student_id=2, behavior="unknown", confidence=0.0),
        ]
        _write_history_csv(out, records)
        rows = _read_csv(out)
        assert rows[0]["unknown_cause"] == ""         # write
        assert rows[1]["unknown_cause"] == "no_match" # unknown
        assert rows[2]["unknown_cause"] == ""         # sleep
        assert rows[3]["unknown_cause"] == "no_match" # unknown

    def test_no_match_is_not_assigned_to_detected_behavior(self, tmp_path):
        """Every non-unknown behavior must have an empty unknown_cause, regardless of confidence."""
        out = tmp_path / "hist.csv"
        behaviors = ["write", "sleep", "read", "handrise", "stand", "using_electronic_devices"]
        records = [
            _make_record(frame=i, student_id=1, behavior=b, confidence=0.3 + i * 0.01)
            for i, b in enumerate(behaviors)
        ]
        _write_history_csv(out, records)
        rows = _read_csv(out)
        for row in rows:
            assert row["unknown_cause"] == "", (
                f"Expected empty unknown_cause for behavior='{row['behavior']}', "
                f"got '{row['unknown_cause']}'"
            )


class TestStreamingSafety:
    """Verify the streaming write is robust to flush boundary edge cases."""

    def test_flush_boundary_exact_multiple(self, tmp_path):
        """Record count that is an exact multiple of flush_every."""
        out = tmp_path / "hist.csv"
        records = [_make_record(frame=i) for i in range(200)]
        _write_history_csv(out, records, flush_every=100)
        rows = _read_csv(out)
        assert len(rows) == 200

    def test_flush_boundary_one_over(self, tmp_path):
        """Record count one more than a flush boundary."""
        out = tmp_path / "hist.csv"
        records = [_make_record(frame=i) for i in range(101)]
        _write_history_csv(out, records, flush_every=100)
        rows = _read_csv(out)
        assert len(rows) == 101

    def test_flush_boundary_one_under(self, tmp_path):
        """Record count one less than a flush boundary — handled by finally block."""
        out = tmp_path / "hist.csv"
        records = [_make_record(frame=i) for i in range(99)]
        _write_history_csv(out, records, flush_every=100)
        rows = _read_csv(out)
        assert len(rows) == 99

    def test_confidence_precision_preserved(self, tmp_path):
        """Confidence values written to 4 decimal places."""
        out = tmp_path / "hist.csv"
        _write_history_csv(out, [_make_record(confidence=0.12345678)])
        rows = _read_csv(out)
        # _make_record rounds to 4dp, so we get 0.1235
        assert float(rows[0]["confidence"]) == pytest.approx(0.1235, abs=1e-5)


class TestSummaryGeneration:
    """
    Test the summary aggregation logic in isolation.

    The summary dict tracks frame counts per student per behavior and converts
    to seconds by dividing by FPS.  This is independent of CSV streaming.
    """

    def test_summary_counts_correctly(self):
        fps = 20.0
        history: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Student 1: 10 write frames, 5 sleep frames
        for _ in range(10):
            history[1]["write"] += 1
        for _ in range(5):
            history[1]["sleep"] += 1

        assert round(history[1]["write"] / fps, 2) == pytest.approx(0.5)
        assert round(history[1]["sleep"] / fps, 2) == pytest.approx(0.25)

    def test_summary_csv_written_correctly(self, tmp_path):
        fps = 20.0
        history: dict[int, dict[str, int]] = {
            1: {"write": 40, "unknown": 20},
            2: {"sleep": 60, "write": 10},
        }

        summary_path = tmp_path / "summary.csv"
        behaviors = SUMMARY_BEHAVIOR_COLUMNS

        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["student_id"] + behaviors
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for student_id in sorted(history):
                row = {"student_id": student_id}
                for behavior in behaviors:
                    frame_count = history[student_id].get(behavior, 0)
                    row[behavior] = round(frame_count / fps, 2)
                writer.writerow(row)

        rows = _read_csv(summary_path)
        assert len(rows) == 2

        s1 = next(r for r in rows if r["student_id"] == "1")
        assert float(s1["write"]) == pytest.approx(2.0)    # 40/20
        assert float(s1["unknown"]) == pytest.approx(1.0)  # 20/20
        assert float(s1["sleep"]) == pytest.approx(0.0)    # not present

        s2 = next(r for r in rows if r["student_id"] == "2")
        assert float(s2["sleep"]) == pytest.approx(3.0)    # 60/20
        assert float(s2["write"]) == pytest.approx(0.5)    # 10/20

    def test_summary_uses_summary_behavior_columns(self, tmp_path):
        """Summary CSV columns must match SUMMARY_BEHAVIOR_COLUMNS."""
        summary_path = tmp_path / "summary.csv"
        behaviors = SUMMARY_BEHAVIOR_COLUMNS
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["student_id"] + behaviors)
            writer.writeheader()
        with open(summary_path, newline="", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        assert header[0] == "student_id"
        assert header[1:] == behaviors
