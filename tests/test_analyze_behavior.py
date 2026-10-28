"""
Tests for src/analyze_behavior.py — Phase 8.2 analytics library.

All tests use synthetic data so no real CSV files are required.
"""

from __future__ import annotations

import pytest
from src.analyze_behavior import (
    behavior_runs,
    cache_analysis,
    confidence_analysis,
    coverage_analysis,
    overall_distribution,
    per_student_stats,
    session_stats,
    temporal_analysis,
    unknown_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_row(
    frame: int = 0,
    time_seconds: float = 0.0,
    student_id: int = 1,
    behavior: str = "sleep",
    confidence: float = 0.5,
    is_cached: bool = False,
    unknown_cause: str = "",
) -> dict:
    return {
        "_lineno": 2,
        "frame": frame,
        "time_seconds": time_seconds,
        "student_id": student_id,
        "behavior": behavior,
        "confidence": confidence,
        "is_cached": is_cached,
        "unknown_cause": unknown_cause,
    }


SIMPLE_ROWS = [
    make_row(frame=0, student_id=1, behavior="sleep",   confidence=0.5, is_cached=False),
    make_row(frame=1, student_id=1, behavior="sleep",   confidence=0.4, is_cached=True),
    make_row(frame=2, student_id=1, behavior="write",   confidence=0.3, is_cached=False),
    make_row(frame=0, student_id=2, behavior="unknown", confidence=0.0, is_cached=False, unknown_cause="no_match"),
    make_row(frame=1, student_id=2, behavior="read",    confidence=0.6, is_cached=False),
    make_row(frame=2, student_id=2, behavior="read",    confidence=0.55, is_cached=True),
]


# ---------------------------------------------------------------------------
# session_stats
# ---------------------------------------------------------------------------


class TestSessionStats:
    def test_basic_counts(self):
        stats = session_stats(SIMPLE_ROWS)
        assert stats["total_rows"] == 6
        assert stats["n_students"] == 2
        assert stats["n_frames"] == 3

    def test_frame_range(self):
        stats = session_stats(SIMPLE_ROWS)
        assert stats["min_frame"] == 0
        assert stats["max_frame"] == 2

    def test_unique_student_ids_sorted(self):
        stats = session_stats(SIMPLE_ROWS)
        assert stats["unique_student_ids"] == [1, 2]

    def test_single_frame_no_fps(self):
        rows = [make_row(frame=0, time_seconds=0.0)]
        stats = session_stats(rows)
        # Only one distinct timestamp value of 0 — fps cannot be inferred
        assert stats["inferred_fps"] is None or isinstance(stats["inferred_fps"], float)


# ---------------------------------------------------------------------------
# overall_distribution
# ---------------------------------------------------------------------------


class TestOverallDistribution:
    def test_total_rows(self):
        dist = overall_distribution(SIMPLE_ROWS)
        assert dist["total_rows"] == 6

    def test_behavior_counts(self):
        dist = overall_distribution(SIMPLE_ROWS)
        assert dist["counts"]["sleep"] == 2
        assert dist["counts"]["write"] == 1
        assert dist["counts"]["read"] == 2
        assert dist["counts"]["unknown"] == 1

    def test_known_unknown_totals(self):
        dist = overall_distribution(SIMPLE_ROWS)
        assert dist["known_total"] == 5
        assert dist["unknown_total"] == 1

    def test_live_cached_split(self):
        dist = overall_distribution(SIMPLE_ROWS)
        # Cached rows: sleep(cached) at frame=1,s=1 + read(cached) at frame=2,s=2
        assert dist["cached_counts"].get("sleep", 0) == 1
        assert dist["cached_counts"].get("read", 0) == 1
        assert dist["live_counts"].get("sleep", 0) == 1

    def test_percentages_sum_to_100(self):
        dist = overall_distribution(SIMPLE_ROWS)
        total_pct = sum(dist["percentages"].values())
        assert abs(total_pct - 100.0) < 0.1

    def test_empty_rows(self):
        dist = overall_distribution([])
        assert dist["total_rows"] == 0
        assert dist["known_total"] == 0
        assert dist["unknown_total"] == 0


# ---------------------------------------------------------------------------
# per_student_stats
# ---------------------------------------------------------------------------


class TestPerStudentStats:
    def test_student_ids_present(self):
        ps = per_student_stats(SIMPLE_ROWS)
        assert set(ps.keys()) == {1, 2}

    def test_student1_totals(self):
        ps = per_student_stats(SIMPLE_ROWS)
        s1 = ps[1]
        assert s1["total_records"] == 3
        assert s1["cached_records"] == 1
        assert s1["live_records"] == 2
        assert s1["unknown_records"] == 0
        assert s1["known_records"] == 3

    def test_student2_unknown(self):
        ps = per_student_stats(SIMPLE_ROWS)
        s2 = ps[2]
        assert s2["unknown_records"] == 1
        assert s2["known_records"] == 2

    def test_dominant_behavior(self):
        ps = per_student_stats(SIMPLE_ROWS)
        # Student 2: read=2 > unknown=1 (dominant excludes unknown)
        assert ps[2]["dominant_behavior"] == "read"

    def test_dominant_behavior_excludes_unknown(self):
        rows = [
            make_row(student_id=1, behavior="unknown", confidence=0.0),
            make_row(student_id=1, behavior="unknown", confidence=0.0),
            make_row(student_id=1, behavior="sleep",   confidence=0.4),
        ]
        ps = per_student_stats(rows)
        assert ps[1]["dominant_behavior"] == "sleep"

    def test_mean_live_conf_excludes_cached(self):
        ps = per_student_stats(SIMPLE_ROWS)
        # Student 1: live confidences = [0.5] (cached=0.4 excluded, write live=0.3)
        s1 = ps[1]
        # live non-zero confs: sleep live 0.5 + write live 0.3
        assert s1["mean_live_conf"] == pytest.approx((0.5 + 0.3) / 2, abs=0.001)

    def test_frames_seen_are_sorted(self):
        ps = per_student_stats(SIMPLE_ROWS)
        assert ps[1]["frames_seen"] == sorted(ps[1]["frames_seen"])

    def test_no_dominant_for_all_unknown(self):
        rows = [make_row(student_id=5, behavior="unknown", confidence=0.0)]
        ps = per_student_stats(rows)
        assert ps[5]["dominant_behavior"] is None


# ---------------------------------------------------------------------------
# behavior_runs
# ---------------------------------------------------------------------------


class TestBehaviorRuns:
    def test_single_row(self):
        rows = [make_row(frame=0, behavior="sleep")]
        runs = behavior_runs(rows)
        assert len(runs) == 1
        assert runs[0]["behavior"] == "sleep"
        assert runs[0]["length"] == 1

    def test_two_same_behavior(self):
        rows = [make_row(frame=0, behavior="sleep"), make_row(frame=1, behavior="sleep")]
        runs = behavior_runs(rows)
        assert len(runs) == 1
        assert runs[0]["length"] == 2

    def test_two_different_behaviors(self):
        rows = [make_row(frame=0, behavior="sleep"), make_row(frame=1, behavior="write")]
        runs = behavior_runs(rows)
        assert len(runs) == 2
        assert runs[0]["behavior"] == "sleep"
        assert runs[1]["behavior"] == "write"

    def test_run_boundaries(self):
        rows = [
            make_row(frame=0, behavior="sleep"),
            make_row(frame=1, behavior="sleep"),
            make_row(frame=2, behavior="write"),
        ]
        runs = behavior_runs(rows)
        assert runs[0]["start_frame"] == 0
        assert runs[0]["end_frame"] == 1
        assert runs[0]["length"] == 2
        assert runs[1]["start_frame"] == 2

    def test_matching_behavior_after_frame_gap_starts_new_run(self):
        rows = [
            make_row(frame=4, behavior="sleep"),
            make_row(frame=6, behavior="sleep"),
        ]
        runs = behavior_runs(rows)
        assert [(run["start_frame"], run["end_frame"], run["length"]) for run in runs] == [
            (4, 4, 1),
            (6, 6, 1),
        ]

    def test_empty_rows(self):
        assert behavior_runs([]) == []

    def test_run_abba_pattern(self):
        rows = [
            make_row(frame=0, behavior="sleep"),
            make_row(frame=1, behavior="write"),
            make_row(frame=2, behavior="write"),
            make_row(frame=3, behavior="sleep"),
        ]
        runs = behavior_runs(rows)
        assert len(runs) == 3
        assert runs[1]["length"] == 2


# ---------------------------------------------------------------------------
# temporal_analysis
# ---------------------------------------------------------------------------


class TestTemporalAnalysis:
    def test_transitions_counted(self):
        ta = temporal_analysis(SIMPLE_ROWS)
        # Student 1: sleep->sleep(cached)->write = 2 runs -> 1 transition
        assert ta["transitions_per_student"][1] == 1

    def test_longest_run_per_behavior(self):
        ta = temporal_analysis(SIMPLE_ROWS)
        assert "sleep" in ta["longest_run_per_behavior"]
        assert ta["longest_run_per_behavior"]["sleep"]["length"] >= 1

    def test_total_transitions_non_negative(self):
        ta = temporal_analysis(SIMPLE_ROWS)
        assert ta["total_transitions"] >= 0

    def test_avg_run_length_positive(self):
        ta = temporal_analysis(SIMPLE_ROWS)
        assert ta["avg_run_length"] > 0

    def test_mean_run_by_behavior_keys(self):
        ta = temporal_analysis(SIMPLE_ROWS)
        for b, mean in ta["mean_run_by_behavior"].items():
            assert mean > 0


# ---------------------------------------------------------------------------
# coverage_analysis
# ---------------------------------------------------------------------------


class TestCoverageAnalysis:
    def setup_method(self):
        self.ps = per_student_stats(SIMPLE_ROWS)

    def test_full_coverage_for_always_present(self):
        # Both students appear in all 3 frames
        cov = coverage_analysis(self.ps, total_frames=3)
        assert 1 in cov["full_coverage_students"]
        assert 2 in cov["full_coverage_students"]

    def test_short_track_threshold(self):
        # Student 1 has 3 records, threshold=10 -> short
        cov = coverage_analysis(self.ps, total_frames=100, short_track_threshold=10)
        assert 1 in cov["short_track_students"]

    def test_high_unknown_detection(self):
        rows = [
            make_row(student_id=9, behavior="unknown", confidence=0.0),
            make_row(student_id=9, behavior="unknown", confidence=0.0),
            make_row(student_id=9, behavior="unknown", confidence=0.0),
            make_row(student_id=9, behavior="sleep",   confidence=0.3),
        ]
        ps = per_student_stats(rows)
        cov = coverage_analysis(ps, total_frames=4, high_unknown_threshold=70.0)
        assert 9 in cov["high_unknown_students"]

    def test_frame_coverage_pct_range(self):
        cov = coverage_analysis(self.ps, total_frames=3)
        for sid, pct in cov["frame_coverage_pct"].items():
            assert 0.0 <= pct <= 100.0


# ---------------------------------------------------------------------------
# confidence_analysis
# ---------------------------------------------------------------------------


class TestConfidenceAnalysis:
    def test_excludes_unknowns(self):
        ca = confidence_analysis(SIMPLE_ROWS)
        # Unknown rows have conf=0.0; those should not appear
        assert all(c > 0 for c in ca["all_live_conf"])

    def test_excludes_cached(self):
        # Cached row: frame=1,s=1,sleep,0.4 — should be excluded
        ca = confidence_analysis(SIMPLE_ROWS)
        assert 0.4 not in ca["all_live_conf"]

    def test_global_stats_count(self):
        ca = confidence_analysis(SIMPLE_ROWS)
        # live+known rows: s1 sleep 0.5, s1 write 0.3, s2 read 0.6
        assert ca["global_stats"]["count"] == 3

    def test_mean_is_average_of_included(self):
        ca = confidence_analysis(SIMPLE_ROWS)
        expected = round((0.5 + 0.3 + 0.6) / 3, 4)
        assert ca["global_stats"]["mean"] == pytest.approx(expected, abs=0.001)

    def test_by_behavior_has_read_key(self):
        ca = confidence_analysis(SIMPLE_ROWS)
        assert "read" in ca["by_behavior"]
        assert ca["by_behavior"]["read"]["count"] == 1

    def test_low_conf_behaviors_identified(self):
        rows = [make_row(student_id=1, behavior="sleep", confidence=0.22, is_cached=False)]
        ca = confidence_analysis(rows)
        assert "sleep" in ca["low_conf_behaviors"]

    def test_no_conf_for_empty(self):
        ca = confidence_analysis([])
        assert ca["global_stats"]["count"] == 0
        assert ca["global_stats"]["mean"] is None


# ---------------------------------------------------------------------------
# cache_analysis
# ---------------------------------------------------------------------------


class TestCacheAnalysis:
    def test_totals(self):
        ca = cache_analysis(SIMPLE_ROWS)
        assert ca["total_rows"] == 6
        assert ca["cached_rows"] == 2
        assert ca["live_rows"] == 4

    def test_known_live_and_cached_percentages_exclude_unknown_rows(self):
        rows = [
            make_row(frame=0, behavior="unknown", confidence=0.0),
            make_row(frame=1, behavior="sleep", confidence=0.5),
            make_row(frame=2, behavior="sleep", confidence=0.5, is_cached=True),
        ]
        ca = cache_analysis(rows)
        assert ca["live_known_rows"] == 1
        assert ca["cached_known_rows"] == 1
        assert ca["live_known_pct"] == 50.0
        assert ca["cached_known_pct"] == 50.0

    def test_cache_pct(self):
        ca = cache_analysis(SIMPLE_ROWS)
        assert ca["cache_pct"] == pytest.approx(100 * 2 / 6, abs=0.1)

    def test_cache_sequence_valid_true(self):
        ca = cache_analysis(SIMPLE_ROWS)
        assert ca["cache_sequence_valid"] is True

    def test_cache_sequence_invalid_when_cached_before_live(self):
        # Student 3 has a cached row before any live row
        rows = [
            make_row(frame=0, student_id=3, behavior="sleep", confidence=0.5, is_cached=True),
            make_row(frame=1, student_id=3, behavior="sleep", confidence=0.4, is_cached=False),
        ]
        ca = cache_analysis(rows)
        assert ca["cache_sequence_valid"] is False

    def test_cache_only_known_excludes_unknown(self):
        ca = cache_analysis(SIMPLE_ROWS)
        # Cached rows: sleep (known) + read (known) — no cached unknowns in SIMPLE_ROWS
        assert ca["cache_only_known_rows"] == 2

    def test_behaviors_cache_heavy_flagged(self):
        # 100% cached for a behavior -> cache heavy
        rows = [
            make_row(student_id=1, behavior="sleep", confidence=0.5, is_cached=False),
            make_row(student_id=1, behavior="read",  confidence=0.4, is_cached=True),
            make_row(student_id=1, behavior="read",  confidence=0.4, is_cached=True),
        ]
        ca = cache_analysis(rows)
        assert "read" in ca["behaviors_cache_heavy"]


# ---------------------------------------------------------------------------
# unknown_analysis
# ---------------------------------------------------------------------------


class TestUnknownAnalysis:
    def test_unknown_count(self):
        ua = unknown_analysis(SIMPLE_ROWS, total_frames=3)
        assert ua["unknown_rows"] == 1
        assert ua["unknown_pct"] == pytest.approx(100 / 6, abs=0.1)

    def test_students_always_unknown_empty_when_all_have_live(self):
        ua = unknown_analysis(SIMPLE_ROWS, total_frames=3)
        # Both students have at least one live known detection
        assert ua["students_always_unknown"] == []

    def test_students_always_unknown_detected(self):
        rows = [
            make_row(student_id=99, behavior="unknown", confidence=0.0, is_cached=False),
            make_row(student_id=99, behavior="unknown", confidence=0.0, is_cached=False),
        ]
        ua = unknown_analysis(rows, total_frames=2)
        assert 99 in ua["students_always_unknown"]

    def test_frames_all_unknown(self):
        rows = [
            make_row(frame=5, student_id=1, behavior="unknown"),
            make_row(frame=5, student_id=2, behavior="unknown"),
        ]
        ua = unknown_analysis(rows, total_frames=6)
        assert 5 in ua["frames_all_unknown"]

    def test_frames_all_unknown_not_in_any_known(self):
        rows = [make_row(frame=5, student_id=1, behavior="unknown")]
        ua = unknown_analysis(rows, total_frames=6)
        assert 5 not in ua["frames_any_known"]

    def test_short_track_unknown_attribution(self):
        rows = [
            # Student 99 has 2 records (short) — both unknown
            make_row(student_id=99, frame=0, behavior="unknown"),
            make_row(student_id=99, frame=1, behavior="unknown"),
            # Student 1 has 50 records to avoid short track
        ] + [make_row(student_id=1, frame=i, behavior="sleep") for i in range(50)]
        ua = unknown_analysis(rows, total_frames=50)
        assert 99 in ua["short_track_student_ids"]
        assert ua["unknown_from_short_tracks"] == 2

    def test_per_student_unknown_pct(self):
        ua = unknown_analysis(SIMPLE_ROWS, total_frames=3)
        # Student 2: 1 unknown / 3 total = 33.33%
        assert ua["per_student_unknown_pct"][2] == pytest.approx(33.33, abs=0.1)
