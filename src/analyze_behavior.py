"""
src/analyze_behavior.py
=======================
Reusable behavioral-analytics library for VisionClass Phase 8.2.

All public functions are pure: they accept Python data structures (lists /
dicts) and return plain dicts / lists.  No CSV I/O, no printing.  This keeps
the functions trivially unit-testable.

I/O helpers (load_history, load_summary) are separated at the bottom
so tests can supply synthetic data without touching the filesystem.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Behavior constants
VALID_BEHAVIORS: frozenset[str] = frozenset(
    {"handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown"}
)
KNOWN_BEHAVIORS: frozenset[str] = VALID_BEHAVIORS - {"unknown"}
SUMMARY_BEHAVIOR_COLUMNS: list[str] = [
    "handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown",
]

Row = dict[str, Any]
SummaryRow = dict[str, Any]


# I/O helpers

def load_history(path) -> list[Row]:
    """Parse classroom_behavior_history.csv into typed dicts."""
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, raw in enumerate(reader, start=2):
            rows.append({
                "_lineno": line_no,
                "frame": int(raw["frame"]),
                "time_seconds": float(raw["time_seconds"]),
                "student_id": int(raw["student_id"]),
                "behavior": raw["behavior"].strip(),
                "confidence": float(raw["confidence"]),
                "is_cached": raw["is_cached"].strip().lower() == "true",
                "unknown_cause": raw.get("unknown_cause", "").strip(),
            })
    return rows


def load_summary(path) -> list[SummaryRow]:
    """Parse classroom_behavior_summary.csv into typed dicts."""
    rows: list[SummaryRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            sid_str = (raw.get("student_id") or "").strip()
            if not sid_str:
                continue
            rows.append({
                "student_id": int(sid_str),
                **{b: float(raw.get(b) or 0) for b in SUMMARY_BEHAVIOR_COLUMNS},
            })
    return rows


# 1. Overall distribution

def overall_distribution(rows: list[Row]) -> dict[str, Any]:
    """Compute aggregate counts/percentages for every behavior."""
    total = len(rows)
    counts: dict[str, int] = defaultdict(int)
    live_counts: dict[str, int] = defaultdict(int)
    cached_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        b = r["behavior"]
        counts[b] += 1
        if r["is_cached"]:
            cached_counts[b] += 1
        else:
            live_counts[b] += 1

    percentages = {b: round(100.0 * c / total, 2) if total else 0.0 for b, c in counts.items()}
    return {
        "total_rows": total,
        "counts": dict(counts),
        "percentages": percentages,
        "known_total": sum(c for b, c in counts.items() if b != "unknown"),
        "unknown_total": counts.get("unknown", 0),
        "live_counts": dict(live_counts),
        "cached_counts": dict(cached_counts),
    }


# 2. Per-student analysis

def per_student_stats(rows: list[Row]) -> dict[int, dict[str, Any]]:
    """Return per-student analytics keyed by student_id."""
    buckets: dict[int, dict] = defaultdict(lambda: {
        "total_records": 0,
        "live_records": 0,
        "cached_records": 0,
        "unknown_records": 0,
        "known_records": 0,
        "behavior_counts": defaultdict(int),
        "live_confidences": [],
        "frames_seen": [],
    })

    for r in rows:
        sid = r["student_id"]
        b = r["behavior"]
        s = buckets[sid]
        s["total_records"] += 1
        s["frames_seen"].append(r["frame"])
        s["behavior_counts"][b] += 1
        if r["is_cached"]:
            s["cached_records"] += 1
        else:
            s["live_records"] += 1
            if r["confidence"] > 0.0:
                s["live_confidences"].append(r["confidence"])
        if b == "unknown":
            s["unknown_records"] += 1
        else:
            s["known_records"] += 1

    result: dict[int, dict[str, Any]] = {}
    for sid, s in buckets.items():
        total = s["total_records"]
        behavior_counts = dict(s["behavior_counts"])
        behavior_pct = {b: round(100.0 * c / total, 2) if total else 0.0 for b, c in behavior_counts.items()}
        known_counts = {b: c for b, c in behavior_counts.items() if b != "unknown"}
        dominant = max(known_counts, key=known_counts.get) if known_counts else None
        confs = s["live_confidences"]
        result[sid] = {
            "total_records": total,
            "live_records": s["live_records"],
            "cached_records": s["cached_records"],
            "unknown_records": s["unknown_records"],
            "known_records": s["known_records"],
            "behavior_counts": behavior_counts,
            "behavior_pct": behavior_pct,
            "dominant_behavior": dominant,
            "live_confidences": confs,
            "mean_live_conf": round(statistics.mean(confs), 4) if confs else None,
            "frames_seen": sorted(set(s["frames_seen"])),
        }
    return result


# 3. Temporal analysis

def _student_sorted_rows(rows: list[Row]) -> dict[int, list[Row]]:
    """Group rows by student_id, sorted by frame."""
    groups: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r["student_id"]].append(r)
    for sid in groups:
        groups[sid].sort(key=lambda r: r["frame"])
    return dict(groups)


def behavior_runs(student_rows: list[Row]) -> list[dict[str, Any]]:
    """
    Identify contiguous behavior runs for a single student's sorted rows.

    Returns a list of dicts: {behavior, start_frame, end_frame, length}.
    """
    if not student_rows:
        return []
    runs: list[dict[str, Any]] = []
    cur_behavior = student_rows[0]["behavior"]
    cur_start = student_rows[0]["frame"]
    cur_end = student_rows[0]["frame"]
    cur_len = 1

    for r in student_rows[1:]:
        if r["behavior"] == cur_behavior and r["frame"] == cur_end + 1:
            cur_end = r["frame"]
            cur_len += 1
        else:
            runs.append({"behavior": cur_behavior, "start_frame": cur_start, "end_frame": cur_end, "length": cur_len})
            cur_behavior = r["behavior"]
            cur_start = r["frame"]
            cur_end = r["frame"]
            cur_len = 1

    runs.append({"behavior": cur_behavior, "start_frame": cur_start, "end_frame": cur_end, "length": cur_len})
    return runs


def temporal_analysis(rows: list[Row]) -> dict[str, Any]:
    """Aggregate temporal statistics across all students."""
    groups = _student_sorted_rows(rows)
    all_runs: list[dict[str, Any]] = []
    transitions_per_student: dict[int, int] = {}

    for sid, srows in groups.items():
        runs = behavior_runs(srows)
        for run in runs:
            run["student_id"] = sid
        all_runs.extend(runs)
        transitions_per_student[sid] = max(0, len(runs) - 1)

    longest: dict[str, dict] = {}
    for run in all_runs:
        b = run["behavior"]
        if b not in longest or run["length"] > longest[b]["length"]:
            longest[b] = run

    run_lengths_by_behavior: dict[str, list[int]] = defaultdict(list)
    for run in all_runs:
        run_lengths_by_behavior[run["behavior"]].append(run["length"])

    mean_run_by_behavior = {
        b: round(statistics.mean(lengths), 2)
        for b, lengths in run_lengths_by_behavior.items()
        if lengths
    }

    total_transitions = sum(transitions_per_student.values())
    n_students = len(transitions_per_student)
    avg_transitions = round(total_transitions / n_students, 2) if n_students else 0.0
    all_lengths = [r["length"] for r in all_runs]
    avg_run_length = round(statistics.mean(all_lengths), 2) if all_lengths else 0.0

    return {
        "all_runs": all_runs,
        "longest_run_per_behavior": longest,
        "transitions_per_student": transitions_per_student,
        "total_transitions": total_transitions,
        "avg_transitions_per_student": avg_transitions,
        "avg_run_length": avg_run_length,
        "run_length_by_behavior": dict(run_lengths_by_behavior),
        "mean_run_by_behavior": mean_run_by_behavior,
    }


# 4. Student coverage analysis

def coverage_analysis(
    per_student: dict[int, dict[str, Any]],
    total_frames: int,
    short_track_threshold: int = 10,
    high_unknown_threshold: float = 80.0,
    low_live_threshold: int = 5,
) -> dict[str, Any]:
    """Classify students by their coverage quality."""
    full_coverage: list[int] = []
    short_track: list[int] = []
    high_unknown: list[int] = []
    high_unknown_pct: dict[int, float] = {}
    strong_coverage: list[int] = []
    low_live: list[int] = []
    frame_coverage_pct: dict[int, float] = {}

    for sid, s in per_student.items():
        n = s["total_records"]
        cov_pct = round(100.0 * n / total_frames, 2) if total_frames else 0.0
        frame_coverage_pct[sid] = cov_pct

        if cov_pct >= 95.0:
            full_coverage.append(sid)
        if n <= short_track_threshold:
            short_track.append(sid)

        unk_pct = round(100.0 * s["unknown_records"] / n, 2) if n else 0.0
        if unk_pct >= high_unknown_threshold:
            high_unknown.append(sid)
            high_unknown_pct[sid] = unk_pct

        known_pct = round(100.0 * s["known_records"] / n, 2) if n else 0.0
        if known_pct >= 80.0 and s["mean_live_conf"] is not None:
            strong_coverage.append(sid)

        # Live known = live records that are not unknown
        live_known_approx = s["live_records"] - s["unknown_records"]
        if live_known_approx <= low_live_threshold:
            low_live.append(sid)

    return {
        "full_coverage_students": sorted(full_coverage),
        "short_track_students": sorted(short_track),
        "high_unknown_students": sorted(high_unknown),
        "high_unknown_pct": high_unknown_pct,
        "strong_coverage_students": sorted(strong_coverage),
        "low_live_evidence": sorted(low_live),
        "frame_coverage_pct": frame_coverage_pct,
    }


# 5. Confidence analysis

def confidence_analysis(rows: list[Row]) -> dict[str, Any]:
    """Compute confidence statistics for live detections."""
    by_behavior: dict[str, list[float]] = defaultdict(list)
    all_confs: list[float] = []

    for r in rows:
        if not r["is_cached"] and r["behavior"] != "unknown" and r["confidence"] > 0.0:
            by_behavior[r["behavior"]].append(r["confidence"])
            all_confs.append(r["confidence"])

    def _stats(confs: list[float]) -> dict[str, Any]:
        if not confs:
            return {"mean": None, "median": None, "min": None, "max": None, "stdev": None, "count": 0}
        return {
            "mean": round(statistics.mean(confs), 4),
            "median": round(statistics.median(confs), 4),
            "min": round(min(confs), 4),
            "max": round(max(confs), 4),
            "stdev": round(statistics.stdev(confs), 4) if len(confs) > 1 else 0.0,
            "count": len(confs),
        }

    by_behavior_stats = {b: _stats(confs) for b, confs in by_behavior.items()}
    low_conf = [b for b, s in by_behavior_stats.items() if s["mean"] is not None and s["mean"] < 0.30]

    return {
        "all_live_conf": all_confs,
        "global_stats": _stats(all_confs),
        "by_behavior": by_behavior_stats,
        "low_conf_behaviors": low_conf,
    }


# 6. Cache contribution analysis

def cache_analysis(rows: list[Row]) -> dict[str, Any]:
    """Analyse the contribution of BehaviorCache to the record set."""
    total = len(rows)
    cached_total = sum(1 for r in rows if r["is_cached"])
    live_total = total - cached_total
    live_known_total = sum(
        1 for r in rows if not r["is_cached"] and r["behavior"] != "unknown"
    )
    cached_known_total = sum(
        1 for r in rows if r["is_cached"] and r["behavior"] != "unknown"
    )
    known_total = live_known_total + cached_known_total

    by_behavior: dict[str, dict] = defaultdict(lambda: {"total": 0, "live": 0, "cached": 0})
    for r in rows:
        b = r["behavior"]
        by_behavior[b]["total"] += 1
        if r["is_cached"]:
            by_behavior[b]["cached"] += 1
        else:
            by_behavior[b]["live"] += 1

    by_behavior_stats: dict[str, dict] = {}
    cache_heavy: list[str] = []
    for b, d in by_behavior.items():
        cp = round(100.0 * d["cached"] / d["total"], 2) if d["total"] else 0.0
        by_behavior_stats[b] = {**d, "cache_pct": cp}
        if cp > 50.0:
            cache_heavy.append(b)

    cache_only_known = sum(1 for r in rows if r["is_cached"] and r["behavior"] != "unknown")

    groups = _student_sorted_rows(rows)
    seq_valid = True
    for sid, srows in groups.items():
        had_live = False
        for r in srows:
            if not r["is_cached"] and r["behavior"] != "unknown":
                had_live = True
            if r["is_cached"] and not had_live:
                seq_valid = False
                break
        if not seq_valid:
            break

    return {
        "total_rows": total,
        "cached_rows": cached_total,
        "live_rows": live_total,
        "live_known_rows": live_known_total,
        "cached_known_rows": cached_known_total,
        "live_known_pct": round(100.0 * live_known_total / known_total, 2) if known_total else 0.0,
        "cached_known_pct": round(100.0 * cached_known_total / known_total, 2) if known_total else 0.0,
        "cache_pct": round(100.0 * cached_total / total, 2) if total else 0.0,
        "by_behavior": by_behavior_stats,
        "cache_only_known_rows": cache_only_known,
        "behaviors_cache_heavy": cache_heavy,
        "cache_sequence_valid": seq_valid,
    }


# 7. Unknown rate analysis

def unknown_analysis(rows: list[Row], total_frames: int) -> dict[str, Any]:
    """Break down the unknown rate into observable causes."""
    unknown_rows = [r for r in rows if r["behavior"] == "unknown"]
    total = len(rows)
    unk_total = len(unknown_rows)

    live_known_per_student: dict[int, int] = defaultdict(int)
    for r in rows:
        if not r["is_cached"] and r["behavior"] != "unknown":
            live_known_per_student[r["student_id"]] += 1

    all_student_ids = {r["student_id"] for r in rows}
    always_unknown = sorted(sid for sid in all_student_ids if live_known_per_student.get(sid, 0) == 0)

    frame_behaviors: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        frame_behaviors[r["frame"]].add(r["behavior"])

    frames_all_unknown = sorted(f for f, behaviors in frame_behaviors.items() if behaviors == {"unknown"})
    frames_any_known = sorted(f for f, behaviors in frame_behaviors.items() if behaviors != {"unknown"})

    track_lengths: dict[int, int] = defaultdict(int)
    for r in rows:
        track_lengths[r["student_id"]] += 1
    short_track_ids = [sid for sid, n in track_lengths.items() if n <= 10]
    short_track_unknown = sum(1 for r in unknown_rows if r["student_id"] in short_track_ids)

    student_totals: dict[int, int] = defaultdict(int)
    student_unk_counts: dict[int, int] = defaultdict(int)
    for r in rows:
        student_totals[r["student_id"]] += 1
        if r["behavior"] == "unknown":
            student_unk_counts[r["student_id"]] += 1
    per_student_unk = {
        sid: round(100.0 * student_unk_counts.get(sid, 0) / student_totals[sid], 2)
        for sid in all_student_ids
    }

    return {
        "unknown_rows": unk_total,
        "total_rows": total,
        "unknown_pct": round(100.0 * unk_total / total, 2) if total else 0.0,
        "students_always_unknown": always_unknown,
        "frames_all_unknown": frames_all_unknown,
        "frames_any_known": frames_any_known,
        "unknown_from_short_tracks": short_track_unknown,
        "short_track_student_ids": short_track_ids,
        "per_student_unknown_pct": per_student_unk,
    }


# 8. Session statistics

def session_stats(rows: list[Row]) -> dict[str, Any]:
    """Compute top-level session/dataset statistics."""
    frames = sorted({r["frame"] for r in rows})
    student_ids = sorted({r["student_id"] for r in rows})
    max_time = max(r["time_seconds"] for r in rows)

    ts_map = {}
    for r in rows:
        if r["frame"] not in ts_map and r["time_seconds"] > 0:
            ts_map[r["frame"]] = r["time_seconds"]
    fps = None
    if ts_map:
        pairs = sorted(ts_map.items())
        deltas = [
            (f2 - f1) / (t2 - t1)
            for (f1, t1), (f2, t2) in zip(pairs, pairs[1:])
            if t2 > t1 and f2 > f1
        ]
        if deltas:
            fps = round(statistics.mean(deltas), 3)

    return {
        "total_rows": len(rows),
        "n_students": len(student_ids),
        "n_frames": len(frames),
        "min_frame": min(frames),
        "max_frame": max(frames),
        "duration_seconds": max_time,
        "inferred_fps": fps,
        "unique_student_ids": student_ids,
    }
