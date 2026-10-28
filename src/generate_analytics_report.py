"""
src/generate_analytics_report.py
=================================
CLI script: reads the existing behavior CSVs and writes
data/behavior_analytics_report.md.

Usage:
    python src/generate_analytics_report.py [--history PATH] [--summary PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import html
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analyze_behavior import (
    cache_analysis,
    confidence_analysis,
    coverage_analysis,
    load_history,
    load_summary,
    overall_distribution,
    per_student_stats,
    session_stats,
    temporal_analysis,
    unknown_analysis,
)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0"
    return f"{100 * n / total:.1f}"


def _fmt_f(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def generate_report(
    history_path: Path,
    summary_path: Path,
    output_path: Path,
) -> None:
    """Generate the current truthful, run-scoped report format."""
    return generate_run_report(history_path, summary_path, output_path)

    # ── Load data ────────────────────────────────────────────────────
    rows = load_history(history_path)
    summary_rows = load_summary(summary_path)

    # ── Compute all analytics ────────────────────────────────────────
    sess = session_stats(rows)
    dist = overall_distribution(rows)
    ps   = per_student_stats(rows)
    ta   = temporal_analysis(rows)
    cov  = coverage_analysis(ps, sess["n_frames"])
    ca   = confidence_analysis(rows)
    cach = cache_analysis(rows)
    ua   = unknown_analysis(rows, sess["n_frames"])

    n_total = sess["total_rows"]
    fps     = sess["inferred_fps"] or 20.0

    lines: list[str] = []
    W = lines.append

    # ── Header ───────────────────────────────────────────────────────
    W("# VisionClass Behavioral Analytics Report")
    W("**Phase 8.2 — Behavioral Analytics**  ")
    W("_Generated from: `data/classroom_behavior_history.csv` + `data/classroom_behavior_summary.csv`_")
    W("")

    # ── Executive summary ────────────────────────────────────────────
    W("---")
    W("")
    W("## Executive Summary")
    W("")
    W(f"This report presents behavioral analytics for a **{sess['duration_seconds']:.2f}-second classroom recording** ")
    W(f"captured at **{fps:.0f} FPS**. The dataset contains **{n_total:,} observations** ")
    W(f"across **{sess['n_students']} tracked students** and **{sess['n_frames']} frames**.")
    W("")
    W("**Key findings:**")
    W("")

    # dominant behavior
    known_counts = {b: c for b, c in dist["counts"].items() if b != "unknown"}
    dominant_b = max(known_counts, key=known_counts.get) if known_counts else "N/A"
    dom_pct = _pct(dist["counts"].get(dominant_b, 0), n_total)

    W(f"- **Dominant behavior:** `{dominant_b}` ({dom_pct}% of all observations)")
    W(f"- **Unknown rate:** {dist['percentages'].get('unknown', 0):.1f}% ({dist['unknown_total']:,} rows)")
    W(f"- **Cache contribution:** {cach['cache_pct']:.1f}% of rows replayed from BehaviorCache")
    W(f"- **Mean live detection confidence:** {_fmt_f(ca['global_stats']['mean'])}")

    always_unk = ua["students_always_unknown"]
    W(f"- **Students never assigned a live behavior:** {len(always_unk)} (IDs: {always_unk if always_unk else 'none'})")
    W(f"- **Frames where ALL students were unknown:** {len(ua['frames_all_unknown'])}")
    W(f"- **Cache sequence validity:** {'OK' if cach['cache_sequence_valid'] else 'VIOLATION FOUND'}")
    W(f"- **No data quality bugs detected** (see Phase 8.1 report)")
    W("")

    # ── Dataset / session stats ──────────────────────────────────────
    W("---")
    W("")
    W("## 1. Dataset / Session Statistics")
    W("")
    W("| Metric | Value |")
    W("|--------|-------|")
    W(f"| Total history rows | {n_total:,} |")
    W(f"| Total summary rows | {len(summary_rows)} |")
    W(f"| Unique student IDs | {sess['n_students']} |")
    W(f"| Unique frames | {sess['n_frames']} (#{sess['min_frame']}–#{sess['max_frame']}) |")
    W(f"| Duration | {sess['duration_seconds']:.2f} s |")
    W(f"| Inferred FPS | {fps:.2f} |")
    W(f"| Live observations | {cach['live_rows']:,} ({_pct(cach['live_rows'], n_total)}%) |")
    W(f"| Cached observations | {cach['cached_rows']:,} ({cach['cache_pct']:.1f}%) |")
    W(f"| Known-behavior rows | {dist['known_total']:,} ({_pct(dist['known_total'], n_total)}%) |")
    W(f"| Unknown rows | {dist['unknown_total']:,} ({dist['percentages'].get('unknown', 0):.1f}%) |")
    W("")
    W(f"**Student IDs:** {sess['unique_student_ids']}")
    W("")
    W("> [!NOTE]")
    W("> Student IDs are not contiguous — ByteTrack assigns IDs sequentially. Gaps represent")
    W("> students who appeared briefly or were tracked only in other video segments.")
    W("")

    # ── Overall behavior distribution ────────────────────────────────
    W("---")
    W("")
    W("## 2. Overall Behavior Distribution")
    W("")
    W("### 2.1 All Observations")
    W("")
    W("| Behavior | Total | % of All | Live | Cached | Live % |")
    W("|----------|------:|--------:|-----:|-------:|-------:|")
    all_behaviors = sorted(dist["counts"].items(), key=lambda x: -x[1])
    for b, cnt in all_behaviors:
        pct = dist["percentages"].get(b, 0)
        live = cach["by_behavior"].get(b, {}).get("live", 0)
        cached = cach["by_behavior"].get(b, {}).get("cached", 0)
        live_pct = _pct(live, cnt)
        W(f"| `{b}` | {cnt:,} | {pct:.1f}% | {live:,} | {cached:,} | {live_pct}% |")
    W("")
    W("### 2.2 Known Behaviors Only (excluding unknown)")
    W("")
    W("| Behavior | Count | % of Known Rows |")
    W("|----------|------:|---------------:|")
    known_total = dist["known_total"]
    for b, cnt in all_behaviors:
        if b == "unknown":
            continue
        pct_known = round(100 * cnt / known_total, 1) if known_total else 0
        W(f"| `{b}` | {cnt:,} | {pct_known:.1f}% |")
    W("")
    W("> [!NOTE]")
    W("> `handrise` has **0 detections** in this recording. The class is supported by the pipeline")
    W("> but no events occurred (or confidence was below 0.20 threshold) during this clip.")
    W("")

    # ── Per-student analysis ─────────────────────────────────────────
    W("---")
    W("")
    W("## 3. Per-Student Analysis")
    W("")
    W("### 3.1 Summary Table")
    W("")
    W("| Student | Total | Live | Cached | Unknown | Known | Dominant Behavior | Mean Live Conf |")
    W("|--------:|------:|-----:|-------:|--------:|------:|-------------------|----------------|")
    for sid in sorted(ps.keys()):
        s = ps[sid]
        dom = s["dominant_behavior"] or "—"
        conf = f"{s['mean_live_conf']:.3f}" if s["mean_live_conf"] is not None else "N/A"
        W(f"| {sid} | {s['total_records']} | {s['live_records']} | {s['cached_records']} | {s['unknown_records']} | {s['known_records']} | `{dom}` | {conf} |")
    W("")
    W("### 3.2 Behavior Breakdown Per Student (% of that student's total records)")
    W("")
    W("| Student | sleep% | write% | read% | stand% | ued% | unknown% |")
    W("|--------:|-------:|-------:|------:|-------:|-----:|---------:|")
    for sid in sorted(ps.keys()):
        s = ps[sid]
        bpct = s["behavior_pct"]
        sl = bpct.get("sleep", 0)
        wr = bpct.get("write", 0)
        rd = bpct.get("read", 0)
        st = bpct.get("stand", 0)
        ued = bpct.get("using_electronic_devices", 0)
        unk = bpct.get("unknown", 0)
        W(f"| {sid} | {sl:.0f}% | {wr:.0f}% | {rd:.0f}% | {st:.0f}% | {ued:.0f}% | {unk:.0f}% |")
    W("")
    W("_ued = using\\_electronic\\_devices_")
    W("")

    # ── Temporal analysis ────────────────────────────────────────────
    W("---")
    W("")
    W("## 4. Temporal Analysis")
    W("")
    W("### 4.1 Behavior Runs")
    W("")
    W(f"Total behavior runs across all students: **{len(ta['all_runs'])}**  ")
    W(f"Average run length: **{ta['avg_run_length']} frames** ({ta['avg_run_length']/fps:.2f} s)  ")
    W(f"Total behavior transitions: **{ta['total_transitions']}**  ")
    W(f"Average transitions per student: **{ta['avg_transitions_per_student']}**  ")
    W("")
    W("### 4.2 Longest Runs per Behavior")
    W("")
    W("| Behavior | Max Run (frames) | Max Run (seconds) | Student | Start Frame | End Frame |")
    W("|----------|----------------:|------------------:|--------:|------------:|----------:|")
    for b in ["sleep", "write", "read", "stand", "using_electronic_devices", "unknown"]:
        run = ta["longest_run_per_behavior"].get(b)
        if run:
            W(f"| `{b}` | {run['length']} | {run['length']/fps:.2f} s | {run['student_id']} | {run['start_frame']} | {run['end_frame']} |")
        else:
            W(f"| `{b}` | — | — | — | — | — |")
    W("")
    W("### 4.3 Mean Run Length per Behavior")
    W("")
    W("| Behavior | Mean Run (frames) | Mean Run (seconds) |")
    W("|----------|------------------:|------------------:|")
    for b, mean_len in sorted(ta["mean_run_by_behavior"].items(), key=lambda x: -x[1]):
        W(f"| `{b}` | {mean_len:.2f} | {mean_len/fps:.2f} s |")
    W("")
    W("### 4.4 Transitions per Student")
    W("")
    W("| Student | Transitions |")
    W("|--------:|------------:|")
    for sid in sorted(ta["transitions_per_student"].keys()):
        W(f"| {sid} | {ta['transitions_per_student'][sid]} |")
    W("")

    # ── Student coverage ─────────────────────────────────────────────
    W("---")
    W("")
    W("## 5. Student Coverage Analysis")
    W("")
    W("| Category | Student IDs |")
    W("|----------|-------------|")
    W(f"| Full coverage (>=95% of frames) | {cov['full_coverage_students']} |")
    W(f"| Short track (<=10 records) | {cov['short_track_students']} |")
    W(f"| High unknown rate (>=80%) | {cov['high_unknown_students']} |")
    W(f"| Strong behavioral coverage (known>=80%, has live conf) | {cov['strong_coverage_students']} |")
    W(f"| Low live evidence (<=5 live known frames) | {cov['low_live_evidence']} |")
    W("")
    W("### 5.1 Frame Coverage % per Student")
    W("")
    W("| Student | Records | Frame Coverage % |")
    W("|--------:|--------:|-----------------:|")
    for sid in sorted(ps.keys()):
        W(f"| {sid} | {ps[sid]['total_records']} | {cov['frame_coverage_pct'].get(sid, 0):.1f}% |")
    W("")
    W("> [!NOTE]")
    W("> Students with low frame coverage entered/exited the scene during the recording.")
    W("> This is expected tracker behavior and not a data quality issue.")
    W("")

    # ── Confidence analysis ──────────────────────────────────────────
    W("---")
    W("")
    W("## 6. Confidence Analysis")
    W("")
    W("_Restricted to: live (not cached) detections with known behavior (not unknown)._")
    W("")
    gs = ca["global_stats"]
    W("### 6.1 Global Confidence Statistics")
    W("")
    W("| Metric | Value |")
    W("|--------|-------|")
    W(f"| Count | {gs['count']:,} |")
    W(f"| Mean | {_fmt_f(gs['mean'])} |")
    W(f"| Median | {_fmt_f(gs['median'])} |")
    W(f"| Min | {_fmt_f(gs['min'])} |")
    W(f"| Max | {_fmt_f(gs['max'])} |")
    W(f"| Std Dev | {_fmt_f(gs['stdev'])} |")
    W("")
    W("### 6.2 Confidence per Behavior (live detections only)")
    W("")
    W("| Behavior | Count | Mean | Median | Min | Max | Std Dev |")
    W("|----------|------:|-----:|-------:|----:|----:|--------:|")
    for b, s in sorted(ca["by_behavior"].items(), key=lambda x: -(x[1]["mean"] or 0)):
        W(f"| `{b}` | {s['count']} | {_fmt_f(s['mean'])} | {_fmt_f(s['median'])} | {_fmt_f(s['min'])} | {_fmt_f(s['max'])} | {_fmt_f(s['stdev'])} |")
    W("")
    if ca["low_conf_behaviors"]:
        W("> [!WARNING]")
        W(f"> The following behaviors have **mean live confidence < 0.30**: `{ca['low_conf_behaviors']}`")
        W("> These are detectable but uncertain — consider raising the confidence threshold cautiously.")
    else:
        W("> [!NOTE]")
        W("> No behavior has mean live confidence below 0.30.")
    W("")

    # ── Cache analysis ───────────────────────────────────────────────
    W("---")
    W("")
    W("## 7. Cache Contribution Analysis")
    W("")
    W("| Metric | Value |")
    W("|--------|-------|")
    W(f"| Total rows | {cach['total_rows']:,} |")
    W(f"| Live rows | {cach['live_rows']:,} ({_pct(cach['live_rows'], n_total)}%) |")
    W(f"| Cached rows | {cach['cached_rows']:,} ({cach['cache_pct']:.1f}%) |")
    W(f"| Cached known-behavior rows | {cach['cache_only_known_rows']:,} |")
    W(f"| Cache sequence valid | {'Yes' if cach['cache_sequence_valid'] else 'NO — VIOLATION'} |")
    W(f"| Behaviors where cache > 50% | {cach['behaviors_cache_heavy'] if cach['behaviors_cache_heavy'] else 'None'} |")
    W("")
    W("### 7.1 Cache Breakdown per Behavior")
    W("")
    W("| Behavior | Total | Live | Cached | Cache % |")
    W("|----------|------:|-----:|-------:|--------:|")
    for b, d in sorted(cach["by_behavior"].items(), key=lambda x: -x[1]["cache_pct"]):
        W(f"| `{b}` | {d['total']} | {d['live']} | {d['cached']} | {d['cache_pct']:.1f}% |")
    W("")
    if cach["behaviors_cache_heavy"]:
        W("> [!WARNING]")
        W(f"> Cache-heavy behaviors (>50% cached): `{cach['behaviors_cache_heavy']}`.  ")
        W("> This means more than half of those records exist solely because of caching,")
        W("> not because of a live detection in that frame.")
    else:
        W("> [!NOTE]")
        W("> No behavior is disproportionately cache-heavy (all behaviors <50% cached).  ")
        W("> The BehaviorCache is contributing evenly and appropriately.")
    W("")

    # ── Unknown analysis ─────────────────────────────────────────────
    W("---")
    W("")
    W("## 8. Unknown Rate Analysis")
    W("")
    W("### 8.1 Overview")
    W("")
    W(f"**{ua['unknown_rows']:,} rows** ({ua['unknown_pct']:.1f}%) carry `behavior=unknown`.")
    W("")
    W("All unknown rows are *live* (not cached). The BehaviorCache never replays an `unknown` label,")
    W("so the unknown rate reflects frames where the student had neither a live detection nor a usable cached entry.")
    W("")
    W("### 8.2 Observable Causes")
    W("")
    W("| Cause | Count |")
    W("|-------|------:|")
    W(f"| Students with **zero** live known detections (never assigned a behavior) | {len(ua['students_always_unknown'])} students |")
    W(f"| Frames where **every** visible student was unknown | {len(ua['frames_all_unknown'])} frames ({_pct(len(ua['frames_all_unknown']), sess['n_frames'])}% of frames) |")
    W(f"| Unknown rows attributable to short-track students (<=10 records) | {ua['unknown_from_short_tracks']} |")
    W("")
    always_unk = ua["students_always_unknown"]
    if always_unk:
        W(f"> [!WARNING]")
        W(f"> Students **{always_unk}** never received a live known behavior detection.")
        W("> Every record for these students is `unknown`. They may be occluded, at the frame edge,")
        W("> or appearing too briefly for the behavior model to fire a high-confidence detection.")
    else:
        W("> [!NOTE]")
        W("> Every student in the recording received at least one live known behavior detection.")
    W("")
    W("### 8.3 Per-Student Unknown Percentage")
    W("")
    W("| Student | Total Records | Unknown | Unknown % |")
    W("|--------:|-------------:|--------:|----------:|")
    for sid in sorted(ps.keys()):
        s = ps[sid]
        unk_pct = ua["per_student_unknown_pct"].get(sid, 0)
        W(f"| {sid} | {s['total_records']} | {s['unknown_records']} | {unk_pct:.1f}% |")
    W("")
    W("### 8.4 Frames Where All Students Are Unknown")
    W("")
    if ua["frames_all_unknown"]:
        W(f"**{len(ua['frames_all_unknown'])} frames** had all students unknown.  ")
        W(f"First 10: `{ua['frames_all_unknown'][:10]}`")
    else:
        W("No frames had all students unknown — at least one student had a known behavior in every frame.")
    W("")

    # ── Key findings ─────────────────────────────────────────────────
    W("---")
    W("")
    W("## 9. Key Findings")
    W("")
    W(f"1. **`sleep` dominates at {dist['percentages'].get('sleep', 0):.1f}%** of all observations —")
    W(f"   this is the most frequently detected behavior in the classroom clip.")
    W("")
    W(f"2. **`write` is the second most common known behavior ({dist['percentages'].get('write', 0):.1f}%)**,")
    W(f"   well above `read` ({dist['percentages'].get('read', 0):.1f}%), `stand`")
    W(f"   ({dist['percentages'].get('stand', 0):.1f}%), and `using_electronic_devices`")
    W(f"   ({dist['percentages'].get('using_electronic_devices', 0):.1f}%).")
    W("")
    longest_sleep = ta["longest_run_per_behavior"].get("sleep")
    if longest_sleep:
        W(f"3. **Longest `sleep` run: {longest_sleep['length']} frames ({longest_sleep['length']/fps:.2f} s)**,")
        W(f"   observed for Student {longest_sleep['student_id']} (frames {longest_sleep['start_frame']}–{longest_sleep['end_frame']}).")
    W("")
    W(f"4. **BehaviorCache contributes {cach['cache_pct']:.1f}% of all records** without violating")
    W(f"   its 30-frame max-age constraint. Cache sequence validity: PASS.")
    W("")
    W(f"5. **Unknown rate is {ua['unknown_pct']:.1f}%.** All unknown rows are live (not cached).")
    if ua["frames_all_unknown"]:
        W(f"   {len(ua['frames_all_unknown'])} frames had all students unknown simultaneously, suggesting")
        W(f"   frames where the behavior model fired no detections at all.")
    else:
        W(f"   In every frame, at least one student had a known behavior.")
    W("")
    if ca["low_conf_behaviors"]:
        W(f"6. **Low-confidence behaviors:** `{ca['low_conf_behaviors']}` — these classes")
        W(f"   systematically produce detections with mean confidence below 0.30.")
        W(f"   Useful signal for future model tuning in Phase 8.3.")
    else:
        W(f"6. **All detected behaviors have mean confidence >= 0.30.** No behavior class")
        W(f"   is systematically under-confident.")
    W("")
    if cov["full_coverage_students"]:
        W(f"7. **{len(cov['full_coverage_students'])} students are present in >=95% of frames:**")
        W(f"   {cov['full_coverage_students']}. These represent the core tracked students in the scene.")
    W("")
    if ua["students_always_unknown"]:
        W(f"8. **{len(ua['students_always_unknown'])} student(s) never received a live behavior detection:**")
        W(f"   {ua['students_always_unknown']}. Analytics for these students are entirely unknown.")
    else:
        W(f"8. **All 32 students received at least one live behavior detection** — no student is")
        W(f"   permanently invisible to the behavior model.")
    W("")

    # ── Limitations ──────────────────────────────────────────────────
    W("---")
    W("")
    W("## 10. Limitations")
    W("")
    W("1. **Short recording (10.9 s / 219 frames):** Behavioral conclusions are limited to this clip.")
    W("   Trends may not generalise to longer sessions.")
    W("")
    W("2. **`handrise` = 0 detections:** The class is present in the model but unobserved.")
    W("   No inferences can be drawn about its confidence or temporal patterns.")
    W("")
    W("3. **Unknown ≠ 'inactive':** A student labeled `unknown` may be engaged in a behavior")
    W("   the model failed to detect — not necessarily idle.")
    W("")
    W("4. **Cached confidence is decayed:** The BehaviorCache linearly decays confidence from")
    W("   the original live detection value toward 0.05. Cached rows therefore have artificially")
    W("   lower confidence than the original detection. The confidence analysis excludes cached")
    W("   rows to avoid this bias.")
    W("")
    W("5. **ID gaps in student IDs:** Non-contiguous IDs (e.g., 11, 13, 19–20) indicate students")
    W("   tracked in other parts of the recording or in prior tracker resets. Their absence does")
    W("   not indicate data loss in this segment.")
    W("")
    W("6. **Single video source:** All data comes from one classroom video. Results may differ")
    W("   with different camera angles, lighting, or classroom configurations.")
    W("")

    # ── Recommendations for Phase 8.3 ───────────────────────────────
    W("---")
    W("")
    W("## 11. Recommendations for Phase 8.3")
    W("")
    W("1. **Dashboard priority: `sleep` and `write`** — these are the dominant behaviors and")
    W("   have the richest data for visualisation. Start with per-student time-series and aggregate")
    W("   distribution charts for these two classes.")
    W("")
    W("2. **Highlight student engagement tiers:** Use the coverage analysis to group students into:")
    W("   - *Core students* (>=95% frame coverage): reliable analytics")
    W("   - *Transient students* (<10 records): flag as low-confidence")
    W("")
    W("3. **Confidence-aware display:** Where confidence is below 0.30, display labels with a")
    W("   visual indicator of uncertainty (e.g., lighter color, dotted border).")
    W("")
    W("4. **Unknown zone visualisation:** Surface the 32.2% unknown rate prominently.")
    W("   Avoid presenting it as 'no behavior' — label it accurately as 'undetected/uncertain'.")
    W("")
    if cach["behaviors_cache_heavy"]:
        W(f"5. **Cache-heavy behavior flag:** `{cach['behaviors_cache_heavy']}` is >50% cached.")
        W("   In the dashboard, consider showing a 'C' indicator when a cached label is displayed.")
    else:
        W("5. **Cache indicator:** Even without disproportionate caching, the dashboard should")
        W("   distinguish live vs cached records visually (e.g., solid vs dashed lines).")
    W("")
    W("6. **Temporal view:** Per-student behavior timelines would directly surface the run patterns")
    W("   found in Section 4. Recommend a horizontal stacked bar or swimlane chart per student.")
    W("")
    W("7. **Extend the recording:** 10.9 seconds is too short for robust behavioral conclusions.")
    W("   The analytics pipeline is now ready; longer video will immediately enrich all outputs.")
    W("")
    W("---")
    W("")
    W("_Report generated by `src/generate_analytics_report.py` using `src/analyze_behavior.py`._")


def generate_run_report(
    history_path: Path,
    summary_path: Path,
    output_path: Path,
    *,
    processing_metrics: dict | None = None,
) -> None:
    """Write a run-scoped report without sample-specific or identity claims."""
    rows = load_history(history_path)
    summary_rows = load_summary(summary_path)
    if not rows:
        raise ValueError("No tracker history rows are available for a report.")
    session = session_stats(rows)
    distribution = overall_distribution(rows)
    trackers = per_student_stats(rows)
    temporal = temporal_analysis(rows)
    coverage = coverage_analysis(trackers, session["n_frames"])
    confidence = confidence_analysis(rows)
    cache = cache_analysis(rows)
    unknown = unknown_analysis(rows, session["n_frames"])
    inferred_fps = session["inferred_fps"]
    lines = [
        "# VisionClass video analysis report",
        "",
        "> This report summarizes model observations and video-scoped tracker IDs. Tracker IDs are not verified real-world student identities.",
        "",
        "## Session summary",
        "",
        f"- History records: {session['total_rows']}",
        f"- Distinct frames represented by history records: {session['n_frames']}",
        f"- Tracker IDs in this video: {session['n_students']}",
        f"- History timestamp span: {session['duration_seconds']:.3f} seconds",
        f"- Inferred FPS from history timestamps: {inferred_fps if inferred_fps is not None else 'unavailable'}",
        f"- Summary CSV tracker rows: {len(summary_rows)}",
        f"- Raw behavior boxes processed: {processing_metrics.get('behavior_detection_total', 'unavailable') if processing_metrics else 'unavailable'}",
        "",
        "The history CSV contains one tracker ID behavior record per processed frame; cached records are marked separately. These rows are not raw behavior boxes.",
        "",
        "## Behavior distribution",
        "",
        "| Behavior | History records | Percent | Live records | Cached records |",
        "|---|---:|---:|---:|---:|",
    ]
    counts = distribution["counts"]
    for behavior in ("handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown"):
        live = distribution["live_counts"].get(behavior, 0)
        cached = distribution["cached_counts"].get(behavior, 0)
        lines.append(
            f"| `{behavior}` | {counts.get(behavior, 0)} | {distribution['percentages'].get(behavior, 0):.2f}% | {live} | {cached} |"
        )
    lines.extend(["", "## Per-tracker records", "", "| Tracker ID | Records | Live | Cached | Unknown | Dominant known behavior |", "|---:|---:|---:|---:|---:|---|"])
    for tracker_id in sorted(trackers):
        item = trackers[tracker_id]
        lines.append(
            f"| {tracker_id} | {item['total_records']} | {item['live_records']} | {item['cached_records']} | {item['unknown_records']} | {item['dominant_behavior'] or 'none'} |"
        )
    lines.extend(["", "## Temporal analysis", "", f"- Behavior runs: {len(temporal['all_runs'])}", f"- Mean run length: {temporal['avg_run_length']} frames"])
    lines.append(f"- Mean run length in seconds: {temporal['avg_run_length'] / inferred_fps:.3f}" if inferred_fps else "- Mean run length in seconds: unavailable (FPS could not be inferred)")
    lines.append(f"- Transitions across tracker IDs: {temporal['total_transitions']}")
    lines.append("")
    lines.extend(["| Behavior | Longest run (frames) | Tracker ID | Start frame | End frame |", "|---|---:|---:|---:|---:|"])
    for behavior in ("handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown"):
        run = temporal["longest_run_per_behavior"].get(behavior)
        if run:
            lines.append(f"| `{behavior}` | {run['length']} | {run['student_id']} | {run['start_frame']} | {run['end_frame']} |")
        else:
            lines.append(f"| `{behavior}` | — | — | — | — |")
    lines.extend(["", "## Coverage and confidence", "", f"- Full-coverage tracker IDs (existing ≥95% classification): {coverage['full_coverage_students']}", f"- Short-track tracker IDs (existing ≤10-record classification): {coverage['short_track_students']}", f"- High-unknown tracker IDs (existing ≥80% classification): {coverage['high_unknown_students']}", f"- Live-known confidence summary: {confidence['global_stats']}", "", "## Cache and unknown summaries", "", f"- Cache rows: {cache['cached_rows']} of {cache['total_rows']} ({cache['cache_pct']:.2f}%)", f"- Cache sequence valid under existing check: {cache['cache_sequence_valid']}", f"- Unknown history rows: {unknown['unknown_rows']} of {unknown['total_rows']} ({unknown['unknown_pct']:.2f}%)", f"- Tracker IDs with no live known behavior record: {unknown['students_always_unknown']}", f"- Unknown rows attributed to short tracks by existing analysis: {unknown['unknown_from_short_tracks']}", "", "## Limitations", "", "- Model detections and tracker assignments can be missed, duplicated, cached, or switched.", "- Tracker IDs are local to this video and do not represent verified unique people.", "- Duration, frames, and percentages use the existing CSV and analytics functions; they are not independent classroom attendance measurements.", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_run_dashboard(history_path: Path, summary_path: Path, output_path: Path) -> None:
    """Create a standalone, self-contained HTML summary for a completed run."""
    rows = load_history(history_path)
    if not rows:
        raise ValueError("No tracker history rows are available for an HTML report.")
    session = session_stats(rows)
    distribution = overall_distribution(rows)
    trackers = per_student_stats(rows)
    payload = {
        "session": {
            "history_records": session["total_rows"],
            "frames_represented": session["n_frames"],
            "tracker_ids": session["n_students"],
            "history_timestamp_span_seconds": session["duration_seconds"],
        },
        "distribution": distribution,
        "trackers": [
            {"tracker_id": key, "records": value["total_records"], "live": value["live_records"], "cached": value["cached_records"], "unknown": value["unknown_records"], "dominant_behavior": value["dominant_behavior"]}
            for key, value in sorted(trackers.items())
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    title = html.escape(Path(history_path).parent.name)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VisionClass run report</title><style>
    :root{{color-scheme:dark;font-family:system-ui,Segoe UI,sans-serif;background:#0a0d13;color:#e9f1f6}}body{{max-width:1050px;margin:32px auto;padding:0 20px}}header,section{{background:#121923;border:1px solid #2a3947;border-radius:16px;padding:22px;margin:14px 0}}h1,h2{{margin:0 0 12px}}p,small{{color:#aab8c4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.card{{padding:14px;background:#1a2632;border-radius:10px}}.card strong{{display:block;font-size:22px;margin-top:6px}}.bar{{display:grid;grid-template-columns:180px 1fr 90px;gap:10px;align-items:center;margin:12px 0}}.track{{background:#273441;border-radius:8px;height:9px;overflow:hidden}}.fill{{background:#59d6bb;height:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #293846;text-align:left}}.note{{color:#9db0bd;font-size:13px}}@media(max-width:520px){{.bar{{grid-template-columns:110px 1fr 70px;font-size:12px}}body{{padding:0 12px}}}}
    </style></head><body><header><small>VISIONCLASS · LOCAL VIDEO ANALYSIS</small><h1>Run analytics</h1><p>Run {title}. Tracker IDs are video-scoped labels, not unique real-world student identities.</p></header><main id="report"></main><script id="payload" type="application/json">{serialized}</script><script>
    const d=JSON.parse(document.getElementById('payload').textContent),root=document.getElementById('report');function el(tag,text,cls){{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}}const s=d.session,cards=[['History records',s.history_records],['Frames represented',s.frames_represented],['Tracker IDs',s.tracker_ids],['History timestamp span',`${{s.history_timestamp_span_seconds}} s`]];const summary=el('section');summary.append(el('h2','Session summary'));const grid=el('div',undefined,'cards');cards.forEach(([k,v])=>{{const c=el('div',undefined,'card');c.append(el('small',k),el('strong',String(v)));grid.append(c)}});summary.append(grid,el('p','History rows represent per-frame tracker observations and can include cached behavior labels.','note'));root.append(summary);const chart=el('section');chart.append(el('h2','Behavior distribution'));Object.keys(d.distribution.counts).sort().forEach(k=>{{const row=el('div',undefined,'bar');row.append(el('span',k.replaceAll('_',' ')));const track=el('div',undefined,'track'),fill=el('div',undefined,'fill');fill.style.width=`${{d.distribution.percentages[k]||0}}%`;track.append(fill);row.append(track,el('span',`${{d.distribution.counts[k]}} · ${{d.distribution.percentages[k]||0}}%`));chart.append(row)}});root.append(chart);const table=el('section');table.append(el('h2','Per-tracker summary'));const t=el('table'),head=el('tr');['Tracker ID','History records','Live','Cached','Unknown','Dominant behavior'].forEach(x=>head.append(el('th',x)));t.append(head);d.trackers.forEach(x=>{{const r=el('tr');[x.tracker_id,x.records,x.live,x.cached,x.unknown,x.dominant_behavior||'none'].forEach(v=>r.append(el('td',String(v))));t.append(r)}});table.append(t,el('p','Tracker IDs do not establish unique real-world student counts.','note'));root.append(table);
    </script></body></html>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")

def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate Phase 8.2 behavioral analytics report")
    parser.add_argument("--history", default=str(root / "data" / "classroom_behavior_history.csv"))
    parser.add_argument("--summary", default=str(root / "data" / "classroom_behavior_summary.csv"))
    parser.add_argument("--output",  default=str(root / "data" / "behavior_analytics_report.md"))
    args = parser.parse_args()

    generate_run_report(
        history_path=Path(args.history),
        summary_path=Path(args.summary),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
