"""
src/build_dashboard.py
======================
Generates a static HTML dashboard for VisionClass Behavioral Analytics (Phase 8.3).
Reuses src.analyze_behavior for all computations and embeds the results into a
premium Vanilla HTML/JS/CSS single-page application.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from src.generate_analytics_report import generate_run_dashboard as generate_dashboard

def generate_interactive_dashboard(
    history_path: Path,
    summary_path: Path,
    output_path: Path,
    tests_passing: str = "Not run",
) -> None:
    """Generate the detailed interactive dashboard from historical CSV data."""

    # 1. Load data
    rows = load_history(history_path)

    # 2. Run analytics
    sess = session_stats(rows)
    dist = overall_distribution(rows)
    ps = per_student_stats(rows)
    ta = temporal_analysis(rows)
    summary_rows = load_summary(summary_path)
    cov = coverage_analysis(ps, sess["n_frames"])
    ca = confidence_analysis(rows)
    cach = cache_analysis(rows)
    ua = unknown_analysis(rows, sess["n_frames"])
    seen = {(row["student_id"], row["frame"]) for row in rows}
    duplicate_rows = len(rows) - len(seen)
    valid_behaviors = {
        "handrise", "read", "sleep", "stand", "using_electronic_devices", "write", "unknown",
    }
    invalid_rows = sum(
        not isinstance(row["frame"], int)
        or row["frame"] < 0
        or not isinstance(row["student_id"], int)
        or row["student_id"] < 0
        or row["behavior"] not in valid_behaviors
        or not isinstance(row["confidence"], (int, float))
        or not 0 <= row["confidence"] <= 1
        for row in rows
    )
    cache_violations = sum(row["is_cached"] and row["behavior"] == "unknown" for row in rows)
    counts: dict[int, dict[str, int]] = {}
    for row in rows:
        tracker_counts = counts.setdefault(row["student_id"], {})
        tracker_counts[row["behavior"]] = tracker_counts.get(row["behavior"], 0) + 1
    summary_by_id = {row["student_id"]: row for row in summary_rows}
    fps = sess["inferred_fps"]
    summary_behaviors = set(summary_rows[0]) - {"student_id"} if summary_rows else set()
    summary_matches = bool(fps) and set(summary_by_id) == set(counts) and all(
        round(summary_by_id[tracker_id].get(behavior, 0.0), 2)
        == round(counts[tracker_id].get(behavior, 0) / fps, 2)
        for tracker_id in counts
        for behavior in summary_behaviors
    )
    quality = {
        "status": "CLEAN" if not (duplicate_rows or invalid_rows or cache_violations) and summary_matches else "ISSUES",
        "duplicates": duplicate_rows,
        "invalid_rows": invalid_rows,
        "cache_violations": cache_violations,
        "summary_match": summary_matches,
        "tests_passing": tests_passing,
    }
    quality_color = "#10b981" if quality["status"] == "CLEAN" else "#ef4444"

    # 3. Construct JSON payload
    data_payload = {
        "session": sess,
        "distribution": dist,
        "students": ps,
        "temporal": ta,
        "coverage": cov,
        "confidence": ca,
        "cache": cach,
        "unknown": ua,
        "quality": quality,
        "raw_history": rows  # needed for timeline
    }

    json_str = json.dumps(data_payload)

    # 4. Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionClass Behavioral Analytics</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #0f1115;
            --surface: rgba(26, 29, 36, 0.7);
            --surface-hover: rgba(40, 44, 52, 0.9);
            --border: rgba(255, 255, 255, 0.1);
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);

            /* Behavior Colors */
            --color-sleep: #8b5cf6;
            --color-write: #10b981;
            --color-read: #f59e0b;
            --color-stand: #ef4444;
            --color-ued: #06b6d4;
            --color-handrise: #ec4899;
            --color-unknown: #475569;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            padding-bottom: 4rem;
        }}

        header {{
            background: rgba(15, 17, 21, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 1.5rem 3rem;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .header-meta {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-align: right;
        }}

        main {{
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        .full-width {{ grid-column: 1 / -1; }}

        .card {{
            background: var(--surface);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: border-color 0.3s ease;
        }}

        .card:hover {{ border-color: rgba(255, 255, 255, 0.2); }}

        .card h2 {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Stats Grid */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }}

        .stat-box {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }}

        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.25rem;
        }}

        .stat-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        /* Progress Bars */
        .bar-container {{ margin-bottom: 1rem; }}
        .bar-header {{ display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem; }}
        .bar-bg {{ background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; transition: width 0.5s ease; }}

        /* Table */
        .table-container {{ overflow-x: auto; max-height: 500px; overflow-y: auto; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; }}
        th {{ position: sticky; top: 0; background: #1e222b; padding: 0.75rem; color: #fff; font-weight: 600; cursor: pointer; border-bottom: 1px solid var(--border); }}
        th:hover {{ background: #2a2f3a; }}
        td {{ padding: 0.75rem; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        tr:hover td {{ background: rgba(255,255,255,0.02); }}

        /* Tags */
        .tag {{ padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; display: inline-block; margin-right: 0.25rem; }}
        .tag-core {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .tag-transient {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .tag-unk {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
        .tag-low {{ background: rgba(236, 72, 153, 0.2); color: #f472b6; }}

        /* Timeline */
        .timeline-container {{ position: relative; height: 120px; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.3); overflow: visible; }}
        .timeline-track {{ position: absolute; top: 20px; bottom: 20px; left: 10px; right: 10px; }}
        .timeline-segment {{ height: 100%; position: absolute; top: 0; border-right: 1px solid var(--bg); cursor: pointer; transition: opacity 0.2s; }}
        .timeline-segment:hover {{ opacity: 0.8; }}
        .timeline-tooltip {{ position: absolute; top: 0; left: 50%; transform: translateX(-50%); background: #fff; color: #000; padding: 4px 8px; border-radius: 4px; font-size: 0.7rem; white-space: nowrap; pointer-events: none; opacity: 0; transition: opacity 0.2s; z-index: 10; font-weight: 600; }}
        .timeline-select {{ border: 0; padding: 0; background: none; color: inherit; font: inherit; text-decoration: underline; cursor: pointer; }}
        .timeline-segment:hover .timeline-tooltip {{ opacity: 1; }}

        /* Markers */
        .legend {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; font-size: 0.8rem; }}
        .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}

        .pattern-cached {{ background-image: repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(255,255,255,0.3) 3px, rgba(255,255,255,0.3) 6px); }}

        .alert-box {{ background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 1rem; border-radius: 4px; margin-top: 1rem; font-size: 0.85rem; }}
        .info-box {{ background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: 0.85rem; }}
    </style>
</head>
<body>

<header>
    <h1>VisionClass Analytics</h1>
    <div class="header-meta">
        <div>Phase 8.3 Dashboard</div>
        <div id="datasource-info">Loading...</div>
    </div>
</header>

<main>
    <!-- 1. OVERVIEW -->
    <div class="card full-width">
        <h2>1. Overview</h2>
        <div class="stats-grid" id="overview-stats"></div>
    </div>

    <!-- 2. BEHAVIOR DISTRIBUTION -->
    <div class="card">
        <h2>2. Behavior Distribution</h2>
        <div class="info-box">
            <strong>Note:</strong> Unknown means no usable behavior detection was available. It does NOT mean inactive.
        </div>
        <div id="behavior-bars"></div>
        <br>
        <h3 style="font-size:0.9rem; margin-bottom:1rem; color:var(--text-muted)">Known Behavior Distribution (Excl. Unknown)</h3>
        <div id="known-behavior-bars"></div>
    </div>

    <!-- 5. CONFIDENCE & 6. CACHE -->
    <div class="card">
        <h2>3. Confidence & Cache Analysis</h2>
        <div id="conf-cache-content"></div>
    </div>

    <!-- 7. UNKNOWN ANALYSIS -->
    <div class="card full-width">
        <h2>4. Unknown Analysis</h2>
        <div class="info-box">
            <strong>Unknown means no usable behavior detection was available for that observation. It does not mean inactive.</strong>
        </div>
        <div class="stats-grid" style="margin-bottom:1rem;" id="unknown-stats"></div>
        <div id="unknown-details" style="font-size: 0.85rem; color: var(--text-muted);"></div>
    </div>

    <!-- 3. STUDENT ANALYTICS -->
    <div class="card full-width">
        <h2>5. Student Analytics (Select a row to view timeline)</h2>
        <div class="table-container">
            <table id="student-table">
                <thead>
                    <tr>
                        <th data-sort="id" onclick="sortTable('id')">ID ↕</th>
                        <th data-sort="obs" onclick="sortTable('obs')">Obs ↕</th>
                        <th data-sort="cov" onclick="sortTable('cov')">Coverage % ↕</th>
                        <th data-sort="known" onclick="sortTable('known')">Known % ↕</th>
                        <th data-sort="unk" onclick="sortTable('unk')">Unknown % ↕</th>
                        <th data-sort="dom" onclick="sortTable('dom')">Dominant ↕</th>
                        <th data-sort="conf" onclick="sortTable('conf')">Mean Conf ↕</th>
                        <th data-sort="live" onclick="sortTable('live')">Non-cached/Cached ↕</th>
                        <th>Flags</th>
                    </tr>
                </thead>
                <tbody id="student-table-body"></tbody>
            </table>
        </div>
    </div>

    <!-- 4. TIMELINE -->
    <div class="card full-width" id="timeline-card" style="display: none;">
        <h2>6. Temporal Timeline: Student <span id="timeline-sid"></span></h2>
        <div class="info-box" style="padding: 0.5rem; margin-bottom: 0.5rem;">
            Dashed patterns indicate <strong>Cached</strong> observations (propagated from past live detections).
        </div>
        <div class="timeline-container">
            <div class="timeline-track" id="timeline-track"></div>
        </div>
        <div class="legend" id="timeline-legend"></div>

        <div id="temporal-stats" style="margin-top: 1rem; font-size: 0.85rem; display: flex; gap: 2rem;"></div>
    </div>

    <!-- 9. DATA QUALITY -->
    <div class="card">
        <h2>7. Data Quality (Phase 8.1)</h2>
        <div class="stats-grid">
            <div class="stat-box"><div class="stat-value" style="color:{quality_color};">{quality["status"]}</div><div class="stat-label">Status</div></div>
            <div class="stat-box"><div class="stat-value">{quality["invalid_rows"] + quality["duplicates"]}</div><div class="stat-label">Invalid or Duplicate Rows</div></div>
            <div class="stat-box"><div class="stat-value">{quality["cache_violations"]}</div><div class="stat-label">Cache Violations</div></div>
            <div class="stat-box"><div class="stat-value">{"PASS" if quality["summary_match"] else "FAIL"}</div><div class="stat-label">Summary Match</div></div>
            <div class="stat-box"><div class="stat-value">{quality["tests_passing"]}</div><div class="stat-label">Tests Passing</div></div>
        </div>
    </div>

    <div class="card">
        <h2>8. Temporal Stats (Global)</h2>
        <div id="global-temporal-stats" style="font-size: 0.9rem;"></div>
    </div>
</main>

<script>
    // Embedded Data Payload
    const DATA = {json_str};

    const COLORS = {{
        'sleep': 'var(--color-sleep)',
        'write': 'var(--color-write)',
        'read': 'var(--color-read)',
        'stand': 'var(--color-stand)',
        'using_electronic_devices': 'var(--color-ued)',
        'handrise': 'var(--color-handrise)',
        'unknown': 'var(--color-unknown)'
    }};

    let currentSortColumn = 'id';
    let currentSortAsc = true;

    // Initialization
    function init() {{
        renderHeader();
        renderOverview();
        renderBehaviorDistribution();
        renderConfCache();
        renderUnknown();
        renderStudentTable();
        renderGlobalTemporal();
    }}

    function renderHeader() {{
        const s = DATA.session;
        document.getElementById('datasource-info').innerHTML =
            `classroom_behavior_history.csv | classroom_behavior_summary.csv<br>${{s.duration_seconds.toFixed(1)}}s @ ${{s.inferred_fps.toFixed(1)}} FPS`;
    }}

    function renderOverview() {{
        const s = DATA.session;
        const c = DATA.cache;
        const u = DATA.unknown;

        document.getElementById('overview-stats').innerHTML = `
            <div class="stat-box"><div class="stat-value">${{s.n_students}}</div><div class="stat-label">Tracked Students</div></div>
            <div class="stat-box"><div class="stat-value">${{s.n_frames}}</div><div class="stat-label">Total Frames</div></div>
            <div class="stat-box"><div class="stat-value">${{s.total_rows.toLocaleString()}}</div><div class="stat-label">Observations</div></div>
            <div class="stat-box"><div class="stat-value">${{c.live_known_pct.toFixed(1)}}%</div><div class="stat-label">Live Known %</div></div>
            <div class="stat-box"><div class="stat-value">${{c.cached_known_pct.toFixed(1)}}%</div><div class="stat-label">Cached Known %</div></div>
            <div class="stat-box"><div class="stat-value" style="color:var(--color-unknown);">${{u.unknown_pct.toFixed(1)}}%</div><div class="stat-label">Unknown %</div></div>
        `;
    }}

    function renderBehaviorDistribution() {{
        const d = DATA.distribution;
        let html = '';
        let htmlKnown = '';

        const sortedBehaviors = Object.keys(COLORS).map(b => ({{
            name: b,
            count: d.counts[b] || 0,
            pct: d.percentages[b] || 0
        }})).sort((a,b) => b.count - a.count);

        sortedBehaviors.forEach(b => {{
            const label = b.name === 'unknown' ? 'Unknown / Undetected' : b.name;
            const color = COLORS[b.name];
            html += `
                <div class="bar-container">
                    <div class="bar-header">
                        <span>${{label}}</span>
                        <span>${{b.pct.toFixed(1)}}% (${{b.count.toLocaleString()}})</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: ${{b.pct}}%; background-color: ${{color}};"></div>
                    </div>
                </div>
            `;

            if (b.name !== 'unknown') {{
                const pctKnown = d.known_total > 0 ? (b.count / d.known_total) * 100 : 0;
                htmlKnown += `
                    <div class="bar-container">
                        <div class="bar-header">
                            <span>${{label}}</span>
                            <span>${{pctKnown.toFixed(1)}}% (${{b.count.toLocaleString()}})</span>
                        </div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: ${{pctKnown}}%; background-color: ${{color}};"></div>
                        </div>
                    </div>
                `;
            }}
        }});

        document.getElementById('behavior-bars').innerHTML = html;
        document.getElementById('known-behavior-bars').innerHTML = htmlKnown;
    }}

    function renderConfCache() {{
        const ca = DATA.confidence;
        const c = DATA.cache;

        let html = `
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                <div class="stat-box"><div class="stat-value">${{ca.global_stats.mean.toFixed(3)}}</div><div class="stat-label">Mean Live Conf</div></div>
                <div class="stat-box"><div class="stat-value">${{ca.global_stats.median.toFixed(3)}}</div><div class="stat-label">Median Live Conf</div></div>
            </div>

            <h3 style="font-size:0.85rem; margin-bottom:0.5rem; color:var(--text-muted);">Confidence by Behavior (Live Only, excludes Unknown/Cached)</h3>
        `;

        Object.keys(ca.by_behavior).sort((a,b) => ca.by_behavior[b].mean - ca.by_behavior[a].mean).forEach(b => {{
            const stat = ca.by_behavior[b];
            if(stat.mean !== null) {{
                const warning = stat.mean < 0.3 ? '⚠️ ' : '';
                const warnColor = stat.mean < 0.3 ? 'color:#ef4444;' : '';
                html += `<div style="font-size:0.85rem; display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                    <span><span style="color:${{COLORS[b]}}">●</span> ${{b}}</span>
                    <span style="${{warnColor}}">${{warning}}${{stat.mean.toFixed(3)}} (n=${{stat.count}})</span>
                </div>`;
            }}
        }});

        if (ca.low_conf_behaviors.length > 0) {{
            html += `<div class="alert-box">Low confidence classes identified: ${{ca.low_conf_behaviors.join(', ')}}</div>`;
        }}

        html += `
            <hr style="border-color:var(--border); margin:1.5rem 0;">
            <h3 style="font-size:0.85rem; margin-bottom:0.5rem; color:var(--text-muted);">Cache Contribution</h3>
            <div style="font-size:0.85rem; margin-bottom: 1rem;">
                <p style="margin-bottom:0.5rem;"><em>Cached labels are propagated from past detections, not new model detections.</em></p>
                Live known detections: <strong>${{c.live_known_rows.toLocaleString()}} (${{c.live_known_pct.toFixed(1)}}% of known observations)</strong><br>
                Cached known observations: <strong>${{c.cached_known_rows.toLocaleString()}} (${{c.cached_known_pct.toFixed(1)}}% of known observations)</strong><br>
                Non-cached observations (including unknown): <strong>${{c.live_rows.toLocaleString()}}</strong><br>
            </div>
            <h3 style="font-size:0.85rem; margin-bottom:0.5rem; margin-top:1rem; color:var(--text-muted);">Cache Percentage by Behavior</h3>
        `;

        Object.keys(c.by_behavior).sort((a,b) => c.by_behavior[b].cache_pct - c.by_behavior[a].cache_pct).forEach(b => {{
            const stat = c.by_behavior[b];
            if(stat.total > 0 && b !== 'unknown') {{
                html += `<div style="font-size:0.85rem; display:flex; justify-content:space-between; margin-bottom:0.25rem;">
                    <span><span style="color:${{COLORS[b]}}">●</span> ${{b}}</span>
                    <span>${{stat.cache_pct.toFixed(1)}}% cached</span>
                </div>`;
            }}
        }});

        document.getElementById('conf-cache-content').innerHTML = html;
    }}

    function renderUnknown() {{
        const u = DATA.unknown;

        document.getElementById('unknown-stats').innerHTML = `
            <div class="stat-box"><div class="stat-value">${{u.unknown_pct.toFixed(1)}}%</div><div class="stat-label">Unknown %</div></div>
            <div class="stat-box"><div class="stat-value">${{u.unknown_rows.toLocaleString()}}</div><div class="stat-label">Unknown Obs</div></div>
            <div class="stat-box"><div class="stat-value">${{u.students_always_unknown.length}}</div><div class="stat-label">Always Unknown Students</div></div>
            <div class="stat-box"><div class="stat-value">${{u.unknown_from_short_tracks}}</div><div class="stat-label">Short-track Unknowns</div></div>
        `;

        document.getElementById('unknown-details').innerHTML = `
            <p><strong>Always Unknown IDs (${{u.students_always_unknown.length}}):</strong> ${{u.students_always_unknown.length > 0 ? u.students_always_unknown.join(', ') : 'None'}}</p>
            <p style="margin-top:0.5rem;"><strong>High Unknown IDs (>=80%):</strong> ${{DATA.coverage.high_unknown_students.join(', ')}}</p>
            <p style="margin-top:0.5rem;">These students never or rarely received a live behavior detection above threshold.</p>
        `;
    }}

    function sortTable(column) {{
        if (currentSortColumn === column) {{
            currentSortAsc = !currentSortAsc;
        }} else {{
            currentSortColumn = column;
            currentSortAsc = false; // Default descending for stats
        }}
        renderStudentTable();
    }}

    function renderStudentTable() {{
        const students = DATA.students;
        const cov = DATA.coverage;
        const tbody = document.getElementById('student-table-body');

        let rowsArr = Object.keys(students).map(sid => {{
            return {{
                sid: parseInt(sid),
                s: students[sid]
            }};
        }});

        // Sorting logic
        rowsArr.sort((a, b) => {{
            let valA, valB;
            switch(currentSortColumn) {{
                case 'id': valA = a.sid; valB = b.sid; break;
                case 'obs': valA = a.s.total_records; valB = b.s.total_records; break;
                case 'cov': valA = cov.frame_coverage_pct[a.sid]; valB = cov.frame_coverage_pct[b.sid]; break;
                case 'known': valA = (100 * a.s.known_records / a.s.total_records); valB = (100 * b.s.known_records / b.s.total_records); break;
                case 'unk': valA = (100 * a.s.unknown_records / a.s.total_records); valB = (100 * b.s.unknown_records / b.s.total_records); break;
                case 'dom': valA = a.s.dominant_behavior || 'Z'; valB = b.s.dominant_behavior || 'Z'; break;
                case 'conf': valA = a.s.mean_live_conf || 0; valB = b.s.mean_live_conf || 0; break;
                case 'live': valA = a.s.live_records; valB = b.s.live_records; break;
            }}

            if (valA < valB) return currentSortAsc ? -1 : 1;
            if (valA > valB) return currentSortAsc ? 1 : -1;
            return 0;
        }});

        let html = '';
        rowsArr.forEach(item => {{
            const sid = item.sid;
            const s = item.s;
            const isCore = cov.full_coverage_students.includes(sid);
            const isTrans = cov.short_track_students.includes(sid);
            const isHighUnk = cov.high_unknown_students.includes(sid);
            const isLowLive = cov.low_live_evidence.includes(sid);

            let flags = '';
            if (isCore) flags += '<span class="tag tag-core">Core</span>';
            if (isTrans) flags += '<span class="tag tag-transient">Transient</span>';
            if (isHighUnk) flags += '<span class="tag tag-unk">High Unk</span>';
            if (isLowLive) flags += '<span class="tag tag-low">Low Live</span>';

            const dom = s.dominant_behavior ? `<span style="color:${{COLORS[s.dominant_behavior]}}">${{s.dominant_behavior}}</span>` : '—';
            const mconf = s.mean_live_conf ? s.mean_live_conf.toFixed(3) : '—';

            const covPct = cov.frame_coverage_pct[sid].toFixed(1);
            const knownPct = (100 * s.known_records / s.total_records).toFixed(1);
            const unkPct = (100 * s.unknown_records / s.total_records).toFixed(1);

            html += `
                <tr>
                    <td><button type="button" class="timeline-select" onclick="showTimeline(${{sid}})">${{sid}}</button></td>
                    <td>${{s.total_records}}</td>
                    <td>${{covPct}}%</td>
                    <td>${{knownPct}}%</td>
                    <td>${{unkPct}}%</td>
                    <td>${{dom}}</td>
                    <td>${{mconf}}</td>
                    <td>${{s.live_records}} / ${{s.cached_records}}</td>
                    <td>${{flags}}</td>
                </tr>
            `;
        }});
        tbody.innerHTML = html;
    }}

    function showTimeline(sid) {{
        document.getElementById('timeline-card').style.display = 'block';
        document.getElementById('timeline-sid').innerText = sid;

        const rows = DATA.raw_history.filter(r => r.student_id === parseInt(sid));
        rows.sort((a,b) => a.frame - b.frame);

        const track = document.getElementById('timeline-track');
        let html = '';

        const minF = DATA.session.min_frame;
        const maxF = DATA.session.max_frame;
        const totalF = maxF - minF + 1;

        // Find runs for stats
        const runs = DATA.temporal.all_runs.filter(r => r.student_id === parseInt(sid));

        // Build blocks
        rows.forEach(r => {{
            const leftPct = ((r.frame - minF) / totalF) * 100;
            const widthPct = Math.max(0.5, (1 / totalF) * 100);
            const color = COLORS[r.behavior] || '#fff';
            const cacheClass = r.is_cached ? 'pattern-cached' : '';

            html += `
                <div class="timeline-segment ${{cacheClass}}"
                     style="left: ${{leftPct}}%; width: ${{widthPct}}%; background-color: ${{color}};">
                    <div class="timeline-tooltip">
                        Frame: ${{r.frame}} (${{r.time_seconds.toFixed(2)}}s)<br>
                        Behavior: ${{r.behavior}}<br>
                        Conf: ${{r.confidence > 0 ? r.confidence.toFixed(3) : 'N/A'}}<br>
                        ${{r.is_cached ? 'Cached' : 'Live'}}
                    </div>
                </div>
            `;
        }});

        track.innerHTML = html;

        // Render legend dynamically based on this student
        const uniqueBehaviors = [...new Set(rows.map(r => r.behavior))];
        let legHtml = '';
        uniqueBehaviors.forEach(b => {{
            legHtml += `<div class="legend-item"><div class="legend-dot" style="background:${{COLORS[b]}}"></div>${{b}}</div>`;
        }});
        document.getElementById('timeline-legend').innerHTML = legHtml;

        // Student temporal stats
        document.getElementById('temporal-stats').innerHTML = `
            <div><strong>Transitions:</strong> ${{runs.length > 0 ? runs.length - 1 : 0}}</div>
            <div><strong>Runs:</strong> ${{runs.length}}</div>
            <div><strong>Longest Run:</strong> ${{runs.length > 0 ? Math.max(...runs.map(r=>r.length)) : 0}} frames</div>
        `;

        // Scroll to timeline
        document.getElementById('timeline-card').scrollIntoView({{behavior: 'smooth', block: 'end'}});
    }}

    function renderGlobalTemporal() {{
        const ta = DATA.temporal;
        let html = `
            <p><strong>Total Runs:</strong> ${{ta.all_runs.length}} | <strong>Total Transitions:</strong> ${{ta.total_transitions}}</p>
            <p><strong>Avg Run Length:</strong> ${{ta.avg_run_length.toFixed(1)}} frames</p>
            <br>
            <table style="width:100%; border:none;">
                <tr style="border-bottom:1px solid var(--border)"><th style="background:transparent; padding:0.2rem;">Behavior</th><th style="background:transparent; padding:0.2rem;">Longest Run</th><th style="background:transparent; padding:0.2rem;">Mean Run Length</th></tr>
        `;
        Object.keys(ta.longest_run_per_behavior).forEach(b => {{
            const r = ta.longest_run_per_behavior[b];
            const m = ta.mean_run_by_behavior[b] || 0;
            html += `<tr><td style="padding:0.2rem; color:${{COLORS[b]}};">${{b}}</td><td style="padding:0.2rem;">${{r.length}} frames (S${{r.student_id}})</td><td style="padding:0.2rem;">${{m.toFixed(1)}} frames</td></tr>`;
        }});
        html += `</table>`;
        document.getElementById('global-temporal-stats').innerHTML = html;
    }}

    document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Dashboard successfully generated at: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML Dashboard for Phase 8.3")
    parser.add_argument("--history", default=str(_ROOT / "data" / "classroom_behavior_history.csv"))
    parser.add_argument("--summary", default=str(_ROOT / "data" / "classroom_behavior_summary.csv"))
    parser.add_argument("--output",  default=str(_ROOT / "data" / "behavior_analytics_dashboard.html"))
    parser.add_argument("--tests-passing", default="Not run")
    parser.add_argument("--interactive-dashboard", action="store_true")
    args = parser.parse_args()

    if args.interactive_dashboard:
        generate_interactive_dashboard(
            history_path=Path(args.history),
            summary_path=Path(args.summary),
            output_path=Path(args.output),
            tests_passing=args.tests_passing,
        )
    else:
        generate_dashboard(
            history_path=Path(args.history),
            summary_path=Path(args.summary),
            output_path=Path(args.output),
        )

if __name__ == "__main__":
    main()
