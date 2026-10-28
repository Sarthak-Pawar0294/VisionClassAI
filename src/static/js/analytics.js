(() => {
  const $ = (id) => document.getElementById(id);
  let runs = [];
  let bundle = null;
  let selectedRun = "";
  let selectedTracker = "";
  let timelineOffset = 0;

  function node(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function number(value, digits = 0) {
    return Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, {maximumFractionDigits:digits}) : "—";
  }
  async function json(url) {
    const response = await fetch(url, {cache:"no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }
  function showEmpty(message) {
    $("analytics-content").hidden = true;
    $("run-message").textContent = message;
  }
  async function loadRuns() {
    $("run-message").textContent = "Loading saved runs…";
    try {
      const data = await json("/api/runs");
      runs = data.runs || [];
      const select = $("run-select");
      select.replaceChildren();
      if (!runs.length) {
        select.append(new Option("No saved video runs yet", ""));
        showEmpty("Completed video runs will appear here. Start with the Video Analysis page.");
        return;
      }
      select.append(new Option("Choose a saved run", ""));
      runs.forEach((run) => {
        const title = `${run.display_name || "Video run"} · ${run.status}`;
        select.append(new Option(title, run.run_id));
      });
      $("run-message").textContent = `${runs.length} saved ${runs.length === 1 ? "run" : "runs"}. Run folders remain until manually removed.`;
      const wanted = new URLSearchParams(window.location.search).get("run_id");
      if (wanted && runs.some((run) => run.run_id === wanted)) select.value = wanted;
      else if (selectedRun && runs.some((run) => run.run_id === selectedRun)) select.value = selectedRun;
      if (select.value) await loadAnalytics(select.value);
    } catch (error) { showEmpty(`${error.message} Check the local server and refresh.`); }
  }
  function renderSummary(data) {
    const session = data.session;
    const cards = [
      ["Raw behavior detections", data.processing.raw_behavior_box_detections, "Behavior boxes across processed frames"],
      ["Per-frame tracker records", session.total_rows, "History rows, including cached records"],
      ["Tracker IDs", session.n_tracker_ids, "IDs scoped to this video"],
      ["Frames represented", session.n_frames, "Distinct frames with history rows"],
      ["Recorded video span", `${number(session.duration_seconds, 2)} s`, "Maximum history-row timestamp"],
    ];
    const container = $("summary-grid");
    container.replaceChildren();
    cards.forEach(([label, value, hint]) => {
      const card = node("article", "summary-card");
      card.append(node("span", "", label), node("strong", "", String(value ?? "—")), node("small", "", hint));
      container.append(card);
    });
    $("tracker-count-label").textContent = `${number(session.n_tracker_ids)} tracker IDs`;
    const insights = [
      ["Unknown history rows", `${number(data.unknown.unknown_rows)} · ${number(data.unknown.unknown_pct, 2)}%`, "Existing unknown-rate analysis"],
      ["Cache contribution", `${number(data.cache.cached_rows)} · ${number(data.cache.cache_pct, 2)}%`, "Cached tracker records"],
      ["Behavior runs", number(data.temporal.run_count), "Contiguous runs in tracker history"],
      ["Behavior transitions", number(data.temporal.total_transitions), "Across video-scoped tracker IDs"],
      ["Live-known confidence", data.confidence.global_stats.mean === null ? "Unavailable" : number(data.confidence.global_stats.mean, 4), "Existing live, known, uncached rows"],
      ["Full-coverage tracker IDs", number(data.coverage.full_coverage_tracker_ids.length), "Existing ≥95% of represented frames rule"],
      ["Frames with all records unknown", number(data.unknown.frames_all_unknown_count), "Only frames represented by history rows"],
      ["Unique real-world students", "Not measured", "Tracker IDs do not establish identity"],
    ];
    const insightGrid = $("insight-grid");
    insightGrid.replaceChildren();
    insights.forEach(([label, value, hint]) => {
      const card = node("article", "summary-card");
      card.append(node("span", "", label), node("strong", "", String(value)), node("small", "", hint));
      insightGrid.append(card);
    });
  }
  function renderDistribution(data) {
    const chart = $("behavior-chart");
    chart.replaceChildren();
    const counts = data.distribution.counts || {};
    const percentages = data.distribution.percentages || {};
    const order = ["handrise","read","sleep","stand","using_electronic_devices","write","unknown"];
    order.forEach((behavior) => {
      const count = counts[behavior] || 0;
      const pct = percentages[behavior] || 0;
      const row = node("div", "bar-row");
      row.append(node("span", "bar-label", behavior.replaceAll("_", " ")));
      const track = node("div", "bar-track");
      const fill = node("div", "bar-fill");
      fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
      if (behavior === "unknown") fill.style.background = "linear-gradient(90deg,#778494,#9aa6b2)";
      track.append(fill);
      row.append(track, node("span", "bar-value", `${number(count)} · ${number(pct, 2)}%`));
      chart.append(row);
    });
  }
  function renderTrackerTable(data) {
    const table = $("tracker-table");
    const select = $("tracker-select");
    table.replaceChildren();
    select.replaceChildren(new Option("Choose a tracker", ""));
    Object.entries(data.per_tracker_id || {}).sort((a,b) => Number(a[0])-Number(b[0])).forEach(([id, stats]) => {
      const option = new Option(`Tracker ${id}`, id);
      select.append(option);
      const row = node("tr");
      row.tabIndex = 0;
      row.setAttribute("aria-label", `View timeline for tracker ${id}`);
      const dominant = stats.dominant_behavior || "No known behavior";
      [id, stats.total_records, stats.live_records, stats.cached_records, stats.unknown_records, dominant].forEach((value) => row.append(node("td", "", String(value))));
      row.addEventListener("click", () => { select.value = id; select.dispatchEvent(new Event("change")); });
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); row.click(); } });
      table.append(row);
    });
    const headers = $("duration-head");
    const durationBody = $("duration-table");
    headers.replaceChildren();
    durationBody.replaceChildren();
    const header = node("tr");
    ["Tracker ID", "Hand raised", "Reading", "Sleeping", "Standing", "Using devices", "Writing", "Unknown"].forEach((value) => header.append(node("th", "", value)));
    headers.append(header);
    (data.summary_rows || []).forEach((summary) => {
      const row = node("tr");
      row.append(node("td", "", String(summary.tracker_id)));
      [summary.handrise, summary.read, summary.sleep, summary.stand, summary.using_electronic_devices, summary.write, summary.unknown].forEach((value) => row.append(node("td", "", `${number(value, 2)} s`)));
      durationBody.append(row);
    });
    renderDetailPanels(data);
  }
  function renderDetailPanels(data) {
    const coverage = data.coverage;
    const unknown = data.unknown;
    const confidence = data.confidence;
    const cache = data.cache;
    const coverageLines = [
      `Full coverage IDs (≥95% represented frames): ${coverage.full_coverage_tracker_ids.length}`,
      `Short tracks (≤10 records): ${coverage.short_track_tracker_ids.length}`,
      `High unknown rate (≥80%): ${coverage.high_unknown_tracker_ids.length}`,
      `Strong coverage (≥80% known with live confidence): ${coverage.strong_coverage_tracker_ids.length}`,
      `Low live evidence (≤5 live-known records): ${coverage.low_live_evidence.length}`,
      `Tracker IDs with no live-known record: ${unknown.tracker_ids_always_unknown.length}`,
      `Unknown rows attributed to short tracks: ${unknown.unknown_from_short_tracks}`,
    ];
    const coverageBox = $("coverage-details");
    coverageBox.replaceChildren(...coverageLines.map((line) => node("p", "", line)));
    const global = confidence.global_stats;
    const confidenceLines = [
      `Live-known confidence: ${global.count} rows · mean ${global.mean === null ? "unavailable" : number(global.mean, 4)}`,
      `Median ${global.median === null ? "unavailable" : number(global.median, 4)} · range ${global.min === null ? "unavailable" : `${number(global.min, 4)}–${number(global.max, 4)}`}`,
      `Low-confidence behavior labels (<0.30 mean): ${confidence.low_conf_behaviors.length ? confidence.low_conf_behaviors.join(", ") : "none"}`,
      `Live records: ${cache.live_rows} · cached records: ${cache.cached_rows} (${number(cache.cache_pct, 2)}%)`,
      `Cache sequence check: ${cache.cache_sequence_valid ? "valid" : "flagged"}`,
      `Cache-heavy behavior classes (>50% cached): ${cache.behaviors_cache_heavy.length ? cache.behaviors_cache_heavy.join(", ") : "none"}`,
    ];
    const confidenceBox = $("confidence-details");
    confidenceBox.replaceChildren(...confidenceLines.map((line) => node("p", "", line)));
  }
  async function loadTimeline(reset = true) {
    if (!selectedTracker || !selectedRun) return;
    if (reset) timelineOffset = 0;
    const list = $("timeline-list");
    list.replaceChildren();
    try {
      const data = await json(`/api/runs/${encodeURIComponent(selectedRun)}/timeline?tracker_id=${encodeURIComponent(selectedTracker)}&offset=${timelineOffset}&limit=100`);
      data.rows.forEach((item) => {
        const row = node("div", "timeline-row");
        row.append(node("span", "", `Frame ${item.frame}`), node("strong", "", `${item.behavior}${item.is_cached ? " · cached" : ""}`), node("span", "", `${number(item.time_seconds, 3)} s`));
        list.append(row);
      });
      const next = timelineOffset + data.rows.length;
      if (next < data.total_records && data.rows.length) {
        const button = node("button", "button button-secondary", `Load more (${data.total_records - next} records)`);
        button.type = "button";
        button.addEventListener("click", () => { timelineOffset = next; loadTimeline(false); });
        list.append(button);
      }
      if (!data.rows.length) list.append(node("p", "muted small", "No rows are available for this tracker."));
    } catch (error) { list.append(node("p", "error-card", error.message)); }
  }
  function renderDownloads(run) {
    const list = $("report-links");
    list.replaceChildren();
    const available = run.artifacts || {};
    const labels = {annotated_video:"Annotated video",source_video:"Source video",history_csv:"History CSV",summary_csv:"Summary CSV",analytics_report:"Markdown report",analytics_dashboard:"HTML analytics dashboard"};
    Object.keys(labels).forEach((key) => {
      if (!Object.prototype.hasOwnProperty.call(available, key)) return;
      const link = node("a", "artifact-link", `↓ ${labels[key]}`);
      link.href = `/api/runs/${encodeURIComponent(run.run_id)}/artifacts/${encodeURIComponent(key)}`;
      list.append(link);
    });
    const notes = run.optional_errors || [];
    $("report-notice").textContent = notes.length ? notes.join(" ") : "Only artifacts generated successfully for this run are listed here.";
  }
  async function loadAnalytics(runId) {
    selectedRun = runId;
    bundle = null;
    $("analytics-content").hidden = false;
    $("run-message").textContent = "Loading analysis from the run's history CSV…";
    try {
      const [data, runData] = await Promise.all([
        json(`/api/runs/${encodeURIComponent(runId)}/analytics`),
        json(`/api/runs/${encodeURIComponent(runId)}`),
      ]);
      bundle = data;
      renderSummary(data);
      renderDistribution(data);
      renderTrackerTable(data);
      renderDownloads(runData.run);
      $("run-message").textContent = `Analysis for ${runData.run.display_name || "selected video"}. Counts and calculations use the existing CSV semantics.`;
      selectedTracker = "";
      $("timeline-list").replaceChildren(node("p", "muted small", "Choose a tracker to view its recorded frames."));
    } catch (error) {
      $("analytics-content").hidden = true;
      $("run-message").textContent = `${error.message} A video can complete without optional analytics when no tracker history was produced.`;
      try {
        const {run} = await json(`/api/runs/${encodeURIComponent(runId)}`);
        renderDownloads(run);
        $("analytics-content").hidden = false;
      } catch (_ignored) { /* The run selector message remains useful. */ }
    }
  }
  $("run-select").addEventListener("change", () => {
    const id = $("run-select").value;
    if (id) loadAnalytics(id);
    else showEmpty("Choose a completed video run to see its existing summaries and reports.");
  });
  $("tracker-select").addEventListener("change", () => {
    selectedTracker = $("tracker-select").value;
    loadTimeline(true);
  });
  $("refresh-runs").addEventListener("click", loadRuns);
  loadRuns();
})();
