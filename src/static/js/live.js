(() => {
  const classes = [
    ["handrise", "Hand-raise observations", "#62d9bd"], ["read", "Reading detections", "#87b8ff"],
    ["sleep", "Sleeping detections", "#c49cff"], ["stand", "Standing observations", "#ffbd72"],
    ["using_electronic_devices", "Device-use detections", "#ff8f9a"], ["write", "Writing detections", "#e5d56f"],
  ];
  const $ = (id) => document.getElementById(id);
  const feed = $("live-feed");
  const placeholder = $("feed-placeholder");
  const startButton = $("start-button");
  const stopButton = $("stop-button");
  let currentStreamJob = "";
  let streamRetry = 0;
  let polling = false;
  let historyPolling = false;
  let chartPaused = false;
  let selectedWindow = 1;
  let lastTrendJob = "";
  let historyTimer = 0;

  const cards = new Map();
  const visibleSeries = new Set(classes.map(([key]) => key));
  classes.forEach(([key, label, color]) => {
    const card = document.createElement("article");
    card.className = "behavior-card";
    card.innerHTML = `<div class="class-name"></div><div class="behavior-values"><strong>0</strong><span>boxes</span></div>`;
    card.querySelector(".class-name").textContent = label;
    card.style.setProperty("--series-color", color);
    $("behavior-grid").append(card);
    cards.set(key, card);

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "series-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = true;
    toggle.dataset.class = key;
    toggle.setAttribute("aria-label", `Show ${label.toLowerCase()} trend`);
    const swatch = document.createElement("span");
    swatch.className = "series-swatch";
    swatch.style.backgroundColor = color;
    const text = document.createElement("span");
    text.textContent = label;
    toggleLabel.append(toggle, swatch, text);
    $("series-toggle-list").append(toggleLabel);
  });

  function setDot(id, value) {
    const dot = $(id);
    dot.className = "status-dot";
    if (["open", "running", "available", "connected"].includes(value)) dot.classList.add("good");
    else if (["error", "failed"].includes(value)) dot.classList.add("error");
    else if (["starting", "connecting", "stop_requested"].includes(value)) dot.classList.add("warning");
    else dot.classList.add("idle");
  }

  function duration(seconds) {
    const total = Math.max(0, Math.floor(seconds || 0));
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}` : `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }

  let latestHistory = null;
  const svg = $("trend-chart");
  const svgElement = (name, attributes = {}) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  };

  function renderDistribution(points) {
    const root = $("window-distribution");
    root.replaceChildren();
    const totals = Object.fromEntries(classes.map(([key]) => [key, 0]));
    points.forEach((point) => classes.forEach(([key]) => {
      totals[key] += Math.max(0, Number(point.class_counts && point.class_counts[key]) || 0);
    }));
    const maximum = Math.max(1, ...Object.values(totals));
    classes.forEach(([key, label, color]) => {
      const row = document.createElement("div");
      row.className = "distribution-row";
      const name = document.createElement("span");
      name.className = "distribution-label";
      name.textContent = label;
      const track = document.createElement("span");
      track.className = "distribution-track";
      const fill = document.createElement("span");
      fill.className = "distribution-fill";
      fill.style.width = `${(totals[key] / maximum) * 100}%`;
      fill.style.backgroundColor = color;
      track.append(fill);
      const value = document.createElement("strong");
      value.textContent = totals[key].toLocaleString();
      value.setAttribute("aria-label", `${label}: ${totals[key]} sampled-frame boxes`);
      row.append(name, track, value);
      root.append(row);
    });
    $("distribution-caption").textContent = `Sum of box counts across ${points.length.toLocaleString()} sampled frames in the last ${selectedWindow} ${selectedWindow === 1 ? "minute" : "minutes"}. A box in separate samples is counted again as a later observation.`;
  }

  function renderTrend(history) {
    latestHistory = history;
    const points = Array.isArray(history.points) ? history.points : [];
    const hasPoints = points.length > 0;
    $("trend-empty").hidden = hasPoints;
    svg.removeAttribute("hidden");
    svg.replaceChildren();

    const width = 800;
    const height = 300;
    const left = 58;
    const right = 782;
    const top = 22;
    const bottom = 244;
    const now = Number(history.current_time) || Date.now() / 1000;
    const windowSeconds = selectedWindow * 60;
    const firstX = now - windowSeconds;
    const maxObserved = Math.max(0, ...points.flatMap((point) => classes.map(([key]) => Number(point.class_counts && point.class_counts[key]) || 0)));
    const tickStep = Math.max(1, Math.ceil(maxObserved / 4));
    const maxY = tickStep * 4;
    const title = svgElement("title");
    title.textContent = `Observed behavior box counts in the last ${selectedWindow} ${selectedWindow === 1 ? "minute" : "minutes"}`;
    const description = svgElement("desc");
    description.textContent = `${points.length} sampled webcam frames; each plotted value is the count of behavior boxes in one sampled frame.`;
    svg.append(title, description);

    for (let tick = 0; tick <= 4; tick += 1) {
      const y = bottom - ((tick * tickStep) / maxY) * (bottom - top);
      svg.append(svgElement("line", {x1:left, y1:y, x2:right, y2:y, class:"chart-grid-line"}));
      const label = svgElement("text", {x:left - 10, y:y + 4, "text-anchor":"end", class:"chart-axis-label"});
      label.textContent = String(tick * tickStep);
      svg.append(label);
    }
    svg.append(svgElement("line", {x1:left, y1:top, x2:left, y2:bottom, class:"chart-axis-line"}));
    svg.append(svgElement("line", {x1:left, y1:bottom, x2:right, y2:bottom, class:"chart-axis-line"}));

    const timeLabels = [
      [left, `−${selectedWindow}m`, "start"],
      [(left + right) / 2, selectedWindow === 1 ? "−30s" : `−${selectedWindow / 2}m`, "middle"],
      [right, "Now", "end"],
    ];
    timeLabels.forEach(([x, label, anchor]) => {
      const text = svgElement("text", {x, y:bottom + 25, "text-anchor":anchor, class:"chart-axis-label"});
      text.textContent = label;
      svg.append(text);
    });

    classes.forEach(([key, label, color]) => {
      if (!visibleSeries.has(key) || !hasPoints) return;
      const samples = points.map((point) => {
        const timestamp = Number(point.observed_at) || firstX;
        const count = Math.max(0, Number(point.class_counts && point.class_counts[key]) || 0);
        const x = left + Math.max(0, Math.min(1, (timestamp - firstX) / windowSeconds)) * (right - left);
        const y = bottom - (count / maxY) * (bottom - top);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      });
      if (samples.length > 1) {
        svg.append(svgElement("polyline", {points:samples.join(" "), fill:"none", stroke:color, "stroke-width":"2.5", "stroke-linecap":"round", "stroke-linejoin":"round", class:"trend-line"}));
      } else {
        const [x, y] = samples[0].split(",");
        svg.append(svgElement("circle", {cx:x, cy:y, r:"4", fill:color}));
      }
    });

    const newest = points[points.length - 1];
    const age = history.data_freshness_seconds;
    const ageLabel = Number.isFinite(age) ? `${Math.floor(age)}s ago` : "time unavailable";
    $("trend-summary").textContent = !hasPoints
      ? "Axes and all six behavior series are ready; sampled-frame observations will appear here."
      : visibleSeries.size
        ? `${points.length.toLocaleString()} sampled frames · latest sample frame ${Number(newest.frame_number || 0).toLocaleString()} · updated ${ageLabel}. A sampled box count is an observation, not a unique event or student.`
        : "Choose at least one behavior series to display a trend.";
    const emptyTitle = $("trend-empty").querySelector("strong");
    const emptyMessage = $("trend-empty").querySelector("span");
    if (!hasPoints) {
      const active = ["starting", "running", "stop_requested"].includes(history.status);
      emptyTitle.textContent = active ? "Waiting for webcam observations" : "No webcam observations in this window";
      emptyMessage.textContent = active
        ? "The chart is ready. Zero detections will appear as zero-count samples once frames arrive."
        : "Start a camera session to populate the timestamped trend.";
    }
    renderDistribution(points);
  }

  function updateChartState(history) {
    const state = $("chart-state");
    state.classList.remove("warning", "error");
    if (chartPaused) {
      state.textContent = "Chart updates are paused. Webcam inference, the camera feed, and status checks continue.";
      return;
    }
    if (["completed", "cancelled"].includes(history.status)) {
      state.textContent = history.points && history.points.length
        ? `Session ${history.status}; showing the final ${history.points.length.toLocaleString()} sampled frames retained in memory.`
        : `Session ${history.status}; no webcam observations were retained for this window.`;
      return;
    }
    if (history.status === "error") {
      state.textContent = history.points && history.points.length
        ? `Session ended with an error; showing ${history.points.length.toLocaleString()} earlier sampled frames. Review the connection and inference status above.`
        : "The webcam session ended with an error before observations were available. Review the status and recovery message above.";
      state.classList.add("error");
      return;
    }
    if (history.stale) {
      state.textContent = "Live data is stale: no new webcam frame has arrived for more than five seconds. Check camera connection and inference status.";
      state.classList.add("warning");
    } else if (history.points && history.points.length) {
      const noDetections = history.points.every((point) =>
        Number(point.total_detections) === 0 || classes.every(([key]) => Number(point.class_counts && point.class_counts[key]) === 0)
      );
      state.textContent = noDetections
        ? "Camera and inference are running. No behavior boxes were detected in the sampled frames; the graph is showing genuine zero counts."
        : `Updating from timestamped webcam frame samples · selected window: last ${selectedWindow} ${selectedWindow === 1 ? "minute" : "minutes"}.`;
    } else if (["starting", "running", "stop_requested"].includes(history.status)) {
      state.textContent = "Waiting for the first webcam observation in this time window.";
    } else {
      state.textContent = "No webcam observations in this session yet. Start a camera session to build a trend.";
    }
  }

  async function refreshHistory() {
    if (historyPolling || chartPaused) return;
    historyPolling = true;
    const requestedWindow = selectedWindow;
    try {
      const response = await fetch(`/api/live/history?window=${requestedWindow}`, {cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not refresh live trend history.");
      if (chartPaused || requestedWindow !== selectedWindow) return;
      renderTrend(data);
      updateChartState(data);
      lastTrendJob = data.job_id || "";
    } catch (error) {
      $("chart-state").textContent = `${error.message} The live feed and session controls remain available.`;
      $("chart-state").classList.add("error");
    } finally {
      historyPolling = false;
      if (!chartPaused && latestHistory && ["starting", "running", "stop_requested"].includes(latestHistory.status)) {
        historyTimer = window.setTimeout(refreshHistory, 4000);
      }
    }
  }

  function connectStream(job) {
    if (!job || !job.job_id || job.job_type !== "webcam" || !job.worker_alive) {
      currentStreamJob = "";
      feed.hidden = true;
      placeholder.hidden = false;
      $("stream-chip").textContent = "Stream idle";
      $("stream-status").textContent = "Waiting for session";
      setDot("stream-dot", "idle");
      return;
    }
    if (currentStreamJob !== job.job_id) {
      currentStreamJob = job.job_id;
      streamRetry = 0;
      feed.hidden = false;
      placeholder.hidden = true;
      feed.src = `/api/live/stream?job_id=${encodeURIComponent(job.job_id)}&r=${Date.now()}`;
      $("stream-chip").textContent = "Connecting to stream";
      $("stream-status").textContent = "Connecting";
      setDot("stream-dot", "connecting");
    }
  }

  feed.addEventListener("load", () => {
    $("stream-chip").textContent = "Stream connected";
    $("stream-status").textContent = "Connected";
    setDot("stream-dot", "connected");
    streamRetry = 0;
  });
  feed.addEventListener("error", () => {
    $("stream-chip").textContent = "Reconnecting…";
    $("stream-status").textContent = "Disconnected · retrying";
    setDot("stream-dot", "warning");
    if (currentStreamJob) {
      const jobId = currentStreamJob;
      const delay = Math.min(5000, 500 * (2 ** streamRetry++));
      window.setTimeout(() => {
        if (currentStreamJob === jobId) feed.src = `/api/live/stream?job_id=${encodeURIComponent(jobId)}&r=${Date.now()}`;
      }, delay);
    }
  });

  function render(job) {
    const status = job.status || "idle";
    const active = job.worker_alive && ["starting", "running", "stop_requested"].includes(status);
    startButton.disabled = active;
    stopButton.disabled = !active || status === "stop_requested";
    startButton.textContent = job.job_type === "video" && active ? "Video job active" : status === "starting" ? "Starting…" : "Start camera";
    stopButton.textContent = status === "stop_requested" ? "Stopping…" : "Stop session";
    const labels = {idle:"Ready to start",starting:"Starting session",running:"Session running",stop_requested:"Termination requested",completed:"Session complete",cancelled:"Session stopped",error:"Session error"};
    $("job-label").textContent = job.job_type === "video" && active ? "Video analysis active" : labels[status] || status;
    setDot("job-dot", status);
    $("camera-status").textContent = job.camera_status || "Idle";
    $("inference-status").textContent = job.inference_status || "Idle";
    setDot("camera-dot", job.camera_status || "idle");
    setDot("inference-dot", job.inference_status || "idle");
    $("fps-value").innerHTML = Number.isFinite(job.processing_fps) && job.processing_fps >= 0 ? `${job.processing_fps.toFixed(1)} <small>FPS</small>` : "—";
    const latestConfidence = Array.isArray(job.detections) && job.detections.length ? Number(job.detections[0].confidence) : NaN;
    $("confidence-value").textContent = Number.isFinite(latestConfidence) ? `${(latestConfidence * 100).toFixed(1)}% · latest box` : "—";
    $("frame-label").textContent = job.frame_number ? `Frame ${job.frame_number.toLocaleString()}` : "No frames yet";
    $("latest-frame-chip").textContent = job.frame_number
      ? `Frame ${job.frame_number.toLocaleString()} · ${Number(job.total_detections || 0).toLocaleString()} detections`
      : "No frames yet";
    $("session-duration").textContent = duration(job.duration_seconds);
    $("session-time-label").textContent = job.job_type === "video" ? "Video job time" : "Session time";
    const sessionTotal = job.job_type === "video" ? job.behavior_detection_total : job.session_detection_total;
    $("session-total-label").textContent = job.job_type === "video" ? "Video behavior-box detections" : "Frame-level box observations";
    $("session-total").textContent = Number.isFinite(sessionTotal) ? sessionTotal.toLocaleString() : "—";
    const frameCounts = job.class_counts || {};
    classes.forEach(([key]) => {
      const card = cards.get(key);
      card.querySelector(".behavior-values strong").textContent = (frameCounts[key] || 0).toLocaleString();
    });
    const observationAt = Number(job.latest_observation_at);
    const freshness = observationAt > 0 ? Math.max(0, Date.now() / 1000 - observationAt) : null;
    const stale = active && freshness !== null && freshness > 5;
    if (job.job_type === "video") {
      $("data-freshness").textContent = "Webcam freshness unavailable during video analysis";
      setDot("freshness-dot", "idle");
    } else if (freshness === null) {
      $("data-freshness").textContent = status === "running" ? "Waiting for first frame" : "No observations yet";
      setDot("freshness-dot", status === "error" ? "error" : status === "starting" ? "starting" : "idle");
    } else {
      $("data-freshness").textContent = stale ? `Stale · ${Math.floor(freshness)}s since frame` : freshness < 2 ? "Updated just now" : `Updated ${Math.floor(freshness)}s ago`;
      setDot("freshness-dot", stale ? "starting" : "running");
    }
    const error = job.error;
    const health = $("health-message");
    if (error) {
      health.classList.add("error");
      health.textContent = error;
      $("action-message").textContent = "The local application remains available. Correct the reported model, camera, or inference issue, then start a new session.";
    } else {
      health.classList.remove("error");
      if (status === "stop_requested") {
        health.textContent = "Stop requested. The current inference iteration and resource cleanup will finish before the session is marked stopped.";
      } else if (stale) {
        health.textContent = "No new webcam frame has arrived for more than five seconds. Check the camera connection and permissions; stop and restart the session if frames do not resume.";
      } else if (job.job_type === "video" && active) {
        health.textContent = "A video analysis is running. The webcam feed and live history are unavailable until that job finishes.";
      } else if (status === "starting") {
        health.textContent = "Loading the behavior model and opening the configured webcam.";
      } else if (status === "running" && !job.frame_number) {
        health.textContent = job.camera_status === "open"
          ? "Camera connected; waiting for its first frame and inference result."
          : "Loading the behavior model and opening the configured webcam.";
      } else if (status === "running") {
        health.textContent = Number(job.total_detections) === 0
          ? "Camera and inference are active; this frame contains zero behavior detections. The worker is healthy and will keep checking new frames."
          : "Camera and inference are active. Browser chart updates can be paused independently.";
      } else if (status === "completed" || status === "cancelled") {
        health.textContent = "The camera worker has stopped and released its resources.";
      } else {
        health.textContent = "Start a webcam session to show the feed and recent sampled-frame observations.";
      }
      $("action-message").textContent = job.job_type === "video" && active
        ? "Open Video Analysis to review progress. Stop session requests cancellation of the active job."
        : status === "running"
          ? "Refreshing this page reconnects to the current session; it does not start another worker."
          : "The existing standalone webcam command and legacy dashboard remain available.";
    }
    connectStream(job);
    if ((job.job_id || "") !== lastTrendJob) {
      lastTrendJob = job.job_id || "";
      window.clearTimeout(historyTimer);
      if (!chartPaused) refreshHistory();
    }
  }

  async function poll() {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch("/api/state", {cache: "no-store"});
      if (!response.ok) throw new Error("Could not read VisionClass session state.");
      const data = await response.json();
      $("server-status").textContent = "Connected";
      render(data.job || {});
    } catch (error) {
      $("server-status").textContent = "Disconnected";
      setDot("stream-dot", "error");
      $("health-message").classList.add("error");
      $("health-message").textContent = `${error.message} Check that the local VisionClass server is still running.`;
    } finally {
      polling = false;
      window.setTimeout(poll, 2000);
    }
  }

  async function getToken() {
    const response = await fetch("/api/csrf", {cache: "no-store"});
    if (!response.ok) throw new Error("Could not initialize the local security token. Refresh the page and retry.");
    const token = (await response.json()).csrf_token;
    if (!token) throw new Error("The local security token was unavailable. Refresh the page and retry.");
    return token;
  }

  startButton.addEventListener("click", async () => {
    startButton.disabled = true;
    $("action-message").textContent = "Starting the webcam session…";
    try {
      const token = await getToken();
      const response = await fetch("/api/session/start", {method:"POST", headers:{"X-CSRF-Token":token}});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not start the webcam session.");
      render(data.job);
    } catch (error) {
      $("action-message").textContent = error.message;
      startButton.disabled = false;
    }
  });
  stopButton.addEventListener("click", async () => {
    stopButton.disabled = true;
    $("action-message").textContent = "Termination requested. Waiting for the current iteration and cleanup…";
    try {
      const token = await getToken();
      const response = await fetch("/api/session/stop", {method:"POST", headers:{"X-CSRF-Token":token}});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not request stop.");
      render(data.job);
    } catch (error) {
      $("action-message").textContent = error.message;
      stopButton.disabled = false;
    }
  });
  $("trend-window").addEventListener("change", () => {
    selectedWindow = Number($("trend-window").value);
    window.clearTimeout(historyTimer);
    if (!chartPaused) refreshHistory();
  });
  $("chart-pause").addEventListener("click", () => {
    chartPaused = !chartPaused;
    const button = $("chart-pause");
    button.textContent = chartPaused ? "Resume charts" : "Pause charts";
    button.setAttribute("aria-pressed", String(chartPaused));
    if (latestHistory) updateChartState(latestHistory);
    window.clearTimeout(historyTimer);
    if (!chartPaused) refreshHistory();
  });
  $("series-toggle-list").addEventListener("change", (event) => {
    const input = event.target.closest("input[data-class]");
    if (!input) return;
    if (input.checked) visibleSeries.add(input.dataset.class);
    else visibleSeries.delete(input.dataset.class);
    if (latestHistory) renderTrend(latestHistory);
  });
  poll();
  refreshHistory();
})();
