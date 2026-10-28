(() => {
  const $ = (id) => document.getElementById(id);
  const fileInput = $("video-file");
  const form = $("upload-form");
  const maxBytes = 1024 ** 3;
  const allowed = new Set([".mp4", ".avi", ".mov", ".mkv"]);
  let runId = "";
  let csrfToken = "";
  let pollHandle = 0;

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    $("file-name").textContent = file ? `${file.name} · ${(file.size / (1024 ** 2)).toFixed(1)} MiB` : "No file selected";
    $("upload-error").hidden = true;
  });

  function showError(message) {
    $("upload-error").hidden = false;
    $("upload-error").textContent = message;
  }
  async function getToken() {
    if (csrfToken) return csrfToken;
    const response = await fetch("/api/csrf", {cache:"no-store"});
    if (!response.ok) throw new Error("Could not initialize the local security token. Refresh the page and retry.");
    csrfToken = (await response.json()).csrf_token;
    return csrfToken;
  }
  function setProgress(label, value, detail) {
    $("progress-label").textContent = label;
    $("progress-value").textContent = value === null ? "—" : `${Math.round(value)}%`;
    $("upload-progress").value = value === null ? 0 : Math.max(0, Math.min(100, value));
    $("progress-detail").textContent = detail || "";
  }
  function showArtifacts(run) {
    const list = $("artifact-list");
    list.replaceChildren();
    const artifacts = run.artifacts || {};
    const labels = {source_video:"Source video",annotated_video:"Annotated video",history_csv:"History CSV",summary_csv:"Summary CSV",analytics_report:"Markdown report",analytics_dashboard:"HTML report"};
    Object.keys(labels).forEach((name) => {
      if (!Object.prototype.hasOwnProperty.call(artifacts, name)) return;
      if (name === "annotated_video" && !(run.output_video && run.output_video.validated === true)) return;
      const link = document.createElement("a");
      link.className = "artifact-link";
      link.href = `/api/runs/${encodeURIComponent(run.run_id)}/artifacts/${encodeURIComponent(name)}`;
      link.textContent = `↓ ${labels[name]}`;
      list.append(link);
    });
    const validation = run.output_video || {};
    if (artifacts.annotated_video && validation.validated === true) {
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = `/api/runs/${encodeURIComponent(run.run_id)}/playback?v=${encodeURIComponent(validation.output_bytes || run.run_id)}`;
      video.setAttribute("aria-label", "Annotated classroom behavior video");
      video.setAttribute("playsinline", "");
      video.addEventListener("loadedmetadata", () => {
        const dimensions = `${video.videoWidth}×${video.videoHeight}`;
        const duration = Number.isFinite(video.duration) ? `${video.duration.toFixed(1)} seconds` : "duration unavailable";
        $("playback-status").textContent = `Ready to play · ${dimensions} · ${duration} · ${validation.codec || "H.264 MP4"}`;
      });
      video.addEventListener("error", () => {
        $("playback-status").textContent = "The browser could not decode this video. Download the annotated MP4 and open it in a local media player; confirm that the browser supports H.264 MP4 playback.";
        $("playback-status").classList.add("error");
      });
      video.addEventListener("stalled", () => {
        $("playback-status").textContent = "Video data stopped arriving. Try reloading playback or download the annotated MP4.";
      });
      $("video-player-wrap").replaceChildren(video);
      $("playback-status").classList.remove("error");
      $("playback-status").textContent = "Loading validated annotated output…";
    } else {
      const message = document.createElement("div");
      message.className = "player-placeholder";
      const title = document.createElement("strong");
      const detail = document.createElement("span");
      title.textContent = "Annotated output unavailable";
      detail.textContent = "No validated, playable video was published for this run.";
      message.append(title, detail);
      $("video-player-wrap").replaceChildren(message);
      $("playback-status").classList.add("error");
      $("playback-status").textContent = "The run has no validated annotated MP4. Check the run error or retry after correcting the reported codec/output issue.";
    }
  }
  function renderRun(run) {
    $("result-title").textContent = run.display_name || "Video run";
    $("run-status").textContent = run.status || "Unknown";
    const progress = run.progress || run;
    if (["queued", "running"].includes(run.status)) {
      const total = progress.total_frames || (run.decoder && run.decoder.total_frames);
      const percent = total && progress.processed_frames ? (progress.processed_frames / total) * 100 : null;
      const finalizing = {
        encoding_h264: "Encoding browser-ready H.264 output",
        finalizing_output: "Finalizing annotated output",
        validating_output: "Validating annotated playback",
      }[run.output_stage];
      if (finalizing) {
        setProgress(finalizing, total ? 100 : null, `${(progress.processed_frames || 0).toLocaleString()} frames processed${total ? ` of ${total.toLocaleString()}` : " · total unavailable"}. Playback remains unavailable until output validation finishes.`);
      } else {
        setProgress("Processing video", percent, total ? `${(progress.processed_frames || 0).toLocaleString()} of ${total.toLocaleString()} frames` : `${(progress.processed_frames || 0).toLocaleString()} frames processed · total unavailable`);
      }
      $("progress-section").hidden = false;
      $("cancel-button").hidden = false;
    } else if (run.status === "completed") {
      $("cancel-button").hidden = true;
      const hasPlayableOutput = run.output_video && run.output_video.validated === true && run.artifacts && run.artifacts.annotated_video;
      if (hasPlayableOutput) {
        const completePercent = progress.total_frames ? 100 : null;
        const outputFrames = run.output_video.decoded_output_frames;
        setProgress("Analysis complete", completePercent, `${(progress.processed_frames || 0).toLocaleString()} input frames processed · ${Number.isFinite(outputFrames) ? outputFrames.toLocaleString() : "unknown"} annotated output frames decoded`);
        $("run-status").textContent = "Completed · output validated";
      } else {
        $("run-status").textContent = "Annotated output unavailable";
        setProgress("Analysis output unavailable", null, "This run lacks a validated annotated video, so it is not presented as a complete playable result.");
        showError("The job reported completion without a validated annotated MP4. Retry the run and check the OpenCV/FFmpeg H.264 output configuration.");
      }
      $("progress-section").hidden = false;
      showArtifacts(run);
    } else if (run.status === "cancelled") {
      setProgress("Processing cancelled", null, `${(progress.processed_frames || 0).toLocaleString()} frames processed before cancellation`);
      $("progress-section").hidden = false;
      $("cancel-button").hidden = true;
      showArtifacts(run);
    } else if (run.status === "error") {
      $("progress-section").hidden = false;
      $("cancel-button").hidden = true;
      showError(run.error || "Video processing failed. Check the selected file and model configuration, then retry.");
    }
    const optional = run.optional_errors || [];
    $("optional-errors").hidden = !optional.length;
    $("optional-errors").textContent = optional.join(" ");
  }
  async function pollRun() {
    if (!runId) return;
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, {cache:"no-store"});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not read video run state.");
      renderRun(data.run);
      if (["queued", "running"].includes(data.run.status)) pollHandle = window.setTimeout(pollRun, 1500);
      else $("cancel-button").hidden = true;
    } catch (error) {
      showError(`${error.message} Refresh this page to reconnect to the run.`);
      pollHandle = window.setTimeout(pollRun, 3000);
    }
  }
  function beginRun(id) {
    runId = id;
    window.clearTimeout(pollHandle);
    $("progress-section").hidden = false;
    $("cancel-button").hidden = false;
    $("artifact-list").replaceChildren();
    $("video-player-wrap").innerHTML = '<div class="player-placeholder"><span aria-hidden="true">◌</span><strong>Processing video</strong><span>Annotated playback will be available when the run completes.</span></div>';
    $("playback-status").textContent = "Output is not finalized yet.";
    $("playback-status").classList.remove("error");
    pollRun();
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files && fileInput.files[0];
    $("upload-error").hidden = true;
    if (!file) return showError("Choose a video file first.");
    const suffix = `.${file.name.split(".").pop().toLowerCase()}`;
    if (!allowed.has(suffix)) return showError("Choose an MP4, AVI, MOV, or MKV video.");
    if (file.size > maxBytes) return showError("This video exceeds the 1 GiB payload limit.");
    $("upload-button").disabled = true;
    $("progress-section").hidden = false;
    setProgress("Uploading video", 0, "The file is sent to the local VisionClass process.");
    try {
      const token = await getToken();
      const body = new FormData();
      body.append("video", file, file.name);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/videos");
      xhr.setRequestHeader("X-CSRF-Token", token);
      xhr.upload.addEventListener("progress", (progressEvent) => {
        const percent = progressEvent.lengthComputable ? (progressEvent.loaded / progressEvent.total) * 100 : null;
        setProgress("Uploading video", percent, progressEvent.lengthComputable ? `${(progressEvent.loaded / (1024 ** 2)).toFixed(1)} MiB sent` : "Upload progress is unavailable for this connection.");
      });
      xhr.addEventListener("load", () => {
        let data = {};
        try { data = JSON.parse(xhr.responseText); } catch (_error) { data = {}; }
        $("upload-button").disabled = false;
        if (xhr.status < 200 || xhr.status >= 300) return showError(data.error || `Upload failed (${xhr.status}).`);
        $("upload-error").hidden = true;
        beginRun(data.run_id);
      });
      xhr.addEventListener("error", () => { $("upload-button").disabled = false; showError("The upload connection failed. The local server may have stopped; retry when it is available."); });
      xhr.send(body);
    } catch (error) {
      $("upload-button").disabled = false;
      showError(error.message);
    }
  });
  $("cancel-button").addEventListener("click", async () => {
    if (!runId) return;
    $("cancel-button").disabled = true;
    try {
      const token = await getToken();
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/cancel`, {method:"POST",headers:{"X-CSRF-Token":token}});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not request cancellation.");
      setProgress("Cancellation requested", null, "The current frame will finish before resources close.");
    } catch (error) { showError(error.message); }
    finally { $("cancel-button").disabled = false; }
  });
  async function reconnect() {
    const queryRun = new URLSearchParams(window.location.search).get("run_id");
    if (queryRun) {
      const response = await fetch(`/api/runs/${encodeURIComponent(queryRun)}`);
      if (response.ok) { beginRun(queryRun); return; }
    }
    const response = await fetch("/api/state", {cache:"no-store"});
    if (!response.ok) return;
    const data = await response.json();
    if (data.job && data.job.job_type === "video" && data.job.run_id && data.job.worker_alive) beginRun(data.job.run_id);
  }
  reconnect().catch((error) => { showError(error.message); });
})();
