/**
 * BORDER AI — Indian Border-Tech Surveillance Workstation
 * Smart India Hackathon 2026
 */

// Global State
let currentMode = "REAL"; // 'REAL' or 'DEMO'
let currentVideoFilename = "sample_cctv_night.mp4";
let currentJobId = null;
let pollingInterval = null;
let currentDetections = [];
let allEvents = [];
let activeFilter = "ALL";
let isAnalysisActive = false;
let frameHistory = [];

// DOM Elements
const mainVideo = document.getElementById("mainVideo");
const hudCanvas = document.getElementById("hudCanvas");
const ctx = hudCanvas ? hudCanvas.getContext("2d") : null;

// Tab Names Mapping for Breadcrumbs
const TAB_NAMES = {
  "overview": "OVERVIEW",
  "surveillance": "VIDEO ANALYSIS WORKSTATION",
  "thermal": "LOW-LIGHT & THERMAL INTELLIGENCE",
  "datasets": "DATASET & QUALITY AUDIT HUB",
  "performance": "AI MODEL PERFORMANCE",
  "pipeline": "TRAINING PIPELINE & FUSION"
};

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  fetchModelStatus();
  fetchDatasets();
  setupVideoSync();
});

function initCanvas() {
  function resizeCanvas() {
    if (mainVideo && hudCanvas) {
      hudCanvas.width = mainVideo.clientWidth || 640;
      hudCanvas.height = mainVideo.clientHeight || 480;
    }
  }
  window.addEventListener("resize", resizeCanvas);
  if (mainVideo) {
    mainVideo.addEventListener("loadedmetadata", resizeCanvas);
  }
  setTimeout(resizeCanvas, 300);
}

// ----------------- TAB NAVIGATION -----------------
function switchTab(tabId) {
  document.querySelectorAll(".tab-pane").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));

  const targetPane = document.getElementById(`tab-${tabId}`);
  if (targetPane) {
    targetPane.classList.add("active");
  }

  // Update Breadcrumb
  const breadcrumb = document.getElementById("activeBreadcrumb");
  if (breadcrumb) {
    breadcrumb.innerText = TAB_NAMES[tabId] || tabId.toUpperCase();
  }

  // Highlight active sidebar item
  const buttons = document.querySelectorAll(".nav-item");
  buttons.forEach(btn => {
    if (btn.getAttribute("onclick") && btn.getAttribute("onclick").includes(`'${tabId}'`)) {
      btn.classList.add("active");
    }
  });

  if (tabId === "surveillance") {
    setTimeout(() => {
      if (hudCanvas && mainVideo) {
        hudCanvas.width = mainVideo.clientWidth || 640;
        hudCanvas.height = mainVideo.clientHeight || 480;
      }
    }, 100);
  }
}

// ----------------- MODE SELECTION -----------------
function setMode(mode) {
  currentMode = mode;
  document.getElementById("btnRealMode").classList.toggle("active", mode === "REAL");
  document.getElementById("btnDemoMode").classList.toggle("active", mode === "DEMO");
  
  const hudBadge = document.getElementById("hudModelBadge");
  if (hudBadge) {
    if (mode === "REAL") {
      hudBadge.innerHTML = "MODEL: YOLOv8 + BORDER CORE<br>CONF THRESHOLD: 30%";
    } else {
      hudBadge.innerHTML = "MODEL: DEMO / SIMULATION MODE<br>CONF THRESHOLD: 30%";
    }
  }
}

// ----------------- VIDEO HANDLING -----------------
function loadSampleVideo(type) {
  if (type === "user") {
    currentVideoFilename = "whatsapp_surveillance.mp4";
  } else if (type === "visible") {
    currentVideoFilename = "sample_cctv_night.mp4";
  } else {
    currentVideoFilename = "sample_thermal_night.mp4";
  }
  
  if (mainVideo) {
    mainVideo.src = `/uploads/${currentVideoFilename}`;
    mainVideo.load();
    mainVideo.play().catch(() => {});
  }
  const badge = document.getElementById("hudStatusBadge");
  if (badge) {
    badge.innerText = "SOURCE: " + currentVideoFilename;
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const badge = document.getElementById("hudStatusBadge");
  if (badge) badge.innerText = "UPLOADING...";
  
  try {
    const res = await fetch("/api/upload-video", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      currentVideoFilename = data.filename;
      if (mainVideo) {
        mainVideo.src = data.video_url;
        mainVideo.load();
        mainVideo.play().catch(() => {});
      }
      if (badge) badge.innerText = "LOADED: " + data.filename;
    } else {
      alert("Upload failed: " + data.detail);
    }
  } catch (err) {
    console.error("Upload error:", err);
    alert("Network error while uploading video.");
  }
}

function togglePlayPause() {
  if (!mainVideo) return;
  if (mainVideo.paused) {
    mainVideo.play();
  } else {
    mainVideo.pause();
  }
}

function restartVideo() {
  if (!mainVideo) return;
  mainVideo.currentTime = 0;
  mainVideo.play();
}

// ----------------- AI VIDEO ANALYSIS -----------------
async function startAIAnalysis() {
  const btn = document.getElementById("btnStartAnalysis");
  btn.disabled = true;
  btn.innerText = "ANALYZING...";
  
  const badge = document.getElementById("hudStatusBadge");
  if (badge) {
    badge.innerText = "AI ANALYSIS ACTIVE";
    badge.className = "panel-badge alert";
  }

  const formData = new FormData();
  formData.append("video_filename", currentVideoFilename);
  formData.append("is_thermal", "false");
  formData.append("mode", currentMode);

  try {
    const res = await fetch("/api/analyze-video", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      currentJobId = data.job_id;
      isAnalysisActive = true;
      if (mainVideo) mainVideo.play().catch(() => {});
      startStatusPolling(currentJobId);
    } else {
      alert("Analysis error: " + data.detail);
      btn.disabled = false;
      btn.innerText = "Start Analysis";
    }
  } catch (err) {
    console.error("Analysis initiation error:", err);
    btn.disabled = false;
    btn.innerText = "Start Analysis";
  }
}

function startStatusPolling(jobId) {
  if (pollingInterval) clearInterval(pollingInterval);

  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/analysis-status/${jobId}`);
      if (!res.ok) return;

      const data = await res.json();
      updateDashboardTelemetry(data);

      if (data.status === "COMPLETED" || data.status === "ERROR") {
        clearInterval(pollingInterval);
        isAnalysisActive = false;
        const btn = document.getElementById("btnStartAnalysis");
        if (btn) {
          btn.disabled = false;
          btn.innerText = "Start Analysis";
        }
        const badge = document.getElementById("hudStatusBadge");
        if (badge) {
          badge.innerText = "ANALYSIS COMPLETE";
          badge.className = "panel-badge live";
        }
        
        fetchFrameHistory(jobId);
      }
    } catch (e) {
      console.error("Polling error:", e);
    }
  }, 200);
}

async function fetchFrameHistory(jobId) {
  try {
    const res = await fetch(`/api/detections-history/${jobId}`);
    if (res.ok) {
      const data = await res.json();
      frameHistory = data.history || [];
    }
  } catch (e) {
    console.error("History fetch error:", e);
  }
}

function updateDashboardTelemetry(data) {
  const p = data.progress_percent || 0;
  const pBar = document.getElementById("progressBar");
  if (pBar) pBar.style.width = `${p}%`;
  
  const pText = document.getElementById("progressText");
  if (pText) pText.innerText = `${p}% (Frame ${data.current_frame || 0}/${data.total_frames || 0})`;

  if (data.stats) {
    const totalEl = document.getElementById("valTotalDetections");
    if (totalEl) totalEl.innerText = data.stats.total_detections || 0;
    
    const humanEl = document.getElementById("valHumanCount");
    if (humanEl) humanEl.innerText = data.stats.humans || 0;
    
    const animEl = document.getElementById("valAnimalCount");
    if (animEl) animEl.innerText = data.stats.animals || 0;
    
    const vehEl = document.getElementById("valVehicleCount");
    if (vehEl) vehEl.innerText = data.stats.vehicles || 0;
    
    const unkEl = document.getElementById("valUnknownCount");
    if (unkEl) unkEl.innerText = data.stats.unknown || 0;
    
    const trackEl = document.getElementById("valActiveTracks");
    if (trackEl) trackEl.innerText = data.stats.active_tracks || 0;
    
    const alertEl = document.getElementById("valAlertsCount");
    if (alertEl) alertEl.innerText = data.stats.alerts || 0;
    
    const activeBadge = document.getElementById("activeAlertsBadge");
    if (activeBadge) activeBadge.innerText = `${data.stats.alerts || 0} ALERTS`;
  }

  const fpsEl = document.getElementById("hudFps");
  if (fpsEl) fpsEl.innerText = data.fps ? data.fps.toFixed(1) : "25.0";
  
  const tracksEl = document.getElementById("hudTracksCount");
  if (tracksEl) tracksEl.innerText = data.active_tracks_count || 0;

  if (data.live_detections && data.live_detections.length > 0) {
    currentDetections = data.live_detections;
    renderLiveDetectionsList(data.live_detections);
    checkAndTriggerAlerts(data.live_detections);
  }
}

// ----------------- HUD CANVAS BOUNDING BOX RENDERING (CLEAN CV WORKSTATION) -----------------
function setupVideoSync() {
  function renderFrame() {
    if (mainVideo && !mainVideo.paused && !mainVideo.ended) {
      const curTime = mainVideo.currentTime;
      const mins = Math.floor(curTime / 60);
      const secs = Math.floor(curTime % 60);
      const millis = Math.floor((curTime % 1) * 100);
      
      const tcEl = document.getElementById("hudTimecode");
      if (tcEl) {
        tcEl.innerText = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(2, '0')}`;
      }

      if (frameHistory.length > 0) {
        const approxFrame = Math.floor(curTime * 20);
        const match = frameHistory.find(f => Math.abs(f.frame - approxFrame) <= 2);
        if (match && match.detections) {
          currentDetections = match.detections;
          renderLiveDetectionsList(match.detections);
        }
      }

      drawBoundingBoxes();
    }
    requestAnimationFrame(renderFrame);
  }
  requestAnimationFrame(renderFrame);
}

function drawBoundingBoxes() {
  if (!ctx || !hudCanvas) return;
  ctx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
  if (!currentDetections || currentDetections.length === 0) return;

  const w = hudCanvas.width;
  const h = hudCanvas.height;

  currentDetections.forEach(det => {
    const [nx1, ny1, nx2, ny2] = det.norm_bbox || [0.2, 0.3, 0.4, 0.7];
    const x1 = nx1 * w;
    const y1 = ny1 * h;
    const boxW = Math.max(18, (nx2 - nx1) * w);
    const boxH = Math.max(18, (ny2 - ny1) * h);

    // Clean CV color scheme
    const color = det.class_name === "HUMAN" ? "#EF4444" : (
      det.class_name === "VEHICLE" ? "#3B82F6" : (
        det.class_name === "ANIMAL" ? "#10B981" : "#F59E0B"
      )
    );

    // Crisp thin border
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x1, y1, boxW, boxH);

    // Subtle corner markers
    const cornerSize = Math.min(8, boxW * 0.25);
    ctx.fillStyle = color;
    ctx.fillRect(x1 - 1, y1 - 1, cornerSize, 2);
    ctx.fillRect(x1 - 1, y1 - 1, 2, cornerSize);
    ctx.fillRect(x1 + boxW - cornerSize + 1, y1 - 1, cornerSize, 2);
    ctx.fillRect(x1 + boxW - 1, y1 - 1, 2, cornerSize);

    // Engineering Label in IBM Plex Mono
    const label = `${det.class_name} ${det.confidence}% [${det.display_id || 'TRK'}]`;
    ctx.font = "600 10px 'IBM Plex Mono', monospace";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = "rgba(13, 17, 26, 0.92)";
    ctx.fillRect(x1, y1 - 16, textWidth + 8, 16);
    ctx.fillStyle = color;
    ctx.fillText(label, x1 + 4, y1 - 4);
  });
}

// ----------------- LIVE DETECTIONS & ALERTS FEED -----------------
function renderLiveDetectionsList(detections) {
  const container = document.getElementById("liveDetectionsContainer");
  if (!container) return;
  
  const countEl = document.getElementById("liveDetectionsCount");
  if (countEl) countEl.innerText = `${detections.length} Target(s) Active`;

  if (detections.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.75rem; text-align: center; padding: 1rem; font-family: var(--font-mono);">No active targets in sector</div>`;
    return;
  }

  const rows = detections.map(d => `
    <div class="detection-entry">
      <div style="display: flex; align-items: center; gap: 0.5rem;">
        <span class="det-badge-pill ${d.class_name}">${d.class_name}</span>
        <span style="font-family: var(--font-mono); font-weight: 600; color: var(--text-primary);">${d.display_id || 'Target'}</span>
      </div>
      <div style="display: flex; align-items: center; gap: 0.5rem; font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted);">
        <span>${d.confidence}%</span>
        ${d.dwell_time ? `<span style="color: var(--saffron-primary);">${d.dwell_time}s</span>` : ''}
      </div>
    </div>
  `).join("");

  container.innerHTML = rows;
}

let lastAlertTime = 0;
function checkAndTriggerAlerts(detections) {
  const now = Date.now();
  if (now - lastAlertTime < 2500) return;

  const humans = detections.filter(d => d.class_name === "HUMAN");
  const vehicles = detections.filter(d => d.class_name === "VEHICLE");
  const tcEl = document.getElementById("hudTimecode");
  const curTime = tcEl ? tcEl.innerText : "00:00.00";

  if (humans.length >= 2) {
    lastAlertTime = now;
    pushAlert("MULTIPLE HUMAN PRESENCE", "ATTENTION", `${humans.length} persons detected in perimeter zone.`, curTime);
    addEventToLog(curTime, `Group (${humans.length})`, "Human", "94.8%", "Multiple Human Presence", "ATTENTION", "Simultaneous group detected in surveillance sector");
  } else if (humans.length === 1) {
    const h = humans[0];
    if (h.dwell_time && h.dwell_time > 6.0) {
      lastAlertTime = now;
      pushAlert("SUSTAINED PRESENCE", "ATTENTION", `${h.display_id} stationary in sector for ${h.dwell_time}s`, curTime);
      addEventToLog(curTime, h.display_id, "Human", `${h.confidence}%`, "Sustained Presence", "ATTENTION", "Target stationary beyond time threshold");
    }
  }

  if (vehicles.length > 0 && Math.random() < 0.15) {
    const v = vehicles[0];
    addEventToLog(curTime, v.display_id || "Vehicle", "Vehicle", `${v.confidence}%`, "Vehicle Movement", "INFO", "Vehicle crossing monitored sector perimeter");
  }
}

function pushAlert(type, severity, message, timecode) {
  const container = document.getElementById("alertFeedContainer");
  if (!container) return;

  const alertEl = document.createElement("div");
  alertEl.className = `alert-card-item ${severity}`;
  alertEl.innerHTML = `
    <div class="alert-card-header">
      <span>${type}</span>
      <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);">${timecode}</span>
    </div>
    <div style="color: var(--text-secondary); font-size: 0.72rem;">${message}</div>
  `;
  container.prepend(alertEl);

  while (container.children.length > 10) {
    container.removeChild(container.lastChild);
  }
}

// ----------------- EVENT LOG TABLE & FILTERING -----------------
function addEventToLog(timestamp, trackId, className, confidence, eventType, severity, details) {
  const eventObj = { timestamp, trackId, className, confidence, eventType, severity, details };
  allEvents.unshift(eventObj);
  const badge = document.getElementById("eventCountBadge");
  if (badge) badge.innerText = `${allEvents.length} EVENTS`;
  renderEventsTable();
}

function filterEvents(filter) {
  activeFilter = filter;
  renderEventsTable();
}

function renderEventsTable() {
  const tbody = document.getElementById("eventsTableBody");
  if (!tbody) return;

  let filtered = allEvents;
  if (activeFilter !== "ALL") {
    if (activeFilter === "ATTENTION") {
      filtered = allEvents.filter(e => e.severity === "ATTENTION");
    } else {
      filtered = allEvents.filter(e => e.className.toUpperCase() === activeFilter);
    }
  }

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem; font-family: var(--font-mono);">No events match the selected filter.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.slice(0, 30).map(e => `
    <tr>
      <td class="mono-cell">${e.timestamp}</td>
      <td class="mono-cell" style="font-weight: 600;">${e.trackId}</td>
      <td><span class="det-badge-pill ${e.className.toUpperCase()}">${e.className}</span></td>
      <td class="mono-cell">${e.confidence}</td>
      <td>${e.eventType}</td>
      <td><span class="det-badge-pill ${e.severity === 'ATTENTION' ? 'HUMAN' : (e.severity === 'MONITOR' ? 'UNKNOWN' : 'VEHICLE')}">${e.severity}</span></td>
      <td style="color: var(--text-muted);">${e.details}</td>
    </tr>
  `).join("");
}

// ----------------- DATASETS HUB -----------------
async function fetchDatasets() {
  try {
    const res = await fetch("/api/dataset-status");
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById("datasetCardsContainer");
    if (!container) return;

    container.innerHTML = data.datasets.map(ds => `
      <div class="dataset-inspect-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <div style="font-weight: 600; font-size: 0.88rem; color: var(--text-primary);">${ds.name}</div>
            <div style="font-size: 0.72rem; font-family: var(--font-mono); color: var(--saffron-primary);">${ds.type}</div>
          </div>
          <span class="det-badge-pill ANIMAL">${ds.status}</span>
        </div>

        <p style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.4;">${ds.description}</p>

        <div style="font-family: var(--font-mono); font-size: 0.72rem; display: flex; flex-direction: column; gap: 0.25rem; margin-top: 0.25rem;">
          <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-muted);">Primary Class:</span><span>${ds.primary_class}</span></div>
          <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-muted);">Format:</span><span>${ds.annotation_format}</span></div>
          <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-muted);">Resolution:</span><span>${ds.resolution}</span></div>
          <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-muted);">Samples:</span><span>${ds.samples}</span></div>
          <div style="display: flex; justify-content: space-between;"><span style="color: var(--text-muted);">License:</span><span>${ds.license}</span></div>
        </div>

        <div style="margin-top: auto; padding-top: 0.5rem; border-top: 1px solid var(--border-subtle);">
          <a href="${ds.reference_url}" target="_blank" style="font-size: 0.72rem; color: var(--saffron-primary); text-decoration: none; font-family: var(--font-mono);">
            ↗ View Benchmark Reference
          </a>
        </div>
      </div>
    `).join("");
  } catch (e) {
    console.error("Dataset fetch error:", e);
  }
}

// ----------------- MODEL STATUS & SETTINGS -----------------
async function fetchModelStatus() {
  try {
    const res = await fetch("/api/model-status");
    if (!res.ok) return;
    const info = await res.json();
    const badge = document.getElementById("hudModelBadge");
    if (badge) {
      badge.innerHTML = `MODEL: ${info.model_name}<br>STATUS: ${info.status}`;
    }
  } catch (e) {
    console.error("Model status error:", e);
  }
}

function openSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.classList.add("active");
}

function closeSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.classList.remove("active");
}

async function saveSettings() {
  const conf = parseFloat(document.getElementById("sliderConf").value) / 100.0;
  const iou = parseFloat(document.getElementById("sliderIoU").value) / 100.0;
  const weights = document.getElementById("inpWeights").value;

  const formData = new FormData();
  formData.append("conf", conf);
  formData.append("iou", iou);
  formData.append("weights_name", weights);

  try {
    const res = await fetch("/api/model-config", {
      method: "POST",
      body: formData
    });
    if (res.ok) {
      alert("Settings applied successfully.");
      closeSettingsModal();
      fetchModelStatus();
    }
  } catch (e) {
    console.error("Settings error:", e);
  }
}

// ----------------- REPORT DOWNLOAD -----------------
function downloadReport(format) {
  if (!currentJobId) {
    alert("Please run an AI analysis first before exporting report.");
    return;
  }
  window.open(`/api/report/${currentJobId}?format=${format}`, "_blank");
}
