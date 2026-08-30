/**
 * BORDER AI — Command Center Surveillance Workstation
 * Smart India Hackathon 2026
 * Multi-Object Tracking, Accurate Unique Person Counting, Dynamic Risk Scoring & Degraded CCTV Benchmark
 */

// Global State
let currentMode = "HIGH_ACCURACY"; // 'HIGH_ACCURACY' or 'REAL_TIME'
let currentVideoFilename = "whatsapp_surveillance.mp4";
let currentJobId = null;
let pollingInterval = null;
let isAnalysisRunning = false;
let allEvents = [];
let allAlerts = [];
let activeEventFilter = "ALL";
let trainingPollInterval = null;
let lastIdentities = [];

// DOM Elements
const mainVideo = document.getElementById("mainVideo");
const hudCanvas = document.getElementById("hudCanvas");
const hudCtx = hudCanvas ? hudCanvas.getContext("2d") : null;

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  checkServerHealth();
  fetchModelStatus();
  fetchDegradedBenchmark();
  loadSampleVideo("user");
});

// ----------------- TAB SWITCHING -----------------
function switchTab(tabId) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));

  const targetPanel = document.getElementById(`tab-${tabId}`);
  if (targetPanel) targetPanel.classList.add("active");

  const activeBtn = Array.from(document.querySelectorAll(".nav-btn")).find(b => b.getAttribute("onclick")?.includes(tabId));
  if (activeBtn) activeBtn.classList.add("active");

  if (tabId === "benchmark") {
    fetchDegradedBenchmark();
  } else if (tabId === "multicam") {
    setTimeout(drawSectorMap, 50);
  } else if (tabId === "context") {
    setTimeout(drawZoneCanvas, 50);
  } else if (tabId === "analytics") {
    setTimeout(renderAnalyticsCharts, 50);
  }
}

function setInferenceMode(mode) {
  currentMode = mode;
  document.getElementById("btnHighAccMode")?.classList.toggle("active", mode === "HIGH_ACCURACY");
  document.getElementById("btnRealtimeMode")?.classList.toggle("active", mode === "REAL_TIME");
  
  const statusBadge = document.getElementById("hudStatusBadge");
  if (statusBadge) {
    statusBadge.innerHTML = `<span class="status-pulsing-dot"></span><span>${mode === 'HIGH_ACCURACY' ? 'HIGH ACCURACY (TILED SAHI)' : 'REAL-TIME FAST'}</span>`;
  }
}

// ----------------- VIDEO HANDLING & CANVAS HUD -----------------
function initCanvas() {
  window.addEventListener("resize", resizeCanvas);
  if (mainVideo) {
    mainVideo.addEventListener("loadedmetadata", resizeCanvas);
    mainVideo.addEventListener("timeupdate", onVideoTimeUpdate);
  }
}

function resizeCanvas() {
  if (mainVideo && hudCanvas) {
    hudCanvas.width = mainVideo.clientWidth || 640;
    hudCanvas.height = mainVideo.clientHeight || 360;
  }
}

function loadSampleVideo(type) {
  document.querySelectorAll(".btn-source").forEach(b => b.classList.remove("active"));
  if (type === "user") {
    currentVideoFilename = "whatsapp_surveillance.mp4";
    document.getElementById("srcBtnUser")?.classList.add("active");
  } else if (type === "visible") {
    currentVideoFilename = "sample_cctv_night.mp4";
    document.getElementById("srcBtnVis")?.classList.add("active");
  } else {
    currentVideoFilename = "sample_thermal_night.mp4";
    document.getElementById("srcBtnTherm")?.classList.add("active");
  }

  if (mainVideo) {
    mainVideo.src = `/uploads/${currentVideoFilename}`;
    mainVideo.load();
    mainVideo.play().catch(() => {});
  }

  const hudCamTag = document.getElementById("hudCamTag");
  if (hudCamTag) {
    hudCamTag.innerText = type === "thermal" ? "CAM-03 / SECTOR-08 (THERMAL LWIR)" : "CAM-01 / SECTOR-07 (ALPHA)";
  }
}

async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  if (mainVideo) {
    mainVideo.src = URL.createObjectURL(file);
    mainVideo.load();
    mainVideo.play().catch(() => {});
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload-video", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (res.ok) {
      currentVideoFilename = data.filename;
      console.log("[Upload] Video uploaded successfully:", data.filename);
    }
  } catch (err) {
    console.error("Upload error:", err);
  }
}

async function checkServerHealth() {
  const headerStatus = document.getElementById("systemStatusHeader");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (res.ok) {
      if (headerStatus) headerStatus.innerText = `ONLINE (${data.device || 'AI'})`;
    }
  } catch (err) {
    if (headerStatus) headerStatus.innerText = "CONNECTING...";
    setTimeout(checkServerHealth, 4000);
  }
}

async function fetchModelStatus() {
  try {
    const res = await fetch("/api/model-status");
    const data = await res.json();
    if (res.ok) {
      const badge = document.getElementById("hudModelBadge");
      if (badge) badge.innerText = `MODEL: ${data.model_name} (${data.device})`;
    }
  } catch (err) {}
}

// ----------------- AI ANALYSIS CONTROLS & HUD RENDER -----------------
async function toggleAnalysis() {
  if (isAnalysisRunning) {
    stopAnalysis();
  } else {
    startAnalysis();
  }
}

async function startAnalysis() {
  const btn = document.getElementById("btnRunAnalysis");
  const btnText = document.getElementById("btnRunAnalysisText");
  const progWrapper = document.getElementById("videoProgressWrapper");

  if (btn) btn.classList.add("running");
  if (btnText) btnText.innerText = "STARTING PIPELINE...";
  if (progWrapper) progWrapper.style.display = "block";

  try {
    const formData = new FormData();
    formData.append("video_filename", currentVideoFilename);
    formData.append("is_thermal", currentVideoFilename.includes("thermal"));
    formData.append("mode", currentMode);

    const res = await fetch("/api/analyze-video", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    if (res.ok) {
      currentJobId = data.job_id;
      isAnalysisRunning = true;
      if (btnText) btnText.innerText = "STOP ANALYSIS";
      
      if (mainVideo) {
        mainVideo.currentTime = 0;
        mainVideo.play().catch(() => {});
      }

      startStatusPolling();
    }
  } catch (err) {
    console.error("Failed to start analysis:", err);
    if (btn) btn.classList.remove("running");
    if (btnText) btnText.innerText = "START AI ANALYSIS";
  }
}

function stopAnalysis() {
  isAnalysisRunning = false;
  clearInterval(pollingInterval);
  const btn = document.getElementById("btnRunAnalysis");
  const btnText = document.getElementById("btnRunAnalysisText");
  if (btn) btn.classList.remove("running");
  if (btnText) btnText.innerText = "START AI ANALYSIS";
}

function startStatusPolling() {
  clearInterval(pollingInterval);
  pollingInterval = setInterval(async () => {
    if (!currentJobId) return;

    try {
      const res = await fetch(`/api/analysis-status/${currentJobId}`);
      const data = await res.json();

      if (res.ok) {
        updateTelemetryDisplay(data);

        // Update Progress Bar
        const progFrame = document.getElementById("progFrameText");
        const progPct = document.getElementById("progPctText");
        const progFill = document.getElementById("progBarFill");
        if (progFrame) progFrame.innerText = `Processing Frame: ${data.current_frame} / ${data.total_frames} (${data.fps} FPS)`;
        if (progPct) progPct.innerText = `${data.progress_percent}%`;
        if (progFill) progFill.style.width = `${data.progress_percent}%`;

        if (data.status === "COMPLETED") {
          clearInterval(pollingInterval);
          isAnalysisRunning = false;
          const btn = document.getElementById("btnRunAnalysis");
          const btnText = document.getElementById("btnRunAnalysisText");
          if (btn) btn.classList.remove("running");
          if (btnText) btnText.innerText = "ANALYSIS COMPLETE";

          fetchUniqueIdentities();
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 300);
}

function updateTelemetryDisplay(data) {
  const stats = data.stats || {};
  const uniq = stats.unique_validated || {};
  const visible = stats.current_visible || {};
  const active = stats.active_tracks || {};
  const rawCount = stats.total_raw_detections || 0;

  // 1. Top Telemetry Counters
  const cardActTrk = document.getElementById("cardActiveTracks");
  const cardVisHum = document.getElementById("cardVisibleHumans");
  const cardUniqHum = document.getElementById("cardUniqueHumans");
  const cardUniqVeh = document.getElementById("cardUniqueVehicles");
  const cardUniqAnim = document.getElementById("cardUniqueAnimals");
  const cardTotDet = document.getElementById("cardTotalDetections");

  if (cardActTrk) cardActTrk.innerText = active.total || 0;
  if (cardVisHum) cardVisHum.innerText = visible.humans || 0;
  if (cardUniqHum) cardUniqHum.innerText = uniq.humans || 0;
  if (cardUniqVeh) cardUniqVeh.innerText = uniq.vehicles || 0;
  if (cardUniqAnim) cardUniqAnim.innerText = uniq.animals || 0;
  if (cardTotDet) cardTotDet.innerText = rawCount;

  // HUD Bottom Bar
  const hudFps = document.getElementById("hudFps");
  const hudUniqH = document.getElementById("hudUniqueHumansVal");
  const hudActT = document.getElementById("hudActiveTracksVal");
  if (hudFps && data.fps) hudFps.innerText = data.fps;
  if (hudUniqH) hudUniqH.innerText = uniq.humans || 0;
  if (hudActT) hudActT.innerText = active.total || 0;

  // Quality Report Badge
  if (data.quality_report) {
    const q = data.quality_report;
    const qBadge = document.getElementById("qualityReportBadge");
    if (qBadge) {
      qBadge.innerText = `QUALITY: ${q.quality_label} | LUX: ${q.brightness_lux} | BLUR: ${q.blur_score} | NOISE: ${q.noise_level} | ${q.resolution}`;
    }
  }

  // Draw Live Bounding Boxes on Canvas HUD
  if (data.live_detections) {
    drawBoundingBoxes(data.live_detections);
    renderLiveDetectionsList(data.live_detections);
  }

  // Risk Engine
  if (data.risk_data) {
    updateRiskEngine(data.risk_data);
  }
}

async function fetchUniqueIdentities() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/unique-identities/${currentJobId}`);
    const data = await res.json();
    if (res.ok) {
      lastIdentities = data.identities || [];
      renderUniquePersonsGallery(lastIdentities);
      renderTrackAuditTable(lastIdentities);
    }
  } catch (err) {
    console.error("Failed to fetch unique identities:", err);
  }
}

function renderUniquePersonsGallery(identities) {
  const gallery = document.getElementById("uniquePersonsGallery");
  const badge = document.getElementById("uniquePersonsCountBadge");
  if (!gallery) return;

  const humans = identities.filter(i => i.class === "HUMAN");
  if (badge) badge.innerText = `${humans.length} UNIQUE PEOPLE`;

  if (identities.length === 0) {
    gallery.innerHTML = `<div style="color: var(--text-muted); font-family: var(--font-mono); font-size: 0.72rem; padding: 0.5rem;">No unique identities validated yet...</div>`;
    return;
  }

  gallery.innerHTML = identities.map(c => `
    <div style="min-width: 140px; max-width: 160px; background: var(--bg-card); border: 1px solid var(--border-mid); border-radius: 4px; padding: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem;">
      <div style="width: 100%; height: 90px; background: #000; border-radius: 2px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.08);">
        ${c.thumbnail_url ? `<img src="${c.thumbnail_url}" alt="${c.display_id}" style="width: 100%; height: 100%; object-fit: cover;">` : `<div style="color: var(--text-muted); font-size: 0.65rem;">CROP SAVED</div>`}
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; color: ${c.class === 'HUMAN' ? '#EF4444' : c.class === 'VEHICLE' ? '#3B82F6' : '#10B981'}; display: flex; justify-content: space-between;">
        <span>${c.display_id}</span>
        <span style="color: var(--text-muted); font-size: 0.65rem;">${c.mean_confidence}%</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted);">
        Dwell: <span style="color: #fff;">${c.dwell_time_sec}s</span> | Hits: <span style="color: var(--mod-ui);">${c.hits}</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted);">
        Status: <span style="color: #10B981;">VALIDATED</span>
      </div>
    </div>
  `).join("");
}

function renderTrackAuditTable(records) {
  const tbody = document.getElementById("trackAuditTableBody");
  const badge = document.getElementById("trackAuditCountBadge");
  if (!tbody) return;

  if (badge) badge.innerText = `${records.length} VALIDATED TRACKS`;

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); font-family: var(--font-mono); font-size: 0.7rem; padding: 0.75rem;">No track audit records available</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map(r => `
    <tr>
      <td style="font-family: var(--font-mono); font-weight: 700; color: #fff;">${r.display_id}</td>
      <td><span class="detection-class-tag tag-${r.class.toLowerCase()}">${r.class}</span></td>
      <td style="font-family: var(--font-mono);">${r.mean_confidence}%</td>
      <td style="font-family: var(--font-mono); color: var(--mod-ui);">${r.first_seen}s</td>
      <td style="font-family: var(--font-mono); color: var(--mod-ui);">${r.last_seen}s</td>
      <td style="font-family: var(--font-mono);">${r.dwell_time_sec}s</td>
      <td style="font-family: var(--font-mono); text-align: center;">${r.visible_frames}</td>
      <td><span class="timeline-sev-tag normal">VALIDATED</span></td>
    </tr>
  `).join("");
}

function openUniqueIdentitiesModal() {
  const modal = document.getElementById("uniqueIdentitiesModal");
  const grid = document.getElementById("modalIdentitiesGrid");
  if (!modal || !grid) return;

  if (lastIdentities.length === 0) {
    grid.innerHTML = `<div style="color: var(--text-muted); padding: 2rem; font-family: var(--font-mono);">Run video analysis first to inspect validated identities...</div>`;
  } else {
    grid.innerHTML = lastIdentities.map(c => `
      <div style="background: var(--bg-core); border: 1px solid var(--border-mid); border-radius: 6px; padding: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem;">
        <div style="width: 100%; height: 120px; background: #000; border-radius: 4px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
          ${c.thumbnail_url ? `<img src="${c.thumbnail_url}" style="width: 100%; height: 100%; object-fit: cover;">` : `<span style="font-size: 0.7rem; color: #666;">CROP AVAILABLE</span>`}
        </div>
        <div style="font-family: var(--font-mono); font-weight: 700; color: ${c.class === 'HUMAN' ? '#EF4444' : c.class === 'VEHICLE' ? '#3B82F6' : '#10B981'}; font-size: 0.85rem;">
          ${c.display_id}
        </div>
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-secondary); display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
          <span>Class: <strong style="color: #fff;">${c.class}</strong></span>
          <span>Confidence: <strong style="color: #fff;">${c.mean_confidence}%</strong></span>
          <span>Dwell Time: <strong style="color: #fff;">${c.dwell_time_sec}s</strong></span>
          <span>Observed: <strong style="color: #fff;">${c.hits} frames</strong></span>
          <span>Distance: <strong style="color: #fff;">${c.distance_px} px</strong></span>
          <span>Status: <strong style="color: #10B981;">VALIDATED</strong></span>
        </div>
      </div>
    `).join("");
  }
  modal.style.display = "flex";
}

function closeUniqueIdentitiesModal() {
  const modal = document.getElementById("uniqueIdentitiesModal");
  if (modal) modal.style.display = "none";
}

// ----------------- CANVAS HUD DRAWING -----------------
function drawBoundingBoxes(detections) {
  if (!hudCtx || !hudCanvas) return;
  const cw = hudCanvas.width;
  const ch = hudCanvas.height;
  hudCtx.clearRect(0, 0, cw, ch);

  detections.forEach(d => {
    const nb = d.norm_bbox;
    if (!nb) return;

    const x = nb[0] * cw;
    const y = nb[1] * ch;
    const w = nb[2] * cw;
    const h = nb[3] * ch;
    const color = d.color || "#EF4444";

    // Thin tactical box
    hudCtx.strokeStyle = color;
    hudCtx.lineWidth = 2;
    hudCtx.strokeRect(x, y, w, h);

    // Corner Accents
    const cLen = Math.min(10, w / 4);
    hudCtx.lineWidth = 3;
    hudCtx.beginPath();
    hudCtx.moveTo(x, y + cLen); hudCtx.lineTo(x, y); hudCtx.lineTo(x + cLen, y);
    hudCtx.moveTo(x + w - cLen, y); hudCtx.lineTo(x + w, y); hudCtx.lineTo(x + w, y + cLen);
    hudCtx.moveTo(x, y + h - cLen); hudCtx.lineTo(x, y + h); hudCtx.lineTo(x + cLen, y + h);
    hudCtx.moveTo(x + w - cLen, y + h); hudCtx.lineTo(x + w, y + h); hudCtx.lineTo(x + w, y + h - cLen);
    hudCtx.stroke();

    // Clean Tactical Label Tag
    const tagText = `${d.display_id || d.class} | ${d.confidence}%`;
    hudCtx.font = "bold 10px 'IBM Plex Mono', monospace";
    const textW = hudCtx.measureText(tagText).width;

    hudCtx.fillStyle = "rgba(10, 14, 23, 0.85)";
    hudCtx.fillRect(x, Math.max(0, y - 16), textW + 8, 16);
    hudCtx.strokeStyle = color;
    hudCtx.lineWidth = 1;
    hudCtx.strokeRect(x, Math.max(0, y - 16), textW + 8, 16);

    hudCtx.fillStyle = color;
    hudCtx.fillText(tagText, x + 4, Math.max(12, y - 4));
  });
}

function renderLiveDetectionsList(detections) {
  const list = document.getElementById("liveDetectionsList");
  const badge = document.getElementById("liveDetectionsCountBadge");
  if (!list) return;

  if (badge) badge.innerText = `${detections.length} DETECTIONS`;

  if (detections.length === 0) {
    list.innerHTML = `<div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); padding: 0.5rem;">Perimeter Clear</div>`;
    return;
  }

  list.innerHTML = detections.map(d => `
    <div class="detection-item" style="border-left-color: ${d.color || '#EF4444'};">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="detection-class-tag tag-${d.class.toLowerCase()}">${d.class}</span>
        <span style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted);">${d.confidence}%</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.7rem; font-weight: 700; color: #fff;">${d.display_id || 'TRACK'}</div>
    </div>
  `).join("");
}

function updateRiskEngine(risk) {
  const scoreEl = document.getElementById("riskScoreValue");
  const levelEl = document.getElementById("riskLevelBadge");
  const factorsEl = document.getElementById("riskFactorsList");

  if (scoreEl) scoreEl.innerText = risk.score || 0;
  if (levelEl) {
    levelEl.innerText = `${risk.level || 'LOW'} RISK`;
    levelEl.className = `risk-level-badge ${(risk.level || 'low').toLowerCase()}`;
  }

  if (factorsEl && risk.factors) {
    factorsEl.innerHTML = risk.factors.map(f => `<div style="font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-secondary);">• ${f}</div>`).join("");
  }
}

// ----------------- DEGRADED BENCHMARK FETCH -----------------
async function fetchDegradedBenchmark() {
  try {
    const res = await fetch("/api/degraded-benchmark");
    const data = await res.json();
    if (res.ok) {
      renderDegradedBenchmarkTable(data.quality_matrix);
    }
  } catch (err) {
    console.error("Failed to fetch degraded benchmark:", err);
  }
}

function renderDegradedBenchmarkTable(matrix) {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody || !matrix) return;

  tbody.innerHTML = matrix.map(m => `
    <tr>
      <td style="font-family: var(--font-mono); font-weight: 700; color: #fff;">${m.quality_tier}</td>
      <td style="font-size: 0.72rem; color: var(--text-secondary);">${m.description}</td>
      <td style="font-family: var(--font-mono); color: #EF4444; font-weight: 600;">${m.person_precision}</td>
      <td style="font-family: var(--font-mono); color: #EF4444;">${m.person_recall}</td>
      <td style="font-family: var(--font-mono); color: #3B82F6;">${m.vehicle_precision}</td>
      <td style="font-family: var(--font-mono); color: #10B981;">${m.animal_precision}</td>
      <td style="font-family: var(--font-mono); color: var(--saffron-gov); font-weight: 700;">${m.map50}</td>
      <td style="font-family: var(--font-mono); font-weight: 700; color: ${m.unique_count_error_mae === 0 ? '#10B981' : '#F59E0B'};">${m.unique_count_error_mae}</td>
      <td><span class="timeline-sev-tag normal">${m.status}</span></td>
    </tr>
  `).join("");
}

function onVideoTimeUpdate() {
  if (!mainVideo) return;
  const currentSec = mainVideo.currentTime;
  const mins = Math.floor(currentSec / 60);
  const secs = Math.floor(currentSec % 60);
  const millis = Math.floor((currentSec % 1) * 100);
  const timecodeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(millis).padStart(2, '0')}`;

  const timecodeEl = document.getElementById("hudTimecode");
  if (timecodeEl) timecodeEl.innerText = timecodeStr;
}

// ----------------- SETTINGS & PRESETS -----------------
async function applyPreset(presetName) {
  try {
    const formData = new FormData();
    formData.append("preset", presetName);
    await fetch("/api/model-config", { method: "POST", body: formData });
    console.log(`[Preset] Applied preset: ${presetName}`);
  } catch (err) {}
}

async function updateCustomThresholds() {
  const p = document.getElementById("threshPerson")?.value;
  const v = document.getElementById("threshVehicle")?.value;
  const a = document.getElementById("threshAnimal")?.value;
  const iou = document.getElementById("threshIoU")?.value;

  const formData = new FormData();
  if (p) formData.append("person_thresh", p);
  if (v) formData.append("vehicle_thresh", v);
  if (a) formData.append("animal_thresh", a);
  if (iou) formData.append("iou", iou);

  try {
    await fetch("/api/model-config", { method: "POST", body: formData });
  } catch (err) {}
}

function openSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.style.display = "flex";
}

function closeSettingsModal() {
  const modal = document.getElementById("settingsModal");
  if (modal) modal.style.display = "none";
}

// ----------------- EXPORT FUNCTIONALITY -----------------
function exportData(format) {
  if (!currentJobId) {
    alert("Please run an AI analysis first to generate export data.");
    return;
  }
  window.open(`/api/export/${format}/${currentJobId}`, "_blank");
}
