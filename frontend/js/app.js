/**
 * BORDER AI — Command Center Surveillance Workstation
 * Smart India Hackathon 2026
 * Multi-Object Tracking, Accurate Unique Person Counting, Dynamic Risk Scoring & Training Engine
 */

// Global State
let currentMode = "REAL"; // 'REAL' or 'DEMO'
let currentVideoFilename = "whatsapp_surveillance.mp4";
let currentJobId = null;
let pollingInterval = null;
let isAnalysisRunning = false;
let frameDataHistory = [];
let allEvents = [];
let allAlerts = [];
let activeEventFilter = "ALL";
let trainingPollInterval = null;

// DOM Elements
const mainVideo = document.getElementById("mainVideo");
const hudCanvas = document.getElementById("hudCanvas");
const hudCtx = hudCanvas ? hudCanvas.getContext("2d") : null;

// ----------------- TAB SWITCHING -----------------
function switchTab(tabId) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));

  const targetPanel = document.getElementById(`tab-${tabId}`);
  if (targetPanel) targetPanel.classList.add("active");

  const activeBtn = Array.from(document.querySelectorAll(".nav-btn")).find(b => b.getAttribute("onclick")?.includes(tabId));
  if (activeBtn) activeBtn.classList.add("active");

  if (tabId === "multicam") {
    setTimeout(drawSectorMap, 50);
  } else if (tabId === "context") {
    setTimeout(drawZoneCanvas, 50);
  } else if (tabId === "analytics") {
    setTimeout(renderAnalyticsCharts, 50);
  }
}

function setMode(mode) {
  currentMode = mode;
  document.getElementById("btnRealMode")?.classList.toggle("active", mode === "REAL");
  document.getElementById("btnDemoMode")?.classList.toggle("active", mode === "DEMO");
  
  const statusBadge = document.getElementById("hudStatusBadge");
  if (statusBadge) {
    statusBadge.innerHTML = `<span class="status-pulsing-dot"></span><span>${mode === 'REAL' ? 'REAL AI INFERENCE' : 'DEMO MODE'}</span>`;
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

  const formData = new FormData();
  formData.append("file", file);

  const statusBadge = document.getElementById("hudStatusBadge");
  if (statusBadge) statusBadge.innerText = "UPLOADING FOOTAGE...";

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
      if (statusBadge) statusBadge.innerText = "FOOTAGE LOADED";
    } else {
      alert("Upload failed: " + data.detail);
    }
  } catch (err) {
    console.error("Upload error:", err);
  }
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
  const btnDownload = document.getElementById("btnDownloadAnnotated");

  if (btn) btn.classList.add("running");
  if (btnText) btnText.innerText = "STARTING PIPELINE...";
  if (progWrapper) progWrapper.style.display = "block";
  if (btnDownload) btnDownload.style.display = "none";

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

          // Show Download Annotated Video button
          const btnDownload = document.getElementById("btnDownloadAnnotated");
          if (btnDownload && data.annotated_video_url) {
            btnDownload.href = `/api/annotated-video/${currentJobId}`;
            btnDownload.style.display = "inline-flex";
          }

          fetchHistoryData();
        }
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 300);
}

async function fetchHistoryData() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/detections-history/${currentJobId}`);
    const data = await res.json();
    if (res.ok) {
      frameDataHistory = data.history || [];
      allEvents = data.events_list || [];
      allAlerts = data.alerts_list || [];
      if (data.track_audit_records) renderTrackAuditTable(data.track_audit_records);
      if (data.unique_person_cards) renderUniquePersonsGallery(data.unique_person_cards);
      renderEventsTable();
      renderEvidenceGallery();
    }
  } catch (err) {
    console.error("History fetch error:", err);
  }
}

function updateTelemetryDisplay(data) {
  // Update Top Telemetry Banner
  const cardActTrk = document.getElementById("cardActiveTracks");
  const cardVisHum = document.getElementById("cardVisibleHumans");
  const cardUniqHum = document.getElementById("cardUniqueHumans");
  const cardUniqVeh = document.getElementById("cardUniqueVehicles");
  const cardUniqAnim = document.getElementById("cardUniqueAnimals");
  const cardTotDet = document.getElementById("cardTotalDetections");

  if (cardActTrk) cardActTrk.innerText = data.active_tracks_count || 0;
  if (cardVisHum && data.visible_counts) cardVisHum.innerText = data.visible_counts.humans || 0;
  if (cardUniqHum && data.unique_counts) cardUniqHum.innerText = data.unique_counts.unique_humans || 0;
  if (cardUniqVeh && data.unique_counts) cardUniqVeh.innerText = data.unique_counts.unique_vehicles || 0;
  if (cardUniqAnim && data.unique_counts) cardUniqAnim.innerText = data.unique_counts.unique_animals || 0;
  if (cardTotDet && data.stats) cardTotDet.innerText = data.stats.total_detections || 0;

  // Bottom HUD Bar Values
  const hudFps = document.getElementById("hudFps");
  const hudUniqH = document.getElementById("hudUniqueHumansVal");
  const hudActT = document.getElementById("hudActiveTracksVal");
  if (hudFps && data.fps) hudFps.innerText = data.fps;
  if (hudUniqH && data.unique_counts) hudUniqH.innerText = data.unique_counts.unique_humans || 0;
  if (hudActT) hudActT.innerText = data.active_tracks_count || 0;

  // Dynamic Risk Engine Card
  if (data.risk) updateRiskEngine(data.risk);

  // Live Detections List
  if (data.live_detections) renderLiveDetectionsList(data.live_detections);

  // Behaviour Stream
  if (data.behaviour_log && data.behaviour_log.length > 0) renderBehaviourStream(data.behaviour_log);

  // Alerts List
  if (data.alerts_list && data.alerts_list.length > 0) renderAlertsStream(data.alerts_list);

  // Events Timeline
  if (data.events_list && data.events_list.length > 0) renderTimelineStream(data.events_list);

  // Validated Human Track Audit Table
  if (data.track_audit_records) renderTrackAuditTable(data.track_audit_records);

  // Unique Persons Visual Verification Gallery
  if (data.unique_person_cards) renderUniquePersonsGallery(data.unique_person_cards);
}

async function updateConfidenceThreshold(confVal) {
  try {
    const formData = new FormData();
    formData.append("conf", confVal);
    await fetch("/api/model-config", {
      method: "POST",
      body: formData
    });
    console.log(`[Config] Updated detector confidence threshold to: ${confVal}`);
  } catch (err) {
    console.error("Failed to update confidence threshold:", err);
  }
}

function renderTrackAuditTable(records) {
  const tbody = document.getElementById("trackAuditTableBody");
  const badge = document.getElementById("trackAuditCountBadge");
  if (!tbody) return;

  const validCount = records.filter(r => r.is_validated).length;
  if (badge) badge.innerText = `${validCount} VALIDATED TRACKS`;

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); font-family: var(--font-mono); font-size: 0.7rem; padding: 0.75rem;">No track audit records available</td></tr>`;
    return;
  }

  tbody.innerHTML = records.map(r => {
    const isVal = r.is_validated;
    const tagClass = isVal ? 'normal' : 'attention';
    const statusText = isVal ? 'VALIDATED' : 'PROBATIONARY';
    return `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color: ${isVal ? 'var(--text-primary)' : 'var(--text-muted)'};">${r.display_id}</td>
        <td><span class="detection-class-tag tag-${r.class_name.toLowerCase()}">${r.class_name}</span></td>
        <td style="font-family: var(--font-mono);">${r.confidence}</td>
        <td style="font-family: var(--font-mono); color: var(--mod-ui);">${r.first_seen}</td>
        <td style="font-family: var(--font-mono); color: var(--mod-ui);">${r.last_seen}</td>
        <td style="font-family: var(--font-mono);">${r.duration}</td>
        <td style="font-family: var(--font-mono); text-align: center;">${r.visible_frames}</td>
        <td><span class="timeline-sev-tag ${tagClass}">${statusText}</span></td>
      </tr>
    `;
  }).join("");
}

function renderUniquePersonsGallery(cards) {
  const gallery = document.getElementById("uniquePersonsGallery");
  const badge = document.getElementById("uniquePersonsCountBadge");
  if (!gallery) return;

  if (badge) badge.innerText = `${cards.length} UNIQUE PEOPLE`;

  if (!cards || cards.length === 0) {
    gallery.innerHTML = `<div style="color: var(--text-muted); font-family: var(--font-mono); font-size: 0.72rem; padding: 0.5rem;">No unique person entities consolidated yet...</div>`;
    return;
  }

  gallery.innerHTML = cards.map(c => `
    <div style="min-width: 140px; max-width: 160px; background: var(--bg-card); border: 1px solid var(--border-mid); border-radius: 4px; padding: 0.5rem; display: flex; flex-direction: column; gap: 0.35rem;">
      <div style="width: 100%; height: 90px; background: #000; border-radius: 2px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.08);">
        <img src="${c.thumbnail_url}" alt="${c.display_id}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.style.display='none'">
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; color: #EF4444; display: flex; justify-content: space-between;">
        <span>${c.display_id}</span>
        <span style="color: var(--text-muted); font-size: 0.65rem;">${c.avg_confidence}%</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted);">
        Tracks: <span style="color: var(--mod-ui);">${c.track_segments_str}</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted);">
        Frames: <span style="color: #fff;">${c.observations}</span> (${c.duration})
      </div>
    </div>
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

  // Find corresponding frame snapshot in history
  if (frameDataHistory.length > 0) {
    const matched = frameDataHistory.find(s => Math.abs(parseTimecodeToSeconds(s.timecode) - currentSec) < 0.25) || frameDataHistory[frameDataHistory.length - 1];
    if (matched) {
      drawBoundingBoxes(matched.detections || []);
      if (matched.active_tracks) renderActiveTracksList(matched.active_tracks);
      if (matched.risk) updateRiskEngine(matched.risk);
      
      // Update counters in sync with video time
      if (matched.visible_counts) {
        const cardVisHum = document.getElementById("cardVisibleHumans");
        if (cardVisHum) cardVisHum.innerText = matched.visible_counts.humans || 0;
      }
      if (matched.unique_counts) {
        const cardUniqHum = document.getElementById("cardUniqueHumans");
        const cardUniqVeh = document.getElementById("cardUniqueVehicles");
        const cardUniqAnim = document.getElementById("cardUniqueAnimals");
        if (cardUniqHum) cardUniqHum.innerText = matched.unique_counts.unique_humans || 0;
        if (cardUniqVeh) cardUniqVeh.innerText = matched.unique_counts.unique_vehicles || 0;
        if (cardUniqAnim) cardUniqAnim.innerText = matched.unique_counts.unique_animals || 0;
      }
    }
  }
}

function parseTimecodeToSeconds(tc) {
  if (!tc) return 0;
  const parts = tc.split(":");
  if (parts.length < 2) return 0;
  return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
}

// ----------------- CANVAS HUD DRAWING -----------------
function drawBoundingBoxes(detections) {
  if (!hudCtx || !hudCanvas) return;
  hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);

  const cw = hudCanvas.width;
  const ch = hudCanvas.height;

  // Draw Restricted Zone A Polygon Overlay
  hudCtx.strokeStyle = "rgba(239, 68, 68, 0.45)";
  hudCtx.lineWidth = 1.5;
  hudCtx.setLineDash([4, 4]);
  hudCtx.strokeRect(cw * 0.10, ch * 0.20, cw * 0.50, ch * 0.70);
  hudCtx.setLineDash([]);
  hudCtx.font = "bold 9px 'IBM Plex Mono', monospace";
  hudCtx.fillStyle = "rgba(239, 68, 68, 0.7)";
  hudCtx.fillText("ZONE-A RESTRICTED BUFFER", cw * 0.11, ch * 0.24);

  detections.forEach(det => {
    let x, y, w, h;
    if (det.norm_bbox && det.norm_bbox.length === 4) {
      const [nx1, ny1, nx2, ny2] = det.norm_bbox;
      x = nx1 * cw;
      y = ny1 * ch;
      w = (nx2 - nx1) * cw;
      h = (ny2 - ny1) * ch;
    } else if (det.bbox && det.bbox.length === 4) {
      const [bx1, by1, bx2, by2] = det.bbox;
      x = (bx1 / 640) * cw;
      y = (by1 / 360) * ch;
      w = ((bx2 - bx1) / 640) * cw;
      h = ((by2 - by1) / 360) * ch;
    } else {
      return;
    }

    const cls = det.class_name || "UNKNOWN";
    const conf = det.confidence || 90;
    const trackLabel = det.display_id || `TRACK_${det.track_id || 1}`;
    const label = `${cls} ${conf}% ${trackLabel}`;

    // Color by class
    let color = "#EF4444"; // Human: Red
    if (cls === "ANIMAL") color = "#10B981"; // Animal: Green
    if (cls === "VEHICLE") color = "#3B82F6"; // Vehicle: Blue
    if (cls === "UNKNOWN") color = "#F59E0B"; // Unknown: Amber

    // Draw trajectory breadcrumbs
    if (det.history && det.history.length > 1) {
      hudCtx.strokeStyle = color;
      hudCtx.lineWidth = 1.2;
      hudCtx.beginPath();
      det.history.forEach((pt, idx) => {
        const px = (pt[0] / 640) * cw;
        const py = (pt[1] / 360) * ch;
        if (idx === 0) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      });
      hudCtx.stroke();
    }

    // 1. Draw Main Entity Bounding Box (Solid thin border)
    hudCtx.strokeStyle = color;
    hudCtx.lineWidth = 1.5;
    hudCtx.setLineDash([]);
    hudCtx.strokeRect(x, y, w, h);

    // Corner brackets accent
    const cl = Math.min(8, Math.max(3, w / 4));
    hudCtx.lineWidth = 2.0;
    hudCtx.beginPath();
    hudCtx.moveTo(x, y + cl); hudCtx.lineTo(x, y); hudCtx.lineTo(x + cl, y);
    hudCtx.moveTo(x + w - cl, y); hudCtx.lineTo(x + w, y); hudCtx.lineTo(x + w, y + cl);
    hudCtx.stroke();

    // Top Label Tag Box
    hudCtx.font = "bold 9px 'IBM Plex Mono', monospace";
    const textWidth = hudCtx.measureText(label).width;
    hudCtx.fillStyle = color;
    hudCtx.fillRect(x, Math.max(0, y - 15), textWidth + 8, 15);

    hudCtx.fillStyle = "#FFFFFF";
    hudCtx.fillText(label, x + 4, Math.max(11, y - 4));
  });
}

// ----------------- DYNAMIC RISK ENGINE UI -----------------
function updateRiskEngine(risk) {
  const scoreVal = document.getElementById("riskScoreValue");
  const levelBadge = document.getElementById("riskLevelBadge");

  if (scoreVal) scoreVal.innerText = risk.score;

  if (levelBadge) {
    levelBadge.innerText = `${risk.level} RISK`;
    levelBadge.className = `risk-level-badge ${risk.level.toLowerCase()}`;
  }

  if (risk.factors) {
    const f = risk.factors;
    setFactorBar("factObj", "barObj", f.object_factor, 25);
    setFactorBar("factZone", "barZone", f.zone_factor, 20);
    setFactorBar("factTime", "barTime", f.time_factor, 15);
    setFactorBar("factDur", "barDur", f.duration_factor, 20);
    setFactorBar("factMove", "barMove", f.movement_factor, 10);
    setFactorBar("factBeh", "barBeh", f.behaviour_factor, 10);
  }
}

function setFactorBar(textId, barId, val, maxVal) {
  const tEl = document.getElementById(textId);
  const bEl = document.getElementById(barId);
  if (tEl) tEl.innerText = `${val}/${maxVal}`;
  if (bEl) bEl.style.width = `${Math.min(100, Math.round((val / maxVal) * 100))}%`;
}

// ----------------- LISTS RENDERING -----------------
function renderLiveDetectionsList(detections) {
  const list = document.getElementById("liveDetectionsList");
  const badge = document.getElementById("liveDetectionsCountBadge");
  if (!list) return;

  if (badge) badge.innerText = `${detections.length} DETECTIONS`;

  if (detections.length === 0) {
    list.innerHTML = `<div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); padding: 0.5rem;">Perimeter Clear (No targets detected)</div>`;
    return;
  }

  list.innerHTML = detections.map(d => {
    const tagClass = `tag-${(d.class_name || 'unknown').toLowerCase()}`;
    const sizeStr = d.size_str || `${d.width_px || 42}x${d.height_px || 112}px`;
    return `
      <div class="detection-chip" style="flex-direction: column; align-items: flex-start; gap: 4px;">
        <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
          <span class="detection-class-tag ${tagClass}">${d.display_id || d.class_name}</span>
          <span class="detection-conf">${d.confidence}% CONF</span>
        </div>
        <div style="display: flex; gap: 4px; font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-muted);">
          <span style="background: var(--bg-core); padding: 1px 4px; border-radius: 2px; border: 1px solid var(--border-dim);">${sizeStr}</span>
          <span style="color: var(--mod-track); background: rgba(6, 182, 212, 0.1); padding: 1px 4px; border-radius: 2px;">TRACK ID: #${d.track_id}</span>
        </div>
      </div>
    `;
  }).join("");
}

function renderActiveTracksList(tracks) {
  const list = document.getElementById("activeTracksList");
  const badge = document.getElementById("activeTracksCountBadge");
  if (!list) return;

  if (badge) badge.innerText = `${tracks.length} ACTIVE`;

  if (tracks.length === 0) {
    list.innerHTML = `<div style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted); padding: 0.5rem;">No active tracks</div>`;
    return;
  }

  list.innerHTML = tracks.map(t => {
    const dwell = Math.round(t.dwell_time || 0);
    const pct = Math.min(100, Math.round((dwell / 12) * 100));
    return `
      <div class="track-row">
        <div class="track-row-top">
          <span class="track-id-text">${t.display_id || ('Track #' + t.track_id)}</span>
          <span class="track-zone-badge">${t.zone || 'ZONE-A'}</span>
        </div>
        <div class="track-dwell-bar">
          <span>Dwell: ${dwell}s</span>
          <div class="dwell-progress-bg"><div class="dwell-progress-fill" style="width: ${pct}%;"></div></div>
          <span>${t.confidence}%</span>
        </div>
      </div>
    `;
  }).join("");
}

function renderBehaviourStream(logs) {
  const stream = document.getElementById("behaviourStream");
  const badge = document.getElementById("behaviourCountBadge");
  if (!stream) return;

  if (badge) badge.innerText = `${logs.length} EVENTS`;

  stream.innerHTML = logs.slice(-6).reverse().map(item => `
    <div class="behaviour-card ${item.severity ? item.severity.toLowerCase() : ''}">
      <div class="behaviour-card-top">
        <span class="behaviour-name">${item.title}</span>
        <span class="behaviour-time">${item.time}</span>
      </div>
      <span class="behaviour-desc">${item.desc}</span>
    </div>
  `).join("");
}

function renderAlertsStream(alerts) {
  const stream = document.getElementById("alertCenterStream");
  const badge = document.getElementById("alertCountBadge");
  if (!stream) return;

  if (badge) badge.innerText = `${alerts.length} ALERTS`;

  stream.innerHTML = alerts.slice(-5).reverse().map(alt => `
    <div class="alert-item-card ${(alt.severity || 'normal').toLowerCase()}">
      <div class="alert-item-head">
        <span class="alert-type-title">${alt.type}</span>
        <span class="alert-timecode">${alt.timestamp}</span>
      </div>
      <span class="alert-item-msg">${alt.message}</span>
      <button class="btn-view-evidence" onclick="openEvidenceModal('${alt.id}', '${alt.type}', '${alt.timestamp}')">VIEW EVIDENCE</button>
    </div>
  `).join("");
}

function renderTimelineStream(events) {
  const stream = document.getElementById("timelineStream");
  if (!stream) return;

  stream.innerHTML = events.slice(-8).reverse().map(ev => `
    <div class="timeline-row">
      <span class="timeline-time">${ev.timestamp}</span>
      <span class="timeline-event-name">${ev.event}</span>
      <span class="timeline-sev-tag ${(ev.severity || 'normal').toLowerCase()}">${ev.severity || 'NORMAL'}</span>
    </div>
  `).join("");
}

// ----------------- TACTICAL SECTOR MAP -----------------
function drawSectorMap() {
  const canvas = document.getElementById("sectorMapCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth || 600;
  canvas.height = canvas.clientHeight || 380;

  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#050913";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#141D30";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw Sector Boundaries
  ctx.strokeStyle = "#F59E0B";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(w * 0.15, h * 0.2, w * 0.7, h * 0.6);
  ctx.setLineDash([]);

  ctx.font = "bold 11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = "#F59E0B";
  ctx.fillText("SECTOR-07 (ALPHA BUFFER)", w * 0.16, h * 0.25);

  const cameras = [
    { id: "CAM-01", x: w * 0.2, y: h * 0.5, angle: 0, range: 120 },
    { id: "CAM-02", x: w * 0.8, y: h * 0.5, angle: Math.PI, range: 120 },
    { id: "CAM-03", x: w * 0.5, y: h * 0.75, angle: -Math.PI / 2, range: 140 },
  ];

  cameras.forEach(cam => {
    ctx.fillStyle = "rgba(59, 130, 246, 0.12)";
    ctx.beginPath();
    ctx.moveTo(cam.x, cam.y);
    ctx.arc(cam.x, cam.y, cam.range, cam.angle - 0.4, cam.angle + 0.4);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = "#3B82F6";
    ctx.beginPath();
    ctx.arc(cam.x, cam.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "bold 9px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(cam.id, cam.x - 14, cam.y - 10);
  });
}

// ----------------- VIRTUAL ZONE CANVAS -----------------
function drawZoneCanvas() {
  const canvas = document.getElementById("zoneCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth || 500;
  canvas.height = canvas.clientHeight || 280;

  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = "#0A0F1D";
  ctx.fillRect(0, 0, w, h);

  // Draw Zone A (High Priority Red)
  ctx.fillStyle = "rgba(239, 68, 68, 0.18)";
  ctx.strokeStyle = "#EF4444";
  ctx.lineWidth = 2;
  ctx.strokeRect(w * 0.1, h * 0.15, w * 0.5, h * 0.7);
  ctx.fillRect(w * 0.1, h * 0.15, w * 0.5, h * 0.7);

  ctx.font = "bold 11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = "#EF4444";
  ctx.fillText("ZONE A: RESTRICTED BUFFER (HIGH PRIORITY)", w * 0.12, h * 0.25);

  // Draw Zone B (Medium Priority Amber)
  ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
  ctx.strokeStyle = "#F59E0B";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(w * 0.65, h * 0.15, w * 0.25, h * 0.7);
  ctx.fillRect(w * 0.65, h * 0.15, w * 0.25, h * 0.7);

  ctx.fillStyle = "#F59E0B";
  ctx.fillText("ZONE B (MEDIUM)", w * 0.67, h * 0.25);
}

// ----------------- ANALYTICS CHARTS -----------------
function renderAnalyticsCharts() {
  const c1 = document.getElementById("activityChart");
  if (c1) {
    const ctx = c1.getContext("2d");
    c1.width = c1.clientWidth || 600;
    c1.height = c1.clientHeight || 240;
    const w = c1.width, h = c1.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0B101E";
    ctx.fillRect(0, 0, w, h);

    const data = [12, 8, 15, 42, 98, 145, 120, 110, 165, 210, 180, 79];
    const max = Math.max(...data);
    const stepX = (w - 60) / (data.length - 1);

    ctx.strokeStyle = "#3B82F6";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    data.forEach((val, idx) => {
      const x = 30 + idx * stepX;
      const y = h - 30 - (val / max) * (h - 60);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const c2 = document.getElementById("distributionChart");
  if (c2) {
    const ctx = c2.getContext("2d");
    c2.width = c2.clientWidth || 300;
    c2.height = c2.clientHeight || 240;
    const w = c2.width, h = c2.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0B101E";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2 - 10, radius = 65;
    const slices = [
      { label: "Human", val: 642, color: "#EF4444" },
      { label: "Vehicle", val: 318, color: "#3B82F6" },
      { label: "Animal", val: 280, color: "#10B981" },
      { label: "Unknown", val: 44, color: "#F59E0B" }
    ];
    const total = slices.reduce((a, b) => a + b.val, 0);
    let startAngle = -Math.PI / 2;

    slices.forEach(s => {
      const sliceAngle = (s.val / total) * Math.PI * 2;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
      ctx.closePath();
      ctx.fill();
      startAngle += sliceAngle;
    });

    ctx.fillStyle = "#0B101E";
    ctx.beginPath();
    ctx.arc(cx, cy, 38, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = "bold 12px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#FFFFFF";
    ctx.textAlign = "center";
    ctx.fillText("1,284", cx, cy + 4);
  }
}

// ----------------- DATA & EVIDENCE HUB -----------------
function renderEventsTable() {
  const tbody = document.getElementById("eventsTableBody");
  if (!tbody) return;

  const filtered = allEvents.filter(ev => {
    if (activeEventFilter !== "ALL" && ev.severity !== activeEventFilter) return false;
    return true;
  });

  tbody.innerHTML = filtered.map(ev => `
    <tr>
      <td style="font-family: var(--font-mono);">${ev.id}</td>
      <td style="font-family: var(--font-mono); color: var(--mod-ui);">${ev.timestamp}</td>
      <td style="font-family: var(--font-mono); font-weight: 600;">${ev.track_id}</td>
      <td>${ev.class_name}</td>
      <td style="font-family: var(--font-mono);">${ev.confidence}</td>
      <td>${ev.event} — ${ev.details || ''}</td>
      <td><span class="timeline-sev-tag ${(ev.severity || 'normal').toLowerCase()}">${ev.severity || 'NORMAL'}</span></td>
      <td>
        <button class="btn-view-evidence" onclick="openEvidenceModal('EV-${ev.id}', '${ev.event}', '${ev.timestamp}')">VIEW SNAPSHOT</button>
      </td>
    </tr>
  `).join("");
}

function renderEvidenceGallery() {
  const gallery = document.getElementById("evidenceGallery");
  if (!gallery) return;

  const sampleEvidence = [
    { id: "ALT-0001", type: "RESTRICTED ZONE ENTRY", time: "00:01:14", cam: "CAM-01" },
    { id: "ALT-0002", type: "SUSTAINED LOITERING (>6s)", time: "00:02:10", cam: "CAM-01" },
    { id: "ALT-0003", type: "FAUNA CROSSING", time: "00:01:45", cam: "CAM-03 (Thermal)" }
  ];

  gallery.innerHTML = sampleEvidence.map(ev => `
    <div class="evidence-thumb-card" onclick="openEvidenceModal('${ev.id}', '${ev.type}', '${ev.time}')">
      <div class="evidence-img-box">
        <div style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-muted); text-align: center;">
          [SNAPSHOT CAPTURE]<br><span style="color: var(--saffron-gov); font-weight: 700;">${ev.id}</span>
        </div>
      </div>
      <div class="evidence-meta">
        <div style="font-weight: 700; color: #fff;">${ev.type}</div>
        <div style="color: var(--text-muted);">${ev.cam} • ${ev.time}</div>
      </div>
    </div>
  `).join("");
}

function setEventFilter(filter) {
  activeEventFilter = filter;
  renderEventsTable();
}

function filterEventsTable() {
  const q = document.getElementById("eventSearchInput")?.value.toLowerCase() || "";
  const rows = document.querySelectorAll("#eventsTableBody tr");
  rows.forEach(r => {
    const text = r.innerText.toLowerCase();
    r.style.display = text.includes(q) ? "" : "none";
  });
}

function exportData(format) {
  if (currentJobId) {
    window.location.href = `/api/report/${currentJobId}?format=${format}`;
  } else {
    alert("Run AI Analysis first to export session report.");
  }
}

// ----------------- EVIDENCE LIGHTBOX MODAL -----------------
function openEvidenceModal(id, title, timecode) {
  const modal = document.getElementById("evidenceModal");
  const mId = document.getElementById("modalEvidenceId");
  const mTitle = document.getElementById("modalEvidenceTitle");
  const mDetails = document.getElementById("modalEvidenceDetails");
  const canvas = document.getElementById("modalEvidenceCanvas");

  if (mId) mId.innerText = id;
  if (mTitle) mTitle.innerText = title;
  if (mDetails) mDetails.innerText = `CAMERA: CAM-01 (Sector-07) | TIMECODE: ${timecode} | STATUS: EVIDENCE LOGGED`;

  if (canvas && mainVideo) {
    canvas.width = mainVideo.videoWidth || 640;
    canvas.height = mainVideo.videoHeight || 360;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(mainVideo, 0, 0, canvas.width, canvas.height);

    // Watermark
    ctx.font = "bold 14px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "rgba(6, 9, 17, 0.75)";
    ctx.fillRect(10, 10, 260, 48);
    ctx.fillStyle = "#FF9933";
    ctx.fillText("BORDER AI EVIDENCE ARCHIVE", 18, 30);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "11px 'IBM Plex Mono', monospace";
    ctx.fillText(`CAM-01 • ${timecode} • ${id}`, 18, 48);
  }

  if (modal) modal.classList.add("active");
}

function closeEvidenceModal(e) {
  const modal = document.getElementById("evidenceModal");
  if (modal) modal.classList.remove("active");
}

// ----------------- AI MODEL TRAINING ENGINE -----------------
async function triggerModelTraining() {
  const epochs = document.getElementById("inputTrainEpochs")?.value || 30;
  const batch = document.getElementById("inputTrainBatch")?.value || 16;
  const btn = document.getElementById("btnStartTraining");

  if (btn) {
    btn.disabled = true;
    btn.innerText = "TRAINING...";
  }

  try {
    const formData = new FormData();
    formData.append("epochs", epochs);
    formData.append("batch_size", batch);

    const res = await fetch("/api/train-model", {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      startTrainingPolling();
    }
  } catch (err) {
    console.error("Training error:", err);
    if (btn) {
      btn.disabled = false;
      btn.innerText = "START TRAINING";
    }
  }
}

function startTrainingPolling() {
  clearInterval(trainingPollInterval);
  trainingPollInterval = setInterval(async () => {
    try {
      const res = await fetch("/api/training-status");
      const data = await res.json();

      if (res.ok) {
        document.getElementById("trainStatusLabel").innerText = data.status;
        document.getElementById("trainEpochMsg").innerText = data.message;
        document.getElementById("trainPctText").innerText = `${data.progress_pct}%`;
        document.getElementById("trainProgressBarFill").style.width = `${data.progress_pct}%`;
        document.getElementById("trainLossVal").innerText = data.loss.toFixed(3);
        document.getElementById("trainValMapVal").innerText = `${(data.val_map50 * 100).toFixed(1)}%`;
        document.getElementById("trainPrecisionVal").innerText = `${(data.precision * 100).toFixed(1)}%`;
        document.getElementById("trainRecallVal").innerText = `${(data.recall * 100).toFixed(1)}%`;

        if (data.status === "COMPLETED" || data.status === "FAILED") {
          clearInterval(trainingPollInterval);
          const btn = document.getElementById("btnStartTraining");
          if (btn) {
            btn.disabled = false;
            btn.innerText = "START TRAINING";
          }
        }
      }
    } catch (err) {
      console.error("Training poll error:", err);
    }
  }, 600);
}

// ----------------- INITIAL DATA FETCH -----------------
async function fetchInitialData() {
  try {
    const res = await fetch("/api/dataset-status");
    const data = await res.json();
    if (res.ok && data.datasets) {
      renderDatasetCards(data.datasets);
    }

    const evalRes = await fetch("/api/model-evaluation");
    const evalData = await evalRes.json();
    if (evalRes.ok && evalData.metrics) {
      const m = evalData.metrics;
      document.getElementById("evalMap50").innerText = `${(m.mAP50 * 100).toFixed(1)}%`;
      document.getElementById("evalMap5095").innerText = `${(m.mAP50_95 * 100).toFixed(1)}%`;
      document.getElementById("evalPrecision").innerText = `${(m.precision * 100).toFixed(1)}%`;
      document.getElementById("evalRecall").innerText = `${(m.recall * 100).toFixed(1)}%`;
      document.getElementById("evalF1").innerText = `${(m.f1_score * 100).toFixed(1)}%`;
      document.getElementById("evalIDSwitches").innerText = m.id_switches;
    }
  } catch (err) {
    console.error("Initial fetch error:", err);
  }
}

function renderDatasetCards(datasets) {
  const grid = document.getElementById("datasetCardsGrid");
  if (!grid) return;

  grid.innerHTML = datasets.map(d => `
    <div class="dataset-card">
      <span class="section-badge" style="margin-bottom: 0.5rem; display: inline-block;">${d.modality}</span>
      <h3 style="font-size: 0.95rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.25rem;">${d.name}</h3>
      <p style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.5rem;">${d.purpose}</p>
      <div style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted);">
        CLASSES: <span style="color: var(--text-primary);">${(d.classes || []).join(", ")}</span><br>
        SAMPLES: <span style="color: var(--text-primary);">${d.samples}</span><br>
        STATUS: <span style="color: var(--sev-normal); font-weight: 600;">${d.status}</span>
      </div>
    </div>
  `).join("");
}

// ----------------- INIT -----------------
window.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  fetchInitialData();
  renderEventsTable();
  renderEvidenceGallery();
});
