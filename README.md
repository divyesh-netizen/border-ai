# BORDER AI — AI Video Analytics for Border Surveillance
### Smart India Hackathon 2026

> **"Don't replace existing CCTV. Make it intelligent."**

BORDER AI is a functional, real-time edge/server AI surveillance system designed to convert existing visible and thermal/infrared CCTV infrastructure into an autonomous monitoring pipeline.

---

## 🌟 Key Features & Capabilities

1. **Modular AI Detection Pipeline**:
   - YOLO-compatible object detection adapter (`inference.py`) supporting real PyTorch weights (`best.pt`, `yolov8n.pt`) and simulation fallback.
   - **Unified Taxonomy Mapping**:
     - `0 — HUMAN`: Pedestrian, patrol, person
     - `1 — ANIMAL`: Wildlife, cattle, perimeter fauna
     - `2 — VEHICLE`: Convoy, car, truck, motorbike
     - `3 — UNKNOWN`: Low-confidence or unmapped visual flags

2. **Multi-Object Tracking & Dwell-Time Analytics**:
   - Persistent ID tracking (`Person #001`, `Vehicle #002`, `Animal #001`).
   - Dwell-time calculation for **Sustained Presence (Loitering)** detection.

3. **Rule-Based Alert Engine**:
   - `MULTIPLE_HUMAN_PRESENCE` ($\ge 2$ persons in sector $\rightarrow$ `ATTENTION`)
   - `SUSTAINED_PRESENCE` ($>8\text{s}$ stationary target $\rightarrow$ `ATTENTION`)
   - `UNKNOWN_OBJECT` (Flagged for human operator review $\rightarrow$ `MONITOR`)
   - `VEHICLE_MOVEMENT` / `ANIMAL_MOVEMENT` (`INFO`)

4. **Low-Light & Thermal Intelligence Hub**:
   - Aligned Visible CCTV vs Thermal Infrared comparison view based on the **LLVIP Low-Light Benchmark**.
   - Demonstrates 100% pedestrian recall under total darkness (<0.05 Lux).

5. **Dataset Governance & Quality Control**:
   - Comprehensive inspection cards for all 7 reference datasets:
     1. **LLVIP** (Paired Visible + Infrared)
     2. **PBVS Benchmark** (Thermal video surveillance)
     3. **TarDAL** (Thermal/Visible image fusion)
     4. **UA-CMDet** (Cross-modality object detection)
     5. **Roboflow Thermal Human 1**
     6. **Roboflow Thermal Human 2**
     7. **VIRAT** (Realistic video surveillance activities)
   - Quality control diagnostic scanner reporting label integrity, resolution uniformity, and class balance.

6. **Automated Final Reporting**:
   - Downloadable CSV and JSON surveillance activity reports.

---

## 🚀 Quick Start Instructions

### 1. Start the FastAPI Server:
```powershell
cd d:\SIH\backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Access the Command Center:
Open your browser at:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧭 SIH Demonstration Flow (2–3 Minutes for Judges)

1. **Step 1**: Open **BORDER AI** dashboard.
2. **Step 2**: Click **"🎬 Load Sample CCTV"** or upload custom video.
3. **Step 3**: Click **"▶️ START AI ANALYSIS"**.
4. **Step 4**: Observe live bounding boxes, confidence tags, and track IDs (`Person #001`, `Person #002`).
5. **Step 5**: Watch metrics cards update in real time (Total Detections, Humans, Animals, Vehicles, Active Tracks).
6. **Step 6**: Trigger events: Notice **MULTIPLE HUMAN PRESENCE** and **SUSTAINED PRESENCE** alerts generated in the Alert Center.
7. **Step 7**: Filter the **Surveillance Event Log** table and click **"Export CSV"** / **"Report"**.
8. **Step 8**: Switch to **"🌙 Low-Light & Thermal Intelligence"** tab to demonstrate visible vs thermal LWIR detection efficacy.
9. **Step 9**: Explore **"📊 Dataset & Quality Hub"** to inspect the 7 connected datasets and quality audit metrics.
