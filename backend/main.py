import os
import sys
import shutil
import json
from typing import Optional, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

from inference import ModelAdapter
from video_processor import VideoProcessor
from alert_engine import AlertEngine
from trainer import ModelTrainer
import dataset_manager
from synthetic_samples import generate_sample_surveillance_videos
from report_generator import generate_csv_report, generate_json_report

# Directories
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize ModelAdapter, VideoProcessor & ModelTrainer
model_adapter = ModelAdapter(
    model_path=os.path.join(MODELS_DIR, "best.pt") if os.path.exists(os.path.join(MODELS_DIR, "best.pt")) else "yolov8n.pt",
    conf_threshold=0.35,
    iou_threshold=0.45
)
video_processor = VideoProcessor(model_adapter=model_adapter, output_dir=OUTPUT_DIR)
model_trainer = ModelTrainer(models_dir=MODELS_DIR)

# Generate sample surveillance test videos if needed
sample_vis, sample_therm = generate_sample_surveillance_videos(UPLOAD_DIR)

app = FastAPI(
    title="BORDER AI — Intelligent Video Analytics for Border Surveillance",
    description="Smart India Hackathon 2026: AI-Powered Multi-Object Tracking & Perimeter Intelligence Core",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- REST API Endpoints -----------------

@app.get("/api/health")
async def health_check():
    return {
        "status": "ONLINE",
        "system": "BORDER AI Surveillance Core",
        "sih_year": "2026",
        "model_status": model_adapter.get_model_info()["status"],
        "device": model_adapter.device.upper()
    }

@app.get("/api/model-status")
async def get_model_status():
    return model_adapter.get_model_info()

@app.post("/api/model-config")
async def update_model_config(
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    preset: Optional[str] = Form(None),
    weights_name: Optional[str] = Form(None)
):
    weights_path = os.path.join(MODELS_DIR, weights_name) if weights_name else None
    model_adapter.update_config(conf=conf, iou=iou, weights_path=weights_path, preset=preset)
    return {
        "message": "Configuration updated successfully",
        "info": model_adapter.get_model_info()
    }

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    raw_name = os.path.basename(file.filename) if file.filename else "video.mp4"
    # Clean filename of unsafe characters
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in raw_name)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".webm", ".mkv"]:
        raise HTTPException(status_code=400, detail="Unsupported video format. Allowed: MP4, AVI, MOV, WEBM, MKV.")
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "message": "Video uploaded successfully",
        "filename": safe_name,
        "video_url": f"/uploads/{safe_name}",
        "file_size": os.path.getsize(dest_path)
    }

@app.get("/api/sample-videos")
async def get_sample_videos():
    return {
        "user_video": {
            "name": "whatsapp_surveillance.mp4",
            "url": "/uploads/whatsapp_surveillance.mp4",
            "description": "User Border Surveillance Footage (Multi-Entity Perimeter)"
        },
        "visible_cctv": {
            "name": "sample_cctv_night.mp4",
            "url": "/uploads/sample_cctv_night.mp4",
            "description": "Visible Low-Light Night Perimeter Surveillance (CCTV)"
        },
        "thermal_cctv": {
            "name": "sample_thermal_night.mp4",
            "url": "/uploads/sample_thermal_night.mp4",
            "description": "Aligned Thermal Long-Wave Infrared (LLVIP Benchmark Aligned)"
        }
    }

@app.post("/api/analyze-video")
async def analyze_video(
    video_filename: str = Form(...),
    is_thermal: bool = Form(False),
    mode: Optional[str] = Form("REAL")  # REAL or DEMO
):
    video_path = os.path.join(UPLOAD_DIR, video_filename)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file '{video_filename}' not found in uploads.")
    
    job_id = video_processor.start_processing(video_path=video_path, is_thermal=is_thermal, mode_override=mode)
    return {
        "job_id": job_id,
        "status": "PROCESSING",
        "video_filename": video_filename,
        "mode": mode,
        "message": "AI video analysis and multi-object tracking started"
    }

@app.get("/api/analysis-status/{job_id}")
async def get_analysis_status(job_id: str):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress_percent": job.progress_percent,
        "current_frame": job.current_frame,
        "total_frames": job.total_frames,
        "fps": job.fps,
        "duration_sec": job.duration_sec,
        "live_detections": job.live_detections,
        "active_tracks_count": job.active_tracks_count,
        "annotated_video_url": job.annotated_video_url,
        "evidence_snapshots": job.evidence_snapshots,
        "visible_counts": {
            "humans": job.visible_humans,
            "vehicles": job.visible_vehicles,
            "animals": job.visible_animals,
            "unknown": job.visible_unknown
        },
        "unique_counts": {
            "unique_humans": job.unique_humans,
            "unique_vehicles": job.unique_vehicles,
            "unique_animals": job.unique_animals,
            "unique_unknown": job.unique_unknown,
            "total_unique": job.total_unique_objects
        },
        "risk": job.risk_data,
        "alerts_list": job.alerts_list,
        "events_list": job.events_list,
        "behaviour_log": job.behaviour_log,
        "stats": {
            "total_detections": job.total_detections_count,
            "active_tracks": job.active_tracks_count,
            "unique_humans": job.unique_humans,
            "unique_vehicles": job.unique_vehicles,
            "unique_animals": job.unique_animals,
            "unique_unknown": job.unique_unknown,
            "humans": job.unique_humans,
            "vehicles": job.unique_vehicles,
            "animals": job.unique_animals,
            "unknown": job.unique_unknown,
            "total_unique": job.total_unique_objects,
            "alerts": job.alerts_count,
            "events": job.events_count
        },
        "track_audit_records": job.track_audit_records,
        "unique_person_cards": job.unique_person_cards,
        "error": job.error_message
    }

@app.get("/api/detections-history/{job_id}")
async def get_detections_history(job_id: str):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {
        "job_id": job_id,
        "history": job.frame_data_history,
        "annotated_video_url": job.annotated_video_url,
        "evidence_snapshots": job.evidence_snapshots,
        "track_audit_records": job.track_audit_records,
        "unique_person_cards": job.unique_person_cards,
        "alerts_list": job.alerts_list,
        "events_list": job.events_list,
        "behaviour_log": job.behaviour_log,
        "risk": job.risk_data
    }

@app.get("/api/annotated-video/{job_id}")
async def get_annotated_video(job_id: str):
    job = video_processor.get_job(job_id)
    if not job or not job.annotated_video_path or not os.path.exists(job.annotated_video_path):
        raise HTTPException(status_code=404, detail="Annotated video not found or analysis pending.")
    return FileResponse(job.annotated_video_path, media_type="video/mp4", filename=f"annotated_{job_id}.mp4")

# Training APIs
@app.post("/api/train-model")
async def train_model(
    epochs: int = Form(30),
    batch_size: int = Form(16),
    learning_rate: float = Form(0.001)
):
    started = model_trainer.start_training(epochs=epochs, batch_size=batch_size, lr=learning_rate)
    if not started:
        raise HTTPException(status_code=400, detail="A training run is already in progress.")
    return {
        "message": "Model training initiated successfully.",
        "status": "TRAINING",
        "epochs": epochs
    }

@app.get("/api/training-status")
async def get_training_status():
    return model_trainer.get_status()

# Surveillance Operations Endpoints
@app.get("/api/cameras")
async def get_cameras():
    return {
        "cameras": [
            {
                "id": "CAM-01",
                "name": "Perimeter North Gate",
                "sector": "Sector-07 (Alpha)",
                "status": "ONLINE",
                "modality": "Visible CCTV (HD)",
                "fps": 25.0,
                "resolution": "1920x1080",
                "health": "Optimal",
                "blur": False,
                "obstruction": False,
                "detections": 3,
                "active_tracks": 2,
                "last_event": "Personnel Transit (00:01:14)",
                "risk_score": 28,
                "sample_url": "/uploads/whatsapp_surveillance.mp4"
            },
            {
                "id": "CAM-02",
                "name": "Perimeter Fence West",
                "sector": "Sector-07 (Bravo)",
                "status": "ONLINE",
                "modality": "Visible Low-Light CCTV",
                "fps": 24.5,
                "resolution": "1920x1080",
                "health": "Optimal",
                "blur": False,
                "obstruction": False,
                "detections": 5,
                "active_tracks": 3,
                "last_event": "Sustained Presence (00:03:12)",
                "risk_score": 67,
                "sample_url": "/uploads/sample_cctv_night.mp4"
            },
            {
                "id": "CAM-03",
                "name": "East Ridge Ridge-line",
                "sector": "Sector-08 (Charlie)",
                "status": "ONLINE",
                "modality": "Thermal Long-Wave IR (LWIR)",
                "fps": 30.0,
                "resolution": "1280x720",
                "health": "Optimal (Thermal Signature)",
                "blur": False,
                "obstruction": False,
                "detections": 4,
                "active_tracks": 2,
                "last_event": "Perimeter Fauna Crossing (00:01:45)",
                "risk_score": 35,
                "sample_url": "/uploads/sample_thermal_night.mp4"
            },
            {
                "id": "CAM-04",
                "name": "Riverine Marshland Outpost",
                "sector": "Sector-09 (Delta)",
                "status": "ONLINE",
                "modality": "Dual Visible/Thermal PTZ",
                "fps": 25.0,
                "resolution": "1920x1080",
                "health": "Optimal",
                "blur": False,
                "obstruction": False,
                "detections": 0,
                "active_tracks": 0,
                "last_event": "Clear (00:05:00)",
                "risk_score": 10,
                "sample_url": "/uploads/sample_cctv_night.mp4"
            }
        ]
    }

@app.get("/api/zones")
async def get_zones():
    return {
        "zones": [
            {
                "id": "ZONE-A",
                "name": "Restricted Inner Buffer",
                "sector": "Sector-07",
                "priority": "HIGH",
                "color": "#EF4444",
                "active_rules": ["Intrusion Detection", "Loitering > 6s", "Multi-person Detection"],
                "status": "ACTIVE_MONITORING"
            },
            {
                "id": "ZONE-B",
                "name": "Outer Transit Perimeter",
                "sector": "Sector-07",
                "priority": "MEDIUM",
                "color": "#F59E0B",
                "active_rules": ["Direction Verification", "Vehicle Transit Log"],
                "status": "ACTIVE_MONITORING"
            },
            {
                "id": "ZONE-C",
                "name": "General Approach Corridor",
                "sector": "Sector-07",
                "priority": "LOW",
                "color": "#3B82F6",
                "active_rules": ["Fauna Counting", "Standard Object Classification"],
                "status": "ACTIVE_MONITORING"
            }
        ]
    }

@app.get("/api/analytics-summary")
async def get_analytics_summary():
    return {
        "total_monitoring_hours": 142.5,
        "total_detections_logged": 1284,
        "unique_humans_tracked": 142,
        "unique_vehicles_tracked": 78,
        "unique_animals_tracked": 64,
        "active_tracks": 4,
        "avg_dwell_time_sec": 14.8,
        "alerts_generated": 19,
        "class_breakdown": {
            "Human": 642,
            "Vehicle": 318,
            "Animal": 280,
            "Unknown": 44
        },
        "severity_breakdown": {
            "Normal": 1180,
            "Monitor": 65,
            "Attention": 32,
            "Alert": 7
        },
        "hourly_activity": [
            {"hour": "00:00", "count": 12}, {"hour": "02:00", "count": 8},
            {"hour": "04:00", "count": 15}, {"hour": "06:00", "count": 42},
            {"hour": "08:00", "count": 98}, {"hour": "10:00", "count": 145},
            {"hour": "12:00", "count": 120}, {"hour": "14:00", "count": 110},
            {"hour": "16:00", "count": 165}, {"hour": "18:00", "count": 210},
            {"hour": "20:00", "count": 180}, {"hour": "22:00", "count": 79}
        ]
    }

@app.get("/api/report/{job_id}")
async def get_report(job_id: str, format: str = "json"):
    job = video_processor.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    summary = {
        "duration_sec": job.duration_sec,
        "frames_processed": job.current_frame,
        "total_detections": job.total_detections_count,
        "unique_humans": job.unique_humans,
        "unique_vehicles": job.unique_vehicles,
        "unique_animals": job.unique_animals,
        "unique_unknown": job.unique_unknown,
        "total_unique_objects": job.total_unique_objects,
        "active_tracks_count": job.active_tracks_count,
        "alerts_count": job.alerts_count,
        "events_count": job.events_count,
        "processing_fps": job.fps,
        "risk_summary": job.risk_data
    }

    events = job.events_list if job.events_list else []
    alerts = job.alerts_list if job.alerts_list else []
    
    if format.lower() == "csv":
        csv_data = generate_csv_report(job_id, summary, events)
        return Response(content=csv_data, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=border_ai_report_{job_id}.csv"})
    else:
        json_report = generate_json_report(job_id, summary, events, alerts)
        return JSONResponse(content=json_report)

# Dataset & Benchmark APIs
@app.get("/api/dataset-status")
async def get_dataset_status():
    return {
        "datasets": dataset_manager.get_all_dataset_inspections(),
        "taxonomy": dataset_manager.get_taxonomy_rules(),
        "total_datasets": len(dataset_manager.get_all_dataset_inspections())
    }

@app.get("/api/data-qc")
async def get_data_qc():
    return dataset_manager.get_qc_report()

@app.get("/api/model-evaluation")
async def get_model_evaluation():
    return dataset_manager.get_model_evaluation()

# Static Mounts
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
