import os
import time
import uuid
import cv2
import threading
import numpy as np
from typing import Dict, Any, Optional

from inference import ModelAdapter
from tracker import MultiObjectTracker
from alert_engine import AlertEngine

class VideoProcessingJob:
    def __init__(self, job_id: str, video_path: str, is_thermal: bool = False, mode_override: Optional[str] = None):
        self.job_id = job_id
        self.video_path = video_path
        self.is_thermal = is_thermal
        self.mode_override = mode_override
        self.status = "INITIALIZING" # INITIALIZING, PROCESSING, COMPLETED, ERROR
        self.progress_percent = 0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.duration_sec = 0.0
        self.start_time = time.time()
        
        # Real-time Telemetry
        self.live_detections = []
        self.active_tracks_count = 0
        self.total_detections_count = 0
        self.human_count = 0
        self.animal_count = 0
        self.vehicle_count = 0
        self.unknown_count = 0
        self.alerts_count = 0
        self.events_count = 0
        
        self.annotated_frame_b64 = None
        self.frame_data_history = []
        self.error_message = None

class VideoProcessor:
    def __init__(self, model_adapter: ModelAdapter):
        self.model_adapter = model_adapter
        self.jobs: Dict[str, VideoProcessingJob] = {}
        self.lock = threading.Lock()

    def start_processing(self, video_path: str, is_thermal: bool = False, mode_override: Optional[str] = None) -> str:
        job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
        job = VideoProcessingJob(job_id, video_path, is_thermal=is_thermal, mode_override=mode_override)
        
        with self.lock:
            self.jobs[job_id] = job

        thread = threading.Thread(target=self._process_video_thread, args=(job,), daemon=True)
        thread.start()
        return job_id

    def _process_video_thread(self, job: VideoProcessingJob):
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "ERROR"
            job.error_message = f"Failed to open video source: {job.video_path}"
            return

        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        job.duration_sec = round(job.total_frames / video_fps, 2)
        job.status = "PROCESSING"

        tracker = MultiObjectTracker(max_distance_threshold=140.0, max_misses=10, iou_threshold=0.25)
        alert_engine = AlertEngine(sustained_presence_sec=8.0, multi_human_threshold=2)

        frame_idx = 0
        start_t = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            job.current_frame = frame_idx
            elapsed_video_time = frame_idx / video_fps
            mins = int(elapsed_video_time // 60)
            secs = int(elapsed_video_time % 60)
            millis = int((elapsed_video_time % 1) * 100)
            timecode_str = f"{mins:02d}:{secs:02d}.{millis:02d}"

            # Run ModelAdapter Object Detection
            raw_detections = self.model_adapter.predict(frame, mode_override=job.mode_override)
            
            # Run Multi-Object Tracker
            tracked_detections = tracker.update(raw_detections, elapsed_video_time)
            active_tracks = tracker.get_active_tracks_summary()

            # Run Alert & Temporal Event Engine
            new_alerts = alert_engine.process_frame(tracked_detections, active_tracks, timecode_str, elapsed_video_time)

            # Update Metrics
            job.total_detections_count += len(tracked_detections)
            for d in tracked_detections:
                c = d["class_name"]
                if c == "HUMAN":
                    job.human_count += 1
                elif c == "ANIMAL":
                    job.animal_count += 1
                elif c == "VEHICLE":
                    job.vehicle_count += 1
                else:
                    job.unknown_count += 1

            job.active_tracks_count = tracker.get_active_tracks_count()
            job.alerts_count = len(alert_engine.get_all_alerts())
            job.events_count = len(alert_engine.get_all_events())
            job.live_detections = tracked_detections

            # Calculate Live Processing FPS & Progress
            now = time.time()
            proc_time = now - start_t
            if proc_time > 0:
                job.fps = round(frame_idx / proc_time, 1)

            job.progress_percent = min(100, int((frame_idx / max(1, job.total_frames)) * 100))

            # Store timestamped snapshot (sample every 3 frames for frontend efficiency)
            if frame_idx % 2 == 0 or frame_idx == job.total_frames:
                snapshot = {
                    "frame": frame_idx,
                    "timecode": timecode_str,
                    "detections": tracked_detections,
                    "active_tracks": active_tracks,
                    "active_tracks_count": job.active_tracks_count,
                    "stats": {
                        "human": job.human_count,
                        "animal": job.animal_count,
                        "vehicle": job.vehicle_count,
                        "unknown": job.unknown_count,
                        "total": job.total_detections_count,
                        "alerts": job.alerts_count
                    }
                }
                job.frame_data_history.append(snapshot)

            # Throttle slightly to simulate real-time stream cadence if video is super fast
            time.sleep(0.015)

        cap.release()
        job.status = "COMPLETED"
        job.progress_percent = 100
        print(f"[VideoProcessor] Job {job.job_id} COMPLETED. Processed {job.current_frame} frames.")

    def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, Any]:
        with self.lock:
            return {k: {"status": v.status, "progress": v.progress_percent, "fps": v.fps} for k, v in self.jobs.items()}
