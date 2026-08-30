import os
import time
import uuid
import cv2
import threading
import numpy as np
from typing import Dict, Any, Optional, List

try:
    from backend.inference import ModelAdapter
    from backend.tracker import ByteTracker
    from backend.alert_engine import AlertEngine
    from backend.video_quality import VideoQualityAnalyzer
except ImportError:
    from inference import ModelAdapter
    from tracker import ByteTracker
    from alert_engine import AlertEngine
    from video_quality import VideoQualityAnalyzer

class VideoProcessingJob:
    def __init__(self, job_id: str, video_path: str, is_thermal: bool = False, mode_override: Optional[str] = None, is_person_only: bool = False):
        self.job_id = job_id
        self.video_path = video_path
        self.is_thermal = is_thermal
        self.mode_override = mode_override or "REAL_TIME" # REAL_TIME (Default) or HIGH_ACCURACY
        self.is_person_only = is_person_only
        self.status = "INITIALIZING"
        self.progress_percent = 0
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.duration_sec = 0.0
        self.start_time = time.time()
        self.error_message = None
        self.latency_ms = 0.0
        self.measured_fps = 0.0
        
        # Real-time Telemetry
        self.live_detections = []
        self.active_tracks_count = 0
        self.total_detections_count = 0
        
        # 1. Current Frame Visible Objects
        self.visible_humans = 0
        self.visible_vehicles = 0
        self.visible_animals = 0
        self.visible_unknown = 0
        
        # 2. Active Tracks (alive in tracker memory)
        self.active_humans = 0
        self.active_vehicles = 0
        self.active_animals = 0
        self.active_unknown = 0
        
        # 3. Validated Unique Objects (Identities throughout video)
        self.unique_humans = 0
        self.unique_vehicles = 0
        self.unique_animals = 0
        self.unique_unknown = 0
        self.total_unique_objects = 0
        
        # Risk & Behavioral Events
        self.risk_data = {}
        self.alerts_count = 0
        self.alerts_list: List[Dict[str, Any]] = []
        self.events_list: List[Dict[str, Any]] = []
        self.events_summary = []
        self.unique_identities_gallery = []
        self.quality_report = None
        
        self.annotated_video_path: Optional[str] = None
        self.annotated_video_url: Optional[str] = None
        self.frame_data_history = []
        self.error_message = None


class VideoProcessor:
    def __init__(self, model_adapter: ModelAdapter, output_dir: str):
        self.model_adapter = model_adapter
        self.output_dir = output_dir
        self.thumbnails_dir = os.path.join(output_dir, "thumbnails")
        self.annotated_dir = os.path.join(output_dir, "annotated")
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)
        
        self.jobs: Dict[str, VideoProcessingJob] = {}
        self.lock = threading.Lock()

    def start_processing(self, video_path: str, is_thermal: bool = False, mode_override: Optional[str] = None, is_person_only: bool = False) -> str:
        job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"
        job = VideoProcessingJob(job_id, video_path, is_thermal=is_thermal, mode_override=mode_override, is_person_only=is_person_only)
        
        with self.lock:
            self.jobs[job_id] = job

        thread = threading.Thread(target=self._process_video_thread, args=(job,), daemon=True)
        thread.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def _process_video_thread(self, job: VideoProcessingJob):
        print(f"[VideoProcessor] Starting job {job.job_id} on video: {job.video_path} (Mode: {job.mode_override})")
        
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "ERROR"
            job.error_message = f"Failed to open video file: {os.path.basename(job.video_path)}"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 120:
            fps = 25.0
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = round(total_frames / fps, 2) if total_frames > 0 else 0.0

        job.total_frames = total_frames
        job.fps = fps
        job.duration_sec = duration
        job.status = "PROCESSING"

        # Determine tracker settings based on mode
        is_high_accuracy = (job.mode_override == "HIGH_ACCURACY")
        min_hits = 4 if is_high_accuracy else 2
        use_tiled = is_high_accuracy and (w >= 720 or h >= 540)
        
        tracker = ByteTracker(
            max_age=35,
            min_hits=min_hits,
            iou_threshold=0.30,
            thumbnail_dir=self.thumbnails_dir
        )
        alert_engine = AlertEngine(loiter_time_sec=8.0, crowd_threshold=4)
        
        frame_idx = 0
        processed_frames = 0
        # Adaptive step: for long videos (>300 frames), analyze every 2nd or 3rd frame
        if total_frames > 1000:
            step = 3
        elif total_frames > 400:
            step = 2
        else:
            step = 1

        t_start = time.time()
        while cap.isOpened():
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % step != 0 and frame_idx != total_frames:
                continue

            processed_frames += 1
            timestamp = round(frame_idx / fps, 2)

            # 1. Non-blocking background quality update (every 60 frames)
            if frame_idx % 60 == 1:
                try:
                    job.quality_report = VideoQualityAnalyzer.analyze_frame(frame)
                except Exception:
                    pass

            # 2. Fast Streaming Object Detection (Zero Preprocessing in Fast Mode)
            detections = self.model_adapter.predict(
                frame=frame,
                is_thermal=job.is_thermal,
                mode_override=job.mode_override,
                use_tiled=use_tiled,
                quality_info=job.quality_report
            )

            # Calculate actual measured latency and FPS
            infer_ms = round((time.time() - t_frame_start) * 1000, 1)
            elapsed = max(0.01, time.time() - t_start)
            measured_fps = round(processed_frames / elapsed, 1)
            job.latency_ms = infer_ms
            job.measured_fps = measured_fps

            # 3. Filter Person Only before tracker if requested
            if job.is_person_only:
                detections = [d for d in detections if d["class"] == "HUMAN"]

            # 4. Update Multi-Object Tracker (Kalman Filter + Temporal Validation + Thumbnails)
            tracked_dets = tracker.update(
                detections=detections,
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_frame=frame
            )

            # 4. Temporal Behavior Analysis & Explainable Risk Scoring
            alerts, risk = alert_engine.evaluate_frame(
                detections=tracked_dets,
                active_tracks=tracker.active_tracks,
                frame_idx=frame_idx,
                timestamp=timestamp,
                video_fps=fps
            )

            # 5. Extract Distinct Metrics (Raw Detections vs Current Visible vs Active Tracks vs Unique)
            curr_counts = tracker.get_current_counts(tracked_dets)
            active_counts = tracker.get_active_track_counts()
            unique_counts = tracker.get_unique_counts()

            job.current_frame = frame_idx
            job.progress_percent = int((frame_idx / max(1, total_frames)) * 100)
            job.live_detections = tracked_dets
            job.total_detections_count = tracker.total_raw_detections
            
            # Current Visible in active frame
            job.visible_humans = curr_counts["HUMAN"]
            job.visible_vehicles = curr_counts["VEHICLE"]
            job.visible_animals = curr_counts["ANIMAL"]
            job.visible_unknown = curr_counts["UNKNOWN"]

            # Active Tracks in memory
            job.active_humans = active_counts["HUMAN"]
            job.active_vehicles = active_counts["VEHICLE"]
            job.active_animals = active_counts["ANIMAL"]
            job.active_unknown = active_counts["UNKNOWN"]
            job.active_tracks_count = active_counts["TOTAL"]

            # Validated Unique Identities
            job.unique_humans = unique_counts["HUMAN"]
            job.unique_vehicles = unique_counts["VEHICLE"]
            job.unique_animals = unique_counts["ANIMAL"]
            job.unique_unknown = unique_counts["UNKNOWN"]
            job.total_unique_objects = unique_counts["TOTAL"]

            job.risk_data = risk
            job.alerts_count = len(alert_engine.alert_history)

            # Event Log
            for d in tracked_dets:
                if d.get("is_validated", False):
                    ev = {
                        "event_id": f"EVT_{len(job.events_list) + 1:04d}",
                        "timestamp": timestamp,
                        "frame": frame_idx,
                        "class": d["class"],
                        "track_id": d.get("display_id", f"TRACK_{d.get('track_id', 0)}"),
                        "confidence": d["confidence"],
                        "dwell_time": d.get("dwell_time", 0.0),
                        "status": "VALIDATED"
                    }
                    # Keep latest 100 events
                    if len(job.events_list) < 100:
                        job.events_list.append(ev)

            # Small sleep to yield CPU if processing very fast
            time.sleep(0.01)

        cap.release()

        # Final Job Consolidation
        job.unique_identities_gallery = tracker.get_validated_tracks_gallery()
        job.alerts_list = alert_engine.alert_history
        job.status = "COMPLETED"
        job.progress_percent = 100
        print(f"[VideoProcessor] Job {job.job_id} completed: Unique Humans: {job.unique_humans}, Unique Vehicles: {job.unique_vehicles}, Unique Animals: {job.unique_animals}, Total Raw Detections: {job.total_detections_count}")
