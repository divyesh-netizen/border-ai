import os
import time
import uuid
import cv2
import threading
import numpy as np
from typing import Dict, Any, Optional, List

from inference import ModelAdapter
from tracker import MultiObjectTracker
from alert_engine import AlertEngine

class VideoProcessingJob:
    def __init__(self, job_id: str, video_path: str, is_thermal: bool = False, mode_override: Optional[str] = None):
        self.job_id = job_id
        self.video_path = video_path
        self.is_thermal = is_thermal
        self.mode_override = mode_override
        self.status = "INITIALIZING"
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
        
        # Current Frame Visible Counts
        self.visible_humans = 0
        self.visible_vehicles = 0
        self.visible_animals = 0
        self.visible_unknown = 0
        
        # Validated Unique Counts
        self.unique_humans = 0
        self.unique_vehicles = 0
        self.unique_animals = 0
        self.unique_unknown = 0
        self.total_unique_objects = 0
        
        self.alerts_count = 0
        self.events_count = 0
        
        self.risk_data = {
            "score": 12,
            "level": "LOW",
            "factors": {
                "object_factor": 5,
                "zone_factor": 5,
                "time_factor": 2,
                "duration_factor": 0,
                "movement_factor": 0,
                "behaviour_factor": 0
            }
        }
        self.alerts_list: List[Dict[str, Any]] = []
        self.events_list: List[Dict[str, Any]] = []
        self.behaviour_log: List[Dict[str, Any]] = []
        self.track_audit_records: List[Dict[str, Any]] = []
        self.unique_person_cards: List[Dict[str, Any]] = []
        
        self.annotated_video_path: Optional[str] = None
        self.annotated_video_url: Optional[str] = None
        self.evidence_snapshots: List[Dict[str, Any]] = []
        self.frame_data_history = []
        self.error_message = None


class VideoProcessor:
    def __init__(self, model_adapter: ModelAdapter, output_dir: str):
        self.model_adapter = model_adapter
        self.output_dir = output_dir
        self.evidence_dir = os.path.join(output_dir, "evidence")
        self.annotated_dir = os.path.join(output_dir, "annotated")
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)
        
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

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]], active_tracks: List[Dict[str, Any]], timecode_str: str, fps: float, unique_counts: Dict[str, int]) -> np.ndarray:
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw Virtual Restricted Zone A
        cv2.rectangle(annotated, (int(w * 0.10), int(h * 0.20)), (int(w * 0.60), int(h * 0.90)), (0, 0, 200), 1)
        cv2.putText(annotated, "ZONE-A RESTRICTED BUFFER", (int(w * 0.11), int(h * 0.24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 240), 1)

        # Draw Detections & Tracks
        for d in detections:
            bbox = d.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            cls = d.get("class_name", "UNKNOWN")
            conf = d.get("confidence", 90.0)
            display_id = d.get("display_id", f"{cls}_TRACK_0001")

            if cls == "HUMAN":
                color = (68, 68, 239)   # Red
            elif cls == "VEHICLE":
                color = (246, 130, 59)  # Blue
            elif cls == "ANIMAL":
                color = (129, 185, 16)  # Green
            else:
                color = (11, 158, 245)  # Amber

            # Draw trajectory trail
            history = d.get("history", [])
            if len(history) > 1:
                for i in range(1, len(history)):
                    pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                    pt2 = (int(history[i][0]), int(history[i][1]))
                    cv2.line(annotated, pt1, pt2, color, 1)

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Label
            label = f"{cls} {conf:.0f}% {display_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Top Telemetry HUD
        cv2.rectangle(annotated, (0, 0), (w, 32), (10, 15, 25), -1)
        hud_text_l = f"BORDER AI | TC: {timecode_str} | FPS: {fps:.1f} | ACTIVE: {len(active_tracks)}"
        hud_text_r = f"UNIQUE HUMANS: {unique_counts.get('unique_humans', 0)} | VEHICLES: {unique_counts.get('unique_vehicles', 0)}"
        cv2.putText(annotated, hud_text_l, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1)
        cv2.putText(annotated, hud_text_r, (max(12, w - 380), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 50), 1)

        return annotated

    def _process_video_thread(self, job: VideoProcessingJob):
        cap = cv2.VideoCapture(job.video_path)
        if not cap.isOpened():
            job.status = "ERROR"
            job.error_message = f"Failed to open video source: {job.video_path}"
            return

        job.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        job.duration_sec = round(job.total_frames / video_fps, 2)
        job.status = "PROCESSING"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

        annotated_filename = f"{job.job_id}_annotated.mp4"
        annotated_path = os.path.join(self.annotated_dir, annotated_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(annotated_path, fourcc, video_fps, (width, height))
        job.annotated_video_path = annotated_path
        job.annotated_video_url = f"/outputs/annotated/{annotated_filename}"

        # Production ByteTrack Tracker
        tracker = MultiObjectTracker(high_conf_thresh=self.model_adapter.conf_threshold, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=60, min_track_frames=8)
        alert_engine = AlertEngine(sustained_presence_sec=6.0, multi_human_threshold=2, min_persistence_frames=3)

        job_evidence_dir = os.path.join(self.evidence_dir, job.job_id)
        job_thumbs_dir = os.path.join(job_evidence_dir, "thumbnails")
        os.makedirs(job_thumbs_dir, exist_ok=True)

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

            # 1. Raw Model Inference
            raw_detections = self.model_adapter.predict(frame, is_thermal=job.is_thermal, mode_override=job.mode_override)
            
            # 2. ByteTrack Multi-Object Tracking
            tracked_detections = tracker.update(raw_detections, elapsed_video_time, frame_idx=frame_idx, raw_frame=frame)
            active_tracks = tracker.get_active_tracks_summary()
            unique_counts = tracker.get_validated_unique_counts()

            # 3. Behavior Analysis & Risk Score
            new_alerts = alert_engine.process_frame(tracked_detections, active_tracks, timecode_str, elapsed_video_time, frame_idx=frame_idx)
            risk_summary = alert_engine.compute_risk_score(tracked_detections, active_tracks)

            # 4. Save Evidence Snapshots for Critical Alerts
            for alt in new_alerts:
                if alt.get("evidence_ready", False):
                    snap_name = f"{alt['id']}_{mins:02d}m{secs:02d}s.jpg"
                    snap_path = os.path.join(job_evidence_dir, snap_name)
                    cv2.imwrite(snap_path, frame)
                    evidence_record = {
                        "alert_id": alt["id"],
                        "timestamp": timecode_str,
                        "type": alt["type"],
                        "severity": alt["severity"],
                        "snapshot_url": f"/outputs/evidence/{job.job_id}/{snap_name}",
                        "snapshot_path": snap_path
                    }
                    job.evidence_snapshots.append(evidence_record)

            # 5. Metrics Update
            job.total_detections_count += len(tracked_detections)
            job.active_tracks_count = tracker.get_active_tracks_count()

            job.visible_humans = sum(1 for d in tracked_detections if d["class_name"] == "HUMAN")
            job.visible_vehicles = sum(1 for d in tracked_detections if d["class_name"] == "VEHICLE")
            job.visible_animals = sum(1 for d in tracked_detections if d["class_name"] == "ANIMAL")
            job.visible_unknown = sum(1 for d in tracked_detections if d["class_name"] == "UNKNOWN")

            job.unique_humans = unique_counts["unique_humans"]
            job.unique_vehicles = unique_counts["unique_vehicles"]
            job.unique_animals = unique_counts["unique_animals"]
            job.unique_unknown = unique_counts["unique_unknown"]
            job.total_unique_objects = unique_counts["total_unique_objects"]
            job.track_audit_records = tracker.get_track_audit_records()

            job.alerts_list = alert_engine.get_all_alerts()
            job.events_list = alert_engine.get_all_events()
            job.behaviour_log = alert_engine.get_behaviour_log()
            job.alerts_count = len(job.alerts_list)
            job.events_count = len(job.events_list)
            job.live_detections = tracked_detections
            job.risk_data = risk_summary

            proc_time = time.time() - start_t
            if proc_time > 0:
                job.fps = round(frame_idx / proc_time, 1)
            job.progress_percent = min(100, int((frame_idx / max(1, job.total_frames)) * 100))

            annotated_frame = self._annotate_frame(frame, tracked_detections, active_tracks, timecode_str, job.fps, unique_counts)
            out_writer.write(annotated_frame)

            if frame_idx % 2 == 0 or frame_idx == job.total_frames:
                snapshot = {
                    "frame": frame_idx,
                    "timecode": timecode_str,
                    "detections": tracked_detections,
                    "active_tracks": active_tracks,
                    "active_tracks_count": job.active_tracks_count,
                    "visible_counts": {
                        "humans": job.visible_humans,
                        "vehicles": job.visible_vehicles,
                        "animals": job.visible_animals,
                        "unknown": job.visible_unknown
                    },
                    "unique_counts": unique_counts,
                    "risk": risk_summary,
                    "stats": {
                        "total_detections": job.total_detections_count,
                        "unique_humans": job.unique_humans,
                        "unique_vehicles": job.unique_vehicles,
                        "unique_animals": job.unique_animals,
                        "unique_unknown": job.unique_unknown,
                        "total_unique": job.total_unique_objects,
                        "alerts": job.alerts_count,
                        "events": job.events_count
                    }
                }
                job.frame_data_history.append(snapshot)

            time.sleep(0.005)

        cap.release()
        out_writer.release()

        # Generate Visual Thumbnails for each Unique Person
        unique_persons = tracker.get_consolidated_unique_persons()
        cards = []
        for p in unique_persons:
            thumb_filename = f"person_{p['person_id']:02d}.jpg"
            thumb_path = os.path.join(job_thumbs_dir, thumb_filename)
            thumb_url = f"/outputs/evidence/{job.job_id}/thumbnails/{thumb_filename}"
            
            if p.get("best_crop") is not None and p["best_crop"].size > 0:
                cv2.imwrite(thumb_path, p["best_crop"])
            else:
                # Fallback blank
                blank = np.zeros((80, 80, 3), dtype=np.uint8)
                cv2.imwrite(thumb_path, blank)
                
            cards.append({
                "person_id": p["person_id"],
                "display_id": p["display_id"],
                "track_segments_str": p["track_segments_str"],
                "observations": p["total_observations"],
                "avg_confidence": p["avg_confidence"],
                "first_seen": p["first_seen"],
                "last_seen": p["last_seen"],
                "duration": p["duration"],
                "thumbnail_url": thumb_url
            })
            
        job.unique_person_cards = cards
        job.status = "COMPLETED"
        job.progress_percent = 100
        print(f"[VideoProcessor] Job {job.job_id} COMPLETED. Processed {job.current_frame} frames. Validated unique humans: {job.unique_humans}, Total unique: {job.total_unique_objects}")

    def get_job(self, job_id: str) -> Optional[VideoProcessingJob]:
        with self.lock:
            return self.jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, Any]:
        with self.lock:
            return {k: {"status": v.status, "progress": v.progress_percent, "fps": v.fps} for k, v in self.jobs.items()}
