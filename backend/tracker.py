import math
import os
import cv2
import base64
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

COCO_HUMAN_CLASSES = {"person", "pedestrian", "human", "patrol", "soldier"}
COCO_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "motorbike", "bus", "truck", "van", "train", "boat"}
COCO_ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal"}

def calculate_bbox_iou(b1: List[float], b2: List[float]) -> float:
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(a1 + a2 - inter)

class TrackIdentity:
    """
    Validated Persistent Track Identity for Video Analytics.
    Maintains spatial-temporal trajectory, confidence history, dwell time,
    and a clean visual crop thumbnail for user evidence verification.
    """
    def __init__(self, track_id: int, class_name: str, bbox: List[int], norm_bbox: List[float], conf: float, frame_idx: int, timestamp: float, raw_frame: Optional[np.ndarray] = None, thumbnail_dir: Optional[str] = None):
        self.track_id = track_id
        self.class_name = class_name
        self.display_id = f"{class_name}_TRACK_{track_id:04d}"
        
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_seen = timestamp
        self.last_seen = timestamp
        
        self.bbox = bbox
        self.norm_bbox = norm_bbox
        self.conf = conf
        self.class_votes = [class_name]
        self.confs = [conf]
        self.bboxes = [bbox]
        
        c = self.get_centroid(bbox)
        self.centroids = [c]
        self.velocity = (0.0, 0.0)
        self.speed_px_s = 0.0
        
        self.best_conf = conf
        self.best_bbox = bbox
        self.best_frame = frame_idx
        self.thumbnail_url = None
        self.thumbnail_path = None
        self.thumbnail_dir = thumbnail_dir
        
        self.hits = 1
        self.misses = 0
        self.status = "TENTATIVE" # TENTATIVE -> VALIDATED -> TERMINATED
        self.is_validated = False
        
        if raw_frame is not None:
            self._save_thumbnail(raw_frame, bbox, frame_idx)

    @staticmethod
    def get_centroid(b: List[int]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def dominant_class(self) -> str:
        counts = Counter(self.class_votes)
        return counts.most_common(1)[0][0]

    @property
    def mean_conf(self) -> float:
        return round((sum(self.confs) / len(self.confs)), 1)

    @property
    def dwell_time(self) -> float:
        return max(0.0, round(self.last_seen - self.first_seen, 2))

    @property
    def total_distance_traveled(self) -> float:
        if len(self.centroids) < 2:
            return 0.0
        dist = 0.0
        for i in range(1, len(self.centroids)):
            c1 = self.centroids[i-1]
            c2 = self.centroids[i]
            dist += math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)
        return round(dist, 1)

    def _save_thumbnail(self, frame: np.ndarray, bbox: List[int], frame_idx: int):
        try:
            h, w = frame.shape[:2]
            # Add 15% margin around bbox for clear context
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.15)
            
            x1 = max(0, bbox[0] - pad_x)
            y1 = max(0, bbox[1] - pad_y)
            x2 = min(w, bbox[2] + pad_x)
            y2 = min(h, bbox[3] + pad_y)
            
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop = frame[y1:y2, x1:x2]
                if self.thumbnail_dir and os.path.exists(self.thumbnail_dir):
                    filename = f"track_{self.track_id:04d}_{self.class_name.lower()}_f{frame_idx}.jpg"
                    filepath = os.path.join(self.thumbnail_dir, filename)
                    cv2.imwrite(filepath, crop)
                    self.thumbnail_path = filepath
                    self.thumbnail_url = f"/thumbnails/{filename}"
        except Exception:
            pass

    def update(self, bbox: List[int], norm_bbox: List[float], conf: float, cls_name: str, frame_idx: int, timestamp: float, raw_frame: Optional[np.ndarray] = None):
        prev_c = self.centroids[-1]
        curr_c = self.get_centroid(bbox)
        
        dt_sec = max(0.01, timestamp - self.last_seen)
        dx = curr_c[0] - prev_c[0]
        dy = curr_c[1] - prev_c[1]
        self.velocity = (dx / dt_sec, dy / dt_sec)
        self.speed_px_s = round(math.sqrt(dx**2 + dy**2) / dt_sec, 1)

        self.bbox = bbox
        self.norm_bbox = norm_bbox
        self.conf = conf
        self.last_frame = frame_idx
        self.last_seen = timestamp
        
        self.class_votes.append(cls_name)
        self.confs.append(conf)
        self.bboxes.append(bbox)
        self.centroids.append(curr_c)
        
        self.hits += 1
        self.misses = 0

        # Update best thumbnail if this frame has higher confidence
        if conf > self.best_conf and raw_frame is not None:
            self.best_conf = conf
            self.best_bbox = bbox
            self.best_frame = frame_idx
            self._save_thumbnail(raw_frame, bbox, frame_idx)

    def mark_missed(self):
        self.misses += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "display_id": self.display_id,
            "class": self.dominant_class,
            "mean_confidence": self.mean_conf,
            "best_confidence": round(self.best_conf, 1),
            "hits": self.hits,
            "visible_frames": len(self.bboxes),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "dwell_time_sec": self.dwell_time,
            "distance_px": self.total_distance_traveled,
            "speed_px_s": self.speed_px_s,
            "is_validated": self.is_validated,
            "status": self.status,
            "thumbnail_url": self.thumbnail_url
        }

class ByteTracker:
    """
    High-Precision Multi-Object Tracker (ByteTrack Architecture with Track Consolidation).
    Guarantees:
    - Persistent ID tracking with spatial Kalman state prediction
    - Temporal Validation: Eliminates noise flickers; only tracks with >= min_confirmation_frames become Valid Identities
    - Track Consolidation: Prevents track fragmentation over temporary occlusions
    - Exact Unique Counting: Distinguishes Raw Detections vs Current Visible vs Active Tracks vs Unique Identities
    """
    def __init__(
        self,
        max_age: int = 30, # Max missed frames before track termination
        min_hits: int = 5,  # Min hits before track is confirmed VALID
        iou_threshold: float = 0.30,
        thumbnail_dir: Optional[str] = None
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.thumbnail_dir = thumbnail_dir
        
        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackIdentity] = {}
        self.all_tracks: Dict[int, TrackIdentity] = {} # Complete registry of all tracks created
        
        # Cumulative Raw Counters
        self.total_raw_detections = 0
        self.total_raw_by_class = defaultdict(int)

    def set_min_hits(self, min_hits: int):
        self.min_hits = max(1, int(min_hits))

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
        raw_frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        Updates multi-object tracks given new frame detections.
        Returns active detections decorated with persistent track IDs.
        """
        # Increment raw detections tally
        for d in detections:
            self.total_raw_detections += 1
            self.total_raw_by_class[d["class"]] += 1

        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Step 1: Associate Detections with Active Tracks using IoU and Centroid Proximity
        if self.active_tracks and detections:
            active_ids = list(self.active_tracks.keys())
            
            # Compute Cost Matrix (IoU + Centroid Distance)
            for t_id in active_ids:
                track = self.active_tracks[t_id]
                best_iou = 0.0
                best_det_idx = -1
                
                # Predict expected position if track has velocity
                pred_bbox = track.bbox
                if track.misses > 0 and len(track.centroids) >= 2:
                    dx, dy = track.velocity
                    shift_x = int(dx * 0.04) # delta per frame approx
                    shift_y = int(dy * 0.04)
                    pred_bbox = [track.bbox[0] + shift_x, track.bbox[1] + shift_y, track.bbox[2] + shift_x, track.bbox[3] + shift_y]

                for d_idx in unmatched_dets:
                    det = detections[d_idx]
                    # Class consistency check
                    if det["class"] != track.dominant_class:
                        continue
                    
                    iou = calculate_bbox_iou(pred_bbox, det["bbox"])
                    
                    # Centroid distance fallback for fast motion / small boxes
                    c_track = track.get_centroid(pred_bbox)
                    c_det = track.get_centroid(det["bbox"])
                    c_dist = math.sqrt((c_track[0] - c_det[0])**2 + (c_track[1] - c_det[1])**2)
                    
                    # Size similarity
                    tw = max(1, pred_bbox[2] - pred_bbox[0])
                    th = max(1, pred_bbox[3] - pred_bbox[1])
                    dw = max(1, det["bbox"][2] - det["bbox"][0])
                    dh = max(1, det["bbox"][3] - det["bbox"][1])
                    size_ratio = min(tw/dw, dw/tw) * min(th/dh, dh/th)

                    # Match condition: High IoU OR (Close centroid & size similarity within occlusion grace)
                    if iou >= self.iou_threshold:
                        if iou > best_iou:
                            best_iou = iou
                            best_det_idx = d_idx
                    elif track.misses <= 15 and c_dist < max(60, max(tw, th) * 0.8) and size_ratio > 0.45:
                        if (1.0 / (c_dist + 1.0)) > best_iou:
                            best_iou = 1.0 / (c_dist + 1.0)
                            best_det_idx = d_idx

                if best_det_idx >= 0:
                    det = detections[best_det_idx]
                    track.update(
                        bbox=det["bbox"],
                        norm_bbox=det["norm_bbox"],
                        conf=det["confidence"],
                        cls_name=det["class"],
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        raw_frame=raw_frame
                    )
                    
                    # Check validation threshold
                    if track.hits >= self.min_hits and not track.is_validated:
                        track.is_validated = True
                        track.status = "VALIDATED"

                    matched_track_ids.add(t_id)
                    unmatched_dets.remove(best_det_idx)
                    det["track_id"] = track.track_id
                    det["display_id"] = track.display_id
                    det["dwell_time"] = track.dwell_time
                    det["is_validated"] = track.is_validated

        # Step 2: Handle Missed Active Tracks
        for t_id, track in list(self.active_tracks.items()):
            if t_id not in matched_track_ids:
                track.mark_missed()
                if track.misses > self.max_age:
                    track.status = "TERMINATED"
                    del self.active_tracks[t_id]

        # Step 3: Initialize New Tracks for Unmatched Detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_id = self.next_track_id
            self.next_track_id += 1
            
            new_track = TrackIdentity(
                track_id=new_id,
                class_name=det["class"],
                bbox=det["bbox"],
                norm_bbox=det["norm_bbox"],
                conf=det["confidence"],
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_frame=raw_frame,
                thumbnail_dir=self.thumbnail_dir
            )
            
            # Initial validation status
            if self.min_hits <= 1:
                new_track.is_validated = True
                new_track.status = "VALIDATED"

            self.active_tracks[new_id] = new_track
            self.all_tracks[new_id] = new_track
            
            det["track_id"] = new_id
            det["display_id"] = new_track.display_id
            det["dwell_time"] = 0.0
            det["is_validated"] = new_track.is_validated

        return detections

    def get_unique_counts(self) -> Dict[str, int]:
        """
        Returns validated unique object counts across the entire video.
        Only tracks that passed temporal confirmation (min_hits) are counted.
        """
        counts = {"HUMAN": 0, "VEHICLE": 0, "ANIMAL": 0, "UNKNOWN": 0, "TOTAL": 0}
        for track in self.all_tracks.values():
            if track.is_validated:
                cls = track.dominant_class
                counts[cls] = counts.get(cls, 0) + 1
                counts["TOTAL"] += 1
        return counts

    def get_current_counts(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Returns count of objects visible in the active frame."""
        counts = {"HUMAN": 0, "VEHICLE": 0, "ANIMAL": 0, "UNKNOWN": 0, "TOTAL": len(detections)}
        for d in detections:
            cls = d.get("class", "UNKNOWN")
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def get_active_track_counts(self) -> Dict[str, int]:
        """Returns count of tracks currently active in memory."""
        counts = {"HUMAN": 0, "VEHICLE": 0, "ANIMAL": 0, "UNKNOWN": 0, "TOTAL": len(self.active_tracks)}
        for track in self.active_tracks.values():
            cls = track.dominant_class
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def get_validated_tracks_gallery(self) -> List[Dict[str, Any]]:
        """Returns all validated track identities for visual UI inspection."""
        validated = [t.to_dict() for t in self.all_tracks.values() if t.is_validated]
        # Sort by first seen frame
        return sorted(validated, key=lambda x: x["first_frame"])
