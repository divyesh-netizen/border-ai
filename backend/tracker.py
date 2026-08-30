import math
import os
import cv2
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional

COCO_HUMAN_CLASSES = {"person", "pedestrian", "human", "patrol", "soldier"}
COCO_VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "motorbike", "bus", "truck", "van", "train", "boat"}
COCO_ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal"}

def map_coco_class(raw_name: str) -> str:
    n = raw_name.lower().strip()
    if n in COCO_HUMAN_CLASSES:
        return "HUMAN"
    elif n in COCO_VEHICLE_CLASSES:
        return "VEHICLE"
    elif n in COCO_ANIMAL_CLASSES:
        return "ANIMAL"
    return "UNKNOWN"

class TrackSegment:
    """
    Continuous track segment produced during consecutive detections.
    """
    def __init__(self, track_id: int, class_name: str, bbox: List[int], norm_bbox: List[float], conf: float, frame_idx: int, timestamp: float):
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
        self.best_crop = None
        
        self.hits = 1
        self.misses = 0
        self.status = "ACTIVE" # ACTIVE, LOST, TERMINATED

    @staticmethod
    def get_centroid(b: List[int]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def dominant_class(self) -> str:
        counts = Counter(self.class_votes)
        return counts.most_common(1)[0][0]

    @property
    def mean_conf(self) -> float:
        return round((sum(self.confs) / len(self.confs)) * 100, 1)

    @property
    def mean_centroid(self) -> Tuple[float, float]:
        xs = [c[0] for c in self.centroids]
        ys = [c[1] for c in self.centroids]
        return (sum(xs)/len(xs), sum(ys)/len(ys))

    @property
    def mean_bbox_size(self) -> Tuple[float, float]:
        ws = [b[2]-b[0] for b in self.bboxes]
        hs = [b[3]-b[1] for b in self.bboxes]
        return (sum(ws)/len(ws), sum(hs)/len(hs))

    @property
    def dwell_time(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def update(self, bbox: List[int], norm_bbox: List[float], conf: float, cls_name: str, frame_idx: int, timestamp: float, raw_frame: Optional[Any] = None):
        prev_c = self.centroids[-1]
        curr_c = self.get_centroid(bbox)
        
        dt_sec = max(0.01, timestamp - self.last_seen)
        dx = curr_c[0] - prev_c[0]
        dy = curr_c[1] - prev_c[1]
        self.velocity = (dx / dt_sec, dy / dt_sec)
        self.speed_px_s = math.sqrt(dx**2 + dy**2) / dt_sec

        self.bbox = bbox
        self.norm_bbox = norm_bbox
        self.conf = conf
        self.last_frame = frame_idx
        self.last_seen = timestamp
        
        self.class_votes.append(cls_name)
        self.confs.append(conf)
        self.bboxes.append(bbox)
        self.centroids.append(curr_c)
        if len(self.centroids) > 60:
            self.centroids.pop(0)
            
        if conf > self.best_conf:
            self.best_conf = conf
            self.best_bbox = bbox
            self.best_frame = frame_idx
            if raw_frame is not None:
                h, w = raw_frame.shape[:2]
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])
                if x2 > x1 and y2 > y1:
                    self.best_crop = raw_frame[y1:y2, x1:x2].copy()

        self.hits += 1
        self.misses = 0
        self.status = "ACTIVE"

    def mark_missed(self):
        self.misses += 1
        if self.misses > 2:
            self.status = "LOST"


class MultiObjectTracker:
    """
    Production-grade ByteTrack Multi-Object Tracker with:
    1. 2-Stage Association (High-Confidence + Low-Confidence Recovery)
    2. 60-Frame Long-Term Track Persistence
    3. Person Identity Consolidation (groups fragmented tracks into true unique persons)
    4. Explanatory Track Audit Table & Representative Thumbnail Extraction
    """
    def __init__(self, high_conf_thresh: float = 0.50, low_conf_thresh: float = 0.20, match_thresh: float = 0.25, track_buffer: int = 60, min_track_frames: int = 8):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_track_frames = min_track_frames
        
        self.next_id = 1
        self.active_tracks: Dict[int, TrackSegment] = {}
        self.all_tracks: Dict[int, TrackSegment] = {}

    @staticmethod
    def iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return inter / float(areaA + areaB - inter)

    def _match_tracks(self, detections: List[Dict[str, Any]], track_ids: List[int], current_frame: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(track_ids)

        for d_idx in list(unmatched_dets):
            det = detections[d_idx]
            det_box = det["bbox"]
            det_c = TrackSegment.get_centroid(det_box)
            det_cls = det["class_name"]

            best_tid = None
            best_score = -1.0

            for tid in unmatched_trks:
                track = self.active_tracks[tid]
                if track.dominant_class != det_cls:
                    continue

                frames_missed = current_frame - track.last_frame
                pred_c = (
                    track.centroids[-1][0] + (track.velocity[0] * frames_missed * 0.04),
                    track.centroids[-1][1] + (track.velocity[1] * frames_missed * 0.04)
                )
                
                iou_score = self.iou(det_box, track.bbox)
                c_dist = math.dist(det_c, pred_c)
                
                diag = math.sqrt((det_box[2]-det_box[0])**2 + (det_box[3]-det_box[1])**2) or 100.0
                dist_score = max(0.0, 1.0 - (c_dist / (diag * 2.0)))

                total_score = (iou_score * 0.7) + (dist_score * 0.3)
                if (iou_score > self.match_thresh or dist_score > 0.35) and total_score > best_score:
                    best_score = total_score
                    best_tid = tid

            if best_tid is not None:
                matched.append((d_idx, best_tid))
                unmatched_dets.remove(d_idx)
                unmatched_trks.remove(best_tid)

        return matched, unmatched_dets, unmatched_trks

    def update(self, detections: List[Dict[str, Any]], current_time: float, frame_idx: int = 0, raw_frame: Optional[Any] = None) -> List[Dict[str, Any]]:
        # Filter into high and low confidence for ByteTrack two-stage association
        high_dets = [d for d in detections if d["confidence"] >= (self.high_conf_thresh * 100)]
        low_dets = [d for d in detections if (self.low_conf_thresh * 100) <= d["confidence"] < (self.high_conf_thresh * 100)]

        active_and_lost_ids = [tid for tid, t in self.active_tracks.items() if t.status in ["ACTIVE", "LOST"]]

        # Stage 1: Match high confidence detections
        matches_1, unmatched_high, remaining_tracks = self._match_tracks(high_dets, active_and_lost_ids, frame_idx)

        for d_idx, tid in matches_1:
            d = high_dets[d_idx]
            self.active_tracks[tid].update(d["bbox"], d["norm_bbox"], d["confidence"]/100.0, d["class_name"], frame_idx, current_time, raw_frame)

        # Stage 2: Match low confidence detections with remaining tracks (recovering occluded/fading tracks)
        matches_2, _, final_remaining = self._match_tracks(low_dets, remaining_tracks, frame_idx)

        for d_idx, tid in matches_2:
            d = low_dets[d_idx]
            self.active_tracks[tid].update(d["bbox"], d["norm_bbox"], d["confidence"]/100.0, d["class_name"], frame_idx, current_time, raw_frame)

        # Stage 3: Track Consolidation / Recovery for unmatched high confidence detections
        for d_idx in unmatched_high:
            d = high_dets[d_idx]
            det_box = d["bbox"]
            det_c = TrackSegment.get_centroid(det_box)
            det_cls = d["class_name"]

            reconnected_tid = None
            for past_tid, past_track in self.all_tracks.items():
                if past_tid in self.active_tracks and self.active_tracks[past_tid].status == "ACTIVE":
                    continue
                if past_track.dominant_class != det_cls:
                    continue
                
                frames_lost = frame_idx - past_track.last_frame
                if 0 < frames_lost <= self.track_buffer:
                    last_c = past_track.centroids[-1]
                    dist = math.dist(det_c, last_c)
                    diag = math.sqrt((det_box[2]-det_box[0])**2 + (det_box[3]-det_box[1])**2) or 100.0
                    if dist < diag * 2.0:
                        reconnected_tid = past_tid
                        break

            if reconnected_tid is not None:
                self.all_tracks[reconnected_tid].update(d["bbox"], d["norm_bbox"], d["confidence"]/100.0, d["class_name"], frame_idx, current_time, raw_frame)
                self.active_tracks[reconnected_tid] = self.all_tracks[reconnected_tid]
            else:
                new_id = self.next_id
                self.next_id += 1
                new_seg = TrackSegment(new_id, det_cls, det_box, d["norm_bbox"], d["confidence"]/100.0, frame_idx, current_time)
                if raw_frame is not None:
                    h, w = raw_frame.shape[:2]
                    x1, y1, x2, y2 = max(0, det_box[0]), max(0, det_box[1]), min(w, det_box[2]), min(h, det_box[3])
                    if x2 > x1 and y2 > y1:
                        new_seg.best_crop = raw_frame[y1:y2, x1:x2].copy()
                self.active_tracks[new_id] = new_seg
                self.all_tracks[new_id] = new_seg

        # Stage 4: Mark missed tracks and remove tracks exceeding buffer
        to_remove = []
        for tid in final_remaining:
            track = self.active_tracks[tid]
            track.mark_missed()
            if frame_idx - track.last_frame > self.track_buffer:
                track.status = "TERMINATED"
                to_remove.append(tid)

        for tid in to_remove:
            del self.active_tracks[tid]

        # Return tracked detections for current frame
        out = []
        for tid, track in self.active_tracks.items():
            if track.last_frame == frame_idx:
                out.append({
                    "track_id": track.track_id,
                    "display_id": track.display_id,
                    "class_name": track.dominant_class,
                    "confidence": round(track.conf * 100, 1),
                    "bbox": track.bbox,
                    "norm_bbox": track.norm_bbox,
                    "dwell_time": round(track.dwell_time, 1),
                    "history": track.centroids[-15:],
                    "speed_px_s": round(track.speed_px_s, 1),
                    "status": track.status,
                    "is_validated": track.hits >= self.min_track_frames
                })
        return out

    def get_consolidated_unique_persons(self, spatial_dist_thresh: float = 90.0) -> List[Dict[str, Any]]:
        """
        Consolidates all human track segments into distinct physical UNIQUE PERSON entities.
        Eliminates track fragmentation where the same person is assigned multiple track IDs across a video.
        """
        human_tracks = [t for t in self.all_tracks.values() if t.dominant_class == "HUMAN" and t.hits >= 4]
        human_tracks.sort(key=lambda x: x.first_frame)
        
        person_clusters: List[List[TrackSegment]] = []
        
        for trk in human_tracks:
            best_c_idx = None
            best_dist = float('inf')
            trk_c = trk.mean_centroid
            trk_w, trk_h = trk.mean_bbox_size
            
            for c_idx, cluster in enumerate(person_clusters):
                # Ensure no temporal overlap (a single physical person cannot appear in 2 places simultaneously)
                overlap = False
                for existing in cluster:
                    if max(trk.first_frame, existing.first_frame) < min(trk.last_frame, existing.last_frame) - 5:
                        overlap = True
                        break
                if overlap:
                    continue
                    
                last_trk = cluster[-1]
                last_c = last_trk.mean_centroid
                last_w, last_h = last_trk.mean_bbox_size
                
                dist = math.dist(trk_c, last_c)
                w_ratio = min(trk_w, last_w) / max(trk_w, last_w)
                h_ratio = min(trk_h, last_h) / max(trk_h, last_h)
                
                diag = math.sqrt(trk_w**2 + trk_h**2)
                max_allowed = max(spatial_dist_thresh, diag * 1.5)
                
                if dist < max_allowed and w_ratio > 0.35 and h_ratio > 0.35:
                    if dist < best_dist:
                        best_dist = dist
                        best_c_idx = c_idx
                        
            if best_c_idx is not None:
                person_clusters[best_c_idx].append(trk)
            else:
                person_clusters.append([trk])
                
        unique_persons = []
        for p_idx, cluster in enumerate(person_clusters, start=1):
            total_obs = sum(t.hits for t in cluster)
            all_confs = [c for t in cluster for c in t.confs]
            avg_conf = sum(all_confs)/len(all_confs) if all_confs else 0.50
            best_t = max(cluster, key=lambda t: t.best_conf)
            
            first_t = min(t.first_seen for t in cluster)
            last_t = max(t.last_seen for t in cluster)
            track_ids = [t.track_id for t in cluster]
            track_str = ", ".join([f"TRACK_{tid:03d}" for tid in track_ids])
            
            m1, s1 = int(first_t // 60), int(first_t % 60)
            m2, s2 = int(last_t // 60), int(last_t % 60)
            
            unique_persons.append({
                "person_id": p_idx,
                "display_id": f"PERSON_{p_idx:02d}",
                "track_segments": track_ids,
                "track_segments_str": track_str,
                "total_observations": total_obs,
                "avg_confidence": round(avg_conf * 100, 1),
                "first_seen": f"{m1:02d}:{s1:02d}",
                "last_seen": f"{m2:02d}:{s2:02d}",
                "duration": f"{int(max(0, last_t - first_t))}s",
                "best_crop": best_t.best_crop,
                "best_bbox": best_t.best_bbox,
                "is_validated": total_obs >= self.min_track_frames
            })
            
        return [p for p in unique_persons if p["is_validated"]]

    def get_validated_unique_counts(self) -> Dict[str, int]:
        unique_persons = self.get_consolidated_unique_persons()
        
        valid_vehicles = set()
        valid_animals = set()
        valid_unknown = set()
        
        for t in self.all_tracks.values():
            if t.hits >= self.min_track_frames and t.mean_conf >= (self.high_conf_thresh * 100):
                cls = t.dominant_class
                if cls == "VEHICLE":
                    valid_vehicles.add(t.track_id)
                elif cls == "ANIMAL":
                    valid_animals.add(t.track_id)
                elif cls == "UNKNOWN":
                    valid_unknown.add(t.track_id)

        return {
            "unique_humans": len(unique_persons),
            "unique_vehicles": len(valid_vehicles),
            "unique_animals": len(valid_animals),
            "unique_unknown": len(valid_unknown),
            "total_unique_objects": len(unique_persons) + len(valid_vehicles) + len(valid_animals) + len(valid_unknown)
        }

    def get_track_audit_records(self) -> List[Dict[str, Any]]:
        records = []
        for t in self.all_tracks.values():
            m1, s1 = int(t.first_seen // 60), int(t.first_seen % 60)
            m2, s2 = int(t.last_seen // 60), int(t.last_seen % 60)
            is_val = t.hits >= self.min_track_frames and t.mean_conf >= (self.high_conf_thresh * 100)
            
            records.append({
                "track_id": t.track_id,
                "display_id": t.display_id,
                "class_name": t.dominant_class,
                "confidence": f"{t.mean_conf}%",
                "first_seen": f"{m1:02d}:{s1:02d}",
                "last_seen": f"{m2:02d}:{s2:02d}",
                "duration": f"{int(t.dwell_time)}s",
                "visible_frames": t.hits,
                "status": "VALIDATED" if is_val else "REJECTED_NOISE",
                "is_validated": is_val
            })
        return records

    def get_active_tracks_count(self) -> int:
        return sum(1 for t in self.active_tracks.values() if t.status == "ACTIVE")

    def get_active_tracks_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "track_id": t.track_id,
                "display_id": t.display_id,
                "class_name": t.dominant_class,
                "dwell_time": round(t.dwell_time, 1),
                "confidence": round(t.conf * 100, 1),
                "bbox": t.bbox,
                "norm_bbox": t.norm_bbox,
                "status": t.status,
                "is_validated": t.hits >= self.min_track_frames,
                "speed_px_s": round(t.speed_px_s, 1),
                "total_visible_frames": t.hits
            }
            for t in self.active_tracks.values() if t.status == "ACTIVE"
        ]
