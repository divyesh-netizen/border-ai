import math
import time
from typing import List, Dict, Any, Tuple

class TrackedObject:
    def __init__(self, track_id: int, class_name: str, bbox: List[int], norm_bbox: List[float], confidence: float, timestamp: float):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.norm_bbox = norm_bbox
        self.confidence = confidence
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.hits = 1
        self.misses = 0
        self.history = [self.get_centroid(bbox)]
        self.display_id = self.format_display_id(track_id, class_name)

    @staticmethod
    def format_display_id(track_id: int, class_name: str) -> str:
        prefix = "Person" if class_name == "HUMAN" else (
            "Vehicle" if class_name == "VEHICLE" else (
                "Animal" if class_name == "ANIMAL" else "Obj"
            )
        )
        return f"{prefix} #{track_id:03d}"

    @staticmethod
    def get_centroid(bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, bbox: List[int], norm_bbox: List[float], confidence: float, timestamp: float):
        self.bbox = bbox
        self.norm_bbox = norm_bbox
        self.confidence = confidence
        self.last_seen = timestamp
        self.hits += 1
        self.misses = 0
        centroid = self.get_centroid(bbox)
        self.history.append(centroid)
        if len(self.history) > 30:
            self.history.pop(0)

    @property
    def dwell_time(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class MultiObjectTracker:
    """
    Centroid & IoU matching Multi-Object Tracker for real-time surveillance.
    """
    def __init__(self, max_distance_threshold: float = 120.0, max_misses: int = 8, iou_threshold: float = 0.25):
        self.max_distance_threshold = max_distance_threshold
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    @staticmethod
    def compute_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, detections: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
        """
        Match incoming frame detections to existing tracks and return augmented detections with track IDs.
        """
        updated_detections = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        # Step 1: Match based on IoU and Class
        matches = []
        for d_idx, det in enumerate(detections):
            best_track_id = None
            best_score = 0.0

            det_bbox = det["bbox"]
            det_class = det["class_name"]

            for t_id in unmatched_tracks:
                track = self.tracks[t_id]
                if track.class_name != det_class:
                    continue

                iou = self.compute_iou(det_bbox, track.bbox)
                centroid_dist = math.dist(TrackedObject.get_centroid(det_bbox), track.history[-1])

                # Combined score
                if iou >= self.iou_threshold or centroid_dist < self.max_distance_threshold:
                    score = iou * 100 + (100 - min(100, centroid_dist))
                    if score > best_score:
                        best_score = score
                        best_track_id = t_id

            if best_track_id is not None:
                matches.append((d_idx, best_track_id))
                if best_track_id in unmatched_tracks:
                    unmatched_tracks.remove(best_track_id)
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)

        # Update matched tracks
        for d_idx, t_id in matches:
            det = detections[d_idx]
            track = self.tracks[t_id]
            track.update(det["bbox"], det["norm_bbox"], det["confidence"], current_time)
            
            aug_det = dict(det)
            aug_det["track_id"] = track.track_id
            aug_det["display_id"] = track.display_id
            aug_det["dwell_time"] = round(track.dwell_time, 1)
            aug_det["history"] = track.history[-10:]
            updated_detections.append(aug_det)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detections[d_idx]
            new_track = TrackedObject(
                track_id=self.next_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                norm_bbox=det["norm_bbox"],
                confidence=det["confidence"],
                timestamp=current_time
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

            aug_det = dict(det)
            aug_det["track_id"] = new_track.track_id
            aug_det["display_id"] = new_track.display_id
            aug_det["dwell_time"] = 0.0
            aug_det["history"] = new_track.history
            updated_detections.append(aug_det)

        # Age unmatched tracks
        to_delete = []
        for t_id in unmatched_tracks:
            self.tracks[t_id].misses += 1
            if self.tracks[t_id].misses > self.max_misses:
                to_delete.append(t_id)

        for t_id in to_delete:
            del self.tracks[t_id]

        return updated_detections

    def get_active_tracks_count(self) -> int:
        return len(self.tracks)

    def get_active_tracks_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "track_id": t.track_id,
                "display_id": t.display_id,
                "class_name": t.class_name,
                "dwell_time": round(t.dwell_time, 1),
                "confidence": t.confidence,
                "bbox": t.bbox,
                "norm_bbox": t.norm_bbox
            }
            for t in self.tracks.values()
        ]

    def reset(self):
        self.tracks.clear()
        self.next_id = 1
