import os
import sys
import math
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(BASE_DIR)
uploads_dir = os.path.join(project_dir, "uploads")

COCO_HUMAN = {"person", "pedestrian", "human"}
COCO_VEHICLE = {"bicycle", "car", "motorcycle", "bus", "truck", "van"}
COCO_ANIMAL = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

def map_coco_class(name: str) -> str:
    n = name.lower().strip()
    if n in COCO_HUMAN:
        return "HUMAN"
    elif n in COCO_VEHICLE:
        return "VEHICLE"
    elif n in COCO_ANIMAL:
        return "ANIMAL"
    return "UNKNOWN"

class TrackSegment:
    def __init__(self, track_id: int, class_name: str, bbox: List[int], conf: float, frame_idx: int, timestamp: float):
        self.track_id = track_id
        self.class_name = class_name
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.first_seen = timestamp
        self.last_seen = timestamp
        
        self.confs = [conf]
        self.bboxes = [bbox]
        self.centroids = [((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0)]
        self.best_conf = conf
        self.best_bbox = bbox
        self.best_frame = frame_idx
        self.hits = 1
        self.misses = 0

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
    def mean_conf(self) -> float:
        return sum(self.confs)/len(self.confs)

    def add(self, bbox: List[int], conf: float, frame_idx: int, timestamp: float):
        self.last_frame = frame_idx
        self.last_seen = timestamp
        self.confs.append(conf)
        self.bboxes.append(bbox)
        self.centroids.append(((bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0))
        if conf > self.best_conf:
            self.best_conf = conf
            self.best_bbox = bbox
            self.best_frame = frame_idx
        self.hits += 1
        self.misses = 0

class PersonConsolidator:
    """
    Consolidates fragmented track segments of the same physical individual into a single UNIQUE_PERSON_ID.
    Uses spatial overlap, size consistency, and non-overlapping temporal constraints.
    """
    def __init__(self, spatial_dist_thresh: float = 80.0, size_sim_thresh: float = 0.50):
        self.spatial_dist_thresh = spatial_dist_thresh
        self.size_sim_thresh = size_sim_thresh

    def consolidate(self, human_tracks: List[TrackSegment]) -> List[Dict[str, Any]]:
        # Filter valid tracks (min frames >= 8)
        valid_tracks = [t for t in human_tracks if t.hits >= 8 and t.mean_conf >= 0.45]
        
        # Sort tracks chronologically by first frame
        valid_tracks.sort(key=lambda x: x.first_frame)
        
        person_clusters: List[List[TrackSegment]] = []
        
        for trk in valid_tracks:
            best_cluster_idx = None
            best_dist = float('inf')
            
            trk_c = trk.mean_centroid
            trk_w, trk_h = trk.mean_bbox_size
            
            for c_idx, cluster in enumerate(person_clusters):
                # A cluster cannot contain two tracks that overlap heavily in time (they would be 2 different simultaneous people)
                has_temporal_overlap = False
                for existing_trk in cluster:
                    # Check overlap: if both tracks share more than 5 frames
                    overlap_start = max(trk.first_frame, existing_trk.first_frame)
                    overlap_end = min(trk.last_frame, existing_trk.last_frame)
                    if overlap_end - overlap_start > 5:
                        has_temporal_overlap = True
                        break
                        
                if has_temporal_overlap:
                    continue
                    
                # Compare spatial proximity with the last track in the cluster
                last_trk = cluster[-1]
                last_c = last_trk.mean_centroid
                last_w, last_h = last_trk.mean_bbox_size
                
                dist = math.dist(trk_c, last_c)
                
                # Check bounding box size ratio
                w_ratio = min(trk_w, last_w) / max(trk_w, last_w)
                h_ratio = min(trk_h, last_h) / max(trk_h, last_h)
                
                # If within spatial threshold and similar size
                diag = math.sqrt(trk_w**2 + trk_h**2)
                max_allowed_dist = max(self.spatial_dist_thresh, diag * 1.2)
                
                if dist < max_allowed_dist and w_ratio > 0.4 and h_ratio > 0.4:
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster_idx = c_idx
                        
            if best_cluster_idx is not None:
                person_clusters[best_cluster_idx].append(trk)
            else:
                person_clusters.append([trk])
                
        # Format consolidated unique persons
        unique_persons = []
        for p_idx, cluster in enumerate(person_clusters, start=1):
            total_observations = sum(t.hits for t in cluster)
            all_confs = [c for t in cluster for c in t.confs]
            avg_conf = sum(all_confs)/len(all_confs)
            best_t = max(cluster, key=lambda t: t.best_conf)
            
            first_seen_t = min(t.first_seen for t in cluster)
            last_seen_t = max(t.last_seen for t in cluster)
            
            track_ids_str = ", ".join([f"TRACK_{t.track_id:03d}" for t in cluster])
            
            unique_persons.append({
                "person_id": p_idx,
                "display_id": f"PERSON_{p_idx:02d}",
                "track_segments": [t.track_id for t in cluster],
                "track_segments_str": track_ids_str,
                "total_observations": total_observations,
                "avg_confidence": round(avg_conf * 100, 1),
                "first_seen": first_seen_t,
                "last_seen": last_seen_t,
                "best_bbox": best_t.best_bbox,
                "best_frame": best_t.best_frame,
                "is_validated": total_observations >= 10
            })
            
        return [p for p in unique_persons if p["is_validated"]]

def test_consolidation():
    model = YOLO("yolov8n.pt")
    target_video = "WhatsApp Video 2026-08-29 at 7.44.01 PM.mp4"
    v_path = os.path.join(uploads_dir, target_video)
    
    cap = cv2.VideoCapture(v_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    
    print(f"Testing Person Consolidation on {target_video} ({total_frames} frames)...")
    
    track_segments: Dict[int, TrackSegment] = {}
    active_ids = {}
    next_id = 1
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t = frame_idx / fps
        
        results = model.predict(source=frame, conf=0.35, iou=0.45, device="cpu", verbose=False)
        frame_dets = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                cls_idx = int(b.cls[0].cpu().numpy())
                cls_name = r.names.get(cls_idx, "unknown")
                conf = float(b.conf[0].cpu().numpy())
                box = [int(x) for x in b.xyxy[0].cpu().numpy().tolist()]
                
                if map_coco_class(cls_name) == "HUMAN":
                    frame_dets.append((conf, box))
                    
        # Simple IoU & Distance matching for raw track segments
        unmatched_dets = list(range(len(frame_dets)))
        unmatched_active = list(active_ids.keys())
        
        for d_idx, (conf, box) in enumerate(frame_dets):
            det_c = ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)
            best_tid = None
            best_dist = 70.0
            
            for tid in unmatched_active:
                trk = track_segments[tid]
                last_c = trk.centroids[-1]
                dist = math.dist(det_c, last_c)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid
                    
            if best_tid is not None:
                track_segments[best_tid].add(box, conf, frame_idx, t)
                unmatched_active.remove(best_tid)
                unmatched_dets.remove(d_idx)
                
        for d_idx in unmatched_dets:
            conf, box = frame_dets[d_idx]
            tid = next_id
            next_id += 1
            track_segments[tid] = TrackSegment(tid, "HUMAN", box, conf, frame_idx, t)
            active_ids[tid] = frame_idx
            
        # Age out active tracks if missed for 20 frames
        to_del = []
        for tid in unmatched_active:
            if frame_idx - track_segments[tid].last_frame > 20:
                to_del.append(tid)
        for tid in to_del:
            del active_ids[tid]
            
    cap.release()
    
    consolidator = PersonConsolidator(spatial_dist_thresh=90.0)
    unique_people = consolidator.consolidate(list(track_segments.values()))
    
    print("\n=======================================================")
    print(f"PERSON CONSOLIDATION FINAL AUDIT FOR USER VIDEO:")
    print("=======================================================")
    print(f"Total Raw Track Segments: {len(track_segments)}")
    print(f"CONSOLIDATED UNIQUE HUMANS COUNT: {len(unique_people)}")
    print("\nDetailed Unique Person Entities:")
    for p in unique_people:
        m1, s1 = int(p['first_seen']//60), int(p['first_seen']%60)
        m2, s2 = int(p['last_seen']//60), int(p['last_seen']%60)
        print(f"  {p['display_id']}: Tracks=[{p['track_segments_str']}], Observations={p['total_observations']} frames, AvgConf={p['avg_confidence']}%, Time={m1:02d}:{s1:02d} - {m2:02d}:{s2:02d}")

if __name__ == "__main__":
    test_consolidation()
