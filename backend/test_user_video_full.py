import cv2
import os
import sys
import numpy as np

sys.path.insert(0, 'd:/SIH/backend')
from ultralytics import YOLO
from inference import ModelAdapter, BorderPerimeterDetector
from tracker import MultiObjectTracker

video_path = r"d:\SIH\WhatsApp Video 2026-08-29 at 3.38.08 PM.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Testing on User Video: {total_frames} frames ({total_frames/fps:.1f}s)")

model = YOLO("yolov8n.pt")
tracker = MultiObjectTracker(max_distance_threshold=80.0, max_misses=15, iou_threshold=0.15)

frame_idx = 0
total_human = 0
total_veh = 0
total_anim = 0

sample_log = []

while cap.isOpened() and frame_idx < 300:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    t = frame_idx / fps
    h, w = frame.shape[:2]

    # Multi-scale sensitive inference on frame
    results = model.predict(source=frame, conf=0.14, imgsz=1024, verbose=False)[0]
    detections = []
    
    for box in results.boxes:
        cls_idx = int(box.cls[0])
        raw_name = results.names[cls_idx]
        conf = float(box.conf[0])
        coords = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = coords
        
        # Taxonomy mapping
        if raw_name in ["person", "pedestrian", "human", "man", "woman", "child"]:
            cls = "HUMAN"
        elif raw_name in ["car", "truck", "bus", "motorcycle", "bicycle", "train", "airplane", "boat", "van"]:
            cls = "VEHICLE"
        elif raw_name in ["dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "animal"]:
            cls = "ANIMAL"
        elif raw_name in ["kite", "umbrella", "sports ball"]:
            # Aerial/small unknown movement in border area
            cls = "UNKNOWN"
        else:
            continue

        detections.append({
            "class_name": cls,
            "raw_class": raw_name,
            "confidence": round(conf * 100, 1),
            "bbox": coords,
            "norm_bbox": [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)],
            "color": "#FF3366" if cls == "HUMAN" else ("#00E5FF" if cls == "VEHICLE" else "#00E676")
        })

    tracked = tracker.update(detections, t)
    
    for d in tracked:
        if d['class_name'] == 'HUMAN': total_human += 1
        elif d['class_name'] == 'VEHICLE': total_veh += 1
        elif d['class_name'] == 'ANIMAL': total_anim += 1

    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx}/{total_frames} | Tracks: {tracker.get_active_tracks_count()} | Humans: {total_human}, Veh: {total_veh}, Anim: {total_anim}")
        sample_log.append((frame_idx, [(d['display_id'], d['class_name'], f"{d['confidence']}%", d['bbox']) for d in tracked]))

cap.release()

print("\n=== Summary over 300 frames of User Video ===")
print(f"Total Human Detections: {total_human}")
print(f"Total Vehicle Detections: {total_veh}")
print(f"Total Animal Detections: {total_anim}")
print(f"Active Tracks Count: {tracker.get_active_tracks_count()}")
print("Sample frame outputs:")
for s in sample_log:
    print(f"  Frame {s[0]}: {s[1]}")
