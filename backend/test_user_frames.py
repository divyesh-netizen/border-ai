import cv2
import os
import sys
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
debug_dir = r"d:\SIH\outputs\debug_frames"

for img_name in sorted(os.listdir(debug_dir)):
    if not img_name.endswith(".jpg"): continue
    img_path = os.path.join(debug_dir, img_name)
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]
    
    print(f"\n--- Testing {img_name} ({w}x{h}) ---")
    
    # 1. Standard YOLO at conf=0.25
    res = model.predict(frame, conf=0.25, verbose=False)[0]
    boxes = res.boxes
    print(f"Standard YOLO (conf=0.25): {len(boxes)} detections")
    for b in boxes:
        cls_id = int(b.cls[0])
        name = res.names[cls_id]
        conf = float(b.conf[0])
        coords = [int(v) for v in b.xyxy[0].tolist()]
        print(f"  -> {name} ({conf:.2f}) at {coords}")
        
    # 2. Sensitive Multi-scale / Low-conf YOLO (conf=0.15, imgsz=1280)
    res_sens = model.predict(frame, conf=0.15, imgsz=1280, verbose=False)[0]
    boxes_sens = res_sens.boxes
    print(f"Sensitive YOLO (conf=0.15, imgsz=1280): {len(boxes_sens)} detections")
    for b in boxes_sens:
        cls_id = int(b.cls[0])
        name = res_sens.names[cls_id]
        conf = float(b.conf[0])
        coords = [int(v) for v in b.xyxy[0].tolist()]
        print(f"  -> {name} ({conf:.2f}) at {coords}")
