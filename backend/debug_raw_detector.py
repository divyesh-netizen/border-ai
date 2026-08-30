import os
import sys
import cv2
from ultralytics import YOLO

def test_all_videos():
    print("==================================================")
    print("      DEBUGGING RAW YOLOv8 DETECTOR DIRECTLY     ")
    print("==================================================")
    
    model = YOLO("yolov8n.pt")
    print(f"Model loaded: yolov8n.pt with {len(model.names)} classes")
    print("Class 0:", model.names[0])
    print("Class 1:", model.names[1])
    print("Class 2:", model.names[2])
    print("Class 3:", model.names[3])
    
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
    video_files = [f for f in os.listdir(uploads_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for vf in video_files:
        vpath = os.path.join(uploads_dir, vf)
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[VIDEO] {vf} | Resolution: {w}x{h} | Total Frames: {total_frames}")
        
        # Test frames: 1, 30, 60
        for fidx in [1, 15, 30, 60]:
            if fidx >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            results = model.predict(source=frame, conf=0.25, verbose=False, imgsz=640)
            for r in results:
                boxes = r.boxes
                print(f"  Frame #{fidx}: Found {len(boxes)} raw boxes")
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    raw_cls = model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    xyxy_int = [int(v) for v in xyxy]
                    bw = xyxy_int[2] - xyxy_int[0]
                    bh = xyxy_int[3] - xyxy_int[1]
                    print(f"    [{i+1}] Class ID {cls_id} -> '{raw_cls}' | Conf: {conf*100:.1f}% | Box: {xyxy_int} (w={bw}, h={bh})")
        cap.release()

if __name__ == "__main__":
    test_all_videos()
