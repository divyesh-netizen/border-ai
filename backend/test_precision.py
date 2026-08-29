import sys, os, cv2
sys.path.insert(0, 'd:/SIH/backend')
from inference import ModelAdapter
from tracker import MultiObjectTracker
from synthetic_samples import generate_sample_surveillance_videos

# Generate fresh high-contrast video
vis, therm = generate_sample_surveillance_videos('d:/SIH/uploads')

adapter = ModelAdapter()
tracker = MultiObjectTracker()
cap = cv2.VideoCapture(vis)

frame_num = 0
total_human = 0
total_veh = 0
total_anim = 0

print("Running 100-frame Border Surveillance Detection Pipeline...")

while cap.isOpened() and frame_num < 100:
    ret, frame = cap.read()
    if not ret: break
    frame_num += 1
    t = frame_num / 20.0
    dets = adapter.predict(frame)
    tracked = tracker.update(dets, t)
    
    for d in tracked:
        if d['class_name'] == 'HUMAN': total_human += 1
        elif d['class_name'] == 'VEHICLE': total_veh += 1
        elif d['class_name'] == 'ANIMAL': total_anim += 1
        
    if frame_num in [20, 50, 80]:
        print(f"Frame {frame_num} [Active Tracks: {tracker.get_active_tracks_count()}]:")
        for d in tracked:
            print(f"   -> {d['display_id']} | {d['class_name']} | Conf: {d['confidence']}% | Dwell: {d.get('dwell_time', 0)}s | Box: {d['bbox']}")

cap.release()
print("\n==========================================")
print("🎯 BORDER AI DETECTION PRECISION SUMMARY:")
print(f"   Total Human Detections:   {total_human}")
print(f"   Total Vehicle Detections: {total_veh}")
print(f"   Total Animal Detections:  {total_anim}")
print(f"   Unique Active Targets:    {tracker.get_active_tracks_count()}")
print("==========================================")
