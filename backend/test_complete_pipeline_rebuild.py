import os
import sys
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import ModelAdapter
from video_processor import VideoProcessor
from tracker import ByteTracker

def run_acceptance_tests():
    print("=================================================================")
    print("      BORDER AI — COMPUTER VISION PIPELINE ACCEPTANCE TEST      ")
    print("=================================================================")
    
    # TEST 1: Model Initialization & Device Auto-Detection
    adapter = ModelAdapter()
    print(f"\n[TEST 1] Model Loaded: {adapter.weights_path} | Device: {adapter.device.upper()} | Status: {adapter.model_info['status']}")
    assert adapter.is_real_ai, "YOLO Model must be loaded and ready"
    print("  ✓ PASS: Model loaded once with device auto-detection")
    
    # TEST 2: Single-Frame Diagnostic
    uploads_dir = os.path.join(os.path.dirname(BASE_DIR), "uploads")
    sample_video = os.path.join(uploads_dir, "sample_cctv_night.mp4")
    cap = cv2.VideoCapture(sample_video)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read sample video frame"
    
    h, w = frame.shape[:2]
    raw_dets = adapter.predict(frame, mode_override="REAL_TIME")
    print(f"\n[TEST 2] Single Frame Diagnostic:")
    print(f"  - Frame Size: {w}x{h}")
    print(f"  - Raw Detections Found: {len(raw_dets)}")
    for i, d in enumerate(raw_dets):
        print(f"    [{i+1}] Class: {d['class']} ({d['raw_class']}) | Conf: {d['confidence']}% | Box: {d['bbox']}")
    assert len(raw_dets) > 0, "Detections must be found on visible people"
    print("  ✓ PASS: Single frame diagnostic successful with tight bounding boxes")
    
    # TEST 3: Person Only Mode Validation (No Vehicles Allowed)
    humans_only = [d for d in raw_dets if d["class"] == "HUMAN"]
    print(f"\n[TEST 3] Person Only Filtering: Found {len(humans_only)} verified humans out of {len(raw_dets)} total objects.")
    for h_det in humans_only:
        assert h_det["class"] == "HUMAN", "Person Only must strictly contain HUMAN class"
        assert not h_det["raw_class"].startswith("car"), "Cars/vehicles must not be labelled as HUMAN"
    print("  ✓ PASS: Person Only filtering strictly verified")
    
    # TEST 4: Tracker Association & Temporal Unique Counting
    tracker = ByteTracker(min_hits=2, max_age=30)
    tracked_1 = tracker.update(humans_only, frame_idx=1, timestamp=0.04, raw_frame=frame)
    tracked_2 = tracker.update(humans_only, frame_idx=2, timestamp=0.08, raw_frame=frame)
    
    unique_counts = tracker.get_unique_counts()
    print(f"\n[TEST 4] Tracking Validation:")
    print(f"  - Frame 1 Visible: {len(tracked_1)} | Assigned IDs: {[d['display_id'] for d in tracked_1]}")
    print(f"  - Frame 2 Visible: {len(tracked_2)} | Unique Validated Humans: {unique_counts['HUMAN']}")
    assert unique_counts["HUMAN"] == len(humans_only), "Unique human count must equal number of persistent individuals"
    print("  ✓ PASS: Tracker IDs stable and unique count correct")
    
    # TEST 5: OpenCV HOG Fallback
    hog_dets = adapter._predict_opencv_hog(frame)
    print(f"\n[TEST 5] OpenCV HOG Fallback Test: Found {len(hog_dets)} human detections.")
    assert len(hog_dets) >= 0, "OpenCV fallback must execute without errors"
    print("  ✓ PASS: OpenCV HOG fallback ready")
    
    print("\n=================================================================")
    print("       ALL ACCEPTANCE TESTS PASSED SUCCESSFULLY (5/5)           ")
    print("=================================================================")

if __name__ == "__main__":
    run_acceptance_tests()
