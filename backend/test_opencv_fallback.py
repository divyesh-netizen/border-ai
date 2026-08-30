import os
import sys
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import ModelAdapter

def test_fallback():
    print("=== TESTING OPENCV HOG FALLBACK DETECTOR ===")
    adapter = ModelAdapter()
    
    # Intentionally test fallback method directly
    test_img_path = os.path.join(os.path.dirname(BASE_DIR), "uploads", "sample_cctv_night.mp4")
    cap = cv2.VideoCapture(test_img_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        dets = adapter._predict_opencv_hog(frame)
        print(f"OpenCV HOG Fallback returned {len(dets)} human detections:")
        for d in dets:
            print(f"  * {d['class']} ({d['confidence']}%) | Box: {d['bbox']}")
    else:
        print("Could not read frame from sample video")

if __name__ == "__main__":
    test_fallback()
