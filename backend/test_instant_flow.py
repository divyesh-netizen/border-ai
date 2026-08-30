import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import ModelAdapter
from video_processor import VideoProcessor, VideoProcessingJob

def test_instant_flow():
    print("=== TESTING INSTANT REAL-TIME DETECTION PIPELINE ===")
    m = ModelAdapter()
    vp = VideoProcessor(m, os.path.join(os.path.dirname(BASE_DIR), "outputs"))

    video_path = os.path.join(os.path.dirname(BASE_DIR), "uploads", "sample_cctv_night.mp4")
    job_id = vp.start_processing(video_path, mode_override="REAL_TIME")
    
    start_t = time.time()
    first_det_time = None
    
    while True:
        job = vp.get_job(job_id)
        if job.current_frame > 0 and first_det_time is None:
            first_det_time = time.time() - start_t
            print(f"[FAST START] First Frame reached in {first_det_time:.2f}s! (Frame #{job.current_frame})")
            print(f"  - Detections in Frame: {len(job.live_detections)}")
            for d in job.live_detections:
                print(f"    * {d['display_id']} | {d['class']} ({d['confidence']}%) | bbox: {d['bbox']}")

        if job.status == "COMPLETED":
            total_t = time.time() - start_t
            print(f"\n[COMPLETE] Video fully processed in {total_t:.2f}s!")
            print(f"  - Total Raw Detections: {job.total_detections_count}")
            print(f"  - Unique Humans: {job.unique_humans}")
            print(f"  - Unique Vehicles: {job.unique_vehicles}")
            print(f"  - Thumbnails saved: {len(job.unique_identities_gallery)}")
            break
        elif job.status == "ERROR":
            print(f"[ERROR] {job.error_message}")
            break
        
        time.sleep(0.05)

if __name__ == "__main__":
    test_instant_flow()
