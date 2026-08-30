import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import ModelAdapter
from video_processor import VideoProcessor

def test_user_videos():
    print("==================================================")
    print("    TESTING RAW & TRACKED PERSON DETECTIONS      ")
    print("==================================================")
    
    adapter = ModelAdapter()
    print(f"Model loaded: {adapter.weights_path} on {adapter.device}")
    
    uploads_dir = os.path.join(os.path.dirname(BASE_DIR), "uploads")
    
    # Test all video files in uploads
    for vf in os.listdir(uploads_dir):
        if not vf.endswith((".mp4", ".mov", ".avi")):
            continue
        vpath = os.path.join(uploads_dir, vf)
        vp = VideoProcessor(adapter, os.path.join(os.path.dirname(BASE_DIR), "outputs"))
        
        print(f"\n--- Testing Video: {vf} ---")
        job_id = vp.start_processing(vpath, mode_override="REAL_TIME")
        
        # Poll for 20 frames
        frames_seen = 0
        while frames_seen < 15:
            time.sleep(0.1)
            job = vp.get_job(job_id)
            if job and job.current_frame > frames_seen:
                frames_seen = job.current_frame
                humans_in_frame = [d for d in job.live_detections if d["class"] == "HUMAN"]
                print(f"  Frame #{job.current_frame} | Humans: {len(humans_in_frame)} | Active: {job.active_tracks_count} | Unique Humans: {job.unique_humans}")
                for h in humans_in_frame:
                    print(f"    * {h['display_id']} | Conf: {h['confidence']}% | Box: {h['bbox']}")
            if job and job.status in ["COMPLETED", "ERROR"]:
                break

if __name__ == "__main__":
    test_user_videos()
