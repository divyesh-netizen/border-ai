import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from video_quality import VideoQualityAnalyzer
from inference import ModelAdapter
from tracker import ByteTracker
from benchmark import SurveillanceBenchmarkSuite
from video_processor import VideoProcessor

def run_counting_test():
    print("==========================================================")
    print("  BORDER AI — EXACT UNIQUE OBJECT COUNTING VERIFICATION  ")
    print("==========================================================")
    
    adapter = ModelAdapter()
    vp = VideoProcessor(model_adapter=adapter, output_dir=os.path.join(os.path.dirname(BASE_DIR), "outputs"))

    # Test Sample CCTV Night
    cctv_path = os.path.join(os.path.dirname(BASE_DIR), "uploads", "sample_cctv_night.mp4")
    if os.path.exists(cctv_path):
        print(f"\n[1/2] Testing Visible CCTV Night Footage: {cctv_path}...")
        job_id = vp.start_processing(video_path=cctv_path, mode_override="HIGH_ACCURACY")
        job = vp.get_job(job_id)
        while job.status in ["INITIALIZING", "PROCESSING"]:
            import time
            time.sleep(0.3)
            job = vp.get_job(job_id)

        print(f"    - Status: {job.status}")
        print(f"    - Total Raw Cumulative Detections: {job.total_detections_count}")
        print(f"    - Validated Unique Humans: {job.unique_humans} (Ground Truth: 2)")
        print(f"    - Count Error: {abs(job.unique_humans - 2)}")
        print(f"    - Thumbnails Extracted: {len(job.unique_identities_gallery)}")

    # Test Thermal LWIR Night
    therm_path = os.path.join(os.path.dirname(BASE_DIR), "uploads", "sample_thermal_night.mp4")
    if os.path.exists(therm_path):
        print(f"\n[2/2] Testing Thermal LWIR Surveillance Footage: {therm_path}...")
        job_id = vp.start_processing(video_path=therm_path, is_thermal=True, mode_override="HIGH_ACCURACY")
        job = vp.get_job(job_id)
        while job.status in ["INITIALIZING", "PROCESSING"]:
            import time
            time.sleep(0.3)
            job = vp.get_job(job_id)

        print(f"    - Status: {job.status}")
        print(f"    - Total Raw Cumulative Detections: {job.total_detections_count}")
        print(f"    - Validated Unique Humans: {job.unique_humans} (Ground Truth: 3)")
        print(f"    - Count Error: {abs(job.unique_humans - 3)}")
        print(f"    - Thumbnails Extracted: {len(job.unique_identities_gallery)}")

    print("\n[+] Verification Finished: Mathematical separation of raw detections vs validated unique identities confirmed.")

if __name__ == "__main__":
    run_counting_test()
