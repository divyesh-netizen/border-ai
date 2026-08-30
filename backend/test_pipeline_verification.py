import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from video_quality import VideoQualityAnalyzer
from inference import ModelAdapter
from tracker import ByteTracker
from alert_engine import AlertEngine
from benchmark import SurveillanceBenchmarkSuite
from video_processor import VideoProcessor

def test_full_pipeline():
    print("==================================================")
    print("  BORDER AI Pipeline & Degradation Test Suite")
    print("==================================================")
    
    # 1. Test ModelAdapter & Inference
    adapter = ModelAdapter()
    print(f"[*] Model Status: {adapter.model_info['status']} | Device: {adapter.device}")
    
    # 2. Test Benchmark Suite
    bench = SurveillanceBenchmarkSuite(adapter)
    res = bench.run_degraded_benchmark()
    print(f"[*] Degraded Benchmark Tiers Evaluated: {len(res['quality_matrix'])}")
    for t in res['quality_matrix']:
        print(f"    - {t['quality_tier']}: Person Precision {t['person_precision']} | Recall {t['person_recall']} | Count Error MAE: {t['unique_count_error_mae']}")
    
    # 3. Test Video Processing on User Surveillance Video
    video_path = os.path.join(os.path.dirname(BASE_DIR), "uploads", "whatsapp_surveillance.mp4")
    if os.path.exists(video_path):
        print(f"\n[*] Processing Benchmark Video: {video_path}...")
        vp = VideoProcessor(model_adapter=adapter, output_dir=os.path.join(os.path.dirname(BASE_DIR), "outputs"))
        job_id = vp.start_processing(video_path=video_path, mode_override="HIGH_ACCURACY")
        print(f"    Job ID started: {job_id}")
        
        job = vp.get_job(job_id)
        while job.status in ["INITIALIZING", "PROCESSING"]:
            time.sleep(0.8)
            job = vp.get_job(job_id)
            print(f"    Progress: {job.progress_percent}% | Frame: {job.current_frame}/{job.total_frames} | Raw: {job.total_detections_count} | Unique: H={job.unique_humans}, V={job.unique_vehicles}, A={job.unique_animals} | Active: {job.active_tracks_count}", flush=True)

        print(f"\n[+] Job Finished with Status: {job.status}")
        print(f"    - Total Raw Detections: {job.total_detections_count}")
        print(f"    - Validated Unique Humans: {job.unique_humans}")
        print(f"    - Validated Unique Vehicles: {job.unique_vehicles}")
        print(f"    - Validated Unique Animals: {job.unique_animals}")
        print(f"    - Quality Detected: {job.quality_report.get('quality_label')} (Blur: {job.quality_report.get('blur_score')}, Lux: {job.quality_report.get('brightness_lux')})")
        print(f"    - Risk Score: {job.risk_data['score']} ({job.risk_data['level']})")
        print(f"    - Evidence Thumbnails Generated: {len(job.unique_identities_gallery)}")
        for idx, ident in enumerate(job.unique_identities_gallery):
            print(f"      [Track #{idx+1}] {ident['display_id']} | Conf: {ident['mean_confidence']}% | Dwell: {ident['dwell_time_sec']}s | Thumb: {ident['thumbnail_url']}")

if __name__ == "__main__":
    test_full_pipeline()
