import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from inference import ModelAdapter
from video_processor import VideoProcessor

def test_full_video_pipeline():
    project_dir = os.path.dirname(BASE_DIR)
    uploads_dir = os.path.join(project_dir, "uploads")
    outputs_dir = os.path.join(project_dir, "outputs")
    models_dir = os.path.join(project_dir, "models")
    
    test_video = os.path.join(uploads_dir, "sample_cctv_night.mp4")
    if not os.path.exists(test_video):
        print("Test video not found:", test_video)
        return

    print("Initializing ModelAdapter & VideoProcessor...")
    adapter = ModelAdapter(model_path="yolov8n.pt", conf_threshold=0.30, iou_threshold=0.45)
    processor = VideoProcessor(model_adapter=adapter, output_dir=outputs_dir)

    print(f"Starting video processing job on {test_video}...")
    job_id = processor.start_processing(test_video, is_thermal=False, mode_override="REAL")
    print(f"Job ID: {job_id}")

    # Wait for completion
    while True:
        job = processor.get_job(job_id)
        if not job:
            break
        print(f"  Frame {job.current_frame}/{job.total_frames} ({job.progress_percent}%) | FPS: {job.fps} | Active: {job.active_tracks_count} | Unique Humans: {job.unique_humans} | Total Detections: {job.total_detections_count}")
        if job.status in ["COMPLETED", "ERROR"]:
            break
        time.sleep(0.5)

    print("\n--- JOB FINAL SUMMARY ---")
    print("Status:", job.status)
    print("Duration:", job.duration_sec, "sec")
    print("Frames Processed:", job.current_frame)
    print("Total Detections:", job.total_detections_count)
    print("Unique Humans Tracked:", job.unique_humans)
    print("Unique Vehicles Tracked:", job.unique_vehicles)
    print("Unique Animals Tracked:", job.unique_animals)
    print("Total Unique Objects:", job.total_unique_objects)
    print("Active Tracks:", job.active_tracks_count)
    print("Alerts Count:", job.alerts_count)
    print("Annotated Video Path:", job.annotated_video_path)
    print("Annotated Video Exists:", os.path.exists(job.annotated_video_path) if job.annotated_video_path else False)
    if job.annotated_video_path and os.path.exists(job.annotated_video_path):
        print("Annotated Video Size:", os.path.getsize(job.annotated_video_path), "bytes")

    assert job.status == "COMPLETED"
    assert job.annotated_video_path is not None
    assert os.path.exists(job.annotated_video_path)
    assert job.unique_humans >= 1
    assert job.total_detections_count > job.unique_humans, "Total detections must be strictly greater than unique humans for multi-frame video!"
    print("\n✓ Full Video Pipeline Integration Test PASSED successfully!")

if __name__ == "__main__":
    test_full_video_pipeline()
