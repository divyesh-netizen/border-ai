import csv
import io
import json
from typing import Dict, Any, List

def generate_csv_report(job_id: str, job_summary: Dict[str, Any], events: List[Dict[str, Any]]) -> str:
    """
    Generates a formal CSV surveillance activity log report.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header section
    writer.writerow(["BORDER AI SURVEILLANCE REPORT — SMART INDIA HACKATHON 2026"])
    writer.writerow(["Job ID", job_id])
    writer.writerow(["Video Duration", f"{job_summary.get('duration_sec', 0)}s"])
    writer.writerow(["Frames Processed", job_summary.get('frames_processed', 0)])
    writer.writerow(["Processing Speed", f"{job_summary.get('processing_fps', 0)} FPS"])
    writer.writerow([])
    writer.writerow(["--- ENTITY & TRACKING SUMMARY ---"])
    writer.writerow(["Total Detections Logged (Frame Sum)", job_summary.get('total_detections', 0)])
    writer.writerow(["Total Unique Humans Tracked", job_summary.get('unique_humans', 0)])
    writer.writerow(["Total Unique Vehicles Tracked", job_summary.get('unique_vehicles', 0)])
    writer.writerow(["Total Unique Animals Tracked", job_summary.get('unique_animals', 0)])
    writer.writerow(["Total Unique Objects Tracked", job_summary.get('total_unique_objects', 0)])
    writer.writerow(["Final Active Tracks", job_summary.get('active_tracks_count', 0)])
    writer.writerow(["Total Alerts Generated", job_summary.get('alerts_count', 0)])
    writer.writerow(["Total Events Logged", job_summary.get('events_count', 0)])
    writer.writerow([])
    
    # Event Log Table
    writer.writerow(["Timestamp", "Track ID", "Class", "Confidence", "Event", "Severity", "Details"])
    for ev in events:
        writer.writerow([
            ev.get("timestamp", ""),
            ev.get("track_id", ""),
            ev.get("class_name", ""),
            ev.get("confidence", ""),
            ev.get("event", ""),
            ev.get("severity", ""),
            ev.get("details", "")
        ])

    return output.getvalue()

def generate_json_report(job_id: str, job_summary: Dict[str, Any], events: List[Dict[str, Any]], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates structured JSON analysis report for downstream command systems.
    """
    return {
        "report_type": "BORDER_AI_SURVEILLANCE_ANALYSIS",
        "job_id": job_id,
        "system_status": "ONLINE",
        "summary": job_summary,
        "alerts": alerts,
        "events": events,
        "model_metadata": {
            "taxonomy": ["HUMAN", "VEHICLE", "ANIMAL", "UNKNOWN"],
            "verification_status": "Active Inference Verified",
            "benchmark_reference": "LLVIP + PBVS + VIRAT Surveillance Benchmarks"
        }
    }
