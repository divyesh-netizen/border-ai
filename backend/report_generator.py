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
    writer.writerow(["Total Detections", job_summary.get('total_detections', 0)])
    writer.writerow(["Human Detections", job_summary.get('human_detections', 0)])
    writer.writerow(["Vehicle Detections", job_summary.get('vehicle_detections', 0)])
    writer.writerow(["Animal Detections", job_summary.get('animal_detections', 0)])
    writer.writerow(["Unknown Detections", job_summary.get('unknown_detections', 0)])
    writer.writerow(["Total Alerts", job_summary.get('alerts_count', 0)])
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
            "taxonomy": ["HUMAN", "ANIMAL", "VEHICLE", "UNKNOWN"],
            "verification_status": "Active Inference Verified",
            "benchmark_reference": "LLVIP Low-Light Surveillance Benchmark"
        }
    }
