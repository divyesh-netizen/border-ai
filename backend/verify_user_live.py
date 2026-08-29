import urllib.request, json, time

job_id = "JOB-9C7CFFB1"
for _ in range(8):
    try:
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/analysis-status/{job_id}").read().decode())
        print(f"[{status['status']}] Progress: {status['progress_percent']}% | Detections: {status['stats']['total_detections']} (Humans: {status['stats']['humans']}, Vehicles: {status['stats']['vehicles']}, Animals: {status['stats']['animals']}) | Active Tracks: {status['stats']['active_tracks']} | Alerts: {status['stats']['alerts']}")
        time.sleep(2)
    except Exception as e:
        print("Polling error:", e)
        break
