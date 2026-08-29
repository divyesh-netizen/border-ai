import urllib.request, json, time

job_id = "JOB-1DC46FF0"
for _ in range(12):
    try:
        status = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/analysis-status/{job_id}").read().decode())
        print(f"[{status['status']}] Progress: {status['progress_percent']}% | Detections: {status['stats']['total_detections']} (Humans: {status['stats']['humans']}, Vehicles: {status['stats']['vehicles']}, Animals: {status['stats']['animals']}) | Tracks: {status['stats']['active_tracks']} | Alerts: {status['stats']['alerts']}")
        if status['status'] in ['COMPLETED', 'ERROR']:
            break
        time.sleep(2)
    except Exception as e:
        print("Polling error:", e)
        break
