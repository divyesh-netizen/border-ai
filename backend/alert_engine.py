import time
from typing import List, Dict, Any, Optional

class AlertEngine:
    """
    Surveillance Event & Alert Engine.
    Evaluates detections and tracks to generate structured events with severity:
    - INFO: Routine detections (e.g. single human, known vehicle)
    - MONITOR: Items requiring attention (e.g. multiple humans, unknown object, animal movement)
    - ATTENTION: Priority surveillance events (e.g. sustained loitering/dwell time, large cluster)
    """
    def __init__(self, sustained_presence_sec: float = 10.0, multi_human_threshold: int = 2):
        self.sustained_presence_sec = sustained_presence_sec
        self.multi_human_threshold = multi_human_threshold
        self.alerts: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.alert_counter = 1
        self.event_counter = 1
        self._last_alert_time: Dict[str, float] = {}

    def _should_throttle(self, key: str, cooldown: float = 4.0) -> bool:
        now = time.time()
        if key in self._last_alert_time:
            if now - self._last_alert_time[key] < cooldown:
                return True
        self._last_alert_time[key] = now
        return False

    def process_frame(self, detections: List[Dict[str, Any]], active_tracks: List[Dict[str, Any]], timecode_str: str, current_time: float) -> List[Dict[str, Any]]:
        new_alerts_this_frame = []

        human_detections = [d for d in detections if d["class_name"] == "HUMAN"]
        animal_detections = [d for d in detections if d["class_name"] == "ANIMAL"]
        vehicle_detections = [d for d in detections if d["class_name"] == "VEHICLE"]
        unknown_detections = [d for d in detections if d["class_name"] == "UNKNOWN"]

        # 1. Single Human Detection (INFO)
        if len(human_detections) == 1:
            h = human_detections[0]
            event = {
                "id": self.event_counter,
                "timestamp": timecode_str,
                "track_id": h.get("display_id", f"H{h.get('track_id', 1):03d}"),
                "class_name": "Human",
                "confidence": f"{h['confidence']}%",
                "event": "Human Presence",
                "severity": "INFO",
                "details": f"Single human detected in sector with confidence {h['confidence']}%"
            }
            self.events.append(event)
            self.event_counter += 1

        # 2. Multiple Human Presence (ATTENTION / MONITOR)
        if len(human_detections) >= self.multi_human_threshold:
            if not self._should_throttle("multi_human", cooldown=5.0):
                avg_conf = round(sum(d["confidence"] for d in human_detections) / len(human_detections), 1)
                alert = {
                    "id": f"ALT-{self.alert_counter:04d}",
                    "timestamp": timecode_str,
                    "type": "MULTIPLE HUMAN PRESENCE",
                    "severity": "ATTENTION",
                    "count": len(human_detections),
                    "confidence": f"{avg_conf}%",
                    "message": f"Multiple humans detected ({len(human_detections)} persons active in perimeter)",
                    "details": f"Detected {len(human_detections)} humans at {timecode_str}."
                }
                self.alerts.append(alert)
                new_alerts_this_frame.append(alert)
                self.alert_counter += 1

                self.events.append({
                    "id": self.event_counter,
                    "timestamp": timecode_str,
                    "track_id": f"Group ({len(human_detections)})",
                    "class_name": "Human",
                    "confidence": f"{avg_conf}%",
                    "event": "Multiple Human Presence",
                    "severity": "ATTENTION",
                    "details": f"Simultaneous detection of {len(human_detections)} individuals"
                })
                self.event_counter += 1

        # 3. Sustained Presence / Loitering (ATTENTION)
        for t in active_tracks:
            if t["dwell_time"] >= self.sustained_presence_sec:
                throttle_key = f"sustained_{t['track_id']}"
                if not self._should_throttle(throttle_key, cooldown=12.0):
                    alert = {
                        "id": f"ALT-{self.alert_counter:04d}",
                        "timestamp": timecode_str,
                        "type": "SUSTAINED PRESENCE",
                        "severity": "ATTENTION",
                        "track_id": t["display_id"],
                        "confidence": f"{t['confidence']}%",
                        "duration": f"{int(t['dwell_time'])} sec",
                        "message": f"Sustained presence detected: {t['display_id']} in sector for {int(t['dwell_time'])}s",
                        "details": f"Target lingered beyond threshold ({self.sustained_presence_sec}s)"
                    }
                    self.alerts.append(alert)
                    new_alerts_this_frame.append(alert)
                    self.alert_counter += 1

                    self.events.append({
                        "id": self.event_counter,
                        "timestamp": timecode_str,
                        "track_id": t["display_id"],
                        "class_name": t["class_name"].capitalize(),
                        "confidence": f"{t['confidence']}%",
                        "event": "Sustained Presence",
                        "severity": "ATTENTION",
                        "details": f"Target stationary / loitering for {int(t['dwell_time'])}s"
                    })
                    self.event_counter += 1

        # 4. Unknown Object (MONITOR)
        if unknown_detections:
            for u in unknown_detections:
                if not self._should_throttle("unknown_obj", cooldown=6.0):
                    alert = {
                        "id": f"ALT-{self.alert_counter:04d}",
                        "timestamp": timecode_str,
                        "type": "UNKNOWN OBJECT",
                        "severity": "MONITOR",
                        "confidence": f"{u['confidence']}%",
                        "message": f"Unclassified object in sector (confidence: {u['confidence']}%)",
                        "details": "Detection below high-confidence taxonomy mapping"
                    }
                    self.alerts.append(alert)
                    new_alerts_this_frame.append(alert)
                    self.alert_counter += 1

                    self.events.append({
                        "id": self.event_counter,
                        "timestamp": timecode_str,
                        "track_id": u.get("display_id", "Obj-Unk"),
                        "class_name": "Unknown",
                        "confidence": f"{u['confidence']}%",
                        "event": "Unknown Object Detected",
                        "severity": "MONITOR",
                        "details": "Low-confidence or unmapped visual target"
                    })
                    self.event_counter += 1

        # 5. Vehicle Detected (INFO / MONITOR)
        if vehicle_detections:
            for v in vehicle_detections:
                throttle_key = f"veh_{v.get('track_id', 1)}"
                if not self._should_throttle(throttle_key, cooldown=8.0):
                    self.events.append({
                        "id": self.event_counter,
                        "timestamp": timecode_str,
                        "track_id": v.get("display_id", "Vehicle"),
                        "class_name": "Vehicle",
                        "confidence": f"{v['confidence']}%",
                        "event": "Vehicle Movement",
                        "severity": "INFO",
                        "details": f"Vehicle activity observed: {v['confidence']}%"
                    })
                    self.event_counter += 1

        # 6. Animal Detected (INFO / MONITOR)
        if animal_detections:
            for a in animal_detections:
                throttle_key = f"anim_{a.get('track_id', 1)}"
                if not self._should_throttle(throttle_key, cooldown=8.0):
                    self.events.append({
                        "id": self.event_counter,
                        "timestamp": timecode_str,
                        "track_id": a.get("display_id", "Animal"),
                        "class_name": "Animal",
                        "confidence": f"{a['confidence']}%",
                        "event": "Animal Movement",
                        "severity": "INFO",
                        "details": f"Fauna / animal crossing observed: {a['confidence']}%"
                    })
                    self.event_counter += 1

        return new_alerts_this_frame

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        return self.alerts

    def get_all_events(self) -> List[Dict[str, Any]]:
        return self.events

    def reset(self):
        self.alerts.clear()
        self.events.clear()
        self._last_alert_time.clear()
        self.alert_counter = 1
        self.event_counter = 1
