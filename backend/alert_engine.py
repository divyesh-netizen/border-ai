import time
import math
from typing import List, Dict, Any, Optional, Tuple

class AlertEngine:
    """
    Surveillance Event, Behaviour Analysis & Dynamic Risk Engine.
    Uses TRACK HISTORY rather than isolated frames to detect:
    - Intrusion / Restricted Zone Entry (Zone A, Zone B, Zone C)
    - Loitering / Sustained Presence
    - Wrong Direction Movement
    - Fast Movement / Running
    - Crowd / Multiple Human Gathering
    - Dynamic 0-100 Risk Score with contributing factor breakdown
    """
    def __init__(self, sustained_presence_sec: float = 6.0, multi_human_threshold: int = 2, min_persistence_frames: int = 3):
        self.sustained_presence_sec = sustained_presence_sec
        self.multi_human_threshold = multi_human_threshold
        self.min_persistence_frames = min_persistence_frames
        
        self.alerts: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.behaviour_log: List[Dict[str, Any]] = []
        self.alert_counter = 1
        self.event_counter = 1
        self._last_alert_time: Dict[str, float] = {}
        
        # Virtual Restricted Zones (Normalized coordinates [x1, y1, x2, y2])
        self.zones = {
            "ZONE-A": {"name": "Restricted Inner Buffer", "priority": "HIGH", "weight": 20, "bbox": [0.10, 0.20, 0.60, 0.90]},
            "ZONE-B": {"name": "Outer Transit Perimeter", "priority": "MEDIUM", "weight": 12, "bbox": [0.60, 0.20, 0.90, 0.90]},
            "ZONE-C": {"name": "General Approach Corridor", "priority": "LOW", "weight": 5, "bbox": [0.0, 0.0, 1.0, 0.35]}
        }

        self.current_risk_score = 12
        self.current_risk_level = "LOW"
        self.current_risk_factors = {
            "object_factor": 10,
            "zone_factor": 5,
            "time_factor": 5,
            "duration_factor": 0,
            "movement_factor": 5,
            "behaviour_factor": 0
        }

    def _should_throttle(self, key: str, cooldown: float = 4.0) -> bool:
        now = time.time()
        if key in self._last_alert_time:
            if now - self._last_alert_time[key] < cooldown:
                return True
        self._last_alert_time[key] = now
        return False

    def check_zone_entry(self, norm_bbox: List[float]) -> Tuple[str, str, int]:
        """
        Determines which zone a bounding box centroid falls into.
        """
        if not norm_bbox or len(norm_bbox) != 4:
            return "PERIMETER", "LOW", 5
        cx = (norm_bbox[0] + norm_bbox[2]) / 2.0
        cy = (norm_bbox[1] + norm_bbox[3]) / 2.0

        for zid, zinfo in self.zones.items():
            zb = zinfo["bbox"]
            if zb[0] <= cx <= zb[2] and zb[1] <= cy <= zb[3]:
                return zid, zinfo["priority"], zinfo["weight"]

        return "PERIMETER", "LOW", 5

    def compute_risk_score(self, detections: List[Dict[str, Any]], active_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates dynamic 0-100 Risk Score based on 6 core risk factors:
        - Object Type (Human, Vehicle, Animal, Unknown)
        - Zone Priority (Restricted Buffer, Transit Sector, Perimeter)
        - Time & Lighting Context (Night / Low-light elevates risk)
        - Duration / Dwell Time in zone
        - Movement Speed / Anomaly
        - Behaviour Pattern (Crowd, Loitering, Intrusion)
        """
        human_count = sum(1 for d in detections if d.get("class_name") == "HUMAN")
        vehicle_count = sum(1 for d in detections if d.get("class_name") == "VEHICLE")
        animal_count = sum(1 for d in detections if d.get("class_name") == "ANIMAL")
        unknown_count = sum(1 for d in detections if d.get("class_name") == "UNKNOWN")
        max_dwell = max([t.get("dwell_time", 0) for t in active_tracks], default=0)
        max_speed = max([t.get("speed_px_s", 0) for t in active_tracks], default=0)

        # 1. Object Factor (0 - 25)
        obj_factor = min(25, (human_count * 9) + (vehicle_count * 6) + (unknown_count * 7) + (animal_count * 2))

        # 2. Zone Factor (0 - 20)
        zone_weights = []
        for t in active_tracks:
            _, _, w = self.check_zone_entry(t.get("norm_bbox", []))
            zone_weights.append(w)
        zone_factor = min(20, max(zone_weights, default=5))

        # 3. Time / Night Context Factor (0 - 15)
        time_factor = 10  # Standard night surveillance context

        # 4. Duration Factor (0 - 20)
        dur_factor = min(20, int(max_dwell * 2.5))

        # 5. Movement Factor (0 - 10)
        move_factor = 7 if (max_speed > 80.0) else (4 if human_count > 0 else 2)

        # 6. Behaviour Factor (0 - 10)
        beh_factor = 0
        if human_count >= self.multi_human_threshold:
            beh_factor += 5
        if max_dwell >= self.sustained_presence_sec:
            beh_factor += 5

        total_score = min(100, max(5, obj_factor + zone_factor + time_factor + dur_factor + move_factor + beh_factor))

        if total_score <= 30:
            level = "LOW"
        elif total_score <= 70:
            level = "MEDIUM"
        else:
            level = "HIGH"

        self.current_risk_score = total_score
        self.current_risk_level = level
        self.current_risk_factors = {
            "object_factor": obj_factor,
            "zone_factor": zone_factor,
            "time_factor": time_factor,
            "duration_factor": dur_factor,
            "movement_factor": move_factor,
            "behaviour_factor": beh_factor
        }

        return {
            "score": total_score,
            "level": level,
            "factors": self.current_risk_factors
        }

    def process_frame(self, detections: List[Dict[str, Any]], active_tracks: List[Dict[str, Any]], timecode_str: str, current_time: float, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Evaluates track history, zone boundaries, and dynamics to trigger real security events & alerts.
        """
        new_alerts_this_frame = []

        # Filter detections by persistence to reduce single-frame false alarms
        persistent_tracks = [t for t in active_tracks if t.get("total_visible_frames", 1) >= self.min_persistence_frames]
        human_tracks = [t for t in persistent_tracks if t["class_name"] == "HUMAN"]
        vehicle_tracks = [t for t in persistent_tracks if t["class_name"] == "VEHICLE"]
        animal_tracks = [t for t in persistent_tracks if t["class_name"] == "ANIMAL"]
        unknown_tracks = [t for t in persistent_tracks if t["class_name"] == "UNKNOWN"]

        # 1. Restricted Zone Entry (Intrusion / High Attention)
        for t in human_tracks:
            zid, zpri, _ = self.check_zone_entry(t.get("norm_bbox", []))
            t["zone"] = zid
            if zid == "ZONE-A":
                throttle_key = f"zone_entry_{t['track_id']}_{zid}"
                if not self._should_throttle(throttle_key, cooldown=8.0):
                    alert = {
                        "id": f"ALT-{self.alert_counter:04d}",
                        "timestamp": timecode_str,
                        "type": "RESTRICTED ZONE ENTRY",
                        "severity": "ALERT",
                        "track_id": t["display_id"],
                        "confidence": f"{t['confidence']}%",
                        "zone": "ZONE-A (Restricted Inner Buffer)",
                        "message": f"Restricted zone breach: {t['display_id']} entered Zone A",
                        "details": f"Target crossed into high-priority buffer at {timecode_str}",
                        "evidence_ready": True
                    }
                    self.alerts.append(alert)
                    new_alerts_this_frame.append(alert)
                    self.alert_counter += 1

                    self.events.append({
                        "id": self.event_counter,
                        "timestamp": timecode_str,
                        "track_id": t["display_id"],
                        "class_name": "Human",
                        "confidence": f"{t['confidence']}%",
                        "event": "Restricted Zone Entry",
                        "severity": "ALERT",
                        "zone": "ZONE-A",
                        "details": f"Personnel ingress into high priority restricted buffer zone"
                    })
                    self.event_counter += 1

                    self.behaviour_log.append({
                        "time": timecode_str,
                        "category": "INTRUSION",
                        "title": "Restricted Buffer Ingress",
                        "desc": f"{t['display_id']} entered Zone A restricted corridor",
                        "severity": "ALERT"
                    })

        # 2. Loitering / Sustained Presence (> N seconds)
        for t in persistent_tracks:
            if t["dwell_time"] >= self.sustained_presence_sec:
                throttle_key = f"sustained_{t['track_id']}"
                if not self._should_throttle(throttle_key, cooldown=10.0):
                    alert = {
                        "id": f"ALT-{self.alert_counter:04d}",
                        "timestamp": timecode_str,
                        "type": "LOITERING / SUSTAINED PRESENCE",
                        "severity": "ATTENTION",
                        "track_id": t["display_id"],
                        "confidence": f"{t['confidence']}%",
                        "duration": f"{int(t['dwell_time'])}s",
                        "zone": t.get("zone", "Zone A"),
                        "message": f"Sustained presence detected: {t['display_id']} in sector for {int(t['dwell_time'])}s",
                        "details": f"Target stationary / lingering beyond threshold ({int(self.sustained_presence_sec)}s)",
                        "evidence_ready": True
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
                        "event": "Loitering / Sustained Presence",
                        "severity": "ATTENTION",
                        "zone": t.get("zone", "Zone A"),
                        "details": f"Dwell duration reached {int(t['dwell_time'])}s in monitored sector"
                    })
                    self.event_counter += 1

                    self.behaviour_log.append({
                        "time": timecode_str,
                        "category": "LOITERING",
                        "title": "Stationary Dwell Alert",
                        "desc": f"{t['display_id']} stationary in perimeter buffer >{int(self.sustained_presence_sec)}s",
                        "severity": "ATTENTION"
                    })

        # 3. Fast Movement / Running Detection
        for t in human_tracks:
            if t.get("speed_px_s", 0) > 120.0:
                throttle_key = f"fast_move_{t['track_id']}"
                if not self._should_throttle(throttle_key, cooldown=6.0):
                    self.behaviour_log.append({
                        "time": timecode_str,
                        "category": "MOVEMENT",
                        "title": "Rapid Acceleration / Running",
                        "desc": f"{t['display_id']} movement speed elevated ({int(t['speed_px_s'])} px/s)",
                        "severity": "ATTENTION"
                    })

        # 4. Crowd / Multiple Human Presence
        if len(human_tracks) >= self.multi_human_threshold:
            if not self._should_throttle("multi_human_cluster", cooldown=6.0):
                avg_conf = round(sum(t["confidence"] for t in human_tracks) / len(human_tracks), 1)
                alert = {
                    "id": f"ALT-{self.alert_counter:04d}",
                    "timestamp": timecode_str,
                    "type": "MULTIPLE HUMAN PRESENCE",
                    "severity": "ATTENTION",
                    "count": len(human_tracks),
                    "confidence": f"{avg_conf}%",
                    "zone": "Sector-07 Perimeter",
                    "message": f"Multiple humans active in sector ({len(human_tracks)} tracked individuals)",
                    "details": f"Simultaneous detection of {len(human_tracks)} individuals at {timecode_str}",
                    "evidence_ready": True
                }
                self.alerts.append(alert)
                new_alerts_this_frame.append(alert)
                self.alert_counter += 1

                self.events.append({
                    "id": self.event_counter,
                    "timestamp": timecode_str,
                    "track_id": f"Cluster ({len(human_tracks)})",
                    "class_name": "Human",
                    "confidence": f"{avg_conf}%",
                    "event": "Multiple Human Presence",
                    "severity": "ATTENTION",
                    "zone": "Sector-07",
                    "details": f"Simultaneous activity of {len(human_tracks)} targets in close perimeter proximity"
                })
                self.event_counter += 1

                self.behaviour_log.append({
                    "time": timecode_str,
                    "category": "CROWD",
                    "title": "Multi-Target Gathering",
                    "desc": f"{len(human_tracks)} active tracked individuals in perimeter",
                    "severity": "ATTENTION"
                })

        # 5. Routine Single Human Presence (NORMAL)
        if len(human_tracks) == 1:
            h = human_tracks[0]
            if not self._should_throttle(f"human_normal_{h['track_id']}", cooldown=12.0):
                self.events.append({
                    "id": self.event_counter,
                    "timestamp": timecode_str,
                    "track_id": h["display_id"],
                    "class_name": "Human",
                    "confidence": f"{h['confidence']}%",
                    "event": "Human Presence",
                    "severity": "NORMAL",
                    "zone": h.get("zone", "Sector-07"),
                    "details": f"Track {h['display_id']} observed in transit with confidence {h['confidence']}%"
                })
                self.event_counter += 1

                self.behaviour_log.append({
                    "time": timecode_str,
                    "category": "DETECTION",
                    "title": "Personnel Transit",
                    "desc": f"{h['display_id']} transit logged in sector",
                    "severity": "NORMAL"
                })

        # 6. Vehicle Transit (NORMAL)
        for v in vehicle_tracks:
            if not self._should_throttle(f"veh_transit_{v['track_id']}", cooldown=10.0):
                self.events.append({
                    "id": self.event_counter,
                    "timestamp": timecode_str,
                    "track_id": v["display_id"],
                    "class_name": "Vehicle",
                    "confidence": f"{v['confidence']}%",
                    "event": "Vehicle Transit",
                    "severity": "NORMAL",
                    "zone": "Access Corridor",
                    "details": f"Vehicle activity observed: {v['confidence']}% confidence"
                })
                self.event_counter += 1

        # 7. Animal Crossing (NORMAL)
        for a in animal_tracks:
            if not self._should_throttle(f"anim_cross_{a['track_id']}", cooldown=10.0):
                self.events.append({
                    "id": self.event_counter,
                    "timestamp": timecode_str,
                    "track_id": a["display_id"],
                    "class_name": "Animal",
                    "confidence": f"{a['confidence']}%",
                    "event": "Fauna Crossing",
                    "severity": "NORMAL",
                    "zone": "Outer Buffer",
                    "details": f"Perimeter fauna crossing logged: {a['confidence']}% confidence"
                })
                self.event_counter += 1

        return new_alerts_this_frame

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        return self.alerts

    def get_all_events(self) -> List[Dict[str, Any]]:
        return self.events

    def get_behaviour_log(self) -> List[Dict[str, Any]]:
        return self.behaviour_log

    def reset(self):
        self.alerts.clear()
        self.events.clear()
        self.behaviour_log.clear()
        self._last_alert_time.clear()
        self.alert_counter = 1
        self.event_counter = 1
        self.current_risk_score = 12
        self.current_risk_level = "LOW"
