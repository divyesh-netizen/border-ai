import math
import time
from typing import List, Dict, Any, Tuple, Optional

class AlertEngine:
    """
    Temporal Behavior Analysis & Explainable Risk Scoring Engine for Border Surveillance.
    Evaluates:
    - Intrusion & Zone Crossings
    - Sustained Presence / Loitering (Dwell time > threshold in stationary radius)
    - Multi-Human Grouping & Crowd Density
    - Fast Movement / Running Anomalies
    - Animal & Vehicle Movement Verification
    - Transparent 0-100 Explainable Risk Score
    """
    def __init__(self, loiter_time_sec: float = 8.0, crowd_threshold: int = 4, speed_threshold: float = 120.0):
        self.loiter_time_sec = loiter_time_sec
        self.crowd_threshold = crowd_threshold
        self.speed_threshold = speed_threshold
        
        self.alert_history: List[Dict[str, Any]] = []
        self.fired_event_keys = set()

    def reset(self):
        self.alert_history = []
        self.fired_event_keys = set()

    def evaluate_frame(
        self,
        detections: List[Dict[str, Any]],
        active_tracks: Dict[int, Any],
        frame_idx: int,
        timestamp: float,
        video_fps: float = 25.0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Evaluates temporal behaviors and calculates explainable risk score for the active frame.
        """
        active_alerts = []
        
        # Categorize active validated tracks
        human_tracks = [t for t in active_tracks.values() if t.dominant_class == "HUMAN" and t.is_validated]
        vehicle_tracks = [t for t in active_tracks.values() if t.dominant_class == "VEHICLE" and t.is_validated]
        animal_tracks = [t for t in active_tracks.values() if t.dominant_class == "ANIMAL" and t.is_validated]
        unknown_tracks = [t for t in active_tracks.values() if t.dominant_class == "UNKNOWN"]

        num_humans = len(human_tracks)
        num_vehicles = len(vehicle_tracks)
        num_animals = len(animal_tracks)
        num_unknowns = len(unknown_tracks)

        # 1. Behavior: Loitering / Sustained Presence
        for track in human_tracks:
            if track.dwell_time >= self.loiter_time_sec:
                event_key = f"LOITER_{track.track_id}_{int(track.dwell_time // 10)}"
                if event_key not in self.fired_event_keys:
                    self.fired_event_keys.add(event_key)
                    alert = {
                        "alert_id": f"ALT_{len(self.alert_history) + 1:04d}",
                        "timestamp": round(timestamp, 2),
                        "frame": frame_idx,
                        "type": "SUSTAINED_PRESENCE",
                        "severity": "ATTENTION",
                        "target_id": track.display_id,
                        "description": f"Target {track.display_id} sustained dwell time ({track.dwell_time}s) in perimeter zone",
                        "action_required": "Operator sector review advised",
                        "risk_contribution": 35
                    }
                    active_alerts.append(alert)
                    self.alert_history.append(alert)

        # 2. Behavior: Multiple Human Presence
        if num_humans >= 2:
            event_key = f"MULTI_HUMAN_{frame_idx // int(video_fps * 5)}" # Throttle to once every 5 seconds
            if event_key not in self.fired_event_keys:
                self.fired_event_keys.add(event_key)
                alert = {
                    "alert_id": f"ALT_{len(self.alert_history) + 1:04d}",
                    "timestamp": round(timestamp, 2),
                    "frame": frame_idx,
                    "type": "MULTIPLE_HUMAN_PRESENCE",
                    "severity": "ATTENTION" if num_humans < self.crowd_threshold else "CRITICAL",
                    "target_id": f"GROUP_OF_{num_humans}",
                    "description": f"{num_humans} concurrent validated human targets tracked in sector",
                    "action_required": "Sector patrol notification",
                    "risk_contribution": 25 + (num_humans * 5)
                }
                active_alerts.append(alert)
                self.alert_history.append(alert)

        # 3. Behavior: Crowd Density Event
        if num_humans >= self.crowd_threshold:
            event_key = f"CROWD_{frame_idx // int(video_fps * 5)}"
            if event_key not in self.fired_event_keys:
                self.fired_event_keys.add(event_key)
                alert = {
                    "alert_id": f"ALT_{len(self.alert_history) + 1:04d}",
                    "timestamp": round(timestamp, 2),
                    "frame": frame_idx,
                    "type": "CROWD_FORMATION",
                    "severity": "CRITICAL",
                    "target_id": f"CROWD_N_{num_humans}",
                    "description": f"Crowd anomaly detected: {num_humans} active human tracks in sector",
                    "action_required": "High-priority perimeter alert",
                    "risk_contribution": 45
                }
                active_alerts.append(alert)
                self.alert_history.append(alert)

        # 4. Behavior: Fast Movement / Running Anomaly
        for track in human_tracks:
            if track.speed_px_s >= self.speed_threshold and track.hits >= 8:
                event_key = f"SPEED_{track.track_id}_{frame_idx // int(video_fps * 4)}"
                if event_key not in self.fired_event_keys:
                    self.fired_event_keys.add(event_key)
                    alert = {
                        "alert_id": f"ALT_{len(self.alert_history) + 1:04d}",
                        "timestamp": round(timestamp, 2),
                        "frame": frame_idx,
                        "type": "FAST_MOVEMENT",
                        "severity": "ATTENTION",
                        "target_id": track.display_id,
                        "description": f"Target {track.display_id} velocity anomaly: {track.speed_px_s} px/s",
                        "action_required": "Track trajectory monitoring",
                        "risk_contribution": 20
                    }
                    active_alerts.append(alert)
                    self.alert_history.append(alert)

        # 5. Behavior: Unknown Target Review Flag
        if num_unknowns >= 1:
            event_key = f"UNKNOWN_{frame_idx // int(video_fps * 10)}"
            if event_key not in self.fired_event_keys:
                self.fired_event_keys.add(event_key)
                alert = {
                    "alert_id": f"ALT_{len(self.alert_history) + 1:04d}",
                    "timestamp": round(timestamp, 2),
                    "frame": frame_idx,
                    "type": "UNKNOWN_OBJECT",
                    "severity": "MONITOR",
                    "target_id": "UNKNOWN_TARGET",
                    "description": f"{num_unknowns} low-confidence / unclassified visual target requiring human verification",
                    "action_required": "Visual verification required",
                    "risk_contribution": 10
                }
                active_alerts.append(alert)
                self.alert_history.append(alert)

        # Calculate Transparent Explainable Risk Score (0-100)
        risk_data = self._compute_explainable_risk(
            num_humans=num_humans,
            num_vehicles=num_vehicles,
            num_animals=num_animals,
            num_unknowns=num_unknowns,
            human_tracks=human_tracks
        )

        return active_alerts, risk_data

    def _compute_explainable_risk(
        self,
        num_humans: int,
        num_vehicles: int,
        num_animals: int,
        num_unknowns: int,
        human_tracks: List[Any]
    ) -> Dict[str, Any]:
        """
        Calculates a 0-100 explainable risk score with transparent factor weights.
        """
        score = 0
        factors = []

        # Base presence factor
        if num_humans > 0:
            human_contrib = min(40, num_humans * 15)
            score += human_contrib
            factors.append(f"Human presence ({num_humans} target{'s' if num_humans > 1 else ''}): +{human_contrib}")

        # Loitering factor
        max_dwell = max([t.dwell_time for t in human_tracks], default=0.0)
        if max_dwell > self.loiter_time_sec:
            dwell_contrib = min(30, int((max_dwell - self.loiter_time_sec) * 3) + 15)
            score += dwell_contrib
            factors.append(f"Sustained dwell time ({max_dwell:.1f}s): +{dwell_contrib}")

        # Vehicle factor
        if num_vehicles > 0:
            veh_contrib = min(20, num_vehicles * 10)
            score += veh_contrib
            factors.append(f"Vehicle sector activity ({num_vehicles}): +{veh_contrib}")

        # Crowd factor
        if num_humans >= self.crowd_threshold:
            score += 20
            factors.append(f"Crowd concentration anomaly: +20")

        # Unknown target factor
        if num_unknowns > 0:
            score += 5
            factors.append(f"Unclassified visual flag: +5")

        total_score = min(100, max(0, score))

        if total_score >= 70:
            level = "CRITICAL"
        elif total_score >= 45:
            level = "HIGH"
        elif total_score >= 20:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "score": total_score,
            "level": level,
            "factors": factors if factors else ["Perimeter clear — baseline monitoring: 0"],
            "max_dwell_sec": round(max_dwell, 1),
            "active_humans": num_humans,
            "active_vehicles": num_vehicles,
            "active_animals": num_animals
        }
