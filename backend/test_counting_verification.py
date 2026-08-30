import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tracker import MultiObjectTracker
from alert_engine import AlertEngine

class TestBorderAICountingAndTracking(unittest.TestCase):
    def test_1_person_500_frames(self):
        """
        Test 1: Video contains exactly 1 person for 500 frames.
        Expected:
        - Current Humans: 1
        - Active Tracks: 1
        - Unique Humans: 1 (NEVER 500)
        - Frame Detections: 500
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=60, min_track_frames=4)
        total_detections_logged = 0

        for frame_idx in range(1, 501):
            t = frame_idx / 25.0
            x1 = 100 + int((t * 5) % 200)
            y1 = 150
            x2 = x1 + 40
            y2 = y1 + 100
            
            frame_dets = [{
                "class_name": "HUMAN",
                "raw_class": "person",
                "confidence": 94.5,
                "bbox": [x1, y1, x2, y2],
                "norm_bbox": [x1/640, y1/360, x2/640, y2/360]
            }]
            
            total_detections_logged += len(frame_dets)
            tracked = tracker.update(frame_dets, current_time=t, frame_idx=frame_idx)
            self.assertEqual(len(tracked), 1)
            self.assertEqual(tracked[0]["track_id"], 1)

        counts = tracker.get_validated_unique_counts()
        active = tracker.get_active_tracks_count()

        print(f"\n[Test 1] 1 Person for 500 frames:")
        print(f"  Frame Detections: {total_detections_logged}")
        print(f"  Active Tracks: {active}")
        print(f"  Unique Humans: {counts['unique_humans']}")

        self.assertEqual(total_detections_logged, 500)
        self.assertEqual(active, 1)
        self.assertEqual(counts["unique_humans"], 1, "Must be exactly 1 unique human, never 500!")

    def test_3_people_simultaneous(self):
        """
        Test 2: Video contains exactly 3 people simultaneously for 60 frames.
        Expected:
        - Active Tracks: 3
        - Unique Humans: 3 (NEVER 180)
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=60, min_track_frames=4)
        
        for frame_idx in range(1, 61):
            t = frame_idx / 25.0
            dets = [
                {"class_name": "HUMAN", "confidence": 91.0, "bbox": [50, 100, 90, 200], "norm_bbox": [50/640, 100/360, 90/640, 200/360]},
                {"class_name": "HUMAN", "confidence": 88.0, "bbox": [200, 100, 240, 200], "norm_bbox": [200/640, 100/360, 240/640, 200/360]},
                {"class_name": "HUMAN", "confidence": 95.0, "bbox": [400, 100, 440, 200], "norm_bbox": [400/640, 100/360, 440/640, 200/360]},
            ]
            tracked = tracker.update(dets, current_time=t, frame_idx=frame_idx)
            self.assertEqual(len(tracked), 3)

        counts = tracker.get_validated_unique_counts()
        print(f"\n[Test 2] 3 People Simultaneous:")
        print(f"  Unique Humans: {counts['unique_humans']}")
        self.assertEqual(counts["unique_humans"], 3, "Must be exactly 3 unique humans!")

    def test_person_leaves_then_another_appears(self):
        """
        Test 3: Person A appears first, Person A leaves, Person B appears at a completely different location.
        Expected:
        - Unique Humans: 2
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=10, min_track_frames=4)

        # Person A (frames 1..30 at x=50)
        for frame_idx in range(1, 31):
            t = frame_idx / 25.0
            dets = [{"class_name": "HUMAN", "confidence": 90.0, "bbox": [50, 100, 90, 200], "norm_bbox": [50/640, 100/360, 90/640, 200/360]}]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        # Gap (frames 31..60, no detections)
        for frame_idx in range(31, 61):
            t = frame_idx / 25.0
            tracker.update([], current_time=t, frame_idx=frame_idx)

        # Person B (frames 61..90 at far right x=550)
        for frame_idx in range(61, 91):
            t = frame_idx / 25.0
            dets = [{"class_name": "HUMAN", "confidence": 92.0, "bbox": [550, 100, 590, 200], "norm_bbox": [550/640, 100/360, 590/640, 200/360]}]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        counts = tracker.get_validated_unique_counts()
        print(f"\n[Test 3] Sequential People (A leaves, B enters):")
        print(f"  Unique Humans: {counts['unique_humans']}")
        self.assertEqual(counts["unique_humans"], 2, "Must be exactly 2 unique humans!")

    def test_person_temporary_occlusion(self):
        """
        Test 4: One person disappears temporarily for 5 frames behind an obstacle and reappears.
        Expected:
        - Unique Humans: 1 (Track buffer must recover track without incrementing count).
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=30, min_track_frames=4)

        # Person visible (frames 1..20)
        for frame_idx in range(1, 21):
            t = frame_idx / 25.0
            x = 100 + frame_idx * 2
            dets = [{"class_name": "HUMAN", "confidence": 90.0, "bbox": [x, 100, x+40, 200], "norm_bbox": [x/640, 100/360, (x+40)/640, 200/360]}]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        # Occluded (frames 21..25)
        for frame_idx in range(21, 26):
            t = frame_idx / 25.0
            tracker.update([], current_time=t, frame_idx=frame_idx)

        # Reappears (frames 26..50)
        for frame_idx in range(26, 51):
            t = frame_idx / 25.0
            x = 100 + frame_idx * 2
            dets = [{"class_name": "HUMAN", "confidence": 90.0, "bbox": [x, 100, x+40, 200], "norm_bbox": [x/640, 100/360, (x+40)/640, 200/360]}]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        counts = tracker.get_validated_unique_counts()
        print(f"\n[Test 4] Person Occlusion & Recovery:")
        print(f"  Unique Humans: {counts['unique_humans']}")
        self.assertEqual(counts["unique_humans"], 1, "Must maintain identity across occlusion and report exactly 1 unique human!")

    def test_vehicles_and_animals_only(self):
        """
        Test 5: Video contains vehicles and animals but no people.
        Expected:
        - Unique Humans: 0
        - Unique Vehicles: 1
        - Unique Animals: 1
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=30, min_track_frames=4)

        for frame_idx in range(1, 31):
            t = frame_idx / 25.0
            dets = [
                {"class_name": "VEHICLE", "confidence": 94.0, "bbox": [50, 150, 180, 240], "norm_bbox": [50/640, 150/360, 180/640, 240/360]},
                {"class_name": "ANIMAL", "confidence": 85.0, "bbox": [350, 200, 390, 250], "norm_bbox": [350/640, 200/360, 390/640, 250/360]}
            ]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        counts = tracker.get_validated_unique_counts()
        print(f"\n[Test 5] Vehicles and Animals Only:")
        print(f"  Unique Humans: {counts['unique_humans']}")
        print(f"  Unique Vehicles: {counts['unique_vehicles']}")
        print(f"  Unique Animals: {counts['unique_animals']}")

        self.assertEqual(counts["unique_humans"], 0, "No humans present, must be 0!")
        self.assertEqual(counts["unique_vehicles"], 1)
        self.assertEqual(counts["unique_animals"], 1)

    def test_1_person_plus_1_vehicle(self):
        """
        Test 6: Video contains 1 person + 1 vehicle.
        Expected:
        - Unique Humans: 1
        - Unique Vehicles: 1
        - Total Unique: 2
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=30, min_track_frames=4)

        for frame_idx in range(1, 31):
            t = frame_idx / 25.0
            dets = [
                {"class_name": "HUMAN", "confidence": 92.0, "bbox": [100, 100, 140, 200], "norm_bbox": [100/640, 100/360, 140/640, 200/360]},
                {"class_name": "VEHICLE", "confidence": 89.0, "bbox": [300, 150, 450, 250], "norm_bbox": [300/640, 150/360, 450/640, 250/360]}
            ]
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        counts = tracker.get_validated_unique_counts()
        print(f"\n[Test 6] 1 Person + 1 Vehicle:")
        print(f"  Unique Humans: {counts['unique_humans']}")
        print(f"  Unique Vehicles: {counts['unique_vehicles']}")

        self.assertEqual(counts["unique_humans"], 1)
        self.assertEqual(counts["unique_vehicles"], 1)
        self.assertEqual(counts["total_unique_objects"], 2)

    def test_transient_single_frame_noise_rejected(self):
        """
        Test 7: Transient 1-frame false detection noise.
        Expected:
        - Unique Humans: 0 (filtered out by min_track_frames validation threshold).
        """
        tracker = MultiObjectTracker(high_conf_thresh=0.50, low_conf_thresh=0.20, match_thresh=0.25, track_buffer=10, min_track_frames=4)

        # Single frame flicker at frame 5
        for frame_idx in range(1, 15):
            t = frame_idx / 25.0
            dets = []
            if frame_idx == 5:
                dets.append({"class_name": "HUMAN", "confidence": 52.0, "bbox": [100, 100, 140, 200], "norm_bbox": [100/640, 100/360, 140/640, 200/360]})
            tracker.update(dets, current_time=t, frame_idx=frame_idx)

        counts = tracker.get_validated_unique_counts()
        audit_records = tracker.get_track_audit_records()

        print(f"\n[Test 7] Transient False Positive Rejection:")
        print(f"  Unique Humans: {counts['unique_humans']}")
        print(f"  Track Audit Status: {audit_records[0]['status'] if audit_records else 'N/A'}")

        self.assertEqual(counts["unique_humans"], 0, "Transient 1-frame noise must be rejected!")
        self.assertEqual(audit_records[0]["status"], "REJECTED_NOISE")

if __name__ == "__main__":
    unittest.main()
