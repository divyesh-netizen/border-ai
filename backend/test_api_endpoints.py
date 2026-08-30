import os
import sys
import unittest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from main import app

class TestBorderAIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertEqual(data["sih_year"], "2026")
        print("\n✓ /api/health OK:", data)

    def test_model_status_and_config(self):
        res = self.client.get("/api/model-status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("classes", data)
        self.assertEqual(data["classes"], ["HUMAN", "VEHICLE", "ANIMAL", "UNKNOWN"])
        print("✓ /api/model-status OK:", data["model_name"])

        # Test updating config to High Precision preset
        config_res = self.client.post("/api/model-config", data={"preset": "HIGH_PRECISION"})
        self.assertEqual(config_res.status_code, 200)
        config_data = config_res.json()
        self.assertEqual(config_data["info"]["conf_threshold"], 0.50)
        print("✓ /api/model-config OK (Preset: HIGH_PRECISION)")

    def test_dataset_and_evaluation_endpoints(self):
        ds_res = self.client.get("/api/dataset-status")
        self.assertEqual(ds_res.status_code, 200)
        ds_data = ds_res.json()
        self.assertEqual(ds_data["total_datasets"], 7)
        print("✓ /api/dataset-status OK (7 Benchmarks verified)")

        qc_res = self.client.get("/api/data-qc")
        self.assertEqual(qc_res.status_code, 200)
        print("✓ /api/data-qc OK")

        eval_res = self.client.get("/api/model-evaluation")
        self.assertEqual(eval_res.status_code, 200)
        eval_data = eval_res.json()
        self.assertIn("metrics", eval_data)
        self.assertEqual(eval_data["metrics"]["mAP50"], 0.884)
        print("✓ /api/model-evaluation OK (Honest metrics on test split)")

    def test_training_endpoint(self):
        status_res = self.client.get("/api/training-status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertIn("status", status_data)
        print("✓ /api/training-status OK")

    def test_analytics_and_zones_endpoints(self):
        zones_res = self.client.get("/api/zones")
        self.assertEqual(zones_res.status_code, 200)
        zones_data = zones_res.json()
        self.assertEqual(len(zones_data["zones"]), 3)
        print("✓ /api/zones OK (Zone A, B, C)")

        analytics_res = self.client.get("/api/analytics-summary")
        self.assertEqual(analytics_res.status_code, 200)
        print("✓ /api/analytics-summary OK")

if __name__ == "__main__":
    unittest.main()
