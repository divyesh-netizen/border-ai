import os
import time
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional, Tuple
from video_quality import VideoQualityAnalyzer

# Strict Canonical Class Sets
COCO_HUMAN_CLASSES = {"person", "pedestrian", "human", "patrol", "soldier"}

COCO_VEHICLE_CLASSES = {
    "bicycle", "car", "motorcycle", "motorbike", "airplane", "bus", "train", "truck", "boat", "van", "vehicle"
}

COCO_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal", "wildlife", "deer"
}

UNIFIED_CLASSES = ["HUMAN", "VEHICLE", "ANIMAL", "UNKNOWN"]

CLASS_COLORS = {
    "HUMAN": "#EF4444",   # Tactical Red
    "VEHICLE": "#3B82F6", # Tactical Blue
    "ANIMAL": "#10B981",  # Tactical Green
    "UNKNOWN": "#F59E0B", # Tactical Amber
}

PRESETS = {
    "HIGH_PRECISION": {
        "PERSON_THRESHOLD": 0.65,
        "VEHICLE_THRESHOLD": 0.60,
        "ANIMAL_THRESHOLD": 0.60,
        "UNKNOWN_THRESHOLD": 0.45,
        "min_confirmation_frames": 8
    },
    "BALANCED": {
        "PERSON_THRESHOLD": 0.50,
        "VEHICLE_THRESHOLD": 0.50,
        "ANIMAL_THRESHOLD": 0.50,
        "UNKNOWN_THRESHOLD": 0.35,
        "min_confirmation_frames": 5
    },
    "HIGH_RECALL": {
        "PERSON_THRESHOLD": 0.35,
        "VEHICLE_THRESHOLD": 0.35,
        "ANIMAL_THRESHOLD": 0.35,
        "UNKNOWN_THRESHOLD": 0.25,
        "min_confirmation_frames": 3
    }
}

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0

    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)

def apply_global_nms(detections: List[Dict[str, Any]], iou_threshold: float = 0.50) -> List[Dict[str, Any]]:
    """Performs class-aware Non-Maximum Suppression (NMS) across merged detections."""
    if not detections:
        return []

    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept_dets = []

    for det in sorted_dets:
        box = det["bbox"]
        cls = det["class"]
        
        # Check overlap with already kept detections of the same class
        overlap = False
        for kept in kept_dets:
            if kept["class"] == cls:
                iou = calculate_iou(box, kept["bbox"])
                if iou >= iou_threshold:
                    overlap = True
                    break
        if not overlap:
            kept_dets.append(det)

    return kept_dets

class ModelAdapter:
    """
    High-Precision Multi-Scale Computer Vision Adapter for Surveillance.
    Supports:
    - Real-Time Mode vs High Accuracy Tiled SAHI Detection (small distant humans)
    - Class-specific thresholds (PERSON, VEHICLE, ANIMAL)
    - Controlled CCTV video quality preprocessing (CLAHE, Denoising, Gamma)
    - Strict classification: never maps UNKNOWN to HUMAN
    """
    def __init__(
        self,
        model_path: Optional[str] = "yolov8n.pt",
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        inference_mode: str = "BALANCED" # HIGH_PRECISION, BALANCED, HIGH_RECALL
    ):
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_real_ai = False
        
        # Class-specific thresholds
        self.thresholds = dict(PRESETS.get(inference_mode, PRESETS["BALANCED"]))
        self.conf_threshold = conf_threshold
        self.inference_mode = inference_mode
        self.tiled_inference_enabled = False

        self.model_info = {
            "model_name": "YOLOv8 Surveillance Engine",
            "model_type": "YOLOv8 Object Detector",
            "weights_path": model_path,
            "device": self.device.upper(),
            "status": "INITIALIZING",
            "classes": UNIFIED_CLASSES,
            "thresholds": self.thresholds,
            "iou_threshold": iou_threshold,
            "inference_mode": self.inference_mode,
            "framework": f"PyTorch {torch.__version__} / Ultralytics"
        }
        self.load_model(model_path)

    def load_model(self, model_path: Optional[str] = None):
        if model_path:
            self.model_path = model_path
            self.model_info["weights_path"] = model_path

        try:
            from ultralytics import YOLO
            path_to_load = self.model_path if (self.model_path and os.path.exists(self.model_path)) else "yolov8n.pt"
            print(f"[ModelAdapter] Loading YOLO weights from: {path_to_load} on {self.device.upper()}")
            self.model = YOLO(path_to_load)
            self.is_real_ai = True
            self.model_info["status"] = "LOADED_REAL_AI"
            self.model_info["model_name"] = os.path.basename(path_to_load)
            print(f"[ModelAdapter] Model loaded successfully on {self.device.upper()}.")
        except Exception as e:
            print(f"[ModelAdapter] Failed to load YOLO model: {e}")
            self.is_real_ai = False
            self.model = None
            self.model_info["status"] = "MODEL_UNAVAILABLE"

    def set_preset(self, preset_name: str):
        if preset_name in PRESETS:
            self.inference_mode = preset_name
            self.thresholds.update(PRESETS[preset_name])
            self.model_info["thresholds"] = self.thresholds
            self.model_info["inference_mode"] = preset_name

    def set_class_threshold(self, class_name: str, threshold: float):
        key = f"{class_name.upper()}_THRESHOLD"
        if key in self.thresholds:
            self.thresholds[key] = max(0.10, min(0.95, float(threshold)))
            self.model_info["thresholds"] = self.thresholds

    def map_class(self, raw_class_name: str) -> Tuple[str, str]:
        """
        Maps detector class name into canonical schema (HUMAN, VEHICLE, ANIMAL, UNKNOWN)
        and preserves sub-category where available (e.g. car, dog).
        """
        clean = raw_class_name.lower().strip()
        
        if clean in COCO_HUMAN_CLASSES:
            return "HUMAN", clean
        elif clean in COCO_VEHICLE_CLASSES:
            return "VEHICLE", clean
        elif clean in COCO_ANIMAL_CLASSES:
            return "ANIMAL", clean
        else:
            return "UNKNOWN", clean

    def _get_threshold_for_class(self, canonical_class: str) -> float:
        key = f"{canonical_class.upper()}_THRESHOLD"
        return self.thresholds.get(key, self.conf_threshold)

    def _slice_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates 4 overlapping quadrant tiles (2x2 grid) for small distant object detection.
        """
        h, w = frame.shape[:2]
        half_w = w // 2
        half_h = h // 2
        overlap_x = int(half_w * 0.15)
        overlap_y = int(half_h * 0.15)
        
        slices = [
            # Top-Left
            {"crop": frame[0:min(h, half_h + overlap_y), 0:min(w, half_w + overlap_x)], "x_offset": 0, "y_offset": 0},
            # Top-Right
            {"crop": frame[0:min(h, half_h + overlap_y), max(0, half_w - overlap_x):w], "x_offset": max(0, half_w - overlap_x), "y_offset": 0},
            # Bottom-Left
            {"crop": frame[max(0, half_h - overlap_y):h, 0:min(w, half_w + overlap_x)], "x_offset": 0, "y_offset": max(0, half_h - overlap_y)},
            # Bottom-Right
            {"crop": frame[max(0, half_h - overlap_y):h, max(0, half_w - overlap_x):w], "x_offset": max(0, half_w - overlap_x), "y_offset": max(0, half_h - overlap_y)}
        ]
        return slices

    def predict(
        self,
        frame: np.ndarray,
        is_thermal: bool = False,
        mode_override: Optional[str] = None,
        use_tiled: bool = False,
        quality_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Runs high-precision multi-scale object detection on a video frame.
        """
        if frame is None or frame.size == 0 or self.model is None:
            return []

        h, w = frame.shape[:2]
        all_raw_detections = []

        # 1. Adaptive Quality Preprocessing (evidence preserving, no hallucination)
        proc_frame = VideoQualityAnalyzer.preprocess_frame(
            frame=frame,
            quality_info=quality_info,
            is_thermal=is_thermal,
            apply_enhancement=True
        )

        min_thresh = min(self.thresholds.values())

        try:
            # 2. Full Frame Inference
            full_results = self.model.predict(
                source=proc_frame,
                conf=min_thresh,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                imgsz=640
            )

            for r in full_results:
                boxes = r.boxes
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    raw_name = self.model.names[cls_id]
                    canonical_class, sub_type = self.map_class(raw_name)

                    # Strict class-specific threshold filtering
                    required_conf = self._get_threshold_for_class(canonical_class)
                    if conf < required_conf:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1 = max(0, min(w, int(xyxy[0])))
                    y1 = max(0, min(h, int(xyxy[1])))
                    x2 = max(0, min(w, int(xyxy[2])))
                    y2 = max(0, min(h, int(xyxy[3])))

                    if (x2 - x1) < 6 or (y2 - y1) < 6:
                        continue

                    norm_bbox = [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]
                    all_raw_detections.append({
                        "class": canonical_class,
                        "sub_type": sub_type,
                        "raw_class": raw_name,
                        "confidence": round(conf * 100, 1),
                        "bbox": [x1, y1, x2, y2],
                        "norm_bbox": norm_bbox,
                        "color": CLASS_COLORS.get(canonical_class, CLASS_COLORS["UNKNOWN"]),
                        "source": "FULL_FRAME"
                    })

            # 3. Tiled Inference for Small / Distant Pedestrians (High Accuracy Mode)
            if use_tiled and (w >= 720 or h >= 540):
                slices = self._slice_frame(proc_frame)
                for s in slices:
                    crop = s["crop"]
                    x_off = s["x_offset"]
                    y_off = s["y_offset"]

                    tile_results = self.model.predict(
                        source=crop,
                        conf=min_thresh,
                        iou=self.iou_threshold,
                        device=self.device,
                        verbose=False,
                        imgsz=480
                    )

                    for r in tile_results:
                        for box in r.boxes:
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            raw_name = self.model.names[cls_id]
                            canonical_class, sub_type = self.map_class(raw_name)

                            required_conf = self._get_threshold_for_class(canonical_class)
                            if conf < required_conf:
                                continue

                            t_xyxy = box.xyxy[0].cpu().numpy().tolist()
                            x1 = max(0, min(w, int(t_xyxy[0] + x_off)))
                            y1 = max(0, min(h, int(t_xyxy[1] + y_off)))
                            x2 = max(0, min(w, int(t_xyxy[2] + x_off)))
                            y2 = max(0, min(h, int(t_xyxy[3] + y_off)))

                            if (x2 - x1) < 6 or (y2 - y1) < 6:
                                continue

                            norm_bbox = [round(x1 / w, 4), round(y1 / h, 4), round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]
                            all_raw_detections.append({
                                "class": canonical_class,
                                "sub_type": sub_type,
                                "raw_class": raw_name,
                                "confidence": round(conf * 100, 1),
                                "bbox": [x1, y1, x2, y2],
                                "norm_bbox": norm_bbox,
                                "color": CLASS_COLORS.get(canonical_class, CLASS_COLORS["UNKNOWN"]),
                                "source": "TILED_SAHI"
                            })

            # 4. Global Class-Aware NMS Fusion
            final_detections = apply_global_nms(all_raw_detections, iou_threshold=self.iou_threshold)
            return final_detections

        except Exception as e:
            print(f"[ModelAdapter] Prediction error: {e}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        return dict(self.model_info)
