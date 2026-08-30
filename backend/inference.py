import os
import time
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional

# Strict COCO Taxonomy Mapping to Project Schema
# Only actual person / pedestrian classes map to HUMAN.
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

class ModelAdapter:
    """
    Pure Model-Based Object Detection Architecture for real-time surveillance.
    Uses verified YOLOv8 PyTorch deep learning weights without background heuristic contour guessing.
    Strictly isolates HUMAN, VEHICLE, and ANIMAL classes with configurable confidence thresholds.
    """
    def __init__(self, model_path: Optional[str] = "yolov8n.pt", conf_threshold: float = 0.50, iou_threshold: float = 0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = 640
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.is_real_ai = False
        
        self.model_info = {
            "model_name": "YOLOv8 Deep Learning Surveillance Engine",
            "model_type": "YOLOv8 Object Detector",
            "weights_path": model_path,
            "device": self.device.upper(),
            "status": "INITIALIZING",
            "classes": UNIFIED_CLASSES,
            "input_resolution": f"{self.imgsz}x{self.imgsz}",
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
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
            print(f"[ModelAdapter] Loading YOLO model from: {path_to_load} on {self.device.upper()}")
            self.model = YOLO(path_to_load)
            self.is_real_ai = True
            self.model_info["status"] = "LOADED_REAL_AI"
            self.model_info["model_name"] = os.path.basename(path_to_load)
            print(f"[ModelAdapter] Real YOLO model loaded successfully on {self.device.upper()}.")
        except Exception as e:
            print(f"[ModelAdapter] Failed to load YOLO model: {e}")
            self.is_real_ai = False
            self.model = None
            self.model_info["status"] = "MODEL_UNAVAILABLE"

    def set_confidence_threshold(self, conf: float):
        self.conf_threshold = max(0.10, min(0.95, float(conf)))
        self.model_info["conf_threshold"] = self.conf_threshold

    def map_class(self, raw_class_name: str, confidence: float) -> str:
        """
        Maps raw detector class name into strict project schema.
        Never maps unknown objects, shadows, trees, or noise to HUMAN.
        """
        clean = raw_class_name.lower().strip()
        
        if clean in COCO_HUMAN_CLASSES:
            return "HUMAN"
        elif clean in COCO_VEHICLE_CLASSES:
            return "VEHICLE"
        elif clean in COCO_ANIMAL_CLASSES:
            return "ANIMAL"
        else:
            return "UNKNOWN"

    def predict(self, frame: np.ndarray, is_thermal: bool = False, mode_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs object detection inference on a video frame.
        Applies confidence filtering and NMS IoU suppression.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections = []

        if (mode_override != "DEMO") and (self.model is not None):
            try:
                # Preprocess low-light / thermal frame if requested
                proc_frame = frame
                if is_thermal:
                    # Enhance thermal dynamic range
                    proc_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

                results = self.model.predict(
                    source=proc_frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().tolist()
                        x1, y1, x2, y2 = [int(v) for v in coords]
                        conf = float(box.conf[0].cpu().numpy())
                        cls_idx = int(box.cls[0].cpu().numpy())
                        raw_name = r.names.get(cls_idx, "unknown")

                        # Strictly map to taxonomy
                        unified_cls = self.map_class(raw_name, conf)
                        
                        # Clip bounding box to frame bounds
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(x1 + 1, min(w, x2))
                        y2 = max(y1 + 1, min(h, y2))
                        
                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)

                        # Filter out tiny sub-pixel artifacts (< 8px height)
                        if bh < 8 or bw < 4:
                            continue

                        detections.append({
                            "class_name": unified_cls,
                            "raw_class": raw_name,
                            "confidence": round(conf * 100, 1),
                            "bbox": [x1, y1, x2, y2],
                            "norm_bbox": [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)],
                            "width_px": bw,
                            "height_px": bh,
                            "size_str": f"{bw}x{bh}px",
                            "area_px": bw * bh,
                            "color": CLASS_COLORS.get(unified_cls, "#F59E0B")
                        })
            except Exception as e:
                print(f"[ModelAdapter] Predict error: {e}")

        return detections

    def get_classes(self) -> List[str]:
        return UNIFIED_CLASSES

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info

    def update_config(self, conf: Optional[float] = None, iou: Optional[float] = None, weights_path: Optional[str] = None):
        if conf is not None:
            self.set_confidence_threshold(conf)
        if iou is not None:
            self.iou_threshold = max(0.10, min(0.95, float(iou)))
            self.model_info["iou_threshold"] = self.iou_threshold
        if weights_path is not None and weights_path != self.model_path:
            self.load_model(weights_path)
