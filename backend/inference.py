import os
import time
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional, Tuple

try:
    from backend.video_quality import VideoQualityAnalyzer
except ImportError:
    from video_quality import VideoQualityAnalyzer

# ---------------------------------------------------------------------------
# Class Mapping Tables (COCO-compatible)
# ---------------------------------------------------------------------------
COCO_HUMAN_CLASSES = {"person", "pedestrian", "human", "patrol", "soldier"}

COCO_VEHICLE_CLASSES = {
    "bicycle", "car", "motorcycle", "motorbike", "airplane",
    "bus", "train", "truck", "boat", "van", "vehicle"
}

COCO_ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "animal", "wildlife", "deer"
}

UNIFIED_CLASSES = ["HUMAN", "VEHICLE", "ANIMAL", "UNKNOWN"]

CLASS_COLORS = {
    "HUMAN":   "#EF4444",
    "VEHICLE": "#3B82F6",
    "ANIMAL":  "#10B981",
    "UNKNOWN": "#F59E0B",
}

PRESETS = {
    "HIGH_PRECISION": {
        "PERSON_THRESHOLD":  0.60,
        "VEHICLE_THRESHOLD": 0.55,
        "ANIMAL_THRESHOLD":  0.55,
        "UNKNOWN_THRESHOLD": 0.45,
        "min_confirmation_frames": 8
    },
    "BALANCED": {
        "PERSON_THRESHOLD":  0.40,
        "VEHICLE_THRESHOLD": 0.40,
        "ANIMAL_THRESHOLD":  0.40,
        "UNKNOWN_THRESHOLD": 0.30,
        "min_confirmation_frames": 4
    },
    "HIGH_RECALL": {
        "PERSON_THRESHOLD":  0.30,
        "VEHICLE_THRESHOLD": 0.30,
        "ANIMAL_THRESHOLD":  0.30,
        "UNKNOWN_THRESHOLD": 0.25,
        "min_confirmation_frames": 2
    }
}


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0
    aA = max(1.0, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    aB = max(1.0, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return interArea / float(aA + aB - interArea)


def apply_global_nms(detections, iou_threshold=0.50):
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)
    kept = []
    for det in sorted_dets:
        box = det["bbox"]; cls = det["class"]
        overlap = any(k["class"]==cls and calculate_iou(box, k["bbox"])>=iou_threshold for k in kept)
        if not overlap:
            kept.append(det)
    return kept


class ModelAdapter:
    def __init__(self, model_path=None, conf_threshold=0.40, iou_threshold=0.45, inference_mode="BALANCED"):
        self.model_path = model_path or "yolov8n.pt"
        self.weights_path = self.model_path
        self.iou_threshold = iou_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.hog = None
        self.is_real_ai = False
        preset = PRESETS.get(inference_mode, PRESETS["BALANCED"])
        self.thresholds = dict(preset)
        self.conf_threshold = conf_threshold
        self.inference_mode = inference_mode
        self.tiled_inference_enabled = False
        self.model_info = {
            "model_name": "YOLOv8 Surveillance Engine",
            "model_type": "YOLOv8 Object Detector",
            "weights_path": self.model_path,
            "device": self.device.upper(),
            "status": "INITIALIZING",
            "classes": UNIFIED_CLASSES,
            "thresholds": self.thresholds,
            "iou_threshold": iou_threshold,
            "inference_mode": self.inference_mode,
            "framework": f"PyTorch {torch.__version__} / Ultralytics",
        }
        self.load_model(self.model_path)

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
            self.model_info["weights_path"] = model_path
        try:
            from ultralytics import YOLO
            path = self.model_path if (self.model_path and os.path.exists(self.model_path)) else "yolov8n.pt"
            print(f"[ModelAdapter] Loading YOLO weights from: {path} on {self.device.upper()}")
            self.model = YOLO(path)
            self.is_real_ai = True
            self.model_info["status"] = "LOADED_REAL_AI"
            self.model_info["model_name"] = os.path.basename(path)
            print(f"[ModelAdapter] Model loaded successfully on {self.device.upper()}.")
        except Exception as e:
            print(f"[ModelAdapter] Failed to load YOLO model: {e}")
            self.is_real_ai = False
            self.model = None
            self.model_info["status"] = "MODEL_UNAVAILABLE"

    def set_preset(self, preset_name):
        p = preset_name.upper()
        preset = PRESETS.get(p, PRESETS["BALANCED"])
        self.thresholds = dict(preset)
        self.iou_threshold = 0.40 if p=="HIGH_PRECISION" else (0.50 if p=="HIGH_RECALL" else 0.45)

    def set_class_threshold(self, class_name, value):
        key_map = {"PERSON":"PERSON_THRESHOLD","HUMAN":"PERSON_THRESHOLD","VEHICLE":"VEHICLE_THRESHOLD","ANIMAL":"ANIMAL_THRESHOLD"}
        key = key_map.get(class_name.upper())
        if key:
            self.thresholds[key] = max(0.10, min(0.95, float(value)))

    def set_thresholds(self, person, vehicle, animal, iou):
        self.thresholds["PERSON_THRESHOLD"] = float(person)
        self.thresholds["VEHICLE_THRESHOLD"] = float(vehicle)
        self.thresholds["ANIMAL_THRESHOLD"] = float(animal)
        self.iou_threshold = float(iou)

    def _get_threshold_for_class(self, canonical_class):
        key_map = {"HUMAN":"PERSON_THRESHOLD","VEHICLE":"VEHICLE_THRESHOLD","ANIMAL":"ANIMAL_THRESHOLD","UNKNOWN":"UNKNOWN_THRESHOLD"}
        return self.thresholds.get(key_map.get(canonical_class, "UNKNOWN_THRESHOLD"), 0.35)

    def map_class(self, raw_class_name):
        n = raw_class_name.strip().lower()
        if n in COCO_HUMAN_CLASSES:   return "HUMAN", n
        if n in COCO_VEHICLE_CLASSES: return "VEHICLE", n
        if n in COCO_ANIMAL_CLASSES:  return "ANIMAL", n
        return "UNKNOWN", n

    def _predict_opencv_hog(self, frame):
        h, w = frame.shape[:2]
        detections = []
        try:
            if self.hog is None:
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            scale_w = min(1.0, 640.0 / w)
            proc = cv2.resize(frame, (0, 0), fx=scale_w, fy=scale_w) if scale_w < 1.0 else frame
            rects, weights = self.hog.detectMultiScale(proc, winStride=(8,8), padding=(8,8), scale=1.05)
            inv = 1.0 / scale_w
            for i, (rx, ry, rw, rh) in enumerate(rects):
                x1=max(0,min(w,int(rx*inv))); y1=max(0,min(h,int(ry*inv)))
                x2=max(0,min(w,int((rx+rw)*inv))); y2=max(0,min(h,int((ry+rh)*inv)))
                wt = float(weights[i]) if (weights is not None and i < len(weights)) else 0.5
                conf = min(95.0, max(40.0, round(wt*100, 1)))
                bw, bh = x2-x1, y2-y1
                norm_bbox = [round(x1/w,4), round(y1/h,4), round(bw/w,4), round(bh/h,4)]
                detections.append({"class":"HUMAN","sub_type":"person (opencv hog)","raw_class":"person",
                    "confidence":conf,"bbox":[x1,y1,x2,y2],"norm_bbox":norm_bbox,"color":CLASS_COLORS["HUMAN"],"source":"OPENCV_HOG_FALLBACK"})
        except Exception as e:
            print(f"[ModelAdapter] OpenCV HOG fallback error: {e}")
        return detections

    def _slice_frame(self, frame, tile_size=640, overlap=0.2):
        h, w = frame.shape[:2]
        step = int(tile_size * (1 - overlap))
        slices = []
        y = 0
        while y < h:
            x = 0; y2 = min(h, y + tile_size)
            while x < w:
                x2 = min(w, x + tile_size)
                slices.append({"crop": frame[y:y2, x:x2], "x_offset": x, "y_offset": y})
                x += step
            y += step
        return slices

    def predict(self, frame, is_thermal=False, mode_override=None, use_tiled=False,
                quality_info=None, person_only=False):
        if frame is None or frame.size == 0:
            return []
        if self.model is None:
            self.model_info["status"] = "OPENCV_HOG_FALLBACK"
            return self._predict_opencv_hog(frame)

        h, w = frame.shape[:2]
        all_raw = []
        is_high_accuracy = (mode_override == "HIGH_ACCURACY")

        if is_high_accuracy:
            try:
                proc_frame = VideoQualityAnalyzer.preprocess_frame(
                    frame=frame, quality_info=quality_info, is_thermal=is_thermal, apply_enhancement=True)
            except Exception:
                proc_frame = frame
        else:
            proc_frame = frame

        try:
            results = self.model.predict(source=proc_frame, conf=0.20, iou=self.iou_threshold,
                                         device=self.device, verbose=False, imgsz=640)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    raw_name = self.model.names[cls_id]
                    canonical_class, sub_type = self.map_class(raw_name)
                    if canonical_class == "UNKNOWN":
                        continue
                    if person_only and canonical_class != "HUMAN":
                        continue
                    if conf < self._get_threshold_for_class(canonical_class):
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1=max(0,min(w-1,int(xyxy[0]))); y1=max(0,min(h-1,int(xyxy[1])))
                    x2=max(0,min(w,int(xyxy[2]))); y2=max(0,min(h,int(xyxy[3])))
                    bw, bh = x2-x1, y2-y1
                    if bw < 8 or bh < 10:
                        continue
                    if canonical_class == "HUMAN" and (bw > 2.5 * bh) and conf < 0.80:
                        continue
                    norm_bbox = [round(x1/w,4), round(y1/h,4), round(bw/w,4), round(bh/h,4)]
                    all_raw.append({"class":canonical_class,"sub_type":sub_type,"raw_class":raw_name,
                        "confidence":round(conf*100,1),"bbox":[x1,y1,x2,y2],"norm_bbox":norm_bbox,
                        "color":CLASS_COLORS[canonical_class],"source":"FULL_FRAME"})

            if use_tiled and (w >= 720 or h >= 540):
                for s in self._slice_frame(proc_frame):
                    crop, x_off, y_off = s["crop"], s["x_offset"], s["y_offset"]
                    tile_results = self.model.predict(source=crop, conf=0.20, iou=self.iou_threshold,
                                                      device=self.device, verbose=False, imgsz=480)
                    for r in tile_results:
                        for box in r.boxes:
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            raw_name = self.model.names[cls_id]
                            canonical_class, sub_type = self.map_class(raw_name)
                            if canonical_class == "UNKNOWN":
                                continue
                            if person_only and canonical_class != "HUMAN":
                                continue
                            if conf < self._get_threshold_for_class(canonical_class):
                                continue
                            t = box.xyxy[0].cpu().numpy().tolist()
                            x1=max(0,min(w,int(t[0]+x_off))); y1=max(0,min(h,int(t[1]+y_off)))
                            x2=max(0,min(w,int(t[2]+x_off))); y2=max(0,min(h,int(t[3]+y_off)))
                            bw, bh = x2-x1, y2-y1
                            if bw < 8 or bh < 10:
                                continue
                            norm_bbox = [round(x1/w,4), round(y1/h,4), round(bw/w,4), round(bh/h,4)]
                            all_raw.append({"class":canonical_class,"sub_type":sub_type,"raw_class":raw_name,
                                "confidence":round(conf*100,1),"bbox":[x1,y1,x2,y2],"norm_bbox":norm_bbox,
                                "color":CLASS_COLORS[canonical_class],"source":"TILED_SLICE"})

            if len(all_raw) > 1:
                return apply_global_nms(all_raw, iou_threshold=self.iou_threshold)
            return all_raw

        except Exception as e:
            print(f"[ModelAdapter] YOLO inference exception: {e}. Falling back to OpenCV HOG...")
            self.model_info["status"] = "OPENCV_HOG_FALLBACK"
            return self._predict_opencv_hog(frame)

    def get_model_info(self):
        return dict(self.model_info)
