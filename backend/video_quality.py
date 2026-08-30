import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

class VideoQualityAnalyzer:
    """
    Real-time Degraded CCTV Video Quality Assessment & Adaptive Preprocessing Engine.
    Quantifies:
    - Blur Score (Laplacian variance / Tenengrad sharpness)
    - Luminance / Lux (Mean Grayscale intensity & low-light indicator)
    - Contrast Ratio (RMS standard deviation of pixel intensities)
    - Noise Level (High-frequency sensor noise / compression artifact estimation)
    - Modality (Auto-detects RGB vs Thermal LWIR based on chroma variance)
    """
    def __init__(self):
        self.quality_history = []

    @staticmethod
    def analyze_frame(frame: np.ndarray) -> Dict[str, Any]:
        """
        Extracts mathematical quality metrics from a raw surveillance frame.
        """
        if frame is None or frame.size == 0:
            return {
                "blur_score": 0.0,
                "brightness_lux": 0.0,
                "contrast_ratio": 0.0,
                "noise_level": 0.0,
                "modality": "UNKNOWN",
                "quality_label": "CORRUPT",
                "resolution": "0x0"
            }

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # 1. Blur Score via Laplacian Variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = round(float(lap.var()), 2)

        # 2. Brightness / Lux Level
        brightness_lux = round(float(np.mean(gray)), 2)

        # 3. RMS Contrast Ratio
        contrast_ratio = round(float(np.std(gray)), 2)

        # 4. Noise Level via High-Frequency Residuals
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_residual = cv2.absdiff(gray, blurred_gray)
        noise_level = round(float(np.mean(noise_residual)), 2)

        # 5. Modality Detection (RGB vs Thermal LWIR)
        modality = "VISIBLE_RGB"
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            b, g, r = cv2.split(frame)
            rg_diff = np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
            gb_diff = np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat = np.mean(hsv[:, :, 1])
            if (rg_diff < 4.0 and gb_diff < 4.0) or sat < 12.0:
                modality = "THERMAL_LWIR"
        else:
            modality = "THERMAL_LWIR"

        # 6. Overall Quality Label
        if brightness_lux < 50.0:
            quality_label = "LOW_LIGHT"
        elif blur_score < 45.0:
            quality_label = "BLURRY"
        elif noise_level > 8.0 or contrast_ratio < 25.0:
            quality_label = "COMPRESSED"
        elif blur_score < 80.0:
            quality_label = "MILD_DEGRADATION"
        else:
            quality_label = "CLEAN"

        return {
            "blur_score": blur_score,
            "brightness_lux": brightness_lux,
            "contrast_ratio": contrast_ratio,
            "noise_level": noise_level,
            "modality": modality,
            "quality_label": quality_label,
            "resolution": f"{w}x{h}"
        }

    @staticmethod
    def preprocess_frame(
        frame: np.ndarray,
        quality_info: Optional[Dict[str, Any]] = None,
        is_thermal: bool = False,
        apply_enhancement: bool = True
    ) -> np.ndarray:
        """
        Controlled, evidence-preserving CCTV video restoration.
        Enhances contrast and reduces noise without hallucinating artificial edges.
        """
        if frame is None or frame.size == 0 or not apply_enhancement:
            return frame

        if quality_info is None:
            quality_info = VideoQualityAnalyzer.analyze_frame(frame)

        processed = frame.copy()
        modality = quality_info.get("modality", "VISIBLE_RGB")
        quality_label = quality_info.get("quality_label", "CLEAN")
        brightness = quality_info.get("brightness_lux", 128.0)

        # 1. Thermal Preprocessing (Dynamic range expansion)
        if is_thermal or modality == "THERMAL_LWIR":
            if len(processed.shape) == 3:
                gray_t = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                norm_t = cv2.normalize(gray_t, None, 0, 255, cv2.NORM_MINMAX)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_t = clahe.apply(norm_t)
                processed = cv2.cvtColor(enhanced_t, cv2.COLOR_GRAY2BGR)
            return processed

        # 2. Low-Light Night Enhancement (CLAHE on L-channel in LAB space + Gamma)
        if quality_label == "LOW_LIGHT" or brightness < 60.0:
            try:
                lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Adaptive CLAHE based on darkness level
                clip_lim = 3.0 if brightness < 35.0 else 2.0
                clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=(8, 8))
                l_eq = clahe.apply(l)
                
                # Gentle Gamma Correction (prevents washing out / blowout)
                gamma = 1.35 if brightness < 40.0 else 1.18
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
                l_gamma = cv2.LUT(l_eq, table)
                
                lab_enhanced = cv2.merge((l_gamma, a, b))
                processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            except Exception:
                pass

        # 3. Heavy Noise / Compression Artifact Denoising (Bilateral filter)
        if quality_info.get("noise_level", 0.0) > 7.0 or quality_label == "COMPRESSED":
            try:
                processed = cv2.bilateralFilter(processed, d=5, sigmaColor=35, sigmaSpace=35)
            except Exception:
                pass

        return processed
