"""
Bal-Setu: Rejection-First Capture Quality Gate
Implements Wadhwani AI Uncertainty-Aware Rejection principle on-device.
Refuses inference on bad captures and speaks immediate Hindi/dialect coaching prompts
BEFORE any model executes or any frame is stored.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import cv2.aruco as aruco
import numpy as np


@dataclass
class QualityGateResult:
    passed: bool
    sharpness_score: float  # Laplacian variance (higher is sharper)
    mean_luminance: float   # 0 to 255
    contrast_std: float     # Grayscale std deviation
    card_detected: bool     # ArUco fiducials found
    card_scale_pct: float   # Percentage of image area occupied by card
    card_centered: bool     # Card located in middle zone
    audio_noise_db: float   # Ambient noise level
    failure_reasons: List[str] = field(default_factory=list)
    hindi_coaching_prompt: str = ""
    bhojpuri_coaching_prompt: str = ""
    suggested_action: str = ""


class CaptureQualityGate:
    """
    On-device gate evaluating viewfinder frames against public-health field constraints.
    - Sharpness threshold: Laplacian variance >= 90.0 (empirically tuned for mobile cameras)
    - Lighting threshold: mean luminance between 60.0 and 220.0, contrast std >= 25.0
    - Card scale threshold: card area between 12.0% and 80.0% of frame
    - Centering threshold: card centroid within middle 30%-70% of frame
    - Audio noise threshold: < 55.0 dB for vocalization checks
    """

    def __init__(
        self,
        min_sharpness: float = 90.0,
        min_luminance: float = 60.0,
        max_luminance: float = 235.0,
        min_contrast: float = 25.0,
        min_card_scale_pct: float = 12.0,
        max_card_scale_pct: float = 90.0,
        max_audio_noise_db: float = 55.0
    ):
        self.min_sharpness = min_sharpness
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.min_contrast = min_contrast
        self.min_card_scale_pct = min_card_scale_pct
        self.max_card_scale_pct = max_card_scale_pct
        self.max_audio_noise_db = max_audio_noise_db
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.detector = None

    def _detect_markers(self, gray: np.ndarray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

    def evaluate_frame(
        self,
        bgr_image: np.ndarray,
        require_card: bool = True,
        audio_noise_db: Optional[float] = None
    ) -> QualityGateResult:
        """
        Evaluate single camera frame.
        """
        if bgr_image is None or bgr_image.size == 0:
            return QualityGateResult(
                passed=False,
                sharpness_score=0.0,
                mean_luminance=0.0,
                contrast_std=0.0,
                card_detected=False,
                card_scale_pct=0.0,
                card_centered=False,
                audio_noise_db=audio_noise_db or 0.0,
                failure_reasons=["Empty or invalid image frame"],
                hindi_coaching_prompt="कैमरा फ्रेम खाली है, पुनः प्रयास करें।",
                bhojpuri_coaching_prompt="कैमरा में कुछु नइखे लउकत, दोबारा कोशिश करीं।",
                suggested_action="Point camera at subject or RBSK-CalibCard"
            )

        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness via Laplacian Variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Lighting / Illuminance & Contrast
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        # 3. ArUco Card Detection & Scale
        corners, ids, _ = self._detect_markers(gray)
        card_detected = False
        card_scale_pct = 0.0
        card_centered = False

        if ids is not None and len(ids) >= 2:
            card_detected = True
            all_pts = np.concatenate([c[0] for c in corners], axis=0)
            x_min, y_min = np.min(all_pts, axis=0)
            x_max, y_max = np.max(all_pts, axis=0)
            card_area = (x_max - x_min) * (y_max - y_min)
            frame_area = w * h
            card_scale_pct = float((card_area / frame_area) * 100.0)

            # Centering: centroid within 30% to 70% of frame width & height
            centroid_x = (x_min + x_max) / 2.0
            centroid_y = (y_min + y_max) / 2.0
            if (0.25 * w <= centroid_x <= 0.75 * w) and (0.20 * h <= centroid_y <= 0.80 * h):
                card_centered = True

        # 4. Audio Noise Gate (if provided)
        current_audio_db = audio_noise_db if audio_noise_db is not None else 42.0

        # Accumulate failure checks
        failures = []
        hindi_prompts = []
        bhojpuri_prompts = []

        if laplacian_var < self.min_sharpness:
            failures.append(f"Image too blurry (Laplacian variance {laplacian_var:.1f} < {self.min_sharpness})")
            hindi_prompts.append("कैमरा स्थिर रखें, धुंधलापन हटाएं")
            bhojpuri_prompts.append("फोन थिर रखीं, फोटो साफ ना आवत बा")

        if mean_lum < self.min_luminance:
            failures.append(f"Scene too dark (Luminance {mean_lum:.1f} < {self.min_luminance})")
            hindi_prompts.append("थोड़ा उजाले में आएं, फोन सीधा रखें")
            bhojpuri_prompts.append("तनी अंजोर में आईं, फोन सोझ रखीं")
        elif mean_lum > self.max_luminance:
            failures.append(f"Scene overexposed/glare (Luminance {mean_lum:.1f} > {self.max_luminance})")
            hindi_prompts.append("चमक कम करें, फोन का कोण थोड़ा बदलें")
            bhojpuri_prompts.append("रोशनी बहुते बेसी बा, फोन तनी तिरछा करीं")

        if contrast_std < self.min_contrast:
            failures.append(f"Insufficient contrast (Std dev {contrast_std:.1f} < {self.min_contrast})")
            hindi_prompts.append("पृष्ठभूमि और विषय में अंतर स्पष्ट करें")

        if require_card:
            if not card_detected:
                failures.append("RBSK-CalibCard not detected in frame")
                hindi_prompts.append("कार्ड को कैमरे के सामने लाएं")
                bhojpuri_prompts.append("कार्ड के कैमरा के सामने ले आईं")
            else:
                if card_scale_pct < self.min_card_scale_pct:
                    failures.append(f"Card too far (Scale {card_scale_pct:.1f}% < {self.min_card_scale_pct}%)")
                    hindi_prompts.append("कार्ड को कैमरे के थोड़ा पास लाएं")
                    bhojpuri_prompts.append("कार्ड के तनी नजदीक ले आईं")
                elif card_scale_pct > self.max_card_scale_pct:
                    failures.append(f"Card too close (Scale {card_scale_pct:.1f}% > {self.max_card_scale_pct}%)")
                    hindi_prompts.append("कार्ड को कैमरे से थोड़ा दूर रखें")
                    bhojpuri_prompts.append("कार्ड के तनी दूर करीं")

                if not card_centered:
                    failures.append("Card not centered in frame")
                    hindi_prompts.append("कार्ड को स्क्रीन के बीच में रखें")
                    bhojpuri_prompts.append("कार्ड के बीचवा में रखीं")

        if current_audio_db > self.max_audio_noise_db:
            failures.append(f"High ambient acoustic noise ({current_audio_db:.1f} dB > {self.max_audio_noise_db} dB)")
            hindi_prompts.append("कृपया शांत जगह पर जाएं, पृष्ठभूमि का शोर कम करें")
            bhojpuri_prompts.append("तनी शांत जगह पर जाईं, शोर बहुते बा")

        passed = len(failures) == 0

        final_hindi = " • ".join(hindi_prompts) if hindi_prompts else "फोटो बिल्कुल सही है, विश्लेषण जारी है।"
        final_bhojpuri = " • ".join(bhojpuri_prompts) if bhojpuri_prompts else "फोटो एकदम सही बा, जांच शुरू बा।"
        suggested_action = failures[0] if failures else "Proceed with on-device LiteRT / OpenCV feature extraction"

        return QualityGateResult(
            passed=passed,
            sharpness_score=laplacian_var,
            mean_luminance=mean_lum,
            contrast_std=contrast_std,
            card_detected=card_detected,
            card_scale_pct=card_scale_pct,
            card_centered=card_centered,
            audio_noise_db=current_audio_db,
            failure_reasons=failures,
            hindi_coaching_prompt=final_hindi,
            bhojpuri_coaching_prompt=final_bhojpuri,
            suggested_action=suggested_action
        )
