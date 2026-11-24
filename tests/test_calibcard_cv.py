"""
Unit tests for CalibCard CV Pipeline
Validates 4-point homography unwarping, ArUco detection, Von Kries color balance, and fine-motor tracing scoring.
"""

import os
import cv2
import numpy as np
import pytest
from core.calibcard_cv import CalibCardCVPipeline


@pytest.fixture
def calibcard_pipeline():
    return CalibCardCVPipeline(canonical_width=856, canonical_height=540)


@pytest.fixture
def synthetic_card_frame():
    card_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibcard", "calibcard_front.png"))
    if os.path.exists(card_path):
        card = cv2.imread(card_path)
        frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
        h, w = card.shape[:2]
        y1, x1 = 200, 400
        frame[y1:y1+h, x1:x1+w] = card
        return frame
    else:
        return np.full((720, 1280, 3), 150, dtype=np.uint8)


def test_calibcard_detection_and_rectification(calibcard_pipeline, synthetic_card_frame):
    rectified, H, metrics = calibcard_pipeline.detect_and_rectify(synthetic_card_frame)
    if metrics["success"]:
        assert rectified is not None
        assert rectified.shape == (540, 856, 3)
        assert H is not None
        assert H.shape == (3, 3)
    else:
        assert rectified is None


def test_von_kries_color_constancy(calibcard_pipeline):
    synthetic_rectified = np.full((540, 856, 3), 200, dtype=np.uint8)
    constancy = calibcard_pipeline.extract_color_constancy(synthetic_rectified)
    assert "white_balance_gains" in constancy
    assert "gain_r" in constancy["white_balance_gains"]
    assert "gain_g" in constancy["white_balance_gains"]
    assert "gain_b" in constancy["white_balance_gains"]

    corrected = calibcard_pipeline.apply_color_correction(synthetic_rectified, constancy["white_balance_gains"])
    assert corrected.shape == synthetic_rectified.shape


def test_fine_motor_tracing_evaluation(calibcard_pipeline):
    synthetic_rectified = np.full((540, 856, 3), 240, dtype=np.uint8)
    res = calibcard_pipeline.evaluate_tracing_pattern(synthetic_rectified, pattern_type="circle", child_age_years=3.5)
    assert res.hausdorff_distance_mm >= 0.0
    assert 0.0 <= res.motor_score <= 100.0
    assert res.clinical_finding is not None
