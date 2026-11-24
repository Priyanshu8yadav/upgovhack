"""
Test Rejection-First Capture Quality Gate
"""

import cv2
import numpy as np
import pytest
from core.quality_gate import CaptureQualityGate


@pytest.fixture
def quality_gate():
    return CaptureQualityGate()


def test_blurry_frame_rejection(quality_gate):
    # Create smooth, blurred image (low Laplacian variance)
    smooth_img = np.ones((400, 400, 3), dtype=np.uint8) * 128
    smooth_img = cv2.GaussianBlur(smooth_img, (25, 25), 0)

    res = quality_gate.evaluate_frame(smooth_img, require_card=False)
    assert not res.passed
    assert any("blurry" in r.lower() for r in res.failure_reasons)
    assert "कैमरा स्थिर रखें" in res.hindi_coaching_prompt


def test_underexposed_frame_rejection(quality_gate):
    # Pitch-black or very dark image
    dark_img = np.ones((400, 400, 3), dtype=np.uint8) * 15

    res = quality_gate.evaluate_frame(dark_img, require_card=False)
    assert not res.passed
    assert any("dark" in r.lower() for r in res.failure_reasons)
    assert "उजाले में आएं" in res.hindi_coaching_prompt


def test_overexposed_frame_rejection(quality_gate):
    # Extremely bright, blown-out image
    bright_img = np.ones((400, 400, 3), dtype=np.uint8) * 250

    res = quality_gate.evaluate_frame(bright_img, require_card=False)
    assert not res.passed
    assert any("overexposed" in r.lower() for r in res.failure_reasons)
    assert "चमक कम करें" in res.hindi_coaching_prompt


def test_calibcard_front_acceptance(quality_gate):
    # Load canonical CalibCard front image
    front_img = cv2.imread("calibcard/calibcard_front.png")
    assert front_img is not None

    res = quality_gate.evaluate_frame(front_img, require_card=True)
    assert res.passed
    assert res.card_detected
    assert len(res.failure_reasons) == 0
