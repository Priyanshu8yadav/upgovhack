"""
Unit tests for Hirschberg CLR Ocular Alignment Engine
Validates automated Hirschberg geometry (CLR displacement -> Prism Diopters: Delta = 22 * d),
distinguishing pseudo-strabismus (epicanthic fold) from true esotropia, and leukocoria red reflex.
"""

import numpy as np
import pytest
from core.ocular_alignment import OcularAlignmentEngine


@pytest.fixture
def ocular_engine():
    return OcularAlignmentEngine()


def test_normal_ocular_alignment(ocular_engine):
    dummy_face = np.zeros((300, 300, 3), dtype=np.uint8)
    res = ocular_engine.evaluate_ocular_alignment(dummy_face, simulated_esotropia_mm=0.0)
    assert res.strabismus_detected is False
    assert "Green" in res.routing_tier


def test_pseudo_strabismus_differentiation(ocular_engine):
    dummy_face = np.zeros((300, 300, 3), dtype=np.uint8)
    res = ocular_engine.evaluate_ocular_alignment(dummy_face, simulated_esotropia_mm=0.4)
    assert res.strabismus_detected is False
    assert "Green" in res.routing_tier
    assert "Pseudo-strabismus" in res.hindi_explanation or "सामान्य" in res.hindi_explanation


def test_true_infantile_esotropia(ocular_engine):
    dummy_face = np.zeros((300, 300, 3), dtype=np.uint8)
    res = ocular_engine.evaluate_ocular_alignment(dummy_face, simulated_esotropia_mm=1.5)
    assert res.strabismus_detected is True
    assert res.deviation_prism_diopters == pytest.approx(33.0, rel=0.1)
    assert "Amber" in res.routing_tier


def test_congenital_cataract_leukocoria(ocular_engine):
    dummy_face = np.zeros((300, 300, 3), dtype=np.uint8)
    res = ocular_engine.evaluate_ocular_alignment(dummy_face, simulated_leukocoria_eye="OS")
    assert res.leukocoria_suspected is True
    assert "Complex Red" in res.routing_tier


def test_real_non_blank_photo_ocular_alignment(ocular_engine):
    # Non-blank realistic face frame without simulation override
    real_face = np.random.randint(60, 220, (480, 640, 3), dtype=np.uint8)
    res = ocular_engine.evaluate_ocular_alignment(real_face, simulated_esotropia_mm=0.0)
    assert res is not None
    assert isinstance(res.left_eye.displacement_px[0], float)
    assert isinstance(res.right_eye.displacement_px[0], float)
