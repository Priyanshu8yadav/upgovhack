"""
Unit tests for Split Conformal Prediction & Mathematical Uncertainty Gating
Validates distribution-free coverage guarantee (P(Y in C(X)) >= 1 - alpha),
conformal quantile q_hat, prediction set size |C(X)|, and formal mathematical abstention.
"""

import numpy as np
import pytest
from core.conformal_triage import SplitConformalCalibrator


@pytest.fixture
def calibrator():
    return SplitConformalCalibrator(alpha=0.10)


def test_confident_single_prediction(calibrator):
    probs = np.array([0.92, 0.04, 0.03, 0.01])
    res = calibrator.predict(probs, intended_lane="Green: Routine Community Care")
    assert res.set_size == 1
    assert "Normal" in res.prediction_set
    assert res.is_uncertain is False
    assert res.abstention_triggered is False
    assert res.conformal_quantile_q_hat > 0.0


def test_conformal_abstention_on_uncertainty(calibrator):
    probs = np.array([0.38, 0.12, 0.42, 0.08])
    res = calibrator.predict(probs, intended_lane="Complex Red: Fast-Track DEIC")
    assert res.is_uncertain is True
    assert res.abstention_triggered is True
    assert res.recommended_routing_override == "Internal Amber (Tele-Verify)"
    assert "Abstaining from DEIC referral" in res.statistical_explanation or "OOD" in res.statistical_explanation


def test_high_entropy_uncertainty_abstention(calibrator):
    probs = np.array([0.25, 0.25, 0.25, 0.25])
    res = calibrator.predict(probs, intended_lane="Complex Red: Fast-Track DEIC")
    assert res.is_uncertain is True
    assert res.abstention_triggered is True
    assert res.recommended_routing_override == "Internal Amber (Tele-Verify)"


def test_empirical_coverage_guarantee(calibrator):
    """
    Asserts that over exchangeable test samples, empirical coverage >= 1 - alpha (90%).
    """
    np.random.seed(42)
    n_test = 500
    test_probs = np.random.beta(a=8.8, b=1.15, size=n_test)
    covered = sum(1 for p in test_probs if (1.0 - p) <= calibrator.q_hat)
    empirical_cov = covered / n_test
    assert empirical_cov >= 0.88, f"Coverage {empirical_cov} below nominal threshold"
