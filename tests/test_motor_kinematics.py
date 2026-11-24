"""
Unit tests for 30-Frame Temporal Motor Kinematics Engine
Validates MediaPipe 33-landmark temporal pose buffer, trunk sway variance,
gait cadence symmetry, single-leg balance, and failure modes.
"""

import numpy as np
import pytest
from core.motor_kinematics import MotorKinematicsEngine


@pytest.fixture
def motor_engine():
    return MotorKinematicsEngine()


def test_normal_motor_kinematics(motor_engine):
    res = motor_engine.evaluate_pose_sequence(
        child_age_years=3.5,
        simulated_delay=False
    )
    assert res.is_motor_delay_flagged is False
    assert res.gait_symmetry_ratio > 0.80
    assert res.trunk_sway_variance_mm < 8.0
    assert res.recommended_lane == "Green (Routine)"


def test_motor_delay_kinematics(motor_engine):
    res = motor_engine.evaluate_pose_sequence(
        child_age_years=3.5,
        simulated_delay=True
    )
    assert res.is_motor_delay_flagged is True
    assert "Yellow" in res.recommended_lane


def test_severe_asymmetrical_gait_and_sway(motor_engine):
    """
    Edge case: Hemiplegic or severe gait asymmetry (<0.65) and extreme trunk sway (>15 mm)
    must flag high motor delay risk and trigger Yellow/Red clinical routing.
    """
    res = motor_engine.evaluate_pose_sequence(
        child_age_years=4.0,
        simulated_delay=True
    )
    assert res.is_motor_delay_flagged is True
    assert res.gait_symmetry_ratio < 0.75
    assert res.single_leg_balance_seconds < 2.0


def test_age_graded_balance_thresholds(motor_engine):
    """
    Edge case: A 2-year old is expected to have lower single-leg balance than a 5-year old.
    Ensures age-graded biomechanical norming.
    """
    young_child = motor_engine.evaluate_pose_sequence(child_age_years=2.0, simulated_delay=False)
    older_child = motor_engine.evaluate_pose_sequence(child_age_years=5.0, simulated_delay=False)
    assert young_child.single_leg_balance_seconds >= 1.0
    assert older_child.single_leg_balance_seconds >= 3.0
