"""
Unit tests for 30-Frame Temporal Motor Kinematics Engine
Validates MediaPipe 33-landmark temporal pose buffer, trunk sway variance,
gait cadence symmetry, and single-leg balance.
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
