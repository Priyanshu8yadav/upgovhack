"""
Unit tests for Dialect-Invariant Speech Biomarkers Engine
Validates energy envelope syllable detection, articulation rate, phonation ratio,
F0 pitch variability, ambient noise gating, and edge-case acoustic failure modes.
"""

import numpy as np
import pytest
from core.speech_biomarkers import SpeechBiomarkersEngine


@pytest.fixture
def speech_engine():
    return SpeechBiomarkersEngine()


def test_normal_speech_biomarkers(speech_engine):
    res = speech_engine.evaluate_audio_clip(
        child_age_months=36.0,
        simulated_delay=False,
        ambient_noise_db=48.0
    )
    assert res.noise_gate_passed is True
    assert res.is_speech_delay_flagged is False
    assert res.syllable_rate_per_sec > 2.0
    assert res.phonation_ratio > 0.50
    assert "Green" in res.recommended_routing


def test_speech_delay_detection(speech_engine):
    res = speech_engine.evaluate_audio_clip(
        child_age_months=36.0,
        simulated_delay=True,
        ambient_noise_db=48.0
    )
    assert res.noise_gate_passed is True
    assert res.is_speech_delay_flagged is True
    assert res.syllable_rate_per_sec < 2.5
    assert "Yellow" in res.recommended_routing


def test_noisy_anganwadi_rejection(speech_engine):
    """
    Edge case: Loud classroom or street traffic (>68 dB) must reject capture at the quality gate
    to prevent acoustic false alarms.
    """
    res = speech_engine.evaluate_audio_clip(
        child_age_months=36.0,
        ambient_noise_db=74.0
    )
    assert res.noise_gate_passed is False
    assert "Noise" in res.clinical_finding or res.ambient_noise_db >= 68.0


def test_sparse_vocalization_and_reduced_pitch_variability(speech_engine):
    """
    Edge case: Severe expressive delay / monotone vocalization (reduced F0 variability)
    must flag developmental speech concern.
    """
    res = speech_engine.evaluate_audio_clip(
        child_age_months=48.0,
        simulated_delay=True,
        ambient_noise_db=50.0
    )
    assert res.is_speech_delay_flagged is True
    assert res.f0_pitch_variability_hz < 30.0
