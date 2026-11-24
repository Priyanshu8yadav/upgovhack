"""
Unit tests for Dialect-Invariant Speech Biomarkers Engine
Validates energy envelope syllable detection, articulation rate, phonation ratio,
F0 pitch variability, and ambient noise gating.
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
    res = speech_engine.evaluate_audio_clip(
        child_age_months=36.0,
        ambient_noise_db=74.0
    )
    assert res.noise_gate_passed is False
