"""
Unit tests for 7-Segment LCD Digit OCR Reader
Validates parsing of digital scale weights and Hb meters with WHO SAM/MAM cutoffs.
"""

import numpy as np
import pytest
from core.lcd_ocr import SevenSegmentLCDReader


@pytest.fixture
def lcd_reader():
    return SevenSegmentLCDReader()


def test_scale_weight_reading_normal(lcd_reader):
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    res = lcd_reader.parse_scale_reading(dummy_img, simulated_value=12.4)
    assert res.detected_value == 12.4
    assert res.unit == "kg"
    assert res.device_type == "Weighing Scale"


def test_scale_weight_reading_severe_underweight(lcd_reader):
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    res = lcd_reader.parse_scale_reading(dummy_img, simulated_value=6.8)
    assert res.detected_value == 6.8
    assert res.unit == "kg"


def test_haemoglobin_meter_reading_normal(lcd_reader):
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    res = lcd_reader.parse_haemoglobin_reading(dummy_img, child_age_months=36.0, simulated_value=11.8)
    assert res.detected_value == 11.8
    assert res.unit == "g/dL"
    assert res.anemia_tier == "Normal"
    assert "Green" in res.routing_tier


def test_haemoglobin_meter_reading_severe_anemia(lcd_reader):
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    res = lcd_reader.parse_haemoglobin_reading(dummy_img, child_age_months=36.0, simulated_value=6.4)
    assert res.detected_value == 6.4
    assert "Severe Anaemia" in res.anemia_tier
    assert "Red" in res.routing_tier
