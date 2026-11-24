"""
Test WHO Child Growth LMS Z-Score Engine and Nutritional Assessment
"""

import pytest
from core.growth_standards import WHOGrowthEngine


@pytest.fixture
def growth_engine():
    return WHOGrowthEngine()


def test_normal_child_growth(growth_engine):
    # 24-month-old boy, 12.2 kg, 87.8 cm (close to 50th percentile)
    res = growth_engine.evaluate_child(
        age_months=24.0,
        gender="M",
        weight_kg=12.2,
        height_cm=87.8,
        muac_cm=14.5
    )
    assert abs(res.waz) < 1.0
    assert abs(res.haz) < 1.0
    assert abs(res.whz) < 1.0
    assert res.stunting_status == "Normal"
    assert res.wasting_status == "Normal"
    assert res.clinical_nutrition_tier == "Green (Routine)"


def test_sam_detection_via_whz(growth_engine):
    # Severe Acute Malnutrition: WHZ < -3.0
    res = growth_engine.evaluate_child(
        age_months=18.0,
        gender="F",
        weight_kg=6.5,
        height_cm=78.0,
        muac_cm=12.0
    )
    assert res.whz < -3.0
    assert "SAM" in res.wasting_status
    assert res.clinical_nutrition_tier == "Primary Red (NRC)"


def test_sam_detection_via_muac(growth_engine):
    # Severe Acute Malnutrition: MUAC < 11.5 cm even if weight is borderline
    res = growth_engine.evaluate_child(
        age_months=20.0,
        gender="M",
        weight_kg=9.2,
        height_cm=81.0,
        muac_cm=11.2
    )
    assert "SAM" in res.wasting_status
    assert res.clinical_nutrition_tier == "Primary Red (NRC)"


def test_sam_detection_via_edema(growth_engine):
    # Bilateral pedal edema is diagnostic of SAM (Kwashiorkor) regardless of weight
    res = growth_engine.evaluate_child(
        age_months=30.0,
        gender="F",
        weight_kg=11.5,
        height_cm=88.0,
        has_pedal_edema=True
    )
    assert "SAM" in res.wasting_status
    assert res.clinical_nutrition_tier == "Primary Red (NRC)"


def test_mam_detection(growth_engine):
    # Moderate Acute Malnutrition: -3 <= WHZ < -2
    res = growth_engine.evaluate_child(
        age_months=24.0,
        gender="M",
        weight_kg=9.8,
        height_cm=87.0,
        muac_cm=12.0
    )
    assert "MAM" in res.wasting_status
    assert res.clinical_nutrition_tier == "Yellow (Home Diet Plan)"


def test_stunting_classification(growth_engine):
    # Stunted child: HAZ < -2.0
    res = growth_engine.evaluate_child(
        age_months=36.0,
        gender="M",
        weight_kg=12.0,
        height_cm=86.0  # Normal ~96cm
    )
    assert res.haz < -2.5
    assert "Stunting" in res.stunting_status
