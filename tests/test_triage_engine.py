"""
Test Capacity-Aware 4-Tier Triage Engine
"""

import pytest
from core.triage_engine import BalSetuTriageEngine


@pytest.fixture
def triage_engine():
    return BalSetuTriageEngine()


def test_triage_healthy_child_green(triage_engine):
    summary = triage_engine.evaluate_child(
        child_id="TEST-GREEN-01",
        age_months=24,
        gender="M",
        weight_kg=12.2,
        height_cm=87.8,
        muac_cm=14.0,
        hb_g_dl=12.0
    )
    assert summary.overall_parent_color == "Green"
    assert summary.d1_defects_color == "Green"
    assert summary.d2_deficiencies_color == "Green"
    assert summary.d3_diseases_color == "Green"
    assert summary.d4_developmental_color == "Green"
    assert summary.clinical_routing_lane == "Green (Routine)"
    assert not summary.requires_102_ambulance


def test_triage_speech_delay_yellow_rti(triage_engine):
    # Child with mild speech delay (LEST fail) should enter Yellow 4-week RtI, NOT DEIC!
    summary = triage_engine.evaluate_child(
        child_id="TEST-YELLOW-02",
        age_months=36,
        gender="F",
        weight_kg=13.5,
        height_cm=95.0,
        muac_cm=14.2,
        hb_g_dl=11.2,
        lest_passed=False
    )
    assert summary.overall_parent_color == "Yellow"
    assert summary.d4_developmental_color == "Yellow"
    assert summary.clinical_routing_lane == "Yellow (4-Wk RtI)"
    assert summary.requires_rti_home_plan
    assert not summary.requires_102_ambulance


def test_triage_sam_primary_red_nrc(triage_engine):
    # Severe Acute Malnutrition routes to NRC locally (Primary Red), avoiding DEIC false alarms!
    summary = triage_engine.evaluate_child(
        child_id="TEST-SAM-03",
        age_months=18,
        gender="M",
        weight_kg=7.2,
        height_cm=79.0,
        muac_cm=11.0,
        hb_g_dl=9.5
    )
    assert summary.overall_parent_color == "Red"
    assert summary.d2_deficiencies_color == "Red"
    assert summary.clinical_routing_lane == "Primary Red (AAM/CHC/NRC)"
    assert "Nutrition Rehabilitation Centre" in summary.destination_facility_type
    assert summary.requires_102_ambulance


def test_triage_strabismus_internal_amber_tele(triage_engine):
    # Inward squint routes to Internal Amber (e-Sanjeevani Tele-verify) first
    summary = triage_engine.evaluate_child(
        child_id="TEST-SQUINT-04",
        age_months=30,
        gender="F",
        weight_kg=12.0,
        height_cm=89.0,
        simulated_esotropia_mm=1.5
    )
    assert summary.overall_parent_color == "Red"
    assert summary.clinical_routing_lane == "Internal Amber (Tele-Verify)"
    assert summary.requires_tele_verification


def test_triage_congenital_defect_complex_red(triage_engine):
    # Cleft lip/palate or leukocoria requires Fast-track DEIC / Tertiary surgery
    summary = triage_engine.evaluate_child(
        child_id="TEST-CLEFT-05",
        age_months=6,
        gender="M",
        weight_kg=7.5,
        height_cm=66.0,
        has_cleft_lip_or_palate=True
    )
    assert summary.overall_parent_color == "Red"
    assert summary.d1_defects_color == "Red"
    assert summary.clinical_routing_lane == "Complex Red (DEIC)"
    assert summary.requires_102_ambulance
