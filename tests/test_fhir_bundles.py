"""
Test ABDM FHIR R4 Bundle Generation & Structural Validation
"""

import pytest
from core.triage_engine import BalSetuTriageEngine
from fhir.bundle_generator import ABDMFHIRBundleGenerator
from fhir.validator import FHIRBundleValidator


@pytest.fixture
def fhir_tools():
    triage = BalSetuTriageEngine()
    gen = ABDMFHIRBundleGenerator()
    val = FHIRBundleValidator()
    return triage, gen, val


def test_fhir_bundle_validity_for_healthy_child(fhir_tools):
    triage, gen, val = fhir_tools
    summary = triage.evaluate_child(
        child_id="TEST-FHIR-01",
        age_months=12,
        gender="F",
        weight_kg=8.9,
        height_cm=74.0,
        muac_cm=13.5,
        hb_g_dl=11.5
    )
    bundle = gen.create_screening_bundle(summary, district_name="Lucknow", district_lgd_code=157)

    is_valid, errors = val.validate_document_bundle(bundle)
    assert is_valid, f"Validation errors: {errors}"
    assert bundle["type"] == "document"
    assert bundle["entry"][0]["resource"]["resourceType"] == "Composition"


def test_fhir_bundle_validity_for_referred_child(fhir_tools):
    triage, gen, val = fhir_tools
    summary = triage.evaluate_child(
        child_id="TEST-FHIR-02",
        age_months=8,
        gender="M",
        weight_kg=6.0,
        height_cm=68.0,
        has_clubfoot_talipes=True
    )
    bundle = gen.create_screening_bundle(summary, district_name="Varanasi", district_lgd_code=185)

    is_valid, errors = val.validate_document_bundle(bundle)
    assert is_valid, f"Validation errors: {errors}"

    # Check for ServiceRequest entry in bundle
    sr_entries = [e for e in bundle["entry"] if e["resource"]["resourceType"] == "ServiceRequest"]
    assert len(sr_entries) >= 1
    assert "DEIC" in sr_entries[0]["resource"]["code"]["text"]
