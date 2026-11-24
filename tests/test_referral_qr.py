"""
Test Offline Referral QR Token Cryptography and Facility Handshake
Asserts HMAC-SHA256 tamper-proofing, arrival state transitions, PFMS incentive release,
and edge-case failure modes.
"""

import json
import pytest
from referral.qr_token import OfflineReferralQRGenerator
from referral.facility_handshake import FacilityHandshakeManager


@pytest.fixture
def referral_tools():
    gen = OfflineReferralQRGenerator()
    mgr = FacilityHandshakeManager()
    return gen, mgr


def test_qr_generation_and_tamper_proofing(referral_tools):
    gen, _ = referral_tools
    token = gen.generate_token(
        child_id="TEST-QR-001",
        clinical_lane="Complex Red (DEIC)",
        destination_facility="DEIC Lucknow",
        destination_facility_code="FAC-157-DEIC",
        reason_findings=["Congenital Cataract"],
        requires_102_ambulance=True,
        asha_id="ASHA-LKO-4421"
    )
    token_dict = token.__dict__
    assert gen.verify_payload_signature(token_dict)

    # Tamper with payload (change destination facility)
    tampered = dict(token_dict)
    tampered["destination_facility"] = "Private Clinic"
    assert not gen.verify_payload_signature(tampered), "Tampered signature must fail verification!"

    # Tamper with requires_102 flag
    tampered_102 = dict(token_dict)
    tampered_102["requires_102"] = False
    assert not gen.verify_payload_signature(tampered_102), "Tampered ambulance flag must fail verification!"


def test_facility_arrival_handshake_and_asha_incentive(referral_tools):
    gen, mgr = referral_tools
    token = gen.generate_token(
        child_id="TEST-QR-002",
        clinical_lane="Primary Red (AAM/CHC/NRC)",
        destination_facility="Nutrition Rehabilitation Centre (NRC) - Bahraich",
        destination_facility_code="FAC-123-NRC",
        reason_findings=["Severe Acute Malnutrition (SAM)"],
        requires_102_ambulance=True,
        asha_id="ASHA-BHR-8812"
    )
    mgr.register_new_referral(token)

    # Simulate hospital desk scan
    payload_json = json.dumps(token.__dict__)
    success, msg, rec = mgr.process_facility_arrival_scan(payload_json, clerk_id="CLERK-NRC-01")

    assert success
    assert rec.current_status == "Treatment Initiated"
    assert rec.arrival_timestamp_utc is not None
    assert rec.asha_incentive is not None
    assert rec.asha_incentive.amount_inr == 300.0  # NRC referral rate
    assert "Verified" in rec.asha_incentive.approval_status


def test_corrupted_qr_rejection(referral_tools):
    """
    Edge case: Corrupted, non-JSON or truncated QR scans must be cleanly rejected.
    """
    _, mgr = referral_tools
    success, msg, rec = mgr.process_facility_arrival_scan("CORRUPTED-BASE64-GARBAGE###", clerk_id="CLERK-01")
    assert not success
    assert "error" in msg.lower() or "invalid" in msg.lower() or "signature" in msg.lower() or "malformed" in msg.lower()
