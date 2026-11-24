"""
Bal-Setu: Offline Referral QR Token Generator & Cryptographic Signer
Implements:
1. Zero-PII offline-verifiable referral payload with HMAC-SHA256 signature.
2. 1-tap 102 free ambulance dispatch prompt.
3. QR code PNG and base64 data-URI rendering for paper cards and WhatsApp/SMS delivery.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import uuid
import io
import base64
from typing import Dict, List, Optional
import qrcode


@dataclass
class ReferralPayload:
    referral_id: str
    child_id_hash: str
    clinical_lane: str
    destination_facility: str
    destination_facility_code: str
    reason_findings: List[str]
    requires_102_ambulance: bool
    asha_id: str
    created_at_utc: str
    valid_until_utc: str
    signature: str = ""


class OfflineReferralQRGenerator:
    """
    Generates tamper-proof offline QR tokens for closing the RBSK screening-to-care loop.
    """

    # Demo state key for HMAC signing (in production, loaded from secure HSM / KMS)
    STATE_SECRET_KEY = b"UP_NHM_RBSK_BALSETU_SECRET_2026"

    def __init__(self, secret_key: bytes = STATE_SECRET_KEY):
        self.secret_key = secret_key

    def _sign_payload(self, data: Dict) -> str:
        """Computes HMAC-SHA256 over canonical JSON string without the signature field."""
        clean = {k: v for k, v in data.items() if k != "signature"}
        canonical_str = json.dumps(clean, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return hmac.new(self.secret_key, canonical_str.encode('utf-8'), hashlib.sha256).hexdigest()[:16]

    def verify_payload_signature(self, payload_dict: Dict) -> bool:
        """Verifies HMAC-SHA256 signature authenticity."""
        expected_sig = self._sign_payload(payload_dict)
        given_sig = payload_dict.get("signature", "")
        return hmac.compare_digest(expected_sig, given_sig)

    def generate_token(
        self,
        child_id: str,
        clinical_lane: str,
        destination_facility: str,
        destination_facility_code: str,
        reason_findings: List[str],
        requires_102_ambulance: bool,
        asha_id: str = "ASHA-UP-LKO-4421",
        validity_days: int = 30
    ) -> ReferralPayload:
        """Creates and cryptographically signs an offline referral token."""
        now = datetime.now(timezone.utc)
        valid_until = now + timedelta(days=validity_days)

        child_hash = hashlib.sha256(child_id.encode('utf-8')).hexdigest()[:12]
        ref_id = f"REF-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        payload_dict = {
            "referral_id": ref_id,
            "child_id_hash": child_hash,
            "clinical_lane": clinical_lane,
            "destination_facility": destination_facility,
            "destination_facility_code": destination_facility_code,
            "reason_findings": reason_findings,
            "requires_102_ambulance": requires_102_ambulance,
            "asha_id": asha_id,
            "created_at_utc": now.isoformat(),
            "valid_until_utc": valid_until.isoformat(),
        }

        signature = self._sign_payload(payload_dict)
        payload_dict["signature"] = signature

        return ReferralPayload(**payload_dict)

    def render_qr_image(self, payload: ReferralPayload, box_size: int = 8, border: int = 2) -> bytes:
        """Encodes payload as scannable PNG image."""
        payload_json = json.dumps(asdict(payload), separators=(',', ':'), ensure_ascii=False)
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=box_size,
            border=border
        )
        qr.add_data(payload_json)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#0F172A", back_color="#FFFFFF")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def render_qr_base64(self, payload: ReferralPayload) -> str:
        """Returns base64 data-URI string suitable for inline web display."""
        png_bytes = self.render_qr_image(payload)
        b64 = base64.b64encode(png_bytes).decode('ascii')
        return f"data:image/png;base64,{b64}"
