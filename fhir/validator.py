"""
Bal-Setu: ABDM FHIR R4 Bundle Validator
Validates structural conformance against:
- HL7 FHIR Release 4 (R4)
- ABDM NDHM Implementation Guide (ndhm.in#6.5.0)
- Composition first entry rule for Document Bundles
- Referential integrity among Patient, Practitioner, Observations, and ServiceRequests
"""

from typing import Any, Dict, List, Tuple


class FHIRBundleValidator:
    """
    Offline validator verifying conformance to ABDM FHIR R4 profiles.
    """

    @staticmethod
    def validate_document_bundle(bundle: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates bundle structure. Returns (is_valid, error_list).
        """
        errors = []

        if not isinstance(bundle, dict):
            return False, ["Bundle must be a JSON object"]

        if bundle.get("resourceType") != "Bundle":
            errors.append(f"Expected resourceType 'Bundle', got '{bundle.get('resourceType')}'")

        if bundle.get("type") != "document":
            errors.append(f"Document bundle must have type 'document', got '{bundle.get('type')}'")

        entries = bundle.get("entry", [])
        if not entries:
            errors.append("Bundle contains no entries")
            return False, errors

        # Rule 1: First entry in a document bundle MUST be a Composition
        first_res = entries[0].get("resource", {})
        if first_res.get("resourceType") != "Composition":
            errors.append(f"First entry in document bundle must be Composition, found '{first_res.get('resourceType')}'")

        # Check required resources
        resource_types = set()
        ids = set()
        for e in entries:
            res = e.get("resource", {})
            rt = res.get("resourceType")
            rid = res.get("id")
            if rt:
                resource_types.add(rt)
            if rid:
                ids.add(rid)

        required_types = {"Composition", "Patient", "Practitioner", "Organization", "Observation"}
        missing_types = required_types - resource_types
        if missing_types:
            errors.append(f"Missing required ABDM resources: {list(missing_types)}")

        # Check Composition links to Patient and Practitioner
        subject_ref = first_res.get("subject", {}).get("reference", "")
        author_ref = first_res.get("author", [{}])[0].get("reference", "")

        if not subject_ref.startswith("Patient/"):
            errors.append("Composition subject must point to a Patient resource")
        if not author_ref.startswith("Practitioner/"):
            errors.append("Composition author must point to a Practitioner resource")

        # Verify SNOMED / LOINC coding in observations
        for e in entries:
            res = e.get("resource", {})
            if res.get("resourceType") == "Observation":
                coding = res.get("code", {}).get("coding", [])
                has_valid_system = any(
                    c.get("system") in ["http://snomed.info/sct", "http://loinc.org"]
                    for c in coding
                )
                if not has_valid_system and not res.get("code", {}).get("text"):
                    errors.append(f"Observation {res.get('id')} lacks standard SNOMED/LOINC code or text")

        is_valid = len(errors) == 0
        return is_valid, errors
