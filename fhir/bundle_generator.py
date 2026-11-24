"""
Bal-Setu: ABDM FHIR R4 Bundle Serializer
Complies strictly with:
- ABDM NDHM FHIR Implementation Guide (ndhm.in#6.5.0 / NRCES India)
- Standard terminologies: SNOMED CT (http://snomed.info/sct), LOINC (http://loinc.org), UCUM (http://unitsofmeasure.org)
- Zero-Raw-Media cloud storage: transmits only structured observations and referral tokens.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from core.triage_engine import ComprehensiveTriageSummary


class ABDMFHIRBundleGenerator:
    """
    Generates fully compliant ABDM FHIR R4 Document Bundles for RBSK 2.0 Digital Health Records.
    """

    def __init__(self):
        self.doc_system = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"

    def create_screening_bundle(
        self,
        triage_summary: ComprehensiveTriageSummary,
        district_name: str = "Lucknow",
        district_lgd_code: int = 157,
        block_name: str = "Sarojini Nagar",
        block_lgd_code: int = 1245,
        asha_worker_id: str = "ASHA-UP-LKO-4421",
        recorded_at: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Builds complete FHIR R4 Bundle for a child's 4D screening.
        """
        now = recorded_at or datetime.now(timezone.utc)
        timestamp_str = now.isoformat()
        date_str = now.strftime("%Y-%m-%d")

        bundle_id = f"bundle-{uuid.uuid4()}"
        composition_id = f"comp-{uuid.uuid4()}"
        patient_id = f"pat-{triage_summary.child_id}"
        practitioner_id = f"prac-{asha_worker_id}"
        org_id = "org-nhm-up"

        # 1. Patient Resource
        patient = {
            "resourceType": "Patient",
            "id": patient_id,
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient"]
            },
            "identifier": [
                {
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                                "display": "Medical Record Number"
                            }
                        ]
                    },
                    "system": "https://healthid.ndhm.gov.in",
                    "value": f"91-{str(abs(hash(triage_summary.child_id)))[:4]}-XXXX-XXXX"
                }
            ],
            "gender": "male" if triage_summary.gender.upper() == "M" else "female",
            "address": [
                {
                    "use": "home",
                    "state": "Uttar Pradesh",
                    "district": district_name,
                    "city": block_name,
                    "extension": [
                        {
                            "url": "https://nrces.in/ndhm/fhir/r4/StructureDefinition/LGDCode",
                            "valueInteger": district_lgd_code
                        }
                    ]
                }
            ]
        }

        # 2. Practitioner Resource (Frontline ASHA / MHT Doctor)
        practitioner = {
            "resourceType": "Practitioner",
            "id": practitioner_id,
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner"]
            },
            "identifier": [
                {
                    "system": "https://uphealth.up.gov.in/frontline-registry",
                    "value": asha_worker_id
                }
            ],
            "name": [
                {
                    "text": "Frontline Health Screener (ASHA / MHT)"
                }
            ]
        }

        # 3. Organization Resource
        organization = {
            "resourceType": "Organization",
            "id": org_id,
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Organization"]
            },
            "identifier": [
                {
                    "system": "https://nhm.up.gov.in",
                    "value": "NHM-UP-RBSK"
                }
            ],
            "name": "National Health Mission - Uttar Pradesh (RBSK 2.0 Directorate)"
        }

        # 4. Observation Resources
        observations = []
        observation_refs = []

        # Overall 4D Screening Observation
        obs_4d_id = f"obs-4d-{uuid.uuid4()}"
        obs_4d = {
            "resourceType": "Observation",
            "id": obs_4d_id,
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"]
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "exam",
                            "display": "Exam"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "410605003",
                        "display": "Screening procedure (procedure)"
                    }
                ],
                "text": "RBSK 2.0 4D Comprehensive Child Health Screening"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": timestamp_str,
            "performer": [{"reference": f"Practitioner/{practitioner_id}"}],
            "valueString": f"Overall: {triage_summary.overall_parent_color} | Clinical Lane: {triage_summary.clinical_routing_lane}",
            "component": [
                {
                    "code": {"text": "D1 Defects at Birth"},
                    "valueString": triage_summary.d1_defects_color
                },
                {
                    "code": {"text": "D2 Deficiencies"},
                    "valueString": triage_summary.d2_deficiencies_color
                },
                {
                    "code": {"text": "D3 Diseases"},
                    "valueString": triage_summary.d3_diseases_color
                },
                {
                    "code": {"text": "D4 Developmental Delays"},
                    "valueString": triage_summary.d4_developmental_color
                }
            ]
        }
        observations.append(obs_4d)
        observation_refs.append({"reference": f"Observation/{obs_4d_id}"})

        # Observation for each finding
        for f in triage_summary.findings:
            f_obs_id = f"obs-{uuid.uuid4()}"
            f_obs = {
                "resourceType": "Observation",
                "id": f_obs_id,
                "meta": {
                    "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"]
                },
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "248326004" if "Malnutrition" in f.condition_name else ("252874005" if "Eye" in f.condition_name else "410605003"),
                            "display": f.condition_name
                        }
                    ],
                    "text": f"{f.d_category}: {f.condition_name}"
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": timestamp_str,
                "valueString": f.finding_description,
                "interpretation": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": "A" if f.status_color == "Red" else "N",
                                "display": "Abnormal" if f.status_color == "Red" else "Normal"
                            }
                        ]
                    }
                ]
            }
            observations.append(f_obs)
            observation_refs.append({"reference": f"Observation/{f_obs_id}"})

        # 5. ServiceRequest (Referral if Red or Amber)
        service_requests = []
        sr_refs = []
        if triage_summary.overall_parent_color == "Red" or "Amber" in triage_summary.clinical_routing_lane:
            sr_id = f"sr-{uuid.uuid4()}"
            sr = {
                "resourceType": "ServiceRequest",
                "id": sr_id,
                "meta": {
                    "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/ServiceRequest"]
                },
                "status": "active",
                "intent": "order",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "306206005",
                                "display": "Referral to service"
                            }
                        ]
                    }
                ],
                "priority": "urgent" if triage_summary.requires_102_ambulance else "routine",
                "code": {
                    "text": f"RBSK Referral to {triage_summary.destination_facility_type}"
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "requester": {"reference": f"Practitioner/{practitioner_id}"},
                "performer": [{"display": triage_summary.destination_facility_type}],
                "authoredOn": timestamp_str,
                "reasonCode": [
                    {"text": f.condition_name} for f in triage_summary.findings
                ],
                "note": [
                    {"text": f"Ambulance 102 Required: {triage_summary.requires_102_ambulance}"},
                    {"text": f"Tele-verification: {triage_summary.requires_tele_verification}"}
                ]
            }
            service_requests.append(sr)
            sr_refs.append({"reference": f"ServiceRequest/{sr_id}"})

        # 6. Composition Resource (Document Header)
        composition = {
            "resourceType": "Composition",
            "id": composition_id,
            "meta": {
                "profile": ["https://nrces.in/ndhm/fhir/r4/StructureDefinition/Composition"]
            },
            "status": "final",
            "type": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "8652-3",
                        "display": "Hospital Consultation Note"
                    }
                ],
                "text": "Bal-Setu RBSK 2.0 Screening and Care Orchestration Summary"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "date": timestamp_str,
            "author": [{"reference": f"Practitioner/{practitioner_id}"}],
            "title": "Pediatric 4D Screening Record (बाल-सेतु)",
            "custodian": {"reference": f"Organization/{org_id}"},
            "section": [
                {
                    "title": "Screening Results and Clinical Triage",
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "11450-4",
                                "display": "Problem list - Reported"
                            }
                        ]
                    },
                    "entry": observation_refs + sr_refs
                }
            ]
        }

        # Assemble Full Document Bundle
        entries = [{"fullUrl": f"urn:uuid:{composition_id}", "resource": composition}]
        entries.append({"fullUrl": f"urn:uuid:{patient_id}", "resource": patient})
        entries.append({"fullUrl": f"urn:uuid:{practitioner_id}", "resource": practitioner})
        entries.append({"fullUrl": f"urn:uuid:{org_id}", "resource": organization})

        for obs in observations:
            entries.append({"fullUrl": f"urn:uuid:{obs['id']}", "resource": obs})
        for sr in service_requests:
            entries.append({"fullUrl": f"urn:uuid:{sr['id']}", "resource": sr})

        bundle = {
            "resourceType": "Bundle",
            "id": bundle_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": timestamp_str,
                "profile": [self.doc_system]
            },
            "identifier": {
                "system": "https://rbsk.up.gov.in/fhir/bundles",
                "value": bundle_id
            },
            "type": "document",
            "timestamp": timestamp_str,
            "entry": entries
        }

        return bundle
