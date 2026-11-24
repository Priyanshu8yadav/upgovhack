# Bal-Setu (बाल-सेतु): DPDP Act 2023 & DPDP Rules 2025 Compliance Dossier

## 1. Executive Summary & Statutory Framework

The Government of India notified the **Digital Personal Data Protection (DPDP) Rules, 2025** (G.S.R. 846(E)) on **November 13, 2025**, bringing the substantive enforcement of the **DPDP Act, 2023** into effect with a transition horizon culminating on **May 13, 2027**. 

Under Section 2(f) of the Act:
> *"Child" means an individual who has not completed eighteen years of age.*

Because 100% of Rashtriya Bal Swasthya Karyakram (RBSK 2.0) beneficiaries are between 0 and 18 years, every single health screening encounter falls under the jurisdiction of **Section 9 (Processing of Personal Data of Children)**. Violations of Section 9 incur statutory civil penalties up to **₹200 Crore** (and up to **₹250 Crore** for failure to take reasonable security safeguards under Section 8(5)).

Bal-Setu is architected to be **natively compliant by design** before the May 13, 2027 enforcement deadline.

---

## 2. Key Legal Nuances for Uttar Pradesh Evaluators

### Nuance 1: The Strict Scoping of Fourth Schedule Part A Healthcare Exemptions
Conventional health-tech submissions often claim that healthcare delivery enjoys a "blanket waiver" from DPDP children's data obligations. This is legally incorrect:

1. **Entries 1 and 2 of Part A of the Fourth Schedule** exempt clinical establishments, registered medical practitioners, and allied health professionals from Section 9(1) (Verifiable Parental Consent) and Section 9(3) (Behavioral Tracking Ban) **strictly and exclusively to the extent necessary for**:
   - Delivering medical health services, or
   - Executing an active clinical diagnostic and referral plan.
2. **Non-Clinical Operations Remain Fully Liable:** Secondary data monetization, non-consensual commercial model pre-training, behavioral profiling, or advertising lakes are **NOT exempt**.
3. **Section 9(2) Admits Zero Exceptions:**
   > *"A Data Fiduciary shall not undertake such processing of personal data of a child that is likely to cause any detrimental effect on the well-being of a child."*
   Section 9(2) applies absolutely, without any exemption under the Fourth Schedule.

### Nuance 2: Rural-Viable Verifiable Parental Consent (Rule 10)
Storing physical copies or photo scans of parents' Aadhaar cards on frontline ASHA/AWW smartphones creates toxic liability and violates UIDAI storage circulars. Under Rule 10(1) of the DPDP Rules, 2025, Bal-Setu implements two rural-viable consent mechanisms:

- **Pathway 1 [Rule 10(1)(a) - Existing KYC Linkage]:**
  Reuses verified adult parent/guardian identity already established through the state's **e-Kavach** maternal-child registry and **ABHA Guardian Accounts**. The ASHA enters the mother's 14-digit ABHA ID or RCH ID, inheriting legal parentage verification without capturing new identity documents.
- **Pathway 2 [Rule 10(1)(b) - Identity-Tokenized DigiLocker / Meri Pehchaan Handshake]:**
  For unlinked parents, consent is verified through a zero-storage virtual token issued by DigiLocker / Meri Pehchaan via SMS OTP, paired with an **itemized, spoken Hindi audio notice** explained to the mother, with an immediate, documented revocation mechanism.
- **Pediatric Assent Protocol (ICMR AI Ethics Guidelines 2023):**
  For older children (Ages 7–12: Verbal Assent; Ages 12–18: Written Assent), the app records age-appropriate spoken agreement prior to vision acuity and motor tasks.

---

## 3. "Zero-Raw-Media Cloud" Architectural Safeguards

The single greatest data-protection vulnerability in pediatric AI is uploading raw facial photos, eye close-ups, oral cavity images, and child vocal recordings to centralized cloud buckets.

Bal-Setu solves this fundamentally through **Zero-Raw-Media Edge Processing**:

```
[ Frontline Viewfinder ] 
         │
         ▼
[ On-Device LiteRT / OpenCV ] ──► Extracts Non-Reconstructible Metrics:
         │                        • Hirschberg displacement (mm)
         │                        • Tracing Hausdorff score
         │                        • 7-segment LCD digits (weight, Hb)
         │                        • Prosodic vocalization syllable rate
         │
         ├───► [ IMMEDIATE LOCAL DELETION ] 
         │      Raw photos, video frames, and audio streams 
         │      are wiped from RAM and local flash cache.
         │
         ▼
[ ABDM FHIR R4 Document Bundle (ndhm.in#6.5.0) ]
         • Only structured numerical Z-scores & SNOMED CT / LOINC codes sync.
         • PII is masked (91-XXXX-XXXX-XXXX).
```

### Exception: Specialist Tele-Verification (Internal Amber / Complex Red)
When an uncertain or complex condition requires human review by a Block MO or e-Sanjeevani pediatrician:
1. **Explicit Guardian Authorization:** The parent provides single-purpose biometric or OTP consent for the image consultation.
2. **On-Device Cropping:** Only the calibrated, de-identified region of interest (e.g. crop of the right eye or oral cavity) is extracted, excluding facial landmarks.
3. **AES-256 Envelope Encryption:** The packet is encrypted on-device with the destination facility's public key.
4. **Mandatory 30-Day TTL Auto-Purge:** The encrypted specialist file is automatically erased from the state server 30 days post-consultation.

---

## 4. Hard-Coded Section 9(3) Privacy Lock

Under Section 9(3):
> *"A Data Fiduciary shall not undertake tracking or behavioural monitoring of children or targeted advertising directed at children."*

Bal-Setu enforces this programmatically in every build:

```kotlin
// BalSetu Application Manifest Policy
object DPDPPolicy {
    const val DPDP_S9_3_LOCK = true
    const val ALLOW_COMMERCIAL_ANALYTICS = false
    const val ALLOW_THIRD_PARTY_SDKS = false
    const val ZERO_RAW_MEDIA_CLOUD = true
    const val DATA_RETENTION_DAYS = 30
}
```

- **Zero Third-Party Trackers:** No Google Firebase Analytics, Meta Pixel, Clevertap, or Mixpanel SDKs are linked into the binary.
- **Sovereign Telemetry Only:** Only anonymous aggregated system health metrics (model inference latency, failure-code distributions) sync to UP NIC servers.

---

## 5. DPDP Compliance Matrix

| Statutory Requirement | DPDP Section / Rule | Conventional Approaches | Bal-Setu Implementation |
| :--- | :--- | :--- | :--- |
| **Child Consent Verification** | Section 9(1) & Rule 10 | Unverified parent checkmark or collecting toxic paper Aadhaar scans. | Reuses KYC-verified e-Kavach / ABHA guardian accounts (Rule 10(1)(a)) with spoken Hindi notice. |
| **Well-Being Protection** | Section 9(2) | Generates stressful false alarms ("Your child may have cancer") sending families on wild chases. | Clear 3-color parent language (UC-3) stating child needs routine care or watchful play, never giving scary raw diagnosis. |
| **Tracking / Profiling Ban** | Section 9(3) | Embedded Firebase, Mixpanel, and Meta WhatsApp tracking scripts. | Hard-coded `DPDP_S9_3_LOCK = true`. Zero commercial ad or analytics libraries compiled. |
| **Data Minimization** | Section 6(1) | Uploads and indefinitely stores high-res child facial photos and videos in AWS S3 buckets. | Zero-Raw-Media Cloud: raw images deleted immediately on-device; only structured FHIR observations sync. |
| **Storage Limitation** | Section 8(7) | Unbounded retention of pediatric photos. | 30-Day automated TTL purge for encrypted tele-verify image packets; lifetime retention only for structured health records. |
| **Security Safeguards** | Section 8(5) | Unencrypted SQLite databases on frontline worker phones. | SQLCipher AES-256 database encryption; HMAC-SHA256 tamper-proof referral tokens. |
