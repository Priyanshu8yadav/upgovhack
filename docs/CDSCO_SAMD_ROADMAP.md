# Bal-Setu (बाल-सेतु): CDSCO Software as a Medical Device (SaMD) Regulatory Roadmap

## 1. Regulatory Status under CDSCO July 2026 Final MDSW Guidance

A critical misconception in health-tech innovation is asserting: *"We are only doing screening and triage, not medical diagnosis, so CDSCO regulations do not apply to us."* 

This statement instantly loses credibility with Medical Health evaluators. Under the **Central Drugs Standard Control Organisation (CDSCO) July 2026 final Guidance Document on Medical Device Software** (`CDSCO/MD/GD/MDSW/01/2026`) under the Medical Device Rules, 2017 (MDR-2017):
- Standalone AI software performing **medical screening, risk stratification, or clinical triage** is explicitly regulated as **Software as a Medical Device (SaMD)**.
- CDSCO specifically cites AI screening algorithms for chest X-rays in tuberculosis as regulated SaMD requiring formal licensing.

### Classification: Class B SaMD
Under the IMDRF risk-categorization framework adopted by CDSCO:
- **Significance of Information:** *Drives clinical management* (triaging between routine care, community response-to-intervention, primary care, and urgent tertiary specialist consultation). It does not autonomously provide definitive diagnosis or therapy.
- **State of Healthcare Situation:** Non-critical for routine development, moderate for acute nutritional deficiency, critical for acute congenital anomalies (CHD, leukocoria).
- **Resulting Risk Classification:** **Class B (Low-to-Moderate Risk)**.
- **Licensing Authority:** State Licensing Authority (SLA) of Uttar Pradesh (Ayush Bhawan, Lucknow).
- **Application Forms:** Form MD-3 (Application for Grant of Licence to Manufacture for Sale and Distribution) and Form MD-5 (Final Licence).
- **Audit Requirement:** Rule 20 Notified Body audit (e.g. Intertek, TÜV SÜD India, or UL India). Statutory fee: ₹5,000 per manufacturing site + ₹500 per device family.

---

## 2. Algorithm Change Protocol (ACP) for Continuous Model Refinement

Medical AI deployed across India's regional populations suffers from biological and environmental drift. A major breakthrough in CDSCO's July 2026 guidance is the **Algorithm Change Protocol (ACP)** framework (harmonized with US FDA and IMDRF SaMD guidance):

- **The Problem:** In traditional medical device regulation, modifying neural network weights or retraining on new regional data triggers a mandatory supplemental licence application—paralyzing iterative improvement.
- **The ACP Solution:** When filing the initial Form MD-3/MD-5 dossier, Bal-Setu includes a pre-specified **Algorithm Change Protocol (ACP)** defining:
  1. **Predetermined Modification Boundaries:** Scope of post-market updates limited to:
     - Retraining color constancy multipliers on regional UP skin tones (Fitzpatrick IV–VI).
     - Fine-tuning acoustic vocalization thresholds for regional Purvanchal Bhojpuri and Bundelkhand Bundeli dialects.
     - Recalibrating haemoglobin conjunctival pallor models against regional dietary pigmentation.
  2. **Acceptance Criteria & Verification Gateways:** Mandates that no model update can decrease sensitivity below 90% for urgent Red categories (SAM, leukocoria, CHD).
  3. **Regression Testing Suite:** Automated execution of the 19-test verification suite before any on-device model payload is pushed via secure OTA update.
- **Regulatory Impact:** Enables Bal-Setu to continuously improve accuracy in rural UP under an approved Class B licence without filing recurrent amendment dossiers.

---

## 3. Disciplined 4-Stage Clinical Validation Protocol

Bal-Setu adheres strictly to ICMR's **Ethical Guidelines for AI in Healthcare (2023)** and **Guidelines for Pediatric Biomedical Research (2017)** through a phased four-stage validation strategy:

```
[ Stage 0: Algorithmic Verification ] ──► [ Stage 1: Silent Shadow ] ──► [ Stage 2: Recalibration ] ──► [ Stage 3: Stepped-Wedge ]
  • Nov–Dec 2026             • Months 1–3                   • Months 4–6                   • Months 7–12
  • Synthetic & Public       • KGMU / JNMC Aligarh          • UP Field Tuning              • 2–4 UP Blocks
  • Zero Real Children       • 2,500 Consented Children     • Class B MD-3/MD-5 Filing     • Health Outcome Proof
```

### Stage 0: Algorithmic Verification (Baseline Validation - Nov–Dec 2026)
- **Data Scope:** 100% synthetic cohorts (50,000 UP children) + open licensed research benchmarks. **Zero real children's personal data.**
- **Deliverable:** Working on-device LiteRT/OpenCV pipeline demonstrating deterministic WHO LMS mathematics, CalibCard homography, and ABDM FHIR R4 schema compliance.

### Stage 1: IEC-Approved Silent Shadow Trial (Months 1–3)
- **Regulatory Gate:** Institutional Ethics Committee (IEC) clearance at nodal centers:
  * **King George's Medical University (KGMU), Lucknow** (Department of Pediatrics)
  * **Jawaharlal Nehru Medical College (JNMC), AMU Aligarh** (Nodal DEIC Centre of Excellence)
  * Test Licence filed via the National Single Window System (NSWS) under Form MD-12.
- **Study Design:** Prospective, observational, silent shadow evaluation across **2,500 children (0–6 years)** attending DEIC and high-risk pediatric follow-up clinics in two contrasting UP districts (Lucknow urban/peri-urban + Bahraich/Gorakhpur rural Terai).
- **Comparator Gold Standards:**
  * Nutritional & SAM: InBody digital scale, calibrated infantometer, and venous CBC.
  * Ocular & Squint: Pediatric ophthalmologist slit-lamp, RetCam, and manual Hirschberg prism cover test.
  * Anaemia: Laboratory Automated Hematology Analyzer (Sysmex/Beckman Coulter) + HemoCue Hb 301.
  * Developmental: Clinical psychologist assessment using full DASII (Developmental Assessment Scale for Indian Infants) and Stanford-Binet.
- **Target Performance Milestones:**
  * Sensitivity ≥ 92% and Specificity ≥ 88% on urgent Red conditions (SAM, Congenital Cataract, Clubfoot).
  * Subgroup demographic parity audited across biological sex, rural/urban origin, and Fitzpatrick IV–VI skin tones.

### Stage 2: Local Recalibration & Regulatory Filing (Months 4–6)
- **Application of Field Learnings:** Applying the crucial lesson from the Bihar Haggenmüller study (where out-of-the-box Western models drifted by ±4.43 g/dL and required Indian mucosal recalibration to reach ±2.25 g/dL screening accuracy).
- **Dataset Replacement:** Replace non-commercial academic research image datasets (e.g. DermaCon-IN, SCIN) with ethically consented UP pediatric clinical training data gathered under the Stage 1 IEC protocol.
- **Regulatory Filing:** Lock inference thresholds, complete Risk Management File (ISO 14971), Usability Engineering File (IEC 62366-1), and Software Lifecycle Documentation (IEC 62304), and submit the Class B SaMD application (Form MD-3) with Algorithm Change Protocol to the UP State Licensing Authority.

### Stage 3: Stepped-Wedge Controlled Block Pilot (Months 7–12)
- **Deployment:** 2 to 4 high-burden blocks under the **Uttar Pradesh Startup Procurement Scheme 2026** single-tender corridor.
- **Primary Endpoint:** Increase documented DEIC referral completion rate from UP's baseline ~47% to **>75%**.
- **Secondary Endpoint:** Maintain DEIC false-positive arrival rate below **10%**, preventing specialist overload while reducing median age of developmental delay detection by >6 months.
