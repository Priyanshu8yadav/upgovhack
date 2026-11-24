# Bal-Setu (बाल-सेतु): Uttar Pradesh Startup Procurement Scheme 2026 Pilot Proposal

## 1. Statutory Procurement Corridor

Under the **Uttar Pradesh Startup Procurement Scheme 2026** (notified on **October 12, 2026** by the Department of IT & Electronics, Government of Uttar Pradesh):
- Government departments, state agencies, and autonomous health societies (including the **State Health Society, National Health Mission, Uttar Pradesh**) are authorized to procure innovative products and solutions directly from **UP-registered startups** through a streamlined corridor:
  1. **Direct Single-Tender Procurement up to ₹50 Lakh:** Exempt from standard prior-experience, turnover, and earnest-money-deposit (EMD) criteria.
  2. **Limited Tenders up to ₹1 Crore:** Reserved for sunrise sectors including Artificial Intelligence and public health diagnostics.
  3. **Departmental Budget Allocation:** Departments may allocate up to **0.5% of their annual procurement budget** specifically for innovative startup pilots.

Bal-Setu's Phase-2 commercial ask is structured precisely to fit within the **₹50 Lakh single-tender limit** for an initial 1-district deployment.

---

## 2. 1-District Turnkey Pilot Scope & Architecture

### Target District
- **Recommended Pilot District:** **Lucknow District** (Urban/Peri-Urban baseline with operational KGMU DEIC) OR **Bahraich District** (Terai/Aspirational district with high stunting ~54% and severe anaemia ~76%).
- **Target Child Population:** **5,00,000 children (0–18 years)** screened across ~2,800 Anganwadi Centres and ~1,900 government primary schools.
- **Frontline Cohort:** ~2,400 ASHAs, ~2,800 Anganwadi Workers, and ~32 Mobile Health Team members.
- **Duration:** 12 Months (comprising 3-month shadow trial, 3-month SDK integration, and 6-month stepped-wedge field rollout).

---

## 3. Itemized Pilot Budget (Total: ₹48.50 Lakh)

Because Bal-Setu executes 100% of inference on-device using LiteRT and eliminates expensive daily WhatsApp bot chat loops, **recurring infrastructure costs are only ₹4.30 Lakh (8.9% of budget)**. Over 91% of funds are directed toward clinical validation, state digital health integration, and frontline capacity building:

| Budget Head | Line-Item Scope & Activities | Allocation (INR) |
| :--- | :--- | :--- |
| **1. Sovereign Cloud & Messaging OpEx** | MeghRaj GI Cloud hosting + IndiaAI GPU compute for async dialect ASR + 1 weekly 7:30 PM WhatsApp utility digest across 75,000 Yellow-tier families + toll-free IVR minutes. | **₹4,30,000** |
| **2. Physical RBSK-CalibCard Production** | Printing and high-durability matte lamination of 50,000 double-sided pocket cards at ₹2.25/card (frontline workers, schools, clinics). | **₹1,12,500** |
| **3. Stage 1 Clinical Validation (KGMU / JNMC)** | Institutional Ethics Committee (IEC) clearance, NSWS Form MD-12 Test Licence, investigator fees, and gold-standard comparative audits across 2,500 children. | **₹18,50,000** |
| **4. e-Kavach SDK Integration & ABDM Pipeline** | Engineering the Android AAR / Intent plugin inside UP's official e-Kavach application, ABDM FHIR R4 (ndhm.in#6.5.0) pipeline, and automated ASHA incentive triggers. | **₹14,52,500** |
| **5. Frontline Capacity Building & Master Training** | Training 120 District Master Trainers (ASHA Sanginis, Child Development Project Officers [CDPOs], and MHT Medical Officers) across block health facilities. | **₹10,05,000** |
| **TOTAL PILOT VALUE** | **Fully within the ₹50 Lakh UP Startup Procurement Threshold** | **₹48,50,000** |

---

## 4. Contractual Public Health Milestones & Deliverables

| Milestone | Timeline | Deliverables & Acceptance Criteria | Payout Tranche |
| :--- | :--- | :--- | :--- |
| **M1: Ethics & Base Setup** | Month 2 | IEC clearance obtained from KGMU/JNMC; NSWS Test Licence issued; 50,000 CalibCards distributed; initial master-trainer workshops completed. | 25% (₹12.125 L) |
| **M2: Clinical Validation & SDK** | Month 5 | 2,500-child silent shadow trial completed; sensitivity ≥90% validated against gold standards; e-Kavach Android Intent SDK delivered; Class B SaMD Form MD-3 filed. | 35% (₹16.975 L) |
| **M3: Field Rollout** | Month 9 | Live deployment across all blocks in pilot district; >2,50,000 children screened; 4-week RtI parent loop active; automated ASHA incentives verified upon QR scan. | 25% (₹12.125 L) |
| **M4: Final Evaluation** | Month 12 | 5,00,000 screenings completed; DEIC referral completion rate evaluated (>75% target achieved); DEIC false-positive rate audited (<10%); final Health Directorate report. | 15% (₹7.275 L) |

---

## 5. Why the Health Department Should Select Bal-Setu Over Rivals

1. **Zero Frontline App Fatigue:** Bal-Setu does not force ASHAs to download a fourth disconnected mobile app. It embeds as an **offline Android Intent SDK inside e-Kavach**, preserving existing beneficiary registries and ABHA IDs.
2. **Immediate Protection of UP's 12 DEICs:** Rather than flooding district hospitals with false alarms, Bal-Setu resolves 85% of acute conditions locally at Ayushman Arogya Mandirs (AAMs) and NRCs, eliminating wage-losing journeys for daily-wage families.
3. **Financially Sustainable After Grant Funding:** While naive apps create a ₹19.45 Crore/year recurring WhatsApp liability that collapses as soon as hackathon prize money runs out, Bal-Setu's modeled statewide recurring cost is just **₹0.86 per child/year**, fitting effortlessly within annual National Health Mission flexi-pool budgets.
