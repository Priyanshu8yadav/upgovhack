#!/usr/bin/env python3
"""
Bal-Setu (बाल-सेतु): One-Click Hackathon Demonstration & Health Command Server
National Health Mission (NHM) • UP AI Samadhan Hackathon 2026

Launches:
- UP Health Department 75-District Governance Dashboard: http://localhost:8080/
- Frontline Worker Screener (UC-1, 2, 3):                http://localhost:8080/worker/
- Parent Offline Mini-PWA (UC-5, 6, 7):                 http://localhost:8080/pwa/
- Interactive Triage & ABDM FHIR R4 API:               http://localhost:8080/api/screen/child
- DEIC Capacity Policy Simulator:                       http://localhost:8080/api/policy/simulate
- Subgroup AI Fairness & Conformal Audit:               http://localhost:8080/api/fairness/metrics
"""

import os
import subprocess
import sys


def main():
    print("=" * 78)
    print("  बाल-सेतु (Bal-Setu): Offline-First Edge AI Child Health Screening SDK")
    print("  Government of Uttar Pradesh • Department of Medical Health and Family Welfare")
    print("  National Health Mission (NHM) • Rashtriya Bal Swasthya Karyakram (RBSK 2.0)")
    print("=" * 78)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Ensure CalibCard assets exist
    calib_front = os.path.join(base_dir, "calibcard", "calibcard_front.png")
    if not os.path.exists(calib_front):
        print("\n[1/4] Generating physical ₹2 RBSK-CalibCard optical assets...")
        subprocess.run([sys.executable, os.path.join(base_dir, "calibcard", "generate_calibcard.py")], check=True)
    else:
        print("\n[1/4] RBSK-CalibCard optical calibration targets ready.")

    # 2. Ensure ONNX Edge Models exist
    onnx_skin = os.path.join(base_dir, "models", "skin_pallor_mobilenetv3.onnx")
    if not os.path.exists(onnx_skin):
        print("\n[2/4] Exporting sub-10 MB multi-modal ONNX edge models...")
        subprocess.run([sys.executable, os.path.join(base_dir, "models", "export_models.py")], check=True)
    else:
        print("\n[2/4] Multi-modal ONNX edge AI models ready (Total footprint: 2.18 MB).")

    # 3. Ensure Synthetic 50k cohort summary exists
    cohort_summary = os.path.join(base_dir, "synthetic", "synthetic_cohort_summary.json")
    if not os.path.exists(cohort_summary):
        print("\n[3/4] Simulating 50,000-child UP synthetic cohort across 75 districts...")
        subprocess.run([sys.executable, os.path.join(base_dir, "synthetic", "cohort_generator.py")], check=True)
    else:
        print("\n[3/4] 50,000 UP synthetic cohort and ABDM FHIR R4 bundles ready.")

    # 4. Launch FastAPI Server
    port = int(os.environ.get("PORT", "7860"))
    print(f"\n[4/4] Starting Bal-Setu State Health Command Server on port {port}...")
    print("-" * 78)
    print(f"  🖥️ State Command Dashboard       : http://localhost:{port}/")
    print(f"  📱 Frontline Worker Screener (UC-1/2/3) : http://localhost:{port}/worker/")
    print(f"  🏠 Parent Offline Mini-PWA (UC-5)       : http://localhost:{port}/pwa/")
    print(f"  🏥 Facility Clinical Desk (UC-4)        : http://localhost:{port}/facility/")
    print(f"  🎴 CalibCard Specs & Assets Viewer      : http://localhost:{port}/calibcard-assets/")
    print(f"  📊 DEIC Policy Simulator API            : http://localhost:{port}/api/policy/simulate")
    print(f"  ⚖️ Subgroup Fairness & Drift API        : http://localhost:{port}/api/fairness/metrics")
    print(f"  🎴 CalibCard Front Vector SVG           : http://localhost:{port}/calibcard-assets/calibcard_front.svg")
    print(f"  🎴 CalibCard Back Tracing SVG           : http://localhost:{port}/calibcard-assets/calibcard_back.svg")
    print("-" * 78)
    print("Press Ctrl+C to stop the server.\n")

    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
