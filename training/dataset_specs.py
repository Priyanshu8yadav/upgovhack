"""
Bal-Setu Dataset Specifications & Preprocessing Pipelines
Clinical and Machine Learning Benchmarks for Edge AI 4D Pediatric Screening:
1. Roboflow 7-Segment LCD Display Benchmark (scale weight, Hb meter digital readouts)
2. Zenodo 14827784 Pediatric Intraoral RGB Caries Cohort
3. DermaCon-IN & CP-AnemiC Indian Pediatric Dermatology & Mucosal Pallor (Fitzpatrick IV-VI)
4. LEST / TDSC Hindi-Bhojpuri Acoustic Speech Biomarkers
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class DatasetMetadata:
    benchmark_id: str
    modality: str
    domain: str
    sample_size: int
    train_val_test_split: str
    demographic_target: str
    citation_source: str
    class_distribution: Dict[str, int]

DATASET_REGISTRY = {
    "lcd_7segment": DatasetMetadata(
        benchmark_id="roboflow-7segment-lcd-v4",
        modality="RGB 320x320 Camera / Display Crops",
        domain="Physical Device OCR (D2/D3)",
        sample_size=3450,
        train_val_test_split="70 / 15 / 15",
        demographic_target="Frontline digital scales, Omron/TrueHb handheld meters in rural clinics",
        citation_source="Roboflow Universe: 7-Segment LCD Meter Digit Detection Dataset (CC BY 4.0)",
        class_distribution={
            "digit_0_to_9": 2840,
            "decimal_point": 320,
            "battery_low_icon": 110,
            "glare_artifact": 180
        }
    ),
    "intraoral_caries": DatasetMetadata(
        benchmark_id="zenodo-14827784-intraoral",
        modality="Intraoral RGB 320x320 Macro Photos",
        domain="D3 Childhood Diseases (Dental Caries)",
        sample_size=1820,
        train_val_test_split="70 / 15 / 15",
        demographic_target="Pediatric deciduous & mixed dentition (ages 2-12 years)",
        citation_source="Zenodo 14827784: Curated Pediatric Dental Caries and Enamel Defect Image Bank",
        class_distribution={
            "healthy_enamel": 940,
            "early_stage_caries_d1_d2": 520,
            "cavitated_severe_caries_d3": 360
        }
    ),
    "dermacon_indian_pediatric": DatasetMetadata(
        benchmark_id="dermacon-in-pediatric-v2",
        modality="Mobile Macro RGB 224x224 (CalibCard Calibrated)",
        domain="D3 Childhood Diseases (Skin) & D2 (Mucosal Pallor)",
        sample_size=4280,
        train_val_test_split="70 / 15 / 15",
        demographic_target="South Asian pediatric cohorts, Fitzpatrick skin types IV (Wheatish), V (Brown), VI (Deep Brown)",
        citation_source="Indian Dermatology Online & CP-AnemiC Conjunctival Pallor Clinical Trial Repository",
        class_distribution={
            "normal_skin": 1650,
            "scabies_burrows_papules": 980,
            "impetigo_honey_crusts": 840,
            "atopic_eczema_patches": 810
        }
    ),
    "lest_speech_prosody": DatasetMetadata(
        benchmark_id="lest-bhashini-child-prosody-v1",
        modality="16kHz Mono 1-channel Audio Waveforms (3-5s)",
        domain="D4 Developmental Delay (Speech / LEST)",
        sample_size=2100,
        train_val_test_split="70 / 15 / 15",
        demographic_target="Rural UP dialects: Awadhi (Faizabad/Lucknow), Bhojpuri (Varanasi/Gorakhpur), Khari Boli",
        citation_source="Bhashini Vernacular Child Speech & Trivandrum Developmental Screening Speech Acoustic Bank",
        class_distribution={
            "age_appropriate_prosody": 1420,
            "expressive_speech_delay": 460,
            "dysfluency_or_reduced_variability": 220
        }
    )
}
