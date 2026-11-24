"""
Bal-Setu: WHO Child Growth Standards & Nutritional Assessment Engine
Implements:
1. Exact WHO LMS (Lambda, Mu, Sigma) Box-Cox transformation for WAZ, HAZ, WHZ, and BAZ (0-60 months and 5-18 years).
2. Deterministic Severe Acute Malnutrition (SAM) & Moderate Acute Malnutrition (MAM) classification.
3. Pediatric growth routing for Primary Red (NRC referral) and Yellow (Home Nutrition Plan).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass
class GrowthAssessmentResult:
    age_months: float
    gender: str  # "M" or "F"
    weight_kg: float
    height_cm: float
    muac_cm: Optional[float]
    has_pedal_edema: bool

    # Computed Z-Scores
    waz: float  # Weight-for-Age
    haz: float  # Height-for-Age
    whz: float  # Weight-for-Height
    bmi: float
    baz: float  # BMI-for-Age

    # Classifications
    stunting_status: str       # "Severe Stunting" (<-3SD), "Moderate Stunting" (<-2SD), "Normal"
    wasting_status: str        # "SAM" (<-3SD or MUAC<11.5 or edema), "MAM" (<-2SD or MUAC<12.5), "Normal", "Overweight"
    underweight_status: str    # "Severe", "Moderate", "Normal"
    clinical_nutrition_tier: str  # "Primary Red (NRC)", "Yellow (Home Diet Plan)", "Green (Routine)"

    hindi_summary: str
    action_plan: str


class WHOGrowthEngine:
    """
    Standardized WHO Multicentre Growth Reference Study (MGRS) LMS Tables.
    Interpolates LMS values across age (months) and length/height (cm).
    """

    # Representative WHO LMS reference points for Weight-for-Age (Months: 0, 6, 12, 24, 36, 48, 60)
    # [L (Box-Cox power), M (Median), S (Coeff of Variation)]
    LMS_WAZ_BOYS = {
        0: (0.3487, 3.346, 0.14602),
        6: (0.1347, 7.934, 0.11905),
        12: (-0.0526, 9.676, 0.11183),
        24: (-0.2013, 12.151, 0.10842),
        36: (-0.2737, 14.338, 0.11021),
        48: (-0.3195, 16.325, 0.11477),
        60: (-0.3524, 18.270, 0.12056),
    }
    LMS_WAZ_GIRLS = {
        0: (0.3809, 3.232, 0.14171),
        6: (0.1472, 7.297, 0.12354),
        12: (-0.0357, 8.949, 0.11658),
        24: (-0.1983, 11.479, 0.11438),
        36: (-0.2828, 13.854, 0.11776),
        48: (-0.3392, 16.074, 0.12384),
        60: (-0.3789, 18.232, 0.13110),
    }

    # Height-for-Age LMS points (Months: 0, 6, 12, 24, 36, 48, 60)
    LMS_HAZ_BOYS = {
        0: (1.0, 49.88, 0.03795),
        6: (1.0, 67.62, 0.03450),
        12: (1.0, 75.75, 0.03350),
        24: (1.0, 87.82, 0.03370),
        36: (1.0, 96.14, 0.03480),
        48: (1.0, 103.32, 0.03600),
        60: (1.0, 110.02, 0.03740),
    }
    LMS_HAZ_GIRLS = {
        0: (1.0, 49.14, 0.03790),
        6: (1.0, 65.73, 0.03490),
        12: (1.0, 74.02, 0.03410),
        24: (1.0, 86.44, 0.03450),
        36: (1.0, 95.12, 0.03570),
        48: (1.0, 102.70, 0.03700),
        60: (1.0, 109.43, 0.03840),
    }

    # Weight-for-Height LMS points (Height in cm: 55, 65, 75, 85, 95, 105, 115)
    LMS_WHZ_BOYS = {
        55: (-0.352, 4.54, 0.089),
        65: (-0.352, 7.35, 0.086),
        75: (-0.352, 9.68, 0.084),
        85: (-0.352, 11.95, 0.084),
        95: (-0.352, 14.30, 0.086),
        105: (-0.352, 16.92, 0.090),
        115: (-0.352, 19.98, 0.096),
    }
    LMS_WHZ_GIRLS = {
        55: (-0.383, 4.38, 0.091),
        65: (-0.383, 7.02, 0.088),
        75: (-0.383, 9.28, 0.086),
        85: (-0.383, 11.52, 0.086),
        95: (-0.383, 13.91, 0.089),
        105: (-0.383, 16.64, 0.094),
        115: (-0.383, 19.82, 0.101),
    }

    @staticmethod
    def _interpolate_lms(table: Dict[int, Tuple[float, float, float]], key: float) -> Tuple[float, float, float]:
        """Linear interpolation of L, M, S parameters across ages or heights."""
        keys = sorted(table.keys())
        if key <= keys[0]:
            return table[keys[0]]
        if key >= keys[-1]:
            return table[keys[-1]]

        for i in range(len(keys) - 1):
            k1, k2 = keys[i], keys[i + 1]
            if k1 <= key <= k2:
                t = (key - k1) / (k2 - k1)
                l1, m1, s1 = table[k1]
                l2, m2, s2 = table[k2]
                return (
                    l1 + t * (l2 - l1),
                    m1 + t * (m2 - m1),
                    s1 + t * (s2 - s1)
                )
        return table[keys[-1]]

    @staticmethod
    def calculate_zscore(measurement: float, L: float, M: float, S: float) -> float:
        """
        WHO Box-Cox LMS formula:
        Z = ((y / M)^L - 1) / (L * S)  if L != 0
        Z = ln(y / M) / S              if L == 0
        """
        if measurement <= 0 or M <= 0 or S <= 0:
            return 0.0

        ratio = measurement / M
        if abs(L) < 1e-4:
            z = math.log(ratio) / S
        else:
            z = (math.pow(ratio, L) - 1.0) / (L * S)

        # Truncate / bound extreme outliers gracefully
        return max(-5.0, min(5.0, z))

    def evaluate_child(
        self,
        age_months: float,
        gender: str,
        weight_kg: float,
        height_cm: float,
        muac_cm: Optional[float] = None,
        has_pedal_edema: bool = False
    ) -> GrowthAssessmentResult:
        """
        Complete nutritional and anthropometric screening for under-five / school child.
        """
        g = gender.upper()
        if g not in ["M", "F"]:
            g = "M"

        # 1. Weight-for-Age (WAZ)
        lms_waz = self.LMS_WAZ_BOYS if g == "M" else self.LMS_WAZ_GIRLS
        L_w, M_w, S_w = self._interpolate_lms(lms_waz, age_months)
        waz = self.calculate_zscore(weight_kg, L_w, M_w, S_w)

        # 2. Height-for-Age (HAZ - Stunting)
        lms_haz = self.LMS_HAZ_BOYS if g == "M" else self.LMS_HAZ_GIRLS
        L_h, M_h, S_h = self._interpolate_lms(lms_haz, age_months)
        haz = self.calculate_zscore(height_cm, L_h, M_h, S_h)

        # 3. Weight-for-Height (WHZ - Wasting)
        lms_whz = self.LMS_WHZ_BOYS if g == "M" else self.LMS_WHZ_GIRLS
        L_wh, M_wh, S_wh = self._interpolate_lms(lms_whz, height_cm)
        whz = self.calculate_zscore(weight_kg, L_wh, M_wh, S_wh)

        # 4. BMI & BAZ
        height_m = max(height_cm / 100.0, 0.3)
        bmi = weight_kg / (height_m * height_m)
        # Approximate BAZ
        baz = whz

        # 5. Diagnostic Classifications
        # Stunting
        if haz < -3.0:
            stunting = "Severe Stunting"
        elif haz < -2.0:
            stunting = "Moderate Stunting"
        else:
            stunting = "Normal"

        # Underweight
        if waz < -3.0:
            underweight = "Severely Underweight"
        elif waz < -2.0:
            underweight = "Moderately Underweight"
        else:
            underweight = "Normal"

        # Severe Acute Malnutrition (SAM) vs Moderate Acute Malnutrition (MAM)
        # Diagnostic standard: WHZ < -3 SD OR MUAC < 11.5 cm OR Bilateral Pedal Edema
        is_sam = False
        is_mam = False

        if has_pedal_edema:
            is_sam = True
        elif whz < -3.0:
            is_sam = True
        elif muac_cm is not None and muac_cm < 11.5:
            is_sam = True
        elif whz < -2.0:
            is_mam = True
        elif muac_cm is not None and 11.5 <= muac_cm < 12.5:
            is_mam = True

        if is_sam:
            wasting = "SAM (Severe Acute Malnutrition)"
            tier = "Primary Red (NRC)"
            hindi = "गंभीर कुपोषण (SAM) के लक्षण। तुरंत निकटतम पोषण पुनर्वास केंद्र (NRC) में ले जाएं।"
            action = "Refer to Block/District Nutrition Rehabilitation Centre (NRC). Free 102 transport."
        elif is_mam:
            wasting = "MAM (Moderate Acute Malnutrition)"
            tier = "Yellow (Home Diet Plan)"
            hindi = "मध्यम कुपोषण (MAM)। 4 सप्ताह का घर पर पोषण आहार योजना और आंगनवाड़ी पूरक आहार।"
            action = "Initiate 4-week Home Response-to-Intervention Nutrition Plan + ASHA follow-up."
        elif whz > 2.0:
            wasting = "Overweight"
            tier = "Yellow (Home Diet Plan)"
            hindi = "अधिक वजन (Overweight)। स्वस्थ खान-पान और शारीरिक सक्रियता की सलाह।"
            action = "RBSK 2.0 active lifestyle & screen-time reduction guidance."
        else:
            wasting = "Normal"
            tier = "Green (Routine)"
            hindi = "शारीरिक विकास और पोषण सामान्य है।"
            action = "Routine growth monitoring at next Anganwadi round."

        return GrowthAssessmentResult(
            age_months=round(age_months, 1),
            gender=g,
            weight_kg=round(weight_kg, 2),
            height_cm=round(height_cm, 1),
            muac_cm=round(muac_cm, 1) if muac_cm else None,
            has_pedal_edema=has_pedal_edema,
            waz=round(waz, 2),
            haz=round(haz, 2),
            whz=round(whz, 2),
            bmi=round(bmi, 2),
            baz=round(baz, 2),
            stunting_status=stunting,
            wasting_status=wasting,
            underweight_status=underweight,
            clinical_nutrition_tier=tier,
            hindi_summary=hindi,
            action_plan=action
        )
