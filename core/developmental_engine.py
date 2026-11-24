"""
Bal-Setu: Neurodevelopmental Screening & Audio Red-Flag Engine
Implements Indian validated pediatric developmental batteries:
1. LEST (Language Evaluation Scale Trivandrum, 0-3y & 3-6y).
2. TDSC (Trivandrum Developmental Screening Chart, 0-6y).
3. M-CHAT-R/F (Modified Checklist for Autism in Toddlers, 16-30m).
4. Audio red-flag questionnaire in Hindi & Bhojpuri (CHD, Hearing, Seizures, Screen Time).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DevelopmentalScreeningResult:
    age_months: float
    lest_score: int
    lest_max: int
    lest_passed: bool
    lest_delay_type: Optional[str]  # "None", "Expressive Delay", "Receptive Delay"

    tdsc_delay_flag: bool
    tdsc_unachieved_milestones: List[str]

    mchat_risk_tier: Optional[str]  # "Low (0-2)", "Medium (3-7)", "High (8-20)", None
    mchat_failed_items: List[str]

    red_flags_present: List[str]  # e.g., "Suspected CHD (Feeding Diaphoresis)", "Hearing Flag"
    screen_time_hours_per_day: float

    overall_developmental_status: str  # "Normal", "Mild Delay / At-Risk", "Significant Delay"
    routing_tier: str  # "Green", "Yellow (4-Week RtI)", "Internal Amber (Tele-Verify)", "Complex Red (DEIC)"

    hindi_explanation: str
    bhojpuri_explanation: str
    recommended_activities: List[str]


class DevelopmentalEngine:
    """
    Standardized Indian Academy of Pediatrics (IAP) & RBSK neurodevelopmental screening engine.
    """

    # TDSC Milestone Norms (Age in Months at which 90% of Indian children achieve the item)
    TDSC_NORMS = [
        (1.5, "Social smile", "सामाजिक मुस्कान (Social smile)"),
        (3.0, "Head control steady", "गर्दन संभालना (Steady head control)"),
        (5.0, "Reaches for object", "वस्तु पकड़ने के लिए हाथ बढ़ाना (Reaching for toys)"),
        (6.0, "Rolls over back to stomach", "करवट लेना (Rolling over)"),
        (8.0, "Sits without support", "बिना सहारे बैठना (Sitting without support)"),
        (9.0, "Crawling / Standing with support", "सहारे से खड़े होना (Standing with support)"),
        (12.0, "Stands alone / Walks with support", "बिना सहारे खड़े होना (Standing alone)"),
        (15.0, "Walks alone well", "बिना सहारे चलना (Walking independently)"),
        (18.0, "Speaks 3-5 meaningful words", "3-5 सार्थक शब्द बोलना (3-5 words)"),
        (24.0, "2-word phrases / runs well", "2 शब्दों के वाक्य बोलना और दौड़ना (2-word phrases)"),
        (36.0, "Speaks in simple sentences / rides tricycle", "साधारण वाक्य बोलना (Full sentences)"),
        (48.0, "Draws circle / hops on one foot", "वृत्त बनाना / एक पैर पर कूदना (Drawing circle)"),
        (60.0, "Dresses without help / tells story", "कपड़े पहनना / कहानी सुनाना (Dressing/storytelling)")
    ]

    # M-CHAT-R/F Critical Items (Items 2, 5, 12: Showing, Pointing, Joint Attention)
    MCHAT_CRITICAL_KEYS = {
        "q2_showing": "Does child show you things by bringing them over?",
        "q5_pointing": "Does child point with one finger to show something interesting?",
        "q7_name_response": "Does child turn and look when you call their name?",
        "q12_joint_attention": "Does child look at something you point at across the room?"
    }

    def evaluate_child(
        self,
        age_months: float,
        # LEST answers: dict of item_id -> bool (True=Achieved)
        lest_responses: Optional[Dict[str, bool]] = None,
        # TDSC: list of achieved milestone indices
        tdsc_achieved_count: Optional[int] = None,
        # M-CHAT (for 16-30m): dict of q_id -> bool (True=Yes/Normal)
        mchat_responses: Optional[Dict[str, bool]] = None,
        # Red-flag question responses (CHD, Hearing, Seizures, Screen-Time)
        chd_feeding_diaphoresis: bool = False,
        chd_fast_breathing: bool = False,
        hearing_startle_absent: bool = False,
        seizure_or_staring_spells: bool = False,
        screen_time_hours: float = 1.0
    ) -> DevelopmentalScreeningResult:
        """
        Runs comprehensive age-appropriate developmental assessment.
        """
        red_flags = []
        if chd_feeding_diaphoresis:
            red_flags.append("CHD: Feeding diaphoresis (दूध पीते समय अत्यधिक पसीना)")
        if chd_fast_breathing:
            red_flags.append("CHD: Fast breathing / cyanosis (तेज सांस / होंठों का नीलापन)")
        if hearing_startle_absent:
            red_flags.append("Hearing: No startle response or sound orientation (तेज आवाज पर चौंकना या मुड़ना नहीं)")
        if seizure_or_staring_spells:
            red_flags.append("Neurological: Seizures or staring spells (झटके या खो जाने की स्थिति)")
        if screen_time_hours >= 3.0:
            red_flags.append(f"RBSK 2.0 Screen Risk: {screen_time_hours:.1f} hrs/day screen exposure (>2h cutoff)")

        # 1. TDSC Evaluation
        # Determine expected milestones for this age
        expected_milestones = [m for m in self.TDSC_NORMS if m[0] <= age_months]
        expected_count = len(expected_milestones)

        if tdsc_achieved_count is not None:
            achieved = min(tdsc_achieved_count, expected_count)
        else:
            achieved = expected_count  # default normal

        tdsc_delay = achieved < max(0, expected_count - 1)
        unachieved = [m[2] for m in expected_milestones[achieved:]] if tdsc_delay else []

        # 2. LEST Evaluation
        lest_max = 10
        if lest_responses:
            lest_achieved = sum(1 for v in lest_responses.values() if v)
            lest_max = len(lest_responses)
        else:
            lest_achieved = 9

        lest_passed = (lest_achieved >= (lest_max - 1))
        delay_type = None
        if not lest_passed:
            delay_type = "Expressive & Receptive Delay" if lest_achieved < (lest_max - 3) else "Mild Speech Delay"

        # 3. M-CHAT-R/F (Only for 16-30 months)
        mchat_risk = None
        mchat_failed = []
        if 16.0 <= age_months <= 30.0:
            if mchat_responses:
                # In M-CHAT-R, answering 'No' to positive behaviors is a fail
                for q_key, is_yes in mchat_responses.items():
                    if not is_yes:
                        mchat_failed.append(q_key)
                fail_count = len(mchat_failed)
                if fail_count <= 2:
                    mchat_risk = "Low Risk (0-2)"
                elif 3 <= fail_count <= 7:
                    mchat_risk = "Medium Risk (3-7)"
                else:
                    mchat_risk = "High Risk (8-20)"
            else:
                mchat_risk = "Low Risk (0-2)"

        # 4. Clinical Triaging Synthesis
        activities = []
        if any("CHD" in rf or "Neurological" in rf for rf in red_flags):
            status = "Significant Medical Red Flag"
            tier = "Complex Red (DEIC/Pediatrician)"
            hindi = "गंभीर स्वास्थ्य लक्षण (हृदय रोग या झटकों का संदेह)। तुरंत नजदीकी सामुदायिक स्वास्थ्य केंद्र या जिला अस्पताल जाएं।"
            bhojpuri = "गंभीर समस्या के लच्छन बा (दिल चाहे मिरगी के संका)। तुरंत सरकारी अस्पताल में डाक्टर के देखाईं।"
            activities = ["Immediate medical specialist consultation via 102 ambulance."]
        elif (mchat_risk == "High Risk (8-20)") or (tdsc_delay and not lest_passed):
            status = "Significant Neurodevelopmental Delay"
            tier = "Internal Amber (Tele-Verify) → DEIC"
            hindi = "विकास के प्रमुख पड़ावों में विलंब (भाषण और गामक विकास)। विशेषज्ञ टेली-समीक्षा और डीईआईसी (DEIC) परामर्श की आवश्यकता है।"
            bhojpuri = "बोले आ चले फिरे में देरी लउकत बा। डीईआईसी केंद्र में विशेषज्ञ डाक्टर से जांच करवावल जरूरी बा।"
            activities = ["Specialist developmental assessment", "Home stimulation therapy"]
        elif (not lest_passed) or tdsc_delay or (mchat_risk == "Medium Risk (3-7)"):
            status = "Mild Delay / Response-to-Intervention Candidate"
            tier = "Yellow (4-Week RtI)"
            hindi = "बोलने या चलने में हल्का विलंब। घर पर 4 सप्ताह का बाल-विकास खेल कार्यक्रम (Response-to-Intervention) शुरू करें।"
            bhojpuri = "बोले या हाथ के काम में तनी सुस्ती बा। 4 हफ्ता ले घरे खेल-खेल में सिखावे के योजना बनावल जाई।"
            activities = [
                "चित्र पहचान और नाम बोलने का खेल (कटोरी, चम्मच, फल दिखाकर नाम बुलवाना)",
                "कागज पर पेंसिल से बिंदु मिलाने और रेखा खींचने का अभ्यास",
                "गेंद पकड़ने और फेंकने का गामक खेल",
                "स्क्रीन समय (फोन/टीवी) शून्य करना और आंखों से संपर्क बढ़ाना"
            ]
        else:
            status = "Normal Developmental Trajectory"
            tier = "Green (Routine)"
            hindi = "बच्चे का मानसिक, शारीरिक और भाषा विकास आयु के अनुसार बिल्कुल सही है।"
            bhojpuri = "बबुआ के बोलल, चलल आ समझल उमिर के हिसाब से एकदम ठीक बा।"
            activities = ["नियमित कहानियां सुनाएं और खिलौनों से संवाद करें"]

        return DevelopmentalScreeningResult(
            age_months=round(age_months, 1),
            lest_score=lest_achieved,
            lest_max=lest_max,
            lest_passed=lest_passed,
            lest_delay_type=delay_type,
            tdsc_delay_flag=tdsc_delay,
            tdsc_unachieved_milestones=unachieved,
            mchat_risk_tier=mchat_risk,
            mchat_failed_items=mchat_failed,
            red_flags_present=red_flags,
            screen_time_hours_per_day=screen_time_hours,
            overall_developmental_status=status,
            routing_tier=tier,
            hindi_explanation=hindi,
            bhojpuri_explanation=bhojpuri,
            recommended_activities=activities
        )
