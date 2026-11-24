"""
Bal-Setu: Response-to-Intervention (RtI) Progress Velocity & Auto-Escalation Engine
Implements:
1. 4-Week longitudinal micro-check tracking for Yellow-tier children.
2. Objective progress-velocity (Δ slope) calculation.
3. Automated escalation:
   - Responders (positive slope) graduate to Green.
   - Non-responders (flat/negative slope at Week 3-4) auto-escalate to ASHA & DEIC Amber/Red.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class WeeklyMilestoneCheck:
    week_number: int  # 1, 2, 3, 4
    check_date_utc: str
    speech_score: float  # 0 to 100
    motor_score: float   # 0 to 100
    play_compliance_days: int  # 0 to 7
    combined_score: float
    notes: str


@dataclass
class RtITrajectoryResult:
    child_id: str
    total_weeks_tracked: int
    weekly_scores: List[float]
    velocity_slope: float  # Points per week
    graduation_status: str  # "Graduated to Green", "Improving - Continue RtI", "Non-Responder - Escalated"
    action_tier: str       # "Green", "Yellow", "Amber Tele-Verify", "Complex Red"
    alert_triggered: bool
    hindi_message: str
    bhojpuri_message: str


class ResponseToInterventionTracker:
    """
    Manages longitudinal 4-week home development velocity.
    """

    def __init__(self):
        # Database of active RtI enrollments
        self.enrollments: Dict[str, List[WeeklyMilestoneCheck]] = {}

    def record_weekly_check(
        self,
        child_id: str,
        week_num: int,
        speech_score: float,
        motor_score: float,
        compliance_days: int = 5,
        notes: str = ""
    ) -> WeeklyMilestoneCheck:
        """Records weekly parent game micro-check."""
        combined = round((0.5 * speech_score) + (0.5 * motor_score), 1)
        check = WeeklyMilestoneCheck(
            week_number=week_num,
            check_date_utc=datetime.now(timezone.utc).isoformat(),
            speech_score=round(speech_score, 1),
            motor_score=round(motor_score, 1),
            play_compliance_days=compliance_days,
            combined_score=combined,
            notes=notes or f"Week {week_num} check recorded."
        )
        if child_id not in self.enrollments:
            self.enrollments[child_id] = []
        self.enrollments[child_id].append(check)
        return check

    def evaluate_trajectory(self, child_id: str) -> RtITrajectoryResult:
        """
        Computes progress slope (Δ) and determines graduation vs. escalation.
        """
        checks = self.enrollments.get(child_id, [])
        if not checks:
            # Fallback for demonstration: generate simulated 4-week curve
            return RtITrajectoryResult(
                child_id=child_id,
                total_weeks_tracked=0,
                weekly_scores=[],
                velocity_slope=0.0,
                graduation_status="Enrolled - Awaiting Week 1 Data",
                action_tier="Yellow",
                alert_triggered=False,
                hindi_message="4 सप्ताह का गृह खेल कार्यक्रम सक्रिय है।",
                bhojpuri_message="4 हफ्ता के घरे खेल कार्यक्रम शुरू बा।"
            )

        scores = [c.combined_score for c in checks]
        weeks = [c.week_number for c in checks]

        if len(scores) >= 2:
            # Linear regression slope (points per week)
            x = np.array(weeks)
            y = np.array(scores)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0

        latest_score = scores[-1]
        latest_week = weeks[-1]

        # Decision Logic:
        # If Week 4 completed:
        if latest_week >= 4:
            if latest_score >= 75.0 and slope > 5.0:
                grad_status = "Graduated to Green"
                tier = "Green"
                alert = False
                hindi = "शानदार सुधार! बच्चे ने 4 सप्ताह के खेल कार्यक्रम में सभी पड़ाव हासिल कर लिए हैं। अब स्थिति सामान्य (Green) है।"
                bhojpuri = "बहुते बढ़िया सुधार! बबुआ 4 हफ्ता के खेल में सब सीख गइल। अब एकदम सामान्य (Green) बा।"
            else:
                grad_status = "Non-Responder - Escalated"
                tier = "Amber Tele-Verify → Complex Red"
                alert = True
                hindi = "चेतावनी: 4 सप्ताह के घरेलू अभ्यास के बाद भी अपेक्षित सुधार नहीं दिखा। आशा संगिनी और ब्लॉक डॉक्टर को स्वतः अलर्ट भेजा गया है।"
                bhojpuri = "चेतावनी: 4 हफ्ता के बादो सुधार नइखे लउकत। आशा दीदी आ अस्पताल के डाक्टर के तुरंत खबर भेज दिहल गइल बा।"
        else:
            # Weeks 1-3
            if slope >= 8.0:
                grad_status = "Improving - Strong Trajectory"
                tier = "Yellow"
                alert = False
                hindi = f"सकारात्मक सुधार (सप्ताह {latest_week}): गति +{slope:.1f} अंक/सप्ताह। अभ्यास जारी रखें।"
                bhojpuri = f"सुधार लउकत बा (हफ्ता {latest_week}): बबुआ तेजी से सीखत बा। खेल जारी रखीं।"
            elif latest_week >= 3 and slope < 2.0:
                grad_status = "Flat Progress - Early Warning Alert"
                tier = "Amber Tele-Verify"
                alert = True
                hindi = "सप्ताह 3 चेतावनी: बच्चे की प्रगति रुकी हुई है। आशा कार्यकर्ता को गृह-भ्रमण की सलाह दी गई है।"
                bhojpuri = "हफ्ता 3 चेतावनी: प्रगति रुक गइल बा। आशा दीदी के घरे आवे के सूचना भेजल गइल बा।"
            else:
                grad_status = "In Progress"
                tier = "Yellow"
                alert = False
                hindi = f"सप्ताह {latest_week} का खेल पूरा हुआ। नियमित अभ्यास जारी रखें।"
                bhojpuri = f"हफ्ता {latest_week} के खेल पूरा भइल। रोज अभ्यास करावत रहीं।"

        return RtITrajectoryResult(
            child_id=child_id,
            total_weeks_tracked=len(checks),
            weekly_scores=scores,
            velocity_slope=round(slope, 2),
            graduation_status=grad_status,
            action_tier=tier,
            alert_triggered=alert,
            hindi_message=hindi,
            bhojpuri_message=bhojpuri
        )
