"""
Bal-Setu: Missed-Call Bhashini Dialect IVR Engine
Implements:
1. Free missed-call trigger for rural households without WhatsApp (~35% of UP homes).
2. Voice prompts in 5 UP dialects: Hindi, Bhojpuri, Awadhi, Bundeli, Braj.
3. 90-second audio coaching for household-object developmental games.
4. Keypad confirmation ('1' to confirm, '2' for ASHA callback) synced to e-Kavach.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time


@dataclass
class DialectAudioPrompt:
    dialect_code: str  # "hi", "bho", "awa", "bnd", "braj"
    dialect_name: str
    region: str
    welcome_message: str
    weekly_activity_audio: str
    keypad_prompt: str
    confirmation_success: str


class DialectIVRSimulator:
    """
    Simulates telecom missed-call IVR interactions for low-connectivity rural families.
    """

    PROMPTS: Dict[str, DialectAudioPrompt] = {
        "hi": DialectAudioPrompt(
            dialect_code="hi",
            dialect_name="Hindi (खड़ी बोली)",
            region="All Uttar Pradesh / State Default",
            welcome_message="नमस्कार! राष्ट्रीय बाल स्वास्थ्य कार्यक्रम बाल-सेतु में आपका स्वागत है।",
            weekly_activity_audio="इस सप्ताह का खेल: बच्चे के सामने रसोई की कटोरी, चम्मच और गिलास रखें। एक-एक वस्तु का नाम पूछें और बच्चे को बोलने के लिए प्रोत्साहित करें। ताली बजाकर शाबाशी दें।",
            keypad_prompt="यदि बच्चे ने यह खेल खेला है तो फोन में 1 दबाएं। आशा दीदी से बात करने के लिए 2 दबाएं।",
            confirmation_success="धन्यवाद! आपके बच्चे की प्रगति दर्ज हो गई है। अगले सप्ताह फिर बात करेंगे।"
        ),
        "bho": DialectAudioPrompt(
            dialect_code="bho",
            dialect_name="Bhojpuri (भोजपुरी)",
            region="Purvanchal (Gorakhpur, Varanasi, Deoria, Azamgarh, Ballia)",
            welcome_message="प्रणाम! बाल-सेतु सेवा में राउर स्वागत बा।",
            weekly_activity_audio="एह हफ्ता के खेल: बबुआ के सामने कटोरी, चमचा आ लोटा रखीं। पूछीं कि ई का हव? बबुआ के नाम बोले खातिर कहल जाव। सही बोलला पर ताली बजाईं।",
            keypad_prompt="अगर बबुआ ई खेल खेललसि त फोन में 1 दबाईं। आशा दीदी के घरे बोलावे खातिर 2 दबाईं।",
            confirmation_success="बहुत-बहुत धन्यवाद! बबुआ के प्रगति दर्ज हो गइल बा। अगिला हफ्ता फेर फोन कइल जाई।"
        ),
        "awa": DialectAudioPrompt(
            dialect_code="awa",
            dialect_name="Awadhi (अवधी)",
            region="Central UP (Ayodhya, Lucknow, Barabanki, Sultanpur, Unnao)",
            welcome_message="राम-राम! बाल-सेतु कार्यक्रम मा आपन स्वागत अहइ।",
            weekly_activity_audio="इ हफ्ता का खेल: लरिका के आगे कटोरी, चमचा धरि के पूछो कि इ का अहइ? लरिका जब नाम बोलै त खुश होइके शाबाशी देव।",
            keypad_prompt="जौ लरिका इ खेल खेलिस होय त 1 दबावा। आशा दीदी से बात करे बरे 2 दबावा।",
            confirmation_success="धन्यवाद! लरिका के खेल दर्ज होइ गवा। अगले हफ्ता फिर खबर लीब।"
        ),
        "bnd": DialectAudioPrompt(
            dialect_code="bnd",
            dialect_name="Bundeli (बुंदेली)",
            region="Bundelkhand (Jhansi, Lalitpur, Banda, Chitrakoot, Mahoba)",
            welcome_message="जय राम जी की! बाल-सेतु सेवा में आप को स्वागत है।",
            weekly_activity_audio="ई हफ्ता को खेल: मोड़ा-मोड़ी के सामने कटोरी-चमचा रखो और नाम पूछो। मोड़ा बोले तो पीठ थपथपा के शाबाशी दो।",
            keypad_prompt="अगर मोड़ा ने खेल खेल लओ होवे तो 1 दबाओ। आशा दीदी से पूछवे लाने 2 दबाओ।",
            confirmation_success="बड़ो धन्यवाद! मोड़ा की प्रगति दर्ज हो गई है। अगले हफ्ता फिर फोन करेंहें।"
        ),
        "braj": DialectAudioPrompt(
            dialect_code="braj",
            dialect_name="Braj (ब्रजभाषा)",
            region="Western UP (Mathura, Agra, Aligarh, Firozabad)",
            welcome_message="राधे-राधे! बाल-सेतु कार्यक्रम में आपकौ स्वागत है।",
            weekly_activity_audio="या हफ्ता कौ खेल: लल्ला के आगे कटोरी अरु चम्मच धर के नाम पूछौ। जब लल्ला बोलै तौ लाड़ लड़ाय के शाबाशी द्यौ।",
            keypad_prompt="यदि लल्ला नै खेल खेल्यौ होय तौ 1 दबाओ। आशा बहना कूं बुलवायबे कूं 2 दबाओ।",
            confirmation_success="धन्यवाद! लल्ला की प्रगति दर्ज ह्वै गई। राधे-राधे!"
        )
    }

    @classmethod
    def simulate_call_session(
        cls,
        phone_number: str = "+91-9876543210",
        dialect_code: str = "hi",
        pressed_key: str = "1"
    ) -> Dict:
        """
        Executes a simulated IVR call session.
        """
        prompt = cls.PROMPTS.get(dialect_code, cls.PROMPTS["hi"])

        status = "Completed Successfully" if pressed_key == "1" else ("ASHA Callback Requested" if pressed_key == "2" else "Invalid Input")
        sync_to_ekavach = (pressed_key == "1")

        return {
            "phone_number": phone_number,
            "dialect_code": dialect_code,
            "dialect_name": prompt.dialect_name,
            "call_duration_seconds": 92,
            "call_cost_inr": 0.0,  # Toll-free NHM missed-call line
            "welcome_transcript": prompt.welcome_message,
            "activity_transcript": prompt.weekly_activity_audio,
            "keypad_received": pressed_key,
            "call_status": status,
            "response_message": prompt.confirmation_success if pressed_key == "1" else "आशा कार्यकर्ता को संदेश भेज दिया गया है, वे जल्द संपर्क करेंगी।",
            "ekavach_sync_status": "Synced to ASHA Task List" if sync_to_ekavach else "Follow-up Triggered"
        }
