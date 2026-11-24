"""
Bal-Setu Offline Audio Asset Generator
Pre-generates crisp offline audio prompts in natural Hindi and Bhojpuri for:
- worker_pwa/audio/ (ASHA screening wizard, quality gate HUD alerts, D4 prompts)
- parent_loop/mini_pwa/audio/ (Parent speech games, praise, and dialect IVR)
"""

import os
import subprocess

WORKER_AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "worker_pwa", "audio"))
PARENT_AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "parent_loop", "mini_pwa", "audio"))

PROMPTS = [
    # Worker PWA Prompts (Hindi & Bhojpuri)
    (WORKER_AUDIO_DIR, "step1_hi", "नमस्ते आशा दीदी! बच्चे का विवरण और ई-कवच संख्या दर्ज करें।"),
    (WORKER_AUDIO_DIR, "step1_bho", "प्रणाम आशा दीदी! बबुआ के ई-कवच नंबर आ नाम दर्ज करीं।"),
    (WORKER_AUDIO_DIR, "step2_hi", "कैलिबकार्ड को बच्चे के कंधे के पास रखें। हरा घेरा बनने पर ही फोटो ली जाएगी।"),
    (WORKER_AUDIO_DIR, "step2_bho", "कैलिबकार्ड के बबुआ के कंधा लगे राखीं। हरियर घेरा बनला पर फोटो खींचा जाई।"),
    (WORKER_AUDIO_DIR, "step3_hi", "बच्चे के विकास, पोषण और शारीरिक लक्षणों की पुष्टि करें।"),
    (WORKER_AUDIO_DIR, "step3_bho", "बबुआ के वजन, पोषण आ देह के लक्षण के जांच करीं।"),
    (WORKER_AUDIO_DIR, "step4_hi", "अब बच्चे को फोन दें। ई की दिशा बताएं और चमकते सितारे को छूएं।"),
    (WORKER_AUDIO_DIR, "step4_bho", "अब बबुआ के फोन दीं। ई के मुंह केने बा बताईं आ तारा के छुईं।"),
    (WORKER_AUDIO_DIR, "step5_hi", "जांच पूरी हुई। अभिभावक को रंग का अर्थ समझाएं व डिजिटल टोकन सौंपें।"),
    (WORKER_AUDIO_DIR, "step5_bho", "जांच पूरा भइल। माई-बाप के रँग के मतलब समझाईं आ टोकन दीं।"),
    (WORKER_AUDIO_DIR, "quality_ok_hi", "गुणवत्ता गेट पास! अब आप फोटो ले सकते हैं।"),
    (WORKER_AUDIO_DIR, "quality_blur_hi", "सावधान! फोटो धुंधली है। कृपया कैमरा स्थिर रखें।"),
    (WORKER_AUDIO_DIR, "quality_light_hi", "रोशनी बहुत कम है। कृपया बच्चे को खिड़की या रोशनी के पास लाएं।"),
    (WORKER_AUDIO_DIR, "ocr_success_hi", "वजन मशीन और हीमोग्लोबिन स्क्रीन सफलतापूर्वक पढ़ी गई।"),

    # Parent Mini-PWA Prompts
    (PARENT_AUDIO_DIR, "apple_hi", "यह एक सेब है। बोलो: सेब!"),
    (PARENT_AUDIO_DIR, "bowl_hi", "यह कटोरी है। बोलो: कटोरी!"),
    (PARENT_AUDIO_DIR, "spoon_hi", "यह चम्मच है। बोलो: चम्मच!"),
    (PARENT_AUDIO_DIR, "water_hi", "यह पानी है। बोलो: पानी!"),
    (PARENT_AUDIO_DIR, "praise_hi", "शाबाश! बहुत अच्छा बोला!"),
    (PARENT_AUDIO_DIR, "bho_ivr", "प्रणाम! बाल-सेतु सेवा में राउर स्वागत बा। एह हफ्ता के खेल: बबुआ के सामने कटोरी आ चमचा रखीं। नाम बोले खातिर कही।"),
    (PARENT_AUDIO_DIR, "awa_ivr", "राम-राम! बाल-सेतु कार्यक्रम मा आपन स्वागत अहइ। लरिका के आगे कटोरी-चमचा धरि के नाम पूछो।")
]

def generate():
    for target_dir in [WORKER_AUDIO_DIR, PARENT_AUDIO_DIR]:
        os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("  Bal-Setu: Synthesizing Offline Hindi & Bhojpuri Audio Assets")
    print("=" * 60)

    for out_dir, name, text in PROMPTS:
        wav_path = os.path.join(out_dir, f"{name}.wav")
        m4a_path = os.path.join(out_dir, f"{name}.m4a")

        # 1. Generate clean 22.05kHz 16-bit WAV via macOS Lekha voice
        cmd_say = [
            "say", "-v", "Lekha",
            "-o", wav_path,
            "--data-format=LEI16@22050",
            text
        ]
        res = subprocess.run(cmd_say, capture_output=True)
        if res.returncode != 0:
            print(f"Failed to synthesize {name}: {res.stderr.decode()}")
            continue

        # 2. Compress to AAC .m4a for ultra-lightweight offline caching
        cmd_convert = [
            "afconvert",
            "-f", "m4af",
            "-d", "aac",
            "-b", "48000",
            wav_path,
            m4a_path
        ]
        subprocess.run(cmd_convert, capture_output=True)

        size_wav = os.path.getsize(wav_path) / 1024
        size_m4a = os.path.getsize(m4a_path) / 1024 if os.path.exists(m4a_path) else 0
        print(f"  ✓ {name}: WAV {size_wav:.1f} KB | M4A {size_m4a:.1f} KB -> '{text[:30]}...'")

    print("-" * 60)
    print("All Offline Audio Assets Synthesized & Compressed Successfully!")

if __name__ == "__main__":
    generate()
