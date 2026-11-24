"""
Bal-Setu: ONNX Edge AI Model Exporter & Training Dispatcher
Executes the comprehensive PyTorch edge AI training and single-file ONNX export pipeline
located in training/train_all_models.py.
Trains and exports all 4 single-file ONNX models:
1. skin_pallor_mobilenetv3.onnx (Fitzpatrick IV-VI MobileNetV3-Small)
2. field_lcd_caries_detector.onnx (7-segment LCD meter + intraoral caries)
3. speech_acoustic_prosody.onnx (Dialect-invariant 1D CNN prosody biomarkers)
4. motor_kinematics_classifier.onnx (2-layer GRU biomechanical pose sequence)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.train_all_models import (
    SkinPallorMobileNetV3,
    FieldLCDCariesDetector,
    SpeechAcousticProsodyModel,
    MotorKinematicsClassifier,
    train_skin_pallor_model,
    train_field_lcd_caries_model,
    train_speech_prosody_model,
    train_motor_kinematics_model,
    main
)

if __name__ == "__main__":
    main()
