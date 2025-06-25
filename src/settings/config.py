"""
config.py

Configuration constants for the Streamlit app.

Author: Anthony Morin
Created: 2025-06-20
Project: SensorGuard
License: MIT
"""

import os

# File path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "failure_predictor_support_vector_machine.pkl"
)
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# UI constants
DISPLAY_TABS = ["🔍 About", "📈 Predict One", "📂 Predict Batch", "📚 Libraries"]
DISPLAY_ICON = "🛡️"
DISPLAY_TITLE = "SensorGuard"

# Model constants
FEATURE_LABELS = ["Operational Hours", "Temperature (°C)", "Vibration (mm/s)"]
FEATURE_KEYS = ["Operational_Hours", "Temperature_C", "Vibration_mms"]
