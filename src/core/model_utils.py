"""
model_utils.py

Utilities to load the model and run predictions.

Author: Anthony Morin
Created: 2025-06-20
Project: SensorGuard
License: MIT
"""

import os

import joblib
import pandas as pd

from settings.config import FEATURE_KEYS


def load_model(model_path):
    """
    Load the trained machine learning model from a specified path.

    Args:
        model_path (str): Path to the serialized model file.

    Returns:
        model (sklearn.base.BaseEstimator): Loaded model object.

    Raises:
        FileNotFoundError: If the file at model_path does not exist.
        Exception: If any other error occurs during loading.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def load_scaler(scaler_path):
    """
    Load a serialized scaler object from the specified file path.

    Args:
        scaler_path (str): The file path to the saved scaler (e.g., a .pkl file).

    Returns:
        sklearn.preprocessing.StandardScaler: The deserialized scaler object used for feature normalization.

    Raises:
        FileNotFoundError: If the scaler file does not exist at the given path.
        RuntimeError: If the file exists but cannot be loaded (e.g., due to corruption or incompatibility).
    """
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Model file not found at: {scaler_path}")

    try:
        model = joblib.load(scaler_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def predict_single(model, scaler, input_data: dict) -> str:
    """
    Predict failure using a single input.

    Args:
        model: Trained classifier.
        scaler: Trained StandardScaler.
        input_data (dict): Input features.

    Returns:
        str: Prediction result.
    """
    # Order of features must match training
    values = [
        input_data[FEATURE_KEYS[0]],
        input_data[FEATURE_KEYS[1]],
        input_data[FEATURE_KEYS[2]],
    ]
    values_scaled = scaler.transform([values])
    return model.predict(values_scaled)[0]


def predict_batch(model, scaler, df: pd.DataFrame) -> list:
    """
    Predict failures for a batch of input rows.

    Args:
        model: Trained classifier.
        scaler: Trained StandardScaler.
        df (pd.DataFrame): Input data containing only the required features
                           in the order [Operational_Hours, Vibration_mms, Temperature_C].

    Returns:
        list: Binary prediction results (0 or 1).
    """
    # Ensure feature order consistency
    values = df[FEATURE_KEYS].values  # Extract as numpy array
    values_scaled = scaler.transform(values)
    return model.predict(values_scaled).tolist()
