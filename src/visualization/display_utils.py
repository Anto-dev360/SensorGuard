"""
display_utils.py

UI display functions for the Streamlit app.

Author: Anthony Morin
Created: 2025-06-20
Project: SensorGuard
License: MIT
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from core.model_utils import (load_model, load_scaler, predict_batch,
                              predict_single)
from settings.config import (FEATURE_KEYS, FEATURE_LABELS, MODEL_PATH,
                             SCALER_PATH, TABS)


def display_set_cfg():
    """
    Configure the Streamlit page settings and apply a consistent banner across all tabs.

    Sets the page title, icon, and layout style for the SensorGuard application.
    Also renders a header banner to provide consistent context throughout the application.

    Configuration includes:
    - Page title: 'SensorGuard'
    - Favicon icon: Shield emoji
    - Layout: Centered
    """
    st.set_page_config(page_title="SensorGuard", page_icon="🛡️", layout="centered")

    # Banner header (applies globally if function is reused at the top of each tab)
    st.markdown(
        "<h1 style='text-align: center; color: #1f77b4;'>🛡️ SensorGuard – Industrial Failure Prediction</h1>",
        unsafe_allow_html=True,
    )


def display_create_sidebar():
    """
    Create the main sidebar navigation for the SensorGuard application.

    Provides a radio button interface to navigate between different app sections,
    such as 'About', 'Manual Input', 'Batch Upload', and 'Libraries Used'.

    Returns:
        str: The currently selected section/tab.
    """

    # Stylized sidebar header
    st.sidebar.markdown(
        "<h2 style='color:#1f77b4; margin-bottom: 0;'>🛠️ SensorGuard</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<p style='margin-top: 0; font-size: 0.9em;'>Industrial Failure Prediction Tool</p>",
        unsafe_allow_html=True,
    )

    # Navigation radio button
    selection = st.sidebar.radio("📁 Choose a section:", TABS)

    return selection


def display_about():
    """Display about section."""
    st.title("📊 SensorGuard - Predictive Maintenance")
    st.markdown(
        "SensorGuard uses machine learning to predict if a machine is at risk of failing within 7 days "
        "based on operational and sensor data. Models are trained on labeled historical records."
    )


def display_load_model():
    """
    Load the trained machine learning model and its associated scaler.

    This function attempts to load the model and the scaler from the paths defined
    in the application settings. A spinner is displayed during the loading process,
    and success or failure messages are shown in the Streamlit UI.

    Returns:
        tuple: (model, scaler)
            model: Trained classifier object.
            scaler: Fitted StandardScaler object.

    Raises:
        Displays Streamlit error and stops the app if loading fails.
    """
    # Load model and scaler with user feedback
    with st.spinner("🔄 Loading predictive model and scaler..."):
        try:
            model = load_model(MODEL_PATH)
            scaler = load_scaler(SCALER_PATH)
            st.success("✅ Model and scaler loaded successfully!")
        except FileNotFoundError as e:
            st.error(f"❌ File not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"❌ Unexpected error during model loading: {e}")
            st.stop()

    return model, scaler


def display_single_input(model, scaler):
    """
    Interface for single row prediction from manual input.

    Args:
        model: Trained ML model.
        scaler: Trained StandardScaler.
    """
    st.header("🧪 Manual Input Prediction")

    inputs = {}
    for label, key in zip(FEATURE_LABELS, FEATURE_KEYS):
        value = st.number_input(label, value=0.0)
        inputs[key] = value

    if st.button("Predict"):
        try:
            result = predict_single(model, scaler, inputs)
            if result == 1:
                st.error("⚠️ Prediction: Failure within 7 days")
            else:
                st.success("✔️ Prediction: No failure within 7 days")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


def display_batch_input(model, scaler):
    """
    Interface for batch prediction via CSV upload, scaler

    Args:
        model: Trained classification model.
        scaler: Trained StandardScaler.
    """
    st.header("📂 Predict from CSV File")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing the correct features", type="csv"
    )

    if uploaded_file:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)

            # Check that all required columns are present
            if not all(key in df.columns for key in FEATURE_KEYS):
                st.error(
                    "❌ CSV must contain the following columns:\n"
                    + ", ".join(FEATURE_KEYS)
                )
                return

            # Keep only required features and in the correct order
            df_input = df[FEATURE_KEYS]

            # Make predictions
            predictions = predict_batch(model, scaler, df_input)

            # Add prediction column
            df["Prediction"] = [
                "⚠️ Failure within 7H" if pred else "✔️ No Failure within 7H"
                for pred in predictions
            ]

            # Display results
            st.success("✅ Predictions completed!")
            st.dataframe(df)

            # Show a pie chart of prediction distribution
            result_counts = pd.Series(predictions).value_counts().sort_index()
            labels = ["No Failure", "Failure"]
            values = [result_counts.get(0, 0), result_counts.get(1, 0)]

            fig, ax = plt.subplots()
            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=["green", "red"],
            )
            ax.axis("equal")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error processing the file:\n{e}")


def display_libraries():
    """Display used libraries."""
    st.header("📚 Libraries Used")
    st.markdown(
        """
    - [Streamlit](https://streamlit.io/)
    - [scikit-learn](https://scikit-learn.org/)
    - [pandas](https://pandas.pydata.org/)
    - [joblib](https://joblib.readthedocs.io/)
    """
    )
