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
from settings.config import (DISPLAY_ICON, DISPLAY_TABS, DISPLAY_TITLE,
                             FEATURE_KEYS, FEATURE_LABELS, MODEL_PATH,
                             SCALER_PATH)


def display_set_cfg():
    """
    Configure the Streamlit page settings and apply a consistent banner across all tabs.

    Sets the page title, icon, and layout style for the SensorGuard application.
    Also renders a header banner to provide consistent context throughout the application.
    """
    st.set_page_config(
        page_title=DISPLAY_TITLE, page_icon=DISPLAY_ICON, layout="centered"
    )

    # Banner header (applies globally)
    title = (
        "<h1 style='text-align: center; color: #1f77b4;'>"
        + DISPLAY_ICON
        + " "
        + DISPLAY_TITLE
        + "<br><br>Industrial Failure Prediction<br></h1>"
    )
    st.markdown(title, unsafe_allow_html=True)


def display_create_sidebar():
    """
    Create the main sidebar navigation for the SensorGuard application.

    Provides a radio button interface to navigate between different app sections,
    such as 'About', 'Manual Input', 'Batch Upload', and 'Libraries Used'.

    Returns:
        str: The currently selected section/tab.
    """

    # Stylized sidebar header
    h2_line = (
        "<h2 style='color:#1f77b4; margin-bottom: 0;'>"
        + DISPLAY_ICON
        + " "
        + DISPLAY_TITLE
        + "</h2>"
    )
    st.sidebar.markdown(h2_line, unsafe_allow_html=True)
    st.sidebar.markdown(
        "<p style='margin-top: 0; font-size: 0.9em;'>Industrial Failure Prediction Tool</p>",
        unsafe_allow_html=True,
    )

    # Navigation radio button
    selection = st.sidebar.radio("📁 Choose a section:", DISPLAY_TABS)

    return selection


def display_about():
    """
    Display the About section of the SensorGuard application.

    Provides an overview of the project, its purpose, use cases, and technical approach.
    """
    line = "📊 " + DISPLAY_TITLE + " - Predictive Maintenance Intelligence"
    st.title(line)

    st.markdown(
        """
        ### 🛡️ Empowering Industrial Safety Through Data

        SensorGuard is a **predictive maintenance tool** designed to forecast equipment failures
        within the next **7 days** using real-time sensor data and operational metrics.

        ---
        """
    )

    st.subheader("🎯 Project Objective")
    st.markdown(
        """
        The core goal of SensorGuard is to **reduce unplanned downtime**, **optimize maintenance costs**,
        and **improve equipment reliability** through smart analytics powered by machine learning.
        """
    )

    st.subheader("🔍 What We Predict")
    st.markdown(
        """
        Based on features such as:
        - **Operational Hours**
        - **Temperature (°C)**
        - **Vibration (mm/s)**

        SensorGuard classifies whether a machine is likely to **fail within the next 7 hours**.
        """
    )

    st.subheader("🏭 Example Use Cases")
    st.markdown(
        """
        - Real-time monitoring of industrial assets
        - Smart scheduling of preventive maintenance
        - Root cause analysis of frequent breakdowns
        """
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
    Display the batch prediction interface for CSV input in the Streamlit app.

    This function allows users to upload a CSV file, validates required columns,
    applies feature scaling, makes batch predictions, and displays the results
    in both tabular and pie chart form.

    Args:
        model: Trained classification model.
        scaler: Trained StandardScaler used during training.
    """
    st.header("📂 Predict from CSV File")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing the required features", type="csv"
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Validate required features
            missing_cols = [key for key in FEATURE_KEYS if key not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                return

            # Run predictions
            df_input = df[FEATURE_KEYS]
            predictions = predict_batch(model, scaler, df_input)

            # Create output DataFrame
            result_df = pd.DataFrame()
            if "Machine_ID" in df.columns:
                result_df["Machine_ID"] = df["Machine_ID"]

            for key in FEATURE_KEYS:
                result_df[key] = df[key]

            result_df["Failure within 7H"] = [
                "⚠️ Failure" if pred else "✔️ Functional" for pred in predictions
            ]

            # Display predictions
            st.success("✅ Predictions completed!")
            st.dataframe(result_df)

            # Download link for results
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_data,
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

            # Pie chart visualization
            result_counts = pd.Series(predictions).value_counts().sort_index()
            labels = ["Functional", "Failure"]
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
            plt.close(fig)

        except Exception as e:
            st.error(f"❌ Error processing the file:\n\n{str(e)}")


def display_libraries():
    """
    Display the list of libraries/frameworks used in the project, with a short description and license link.
    """
    st.header("📚 Libraries Used")

    st.markdown(
        """
        - 🖼️ [**Streamlit**](https://streamlit.io/): Interactive web app framework for building beautiful data-driven dashboards in Python.
        - 🤖 [**scikit-learn**](https://scikit-learn.org/): Core machine learning library used for model training, evaluation, and preprocessing (Logistic Regression, SVM, GridSearch...).
        - 🧮 [**pandas**](https://pandas.pydata.org/): Powerful data analysis library for manipulating and cleaning structured datasets (e.g., CSV files).
        - 📊 [**Matplotlib**](https://matplotlib.org/): Visualization library for generating ROC curves, confusion matrices, and performance charts.
        - 💾 [**joblib**](https://joblib.readthedocs.io/): Model serialization tool used to save/load trained models and scalers efficiently.

        ---

        🔓 **License**: This project is released under the [MIT License](https://opensource.org/licenses/MIT), allowing free usage, modification, and distribution.
        """
    )
