"""
app.py

Main Streamlit app for SensorGuard

Author: Anthony Morin
Created: 2025-06-20
Project: SensorGuard
License: MIT
"""

from settings.config import DISPLAY_TABS
from visualization.display_utils import (display_about, display_batch_input,
                                         display_create_sidebar,
                                         display_libraries, display_load_model,
                                         display_set_cfg, display_single_input)


def main():
    """
    Main entry point of the SensorGuard Streamlit application.

    This function initializes the Streamlit page configuration,
    loads the trained machine learning model and scaler, and displays
    the appropriate interface based on the user's selection from the sidebar.

    Tabs available:
        - About: General information about the project
        - Predict One: Manual input of features for a single prediction
        - Predict Batch: Upload a CSV file for batch prediction
        - Libraries: List of libraries used in the project
    """
    # Set Streamlit app configuration (page title, icon, layout)
    display_set_cfg()

    # Load trained model and associated scaler
    model, scaler = display_load_model()

    # Show sidebar navigation and get selected tab
    selection = display_create_sidebar()

    # Render the selected page based on user navigation
    if selection == DISPLAY_TABS[0]:
        # Main page
        display_about()
    elif selection == DISPLAY_TABS[1]:
        # Submit one sample
        display_single_input(model, scaler)
    elif selection == DISPLAY_TABS[2]:
        # Submit a CSV file containing several samples
        display_batch_input(model, scaler)
    elif selection == DISPLAY_TABS[3]:
        # List of embedded libraries
        display_libraries()


if __name__ == "__main__":
    main()
