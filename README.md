
# 🛡️ SensorGuard

**SensorGuard** is a machine learning pipeline designed to predict industrial machine failures within 7 days, using real-time sensor data. The objective is to enhance preventive maintenance, reduce unplanned downtimes, and increase operational efficiency.

---

## 📦 Project Overview

Using a labeled dataset simulating factory conditions in 2040, the project leverages environmental and operational sensor readings (e.g. temperature, vibration, power consumption) to classify whether a failure will occur in the next 7 days.


## 🧠 Model Pipeline

- Data cleaning and preprocessing
- Class balancing
- Feature correlation and selection
- Model training with:
  - **Logistic Regression**
  - **SVM (Support Vector Machine)**
  - **Random Forest**
- Evaluation via:
  - Classification report
  - Confusion matrix
  - ROC AUC Curve
- ✅ Best model selection based on class 1 recall
- 💾 Export of trained model and scaler to `.pkl`
- 📊 Streamlit interface for single and batch predictions


## 📁 Project Structure

```
SensorGuard/
├── data/
│   └── factory_sensor_simulator_2040.csv      # Main dataset
├── models/                                    # Trained model and its scaler
│   ├── failure_predictor_support_vector_machine.pkl
|   └── scaler.pkl
├── notebooks/
│   └── modeling_failure_prediction.ipynb      # Full pipeline: cleaning, training, and model saving
├── src/
│   ├── app.py                                 # Streamlit entrypoint
│   ├── core/
│   │   ├── model_utils.py                     # Prediction logic
│   ├── settings/
│   │   └── config.py                          # Global constants
│   ├── visualization/
│   │   └── display_utils.py                   # Streamlit UI handlers
├── tests                                      # Example to test application
│   ├── sample_input.csv
│   └── sample_input.pdf
├── results                                    # Outputs generated after test runs
│   └── batch_predictions.csv
├── LICENSE
├── requirements.txt
├── README.md
└── .gitignore
```


## 📚 Dataset Description

**File:** `factory_sensor_simulator_2040.csv`
**Rows:** 120,000
**Target:** `Failure_Within_7_Days` (binary label)

**Selected Input Features:**

| Feature               | Description                           |
|------------------------|---------------------------------------|
| `Operational_Hours`    | Total machine operating hours         |
| `Temperature_C`        | Measured temperature in Celsius       |
| `Vibration_mms`        | Machine vibration in mm/s             |

Additional sensor features are explored in the notebook but only the most impactful ones are retained in the final model.


## 📓 Notebook Overview: `modeling_failure_prediction.ipynb`

This Jupyter notebook contains the complete machine learning workflow, from raw data preprocessing to model export. It serves as a reproducible development pipeline and includes the following steps:

1. **Data Loading**: Imports the raw sensor dataset.
2. **Data Cleaning and Preprocessing**:
   - Handling of missing values
   - Feature normalization/scaling
3. **Exploratory Data Analysis**:
   - Correlation heatmaps
   - Distribution plots of features
4. **Class Balancing**: Application of techniques like SMOTE or undersampling to address class imbalance.
5. **Feature Selection**: Based on correlation and importance metrics.
6. **Model Training**:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
7. **Model Evaluation**:
   - Confusion matrix
   - Classification report
   - ROC AUC curves
8. **Best Model Selection**: Based on class 1 recall performance.
9. **Model Export**: Saves the trained model and scaler using `joblib`.

The notebook can be executed independently to reproduce all modeling steps and final artifacts.

⚠️ Warning: This notebook is written in French 🇫🇷

## 🛠️ Libraries Used

- `pandas`, `numpy` for data handling
- `scikit-learn` for ML pipeline
- `matplotlib`, `seaborn` for data visualization
- `joblib` for model persistence
- `streamlit` for interactive front-end


## 🚀 Running the App

> Make sure to create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```

> Install dependencies:

```bash
pip install -r requirements.txt
```

> Launch Streamlit interface:

```bash
cd src
streamlit run app.py
```


## ⚙️ Streamlit Application Features

The Streamlit interface offers a simple and interactive way to use the trained model for failure prediction. It supports both single-record predictions and batch processing.

### 🔎 Manual Input

- Input three key machine parameters manually:
  - `Operational_Hours`
  - `Temperature_C`
  - `Vibration_mms`
- Instantly receive a prediction on whether a failure is likely to occur within the next 7 days.
- Visual feedback and explanations for the prediction provided (e.g., feature contributions).

### 📂 CSV Batch Prediction

- Upload a `.csv` file containing multiple sensor readings.
- The system processes all rows and returns:
  - Individual predictions
  - Summary statistics
  - Downloadable results file (`.csv`)
  - Visual overview of batch predictions (e.g., failure distribution plot)

### 🧪 Underlying Model

- Uses a **Support Vector Machine (SVM)** classifier:
  - Hyperparameters tuned via **GridSearchCV**.
  - **Probability calibration** applied to support threshold tuning.
  - Threshold selected to **maximize recall for the failure class**, minimizing the risk of missing a potential failure.
- Optimized for:
  - Real-time predictions
  - Fast inference speed
  - Deployment readiness (via `.pkl` export)


## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

