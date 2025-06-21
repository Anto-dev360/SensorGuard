
# 🚧 SensorGuard

**SensorGuard** is a machine learning project designed to predict industrial machine failures within 7 days, using real-time sensor data. The objective is to enhance preventive maintenance, reduce unplanned downtimes, and increase operational efficiency.

---

## 📦 Project Overview

Using a labeled dataset simulating factory conditions in 2040, the project leverages environmental and operational sensor readings (e.g. temperature, vibration, power consumption) to classify whether a failure will occur in the next 7 days.

---

## 🧠 Model Pipeline

- Data cleaning and preprocessing
- Class balancing
- Feature correlation and selection
- Model training with:
  - **Logistic Regression**
  - **Random Forest**
- Evaluation via:
  - Classification report
  - Confusion matrix
  - ROC AUC Curve
- ✅ Best model selection and export to `.pkl`
- 📊 Streamlit front-end for single and batch prediction

---

## 📁 Project Structure

```
SensorGuard/
├── data/
│   └── factory_sensor_simulator_2040.csv      # Main dataset
├── models/                                    # Trained model
│   ├── failure_predictor_logistic_regression.pkl
|   └── scaler.pkl
├── notebooks/
│   └── modeling_failure_prediction.ipynb      # Full ML workflow
├── src/
│   ├── app.py                                 # Streamlit entrypoint
│   ├── core/
│   │   ├── model_utils.py                     # Prediction logic
│   ├── settings/
│   │   └── config.py                          # Global constants
│   ├── visualization/
│   │   └── display_utils.py                   # Streamlit UI handlers
|── test                                       # Example to test application
|   ├── sample_input.csv
|   └── sample_input.pdf
├── LICENSE
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📚 Dataset Description

**File:** `factory_sensor_simulator_2040.csv`
**Records:** 120,000
**Label:** `Failure_Within_7_Days` (Boolean)

**Selected Features:**

| Feature               | Description                               |
|------------------------|-------------------------------------------|
| `Operational_Hours`    | Total machine operating hours             |
| `Temperature_C`        | Measured temperature (Celsius)            |
| `Vibration_mms`        | Machine vibration in mm/s                 |

---

## 🛠️ Libraries Used

- `pandas`, `numpy` for data handling
- `scikit-learn` for ML pipeline
- `matplotlib`, `seaborn` for data visualization
- `joblib` for model persistence
- `streamlit` for interactive front-end

---

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

---

## ⚙️ Strealmit application features

### 🔎 Manual Input

- Predict failures by entering 3 key parameters

### 📂 CSV Batch Prediction

- Upload a `.csv` file with multiple records
- View predictions and a summary plot

### 🧪 Model

- Logistic Regression with threshold tuning
- Scalable for deployment and monitoring

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

