<div align="center">

# 🛵 Food Delivery Time Prediction

**Predict how long a food delivery will take, from order to doorstep — powered by a Random Forest model and served through an interactive Streamlit app.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[Overview](#-overview) • [Demo](#-demo) • [Dataset](#-dataset) • [How It Works](#-how-it-works) • [Getting Started](#-getting-started) • [Results](#-results) • [Limitations](#-known-limitations) • [Roadmap](#-roadmap)

</div>

---

## 📦 Overview

Late deliveries hurt customer trust; overly conservative ETAs hurt conversion. This project trains a regression model to estimate **delivery time in minutes** from order- and courier-level signals — distance, traffic, weather, preparation time, vehicle type, and courier experience — and ships it as a self-contained, point-and-click **Streamlit** app.

The project covers the full lifecycle:

```
Raw CSV  →  EDA & Cleaning  →  Feature Engineering  →  ML Pipeline  →  Trained Model  →  Streamlit App
```

**At a glance**

| | |
|---|---|
| 🎯 **Task** | Regression — predict `Delivery_Time_min` |
| 🧠 **Model** | Random Forest Regressor (`n_estimators=345`) inside a scikit-learn `Pipeline` |
| 🗂️ **Inputs** | Distance, traffic level, weather, time of day, vehicle type, preparation time, courier experience |
| 🖥️ **Interface** | Streamlit web app with live predictions |
| 📊 **Notebook** | Full EDA, visualizations, preprocessing, training, and evaluation |

---

## 🎬 Demo

The app takes a few order details and returns an estimated delivery time instantly:

```
┌─────────────────────────────────────────┐
│  🛵  Food Delivery Time Predictor       │
├─────────────────────────────────────────┤
│  Distance (km)        [=======•----] 8.5│
│  Preparation time     [=====•------] 18 │
│  Courier experience   [===•--------] 3  │
│  Weather              [ Rainy      ▾]   │
│  Traffic level        [ Medium     ▾]   │
│  Vehicle type         [ Bike       ▾]   │
│                                         │
│           [ Predict delivery time ]     │
│                                         │
│   ┌───────────────────────────────┐     │
│   │   ESTIMATED DELIVERY TIME     │     │
│   │            42 min             │     │
│   └───────────────────────────────┘     │
└─────────────────────────────────────────┘
```

> 💡 Run `streamlit run app.py` and replace this mock with a real screenshot — see [Getting Started](#-getting-started).

---

## 🗃️ Dataset

The model is trained on `Food_Delivery_Times.csv`, where each row is one completed delivery.

| Column | Type | Description |
|---|---|---|
| `Order_ID` | id | Unique order identifier *(dropped before training)* |
| `Distance_km` | numeric | Distance between restaurant and customer |
| `Weather` | categorical | Weather conditions during delivery (e.g. Clear, Rainy, Foggy) |
| `Traffic_Level` | categorical | Traffic conditions (Low / Medium / High) |
| `Time_of_Day` | categorical | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | categorical | Bike / Scooter / Car |
| `Preparation_Time_min` | numeric | Time the restaurant took to prepare the order |
| `Courier_Experience_yrs` | numeric | Courier's years of experience |
| `Delivery_Time_min` | numeric | **Target** — total delivery time |

### Engineered features

Two additional features are derived during preprocessing and fed into the model alongside the raw columns:

- **`Distance`** — bucketed from `Distance_km`: `Low distance` (≤ 6 km), `Medium distance` (≤ 14 km), `High distance` (> 14 km)
- **`Experience_level`** — bucketed from `Courier_Experience_yrs` using bins `[0, 2, 6, 9]` → `Junior`, `Mind-Level`, `Senior`

---

## ⚙️ How It Works

### 1. Preprocessing pipeline

Built with `ColumnTransformer` so numeric and categorical columns are handled independently, then trained as one fitted object:

```
ColumnTransformer
├── numeric:     [Distance_km, Preparation_Time_min, Courier_Experience_yrs]
│                 └── SimpleImputer(strategy="median")
└── categorical: [Weather, Traffic_Level, Time_of_Day, Vehicle_Type,
│                 Distance, Experience_level]
│                 ├── SimpleImputer(strategy="most_frequent")
│                 └── OneHotEncoder(handle_unknown="ignore")
        ↓
RandomForestRegressor(n_estimators=345)
```

This whole pipeline — imputers, encoder, and trained trees — is saved as **one artifact**, so inference never needs to repeat any fitting logic.

### 2. Inference-time feature engineering

`Distance` and `Experience_level` are plain Python logic computed *before* the pipeline, not learned by it. The Streamlit app reproduces this logic exactly so there's no train/serve mismatch:

```python
def categorize_distance(dist_km):
    if dist_km <= 6:   return "Low distance"
    if dist_km <= 14:  return "Medium distance"
    return "High distance"

def categorize_experience(years, fallback):
    if years <= 2: return "Junior"
    if years <= 6: return "Mind-Level"
    if years <= 9: return "Senior"
    return fallback   # mirrors the notebook's mode-fill for >9 yrs
```

### 3. Serving

`app.py` loads the saved pipeline and a small metadata file (dropdown options, slider ranges, and the fallback experience label), builds a single-row `DataFrame` matching the training schema, and calls `model.predict(...)`.

---

## 📁 Project Structure

```
food-delivery-time-prediction/
├── data_preprocessing_and_model_training.ipynb   # EDA, cleaning, training, evaluation
├── Food_Delivery_Times.csv                       # Raw dataset (not included — bring your own)
├── app.py                                        # Streamlit inference app
├── requirements.txt                              # Python dependencies
├── delivery_time_model.pkl                       # Saved pipeline (generated by the notebook)
├── model_metadata.json                           # Dropdown options & ranges (generated by the notebook)
└── README.md                                     # You are here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- `Food_Delivery_Times.csv` placed alongside the notebook

### 1. Clone & install

```bash
git clone <your-repo-url>
cd food-delivery-time-prediction
pip install -r requirements.txt
```

### 2. Train the model

Run `data_preprocessing_and_model_training.ipynb` end-to-end. At the final cell, export the trained pipeline and metadata:

```python
import joblib, json

joblib.dump(model, "delivery_time_model.pkl")

metadata = {
    "categorical_options": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
    },
    "numeric_ranges": {
        "Distance_km": [float(df["Distance_km"].min()), float(df["Distance_km"].max())],
        "Preparation_Time_min": [float(df["Preparation_Time_min"].min()), float(df["Preparation_Time_min"].max())],
        "Courier_Experience_yrs": [float(df["Courier_Experience_yrs"].min()), float(df["Courier_Experience_yrs"].max())],
    },
    "experience_level_mode_fallback": df["Experience_level"].mode()[0],
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

This produces `delivery_time_model.pkl` and `model_metadata.json` in the project root.

### 3. Launch the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (typically `http://localhost:8501`) and start predicting.

---

## 📊 Results

Metrics from the notebook's evaluation cell (fill in after your own run — they depend on the specific train/test split and dataset):

| Metric | Train | Test |
|---|---|---|
| R² Score | `<fill in>` | `<fill in>` |
| MAE (min) | — | `<fill in>` |
| MSE | — | `<fill in>` |

A close train/test R² gap indicates the forest isn't overfitting; a wide gap is a sign to tune `n_estimators`, `max_depth`, or add regularization via `min_samples_leaf`.

---

## ⚠️ Known Limitations

- **Experience bins don't cover the full range.** `pd.cut` with bins `[0, 2, 6, 9]` produces `NaN` for couriers with more than 9 years' experience, silently patched with the training mode. Consider extending the top bin to `np.inf`.
- **Label typo preserved on purpose.** The mid-tier experience label is `"Mind-Level"` (not "Mid-Level") in the trained encoder's vocabulary — the app intentionally keeps this string so categories match exactly.
- **No random seed on the forest.** `RandomForestRegressor` was trained without `random_state`, so retraining from scratch won't reproduce bit-identical predictions (saving the fitted pipeline avoids this issue for serving).
- **Redundant features.** Both raw (`Distance_km`, `Courier_Experience_yrs`) and bucketed (`Distance`, `Experience_level`) versions are fed to the model — useful for tree splits, but worth revisiting for a leaner feature set.

---

## 🛣️ Roadmap

- [ ] Add `random_state` and cross-validation for reproducible, robust evaluation
- [ ] Hyperparameter tuning via the already-imported `GridSearchCV`
- [ ] Fix the open-ended experience bin and retire the mode-fill workaround
- [ ] Add model explainability (SHAP / feature importance chart) to the app
- [ ] Containerize with Docker for one-command deployment
- [ ] Add automated tests for the feature-engineering functions

---

## 🤝 Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

- Dataset: `Food_Delivery_Times.csv`
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [Streamlit](https://streamlit.io/)

<div align="center">

Made with 🛵 and 🐍

</div>
