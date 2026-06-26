"""
Food Delivery Time Predictor
-----------------------------
Loads the trained pipeline (delivery_time_model.pkl) and metadata
(model_metadata.json) produced at the end of the training notebook, then
serves a simple form to predict Delivery_Time_min for a new order.

Run with:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
)

st.markdown(
    """
    <style>
        .result-card {
            background: linear-gradient(135deg, #2563EB 0%, #1E3A8A 100%);
            color: white;
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            text-align: center;
            margin-top: 1rem;
        }
        .result-card .big {
            font-size: 2.6rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .result-card .label {
            font-size: 0.95rem;
            opacity: 0.85;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Load model + metadata (cached so it only happens once per session)
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "delivery_time_model.pkl"
METADATA_PATH = Path(__file__).parent / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata


model, metadata = load_artifacts()

st.title("🛵 Food Delivery Time Predictor")
st.write(
    "Estimate how long an order will take to arrive, based on distance, "
    "traffic, weather, and courier details."
)

if model is None:
    st.error(
        "Couldn't find **delivery_time_model.pkl** / **model_metadata.json** "
        "next to this app. Run the export cell at the end of the training "
        "notebook first, then place both files in the same folder as "
        "`app.py`."
    )
    st.stop()

cat_options = metadata["categorical_options"]
num_ranges = metadata["numeric_ranges"]
experience_fallback = metadata["experience_level_mode_fallback"]

# --------------------------------------------------------------------------
# Feature engineering — mirrors the notebook EXACTLY so the model sees the
# same derived features at inference time that it saw during training.
# --------------------------------------------------------------------------


def categorize_distance(dist_km: float) -> str:
    if dist_km <= 6:
        return "Low distance"
    elif dist_km <= 14:
        return "Medium distance"
    else:
        return "High distance"


def categorize_experience(years: float, fallback: str) -> str:
    # Notebook bins: [0, 2, 6, 9] -> Junior / Mind-Level / Senior,
    # include_lowest=True, with NaNs (years > 9) filled by the training
    # mode. We replicate that fallback here rather than letting anything
    # above the trained bin range fall through untreated.
    if years <= 2:
        return "Junior"
    elif years <= 6:
        return "Mind-Level"
    elif years <= 9:
        return "Senior"
    else:
        return fallback


# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("predict_form"):
    st.subheader("Order details")

    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.slider(
            "Distance (km)",
            min_value=0.0,
            max_value=max(30.0, float(num_ranges["Distance_km"][1])),
            value=float(num_ranges["Distance_km"][0] + num_ranges["Distance_km"][1]) / 2,
            step=0.5,
        )
        prep_time = st.slider(
            "Preparation time (min)",
            min_value=0.0,
            max_value=max(45.0, float(num_ranges["Preparation_Time_min"][1])),
            value=float(num_ranges["Preparation_Time_min"][0] + num_ranges["Preparation_Time_min"][1]) / 2,
            step=1.0,
        )
    with col2:
        courier_exp = st.slider(
            "Courier experience (yrs)",
            min_value=0.0,
            max_value=max(15.0, float(num_ranges["Courier_Experience_yrs"][1])),
            value=float(num_ranges["Courier_Experience_yrs"][0]),
            step=0.5,
        )
        time_of_day = st.selectbox("Time of day", cat_options["Time_of_Day"])

    st.markdown("&nbsp;")
    col3, col4, col5 = st.columns(3)
    with col3:
        weather = st.selectbox("Weather", cat_options["Weather"])
    with col4:
        traffic = st.selectbox("Traffic level", cat_options["Traffic_Level"])
    with col5:
        vehicle = st.selectbox("Vehicle type", cat_options["Vehicle_Type"])

    submitted = st.form_submit_button("Predict delivery time")

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if submitted:
    row = pd.DataFrame(
        [
            {
                "Distance_km": distance_km,
                "Preparation_Time_min": prep_time,
                "Courier_Experience_yrs": courier_exp,
                "Weather": weather,
                "Traffic_Level": traffic,
                "Time_of_Day": time_of_day,
                "Vehicle_Type": vehicle,
                "Distance": categorize_distance(distance_km),
                "Experience_level": categorize_experience(courier_exp, experience_fallback),
            }
        ]
    )

    prediction = model.predict(row)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Estimated delivery time</div>
            <div class="big">{prediction:.0f} min</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See the exact row sent to the model"):
        st.dataframe(row, use_container_width=True)

st.markdown("---")
st.caption(
    "Model: Random Forest Regressor trained on historical delivery records. "
    "Predictions are estimates, not guarantees."
)
