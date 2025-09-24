import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# =======================
# Load trained models
# =======================
try:
    leak_model = joblib.load(
        "/Users/yebhonelin/Documents/github/Smart-Water-Distribution-Optimization/leak_prediction/leak_detection_model.pkl"
    )   # from leak_pred.ipynb
except Exception:
    leak_model = None
    st.warning("⚠️ Leak model not found. Please place leak_model.pkl in the project folder.")

try:
    pump_model = joblib.load(
        "/Users/yebhonelin/Documents/github/Smart-Water-Distribution-Optimization/water_pump_maintenance/rf_model.pkl"
    )   # from water_pump_maint.ipynb
except Exception:
    pump_model = None
    st.warning("⚠️ Pump model not found. Please place pump_model.pkl in the project folder.")

# =======================
# Preprocess function
# =======================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    FEATURES = [
        "Q-E", "Z-V", "Q-V", "N-V", "P-V", "D-E", "O-E", "DB-E", "S-E", "SS-E", "D-S",
        "PH-E", "DB-S", "SS-S", "D-D", "PH-D", "DB-D", "SS-D", "D-R", "PH-R", "DB-R", "SS-R",
        "D-I", "E-I", "S-I", "SS-I", "D-O", "PH-O", "DB-O", "SS-O", "P", "D", "PH", "DB",
        "SS", "RD-DB", "RD-SS"
    ]
    df = df.reindex(columns=FEATURES)   # ensures correct order & count

    # Encode categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df


# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Smart Water Optimization", layout="wide")

st.title("💧 Smart Water Distribution Optimization")
st.markdown("""
AI-powered leak prediction and pump scheduling.  
Supports **SDG 6 – Clean Water & Sanitation** 🌍
""")

with st.expander("📌 Problem Statement"):
    st.write("""
    - Minimize water loss through leaks  
    - Reduce pump energy consumption  
    - Maintain network pressure  
    - Improve sustainability in urban water management
    """)

st.header("🔎 Leak Prediction")
uploaded_file = st.file_uploader("Upload network data (CSV)", type=["csv"])

if uploaded_file and leak_model:
    raw_data = pd.read_csv(uploaded_file)
    st.subheader("📄 Sample Data")
    st.dataframe(raw_data.head())

    clean_data = preprocess_data(raw_data)

    preds = leak_model.predict_proba(clean_data)[:, 1]
    raw_data["Leak_Probability"] = preds

    st.subheader("📊 Leak Risk Results")
    st.dataframe(raw_data.head(10))

    col1, col2 = st.columns(2)

    with col2:
        threshold = 0.5
        leak_counts = [
            sum(raw_data["Leak_Probability"] >= threshold),
            sum(raw_data["Leak_Probability"] < threshold),
        ]
        fig_pie, ax_pie = plt.subplots(figsize=(2.5, 2.5))
        ax_pie.pie(leak_counts, labels=["High Risk", "Low Risk"], autopct="%1.0f%%", colors=["red", "green"])
        ax_pie.set_title("Leak Risk Distribution", fontsize=10)
        st.pyplot(fig_pie, use_container_width=False)
    
    with col1:
        st.subheader("📋 Leak Risk Summary")
        st.header(f"High Risk Locations: {leak_counts[0]} ({leak_counts[0]/sum(leak_counts)*100:.1f}%)")
        st.header(f"Low Risk Locations: {leak_counts[1]} ({leak_counts[1]/sum(leak_counts)*100:.1f}%)")

st.header("⚡ Pump Scheduling Optimization")

# --- Pump Maintenance Section ---
st.header("⚙️ Pump Predictive Maintenance")

pump_file = st.file_uploader("Upload pump sensor data (CSV)", type=["csv"], key="pump")

if pump_file and pump_model:
    pump_data = pd.read_csv(pump_file)

    st.subheader("📄 Uploaded Pump Data")
    st.dataframe(pump_data.head())

    # Predict Remaining Useful Life or Failure Probabilities
    if hasattr(pump_model, "predict_proba"):
        pump_preds = pump_model.predict_proba(pump_data)[:, 1]
        pump_data["Failure_Probability"] = pump_preds
    else:
        pump_preds = pump_model.predict(pump_data)
        pump_data["RUL_or_Status"] = pump_preds

    st.subheader("📊 Pump Maintenance Predictions")
    st.dataframe(pump_data.head(10))

    # Example visualization
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(pump_preds, bins=20, color="orange", edgecolor="black")
    ax.set_title("Pump Failure / RUL Distribution")
    ax.set_xlabel("Prediction Value")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.success("✅ Predictions generated for uploaded pump sensor data")

# --- SDG Section ---
st.header("🌍 Alignment with SDG 6")
st.success("✅ Saves water | ✅ Saves energy | ✅ Ensures reliable supply")
