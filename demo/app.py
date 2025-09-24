import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

st.title("Water Pump Maintenance Dashboard")

# Load model
model = pickle.load(open("/Users/yebhonelin/Documents/github/Smart-Water-Distribution-Optimization/water_pump_maintenance/rf_model.pkl", "rb"))

# Upload CSV
df = None
uploaded_file = st.file_uploader("Upload sensor data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Sensor Data", df.head())
    df = df.drop(columns=['Unnamed: 0'])
    with st.expander("ℹ️ Sensor Mapping (Unofficial)"):
        st.markdown("""
        Based on domain knowledge and dataset analysis, the following mappings are likely:

        - **Motor Casing Vibration:** sensor_00, sensor_13, sensor_18  
        - **Motor Frequency (Hz):** sensor_01 – sensor_03  
        - **Motor Current (A):** sensor_05  
        - **Motor Active Power (kW):** sensor_06  
        - **Motor Apparent Power (kVA):** sensor_07  
        - **Motor Reactive Power (kVAR):** sensor_08  
        - **Phase Currents (A/B/C):** sensor_10 – sensor_12  
        - **Phase Voltages (AB, BC, CA):** sensor_14, sensor_16, sensor_17  
        - *(sensor_15 likely phase average but not recorded)*  
        - Remaining sensors (~19–51) may capture gearbox and pump values (temperature, vibration, flow, etc.).
        
        ⚠️ Note: These mappings are inferred, not official from the dataset.
        """)

    # Show charts (first 4 numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 4:
        fig, ax = plt.subplots()
        sns.lineplot(data=df[numeric_cols[:4]], ax=ax)
        st.pyplot(fig)
    elif len(numeric_cols) > 0:
        fig, ax = plt.subplots()
        sns.lineplot(data=df[numeric_cols], ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for charting.")

    # Preprocess for prediction
    X = df[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Predict
    predictions = model.predict(X_scaled)
    st.write("Predictions", predictions)
