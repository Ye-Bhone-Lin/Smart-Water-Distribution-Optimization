import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from water_update.water_chat import WaterDataBot
from water_pump_maintenance.pump_model_work import PumpMaintenanceModel
from leak_prediction.leak_model_work import Leak_Model


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="💧 Smart Water Optimization", layout="wide")
st.title("Smart Water Distribution Optimization Dashboard")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Choose a section:", ["Pump Maintenance", "Leak Prediction", "Country Water Condition Update"])

if page == "Pump Maintenance":
    st.header("Pump Maintenance Monitoring")
    with st.expander("📄 Dataset Recommendations / Guidelines"):
        st.markdown("""
        Before uploading your pump sensor CSV, ensure it includes the following sensors/features:

        - **Motor Casing Vibration:** `sensor_00`, `sensor_13`, `sensor_18`  
        - **Motor Frequency (Hz):** `sensor_01` – `sensor_03`  
        - **Motor Current (A):** `sensor_05`  
        - **Motor Active Power (kW):** `sensor_06`  
        - **Motor Apparent Power (kVA):** `sensor_07`  
        - **Motor Reactive Power (kVAR):** `sensor_08`  
        - **Phase Currents (A/B/C):** `sensor_10` – `sensor_12`  
        - **Phase Voltages (AB, BC, CA):** `sensor_14`, `sensor_16`, `sensor_17`  
        - **Remaining sensors (~19–51):** may capture gearbox and pump values (temperature, vibration, flow, etc.)
        
        **Tips:**  
        - Keep column names exactly as listed.  
        - Avoid missing values in required sensors.  
        - CSV format is required for upload.
        """)
    uploaded_file = st.file_uploader("Upload pump sensor CSV", type="csv", key="pump")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        pump_model = PumpMaintenanceModel(
            "water_pump_maintenance/rf_model.pkl"
        )

        try:
            df_pred = pump_model.predict_full(df)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        status_counts = df_pred["Maintenance_Status"].value_counts()

        st.subheader("Pump Status Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Normal Pumps ✅", status_counts.get("NORMAL", 0))
        col2.metric("Recovering Pumps ⚠️", status_counts.get("RECOVERING", 0))
        col3.metric("Broken Pumps ❌", status_counts.get("BROKEN", 0))

        fig, ax = plt.subplots(figsize=(3, 3))  
        ax.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct="%1.0f%%",
            colors=["green", "yellow", "red"],
            textprops={"fontsize": 4},  
        )

        ax.set_title("Pump Maintenance Status Distribution", fontsize=5)  
        plt.tight_layout(pad=0.4)  
        st.pyplot(fig)
        selected_status = st.radio(
            "🔍 Select a status to see details:",
            options=["All"] + list(pump_model.class_labels),
            horizontal=True,
        )
        if selected_status != "All":
            filtered_df = df_pred[df_pred["Maintenance_Status"] == selected_status]
            st.subheader(f"Detailed Pumps: {selected_status}")
        else:
            filtered_df = df_pred
            st.subheader("All Pumps Details")

        st.dataframe(filtered_df)

        st.subheader("🔹 Pump Prediction Probabilities")
        for i, row in filtered_df.iterrows():
            st.subheader(f"**Pump {i+1} — Status: {row['Maintenance_Status']}**")
            
            # Small compact figure
            fig_prob, ax_prob = plt.subplots(figsize=(4, 0.6))  # slightly smaller
            probs = [row[f"Prob_{cls}"] for cls in pump_model.class_labels]
            
            ax_prob.barh(pump_model.class_labels, probs, color=["green", "yellow", "red"])
            ax_prob.set_xlim(0, 1)
            ax_prob.set_xlabel("Probability", fontsize=6)
            ax_prob.set_yticks(range(len(pump_model.class_labels)))
            ax_prob.set_yticklabels(pump_model.class_labels, fontsize=6)
            
            plt.tight_layout()  # removes extra padding
            st.pyplot(fig_prob)
            st.markdown("---")


elif page == "Leak Prediction":
    st.header("Leak Prediction")

    with st.expander("📄 Dataset Recommendations / Required Columns"):
        st.markdown("""
        | Parameters | Desc | Parameters 2 | Desc | Parameters 3 | Desc | Parameters 4 | Desc |
        |----------|------|----------|------|----------|------|----------|------|
        | Q-E     | input flow to plant | ZN-E     | input Zinc to plant | PH-E     | input pH to plant | DBO-E    | input Biological demand of oxygen to plant |
        | DQO-E   | input chemical demand of oxygen | SS-E     | input suspended solids | SSV-E    | input volatile suspended solids | SED-E    | input sediments to plant |
        | COND-E  | input conductivity | PH-P     | input pH to primary settler | DBO-P    | input Biological demand of oxygen to primary settler | SS-P     | input suspended solids to primary settler |
        | SSV-P   | input volatile suspended solids | SED-P    | input sediments to primary settler | COND-P   | input conductivity to primary settler | PH-D     | input pH to secondary settler |
        | DBO-D   | input Biological demand of oxygen to secondary settler | DQO-D    | input chemical demand of oxygen to secondary settler | SS-D     | input suspended solids to secondary settler | SSV-D    | input volatile suspended solids to secondary settler |
        | SED-D   | input sediments to secondary settler | COND-D   | input conductivity to secondary settler | PH-S     | output pH | DBO-S    | output Biological demand of oxygen |
        | DQO-S   | output chemical demand of oxygen | SS-S     | output suspended solids | SSV-S    | output volatile suspended solids | SED-S    | output sediments |
        | COND-S  | output conductivity | RD-DBO-P | performance input Biological demand of oxygen in primary settler | RD-SS-P | performance input suspended solids to primary settler | RD-SED-P | performance input sediments to primary settler |
        | RD-DBO-S | performance input Biological demand of oxygen to secondary settler | RD-DQO-S | performance input chemical demand of oxygen to secondary settler | RD-DBO-G | global performance input Biological demand of oxygen | RD-DQO-G | global performance input chemical demand of oxygen |
        | RD-SS-G | global performance input suspended solids | RD-SED-G | global performance input sediments | - | - | - | - |
        """)
    uploaded_file = st.file_uploader("Upload network data (CSV)", type=["csv"], key="leak")

    if uploaded_file:
        raw_data = pd.read_csv(uploaded_file)

        leak_model = Leak_Model(
            "leak_prediction/leak_detection_model.pkl"
        )
        probs = leak_model.predict_proba(raw_data)
        raw_data["Leak_Probability"] = probs

        st.subheader("Leak Risk Results")
        st.dataframe(raw_data.head(10))

        col1, col2 = st.columns(2)
        threshold = 0.5
        high_risk = sum(probs >= threshold)
        low_risk = sum(probs < threshold)

        with col1:
            st.subheader("📋 Leak Risk Summary")
            st.metric("High Risk Locations ", f"{high_risk} ({high_risk/(high_risk+low_risk)*100:.1f}%)")
            st.metric("Low Risk Locations ", f"{low_risk} ({low_risk/(high_risk+low_risk)*100:.1f}%)")

        with col2:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                [high_risk, low_risk],
                labels=["High Risk", "Low Risk"],
                autopct="%1.0f%%",
                colors=["red", "green"],
            )
            ax.set_title("Leak Risk Distribution", fontsize=10)
            st.pyplot(fig)


elif page == "Country Water Condition Update":
    st.header("💬 USGS Water Condition Update")
    
    if not GROQ_API_KEY:
        st.error("**Error:** GROQ_API_KEY environment variable not found.")
        st.warning("Please set the `GROQ_API_KEY` in your environment or Streamlit secrets to use the chatbot.")
        st.stop()
        
    try:
        bot = WaterDataBot(GROQ_API_KEY)
    except Exception as e:
        st.error(f"Could not initialize WaterDataBot: {e}")
        st.stop()

    st.write("Enter a **USGS site ID** to get the latest water conditions and AI-generated insights.")
    st.info("""
    **How to find a USGS Site ID:**  
    - Go to the USGS Water Data website: [https://waterdata.usgs.gov/nwis](https://waterdata.usgs.gov/nwis)  
    - Search for your river or location.  
    - Copy the 8-digit site ID (e.g., 01646500 for Potomac River near Washington, DC)  
    - Paste it in the input box below.
    """)
    site_id = st.text_input("USGS Site ID", value="01646500", help="Example: 01646500 (Potomac River near Washington, DC)")

    if st.button("Get Water Update", type="primary"):
        if not site_id:
            st.warning("Please enter a valid site ID.")
        else:
            with st.spinner("Fetching data and generating insights..."):
                data = bot.fetch_usgs_data(site_id)
                
                if isinstance(data, dict) and data:
                    st.success(f"✅ Latest Water Update for Site ID: **{site_id}**")
                    
                    for var, (val, ts) in data.items():
                        st.markdown(f"---")
                        st.metric(label=f"**{var}**", value=val, help=f"Recorded at: {ts}")
                        
                        try:
                            insights = bot.generate_insights(var, val, ts)
                            st.write(insights)
                        except Exception as e:
                            st.warning(f"Could not generate insights for {var}: {e}")

                else:
                    st.error(f"Could not fetch data for site ID **{site_id}**. Response: {data}")
