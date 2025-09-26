import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from water_update.water_chat import WaterDataBot
from water_pump_maintenance.pump_model_work import PumpMaintenanceModel
from leak_prediction.leak_model_work import Leak_Model
import streamlit as st
import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from geopy.distance import geodesic

st.secrets['GROQ_API_KEY']

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="💧 Smart Water Optimization", layout="wide")
st.title("Smart Water Distribution Optimization Dashboard")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Choose a section:", ["Pump Maintenance", "Leak Prediction", "Country Water Condition Update", "Network Controls"])

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

elif page == "Network Controls":
    def get_node_coords(G, node):
        """Returns (latitude, longitude) for a given node."""
        return (G.nodes[node]['y'], G.nodes[node]['x'])

    def add_network_attributes(G, num_pumps, num_tanks, valve_percentage):
        """Adds realistic, scattered components with robust, conflict-free logic."""
        try:
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        except ValueError:
            return G 

        if len(G.nodes) < num_pumps + num_tanks + 2:
            st.warning("Network is too small for the number of components requested.")
            return nx.Graph()

        node_attributes = {}
        node_coords_list = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}
        coords_array = np.array(list(node_coords_list.values()))

        hull = ConvexHull(coords_array)
        hull_nodes = {list(node_coords_list.keys())[i] for i in hull.vertices}

        if len(hull_nodes) <= num_tanks:
            num_tanks = max(0, len(hull_nodes) - 1)

        if hull_nodes:
            center_lon, center_lat = coords_array.mean(axis=0)
            center_coords = (center_lat, center_lon)
            hull_nodes_with_dist = sorted(
                [(node, geodesic(get_node_coords(G, node), center_coords).meters) for node in hull_nodes],
                key=lambda item: item[1]
            )
            source_node = hull_nodes_with_dist[0][0]
            node_attributes[source_node] = {'node_type': 'Treatment Plant', 'demand_m3_per_hr': -1000}
            
            tank_candidates = [node for node, dist in hull_nodes_with_dist[1:]]

    
        tanks = []
        if tank_candidates and num_tanks > 0:
            
            first_tank = tank_candidates.pop(-1)
            tanks.append(first_tank)

            while len(tanks) < num_tanks and tank_candidates:
                best_candidate = None
                max_min_dist = -1
                for cand_node in tank_candidates:
                    cand_coords = get_node_coords(G, cand_node)
                    min_dist_to_tanks = min([geodesic(cand_coords, get_node_coords(G, t)).meters for t in tanks])
                    
                    if min_dist_to_tanks > max_min_dist:
                        max_min_dist = min_dist_to_tanks
                        best_candidate = cand_node
                
                if best_candidate:
                    tanks.append(best_candidate)
                    tank_candidates.remove(best_candidate)

        for tank_node in tanks:
            node_attributes[tank_node] = {'node_type': 'Tank', 'capacity_m3': np.random.randint(500, 2000), 'demand_m3_per_hr': 0}
    

        centrality = nx.degree_centrality(G)
        candidates = [node for node, score in sorted(centrality.items(), key=lambda item: item[1], reverse=True) if node not in hull_nodes][:30]
        pumps = []
        if candidates:
            first_pump = candidates.pop(0)
            pumps.append(first_pump)
            while len(pumps) < num_pumps and candidates:
                best_candidate = None; max_min_dist = -1
                for cand_node in candidates:
                    cand_coords = get_node_coords(G, cand_node)
                    min_dist_to_pumps = min([geodesic(cand_coords, get_node_coords(G, p)).meters for p in pumps])
                    if min_dist_to_pumps > max_min_dist:
                        max_min_dist = min_dist_to_pumps; best_candidate = cand_node
                if best_candidate:
                    pumps.append(best_candidate); candidates.remove(best_candidate)

        for pump_node in pumps:
            node_attributes[pump_node] = {'node_type': 'Pump', 'pressure_bar': round(np.random.uniform(5, 8), 2), 'demand_m3_per_hr': 0}

        for node in G.nodes():
            if node not in node_attributes:
                node_attributes[node] = {'node_type': 'Junction', 'demand_m3_per_hr': round(np.random.uniform(0.5, 5), 2)}
        nx.set_node_attributes(G, node_attributes)

        pipe_diameters_mm = [100, 150, 200, 250, 300]
        edge_attributes = {}
        for u, v in G.edges():
            has_valve = True if np.random.rand() < (valve_percentage / 100.0) else False
            edge_attributes[(u, v)] = {
                'diameter_mm': np.random.choice(pipe_diameters_mm), 'roughness': 130, 'has_valve': has_valve
            }
        nx.set_edge_attributes(G, edge_attributes)
        
        return G

    st.title("💧 Realistic Water Network Modelling")
    st.markdown("Generate a synthetic water network with scattered pumps, tanks, and valves based on a real-world location.")

    with st.sidebar:
        st.header("⚙️ Network Controls")
        
        location_options = [
            "Insein District, Yangon, Myanmar",
            "Kamayut District, Yangon, Myanmar",
            "Dagon Myothit District, Yangon, Myanmar"
        ]
        location_query = st.selectbox("Select a Location", location_options)
        skip_probability = st.slider("Node Skip Probability", 0.0, 1.0, 0.25, 0.05, help="Higher values create a sparser network.")
        num_pumps = st.number_input("Number of Pumps", 1, 10, 3)
        num_tanks = st.number_input("Number of Water Tanks", 1, 10, 2)
        valve_percentage = st.slider("Percentage of Pipes with Valves (%)", 0, 100, 20)
        generate_button = st.button("Generate Network", type="primary")

    if generate_button:
        try:
            with st.spinner(f"Fetching street data for {location_query}..."):
                G_street = ox.graph_from_place(location_query, network_type='drive', simplify=True)
                G_street = nx.Graph(G_street.to_undirected())
                G_street = nx.convert_node_labels_to_integers(G_street)
            
            with st.spinner("Generating and enriching water network..."):
                G_water = G_street.copy()
                nodes_to_remove = [node for node in G_water.nodes() if np.random.rand() < skip_probability]
                G_water.remove_nodes_from(nodes_to_remove)
                
                G_water_rich = add_network_attributes(G_water, num_pumps, num_tanks, valve_percentage)
                
                if len(G_water_rich.nodes) == 0:
                    st.warning("⚠️ No network could be generated. Try a lower 'Node Skip Probability'.")
                    st.stop()
            
            st.success("✅ Water network generated.")

            with st.spinner("Preparing map visualization..."):
                nodes_df = pd.DataFrame.from_dict(dict(G_water_rich.nodes(data=True)), orient='index')
                
                pipes_reg_lon, pipes_reg_lat = [], []
                pipes_valve_lon, pipes_valve_lat = [], []

                for u, v, data in G_water_rich.edges(data=True):
                    lon_vals = [G_water_rich.nodes[u]['x'], G_water_rich.nodes[v]['x'], None]
                    lat_vals = [G_water_rich.nodes[u]['y'], G_water_rich.nodes[v]['y'], None]
                    if data.get('has_valve', False):
                        pipes_valve_lon.extend(lon_vals)
                        pipes_valve_lat.extend(lat_vals)
                    else:
                        pipes_reg_lon.extend(lon_vals)
                        pipes_reg_lat.extend(lat_vals)

            fig = go.Figure()
            
            # --- PLOTTING CHANGES START HERE ---

            # CHANGED: go.Scattermapbox -> go.Scattermap
            fig.add_trace(go.Scattermap(mode="lines", lon=pipes_reg_lon, lat=pipes_reg_lat,
                                        line=dict(width=2, color="#4682B4"), name="Pipe", hoverinfo="none"))
            # CHANGED: go.Scattermapbox -> go.Scattermap
            fig.add_trace(go.Scattermap(mode="lines", lon=pipes_valve_lon, lat=pipes_valve_lat,
                                        line=dict(width=3, color="#301934"), name="Pipe with Valve", hoverinfo="none"))

            node_styles = {
                'Junction': {'color': 'red', 'size': 8, 'symbol': 'circle'},
                'Pump': {'color': 'orange', 'size': 14, 'symbol': 'circle'},
                'Treatment Plant': {'color': 'green', 'size': 18, 'symbol': 'circle'},
                'Tank': {'color': 'blue', 'size': 16, 'symbol': 'circle'}
            }

            for node_type, style in node_styles.items():
                df_subset = nodes_df[nodes_df['node_type'] == node_type]
                if not df_subset.empty:
                    hover_texts = [f"<b>{row['node_type']}</b><br>Demand: {row.get('demand_m3_per_hr', 'N/A')} m³/hr" for _, row in df_subset.iterrows()]
                    # CHANGED: go.Scattermapbox -> go.Scattermap
                    fig.add_trace(go.Scattermap(
                        mode="markers", name=node_type, lon=df_subset['x'], lat=df_subset['y'],
                        marker=style, hoverinfo="text", hovertext=hover_texts
                    ))
            
        
            fig.update_layout(
                title=f"Water Network for {location_query}",
                map_style="open-street-map",
                map_center_lon=nodes_df['x'].mean(),
                map_center_lat=nodes_df['y'].mean(),
                map_zoom=12,
                margin={"r":0, "t":40, "l":0, "b":0},
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            


            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Generated Network Data")
            stats_df = nodes_df['node_type'].value_counts().reset_index()
            stats_df.columns = ['Component Type', 'Count']
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Total Nodes", len(G_water_rich.nodes()))
                st.metric("Total Pipes", len(G_water_rich.edges()))
                st.dataframe(stats_df)
            with col2:
                st.info("Node Data Sample (with attributes)")
                display_cols = ['node_type', 'demand_m3_per_hr', 'pressure_bar', 'capacity_m3']
                st.dataframe(nodes_df[[col for col in display_cols if col in nodes_df.columns]].head())

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Could not generate the network. Please try a different location or adjust the parameters.")
    else:
        st.info("Enter a location and click 'Generate Network' to begin.")
