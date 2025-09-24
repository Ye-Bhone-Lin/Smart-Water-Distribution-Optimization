import osmnx as ox
import networkx as nx
import wntr

def generate_synthetic_network(place_name, output_filename):
    """
    Fetches a street network from OSM, converts it to a synthetic water
    network model, and saves it as an EPANET .inp file.
    """
    # 1. Fetch the street network from OpenStreetMap
    print(f"Fetching street network for {place_name}...")
    # Using 'drive' network type simplifies to major roads
    G_osm = ox.graph_from_place(place_name, network_type='drive')

    # Optional: Further simplification if the graph is too large
    # G_simplified = ox.simplify_graph(G_osm)
    G_simplified = G_osm.to_directed() # Ensure graph is directed

    print(f"Found a network with {G_simplified.number_of_nodes()} nodes.")

    # 2. Build a WNTR water network model from the simplified graph
    wn = wntr.network.WaterNetworkModel()

    # 3. Add nodes (junctions) and a water source (reservoir)
    # This is where you add ARBITRARY data
    print("Adding junctions and assigning demands...")

    node_coords = {}
    for i, (node, data) in enumerate(G_simplified.nodes(data=True)):
        node_name = f"J-{node}"
        coords = (data['x'], data['y'])
        node_coords[node_name] = coords

        # Assign a small, arbitrary demand to each junction
        wn.add_junction(node_name,
                        base_demand=0.001, # m^3/s (approx. 1 family)
                        coordinates=coords)

        # Designate the very first node as the water source
        if i == 0:
            source_name = f"R-{node}"
            wn.add_reservoir(source_name,
                            base_head=100, # meters of pressure
                            coordinates=coords)
            # This junction will now be fed by the reservoir
            # Add a pipe to connect the reservoir to the first junction
            wn.add_pipe(f"P-Source-{node}", source_name, node_name,
                        length=10, diameter=0.5, roughness=140)

    # 4. Add pipes based on the streets
    # Again, assigning ARBITRARY hydraulic properties
    print("Adding pipes and assigning properties...")
    for i, (u, v) in enumerate(G_simplified.edges()):
        start_node = f"J-{u}"
        end_node = f"J-{v}"
        pipe_name = f"P-{i}"

        # Get edge data safely
        edge_data = G_simplified.get_edge_data(u, v, default={})
        
        # Use actual street length from OSM
        length = edge_data.get('length', 100)

        # Assign an arbitrary diameter (e.g., 0.3m or 12 inches)
        # [cite_start]The ADB report mentions a new main pipeline of 2.4m diameter[cite: 307],
        # so smaller distribution pipes would be in this range.
        diameter = 0.3

        wn.add_pipe(pipe_name, start_node, end_node,
                    length=length, diameter=diameter, roughness=140)

    # 5. Write the complete model to an EPANET .inp file
    print(f"Writing EPANET file to {output_filename}...")
    wntr.network.io.write_inpfile(wn, output_filename)
    print("Done!")


# --- To run this script ---
if __name__ == '__main__':
    place = "Insein District , Yangon, Myanmar"
    output_file = "Insein_synthetic_network.inp"
    generate_synthetic_network(place, output_file)