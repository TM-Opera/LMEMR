import numpy as np
import geopandas as gpd

# ------------------ Node Importance Computation ------------------

def compute_node_importance_with_mapping(att_path, geojson_path, graph_file, output_geojson=None):
    """
    Compute node importance for each modality and map it to the node IDs represented by the 'GWBH_500' field in the GeoJSON.

    Parameters:
        att_path (str): Path to the .npy file
        geojson_path (str): Path to the GeoJSON file, which contains the 'GWBH_500' field
        graph_file (str): Path to the graph file containing node ID mapping
        output_geojson (str or None): Output path

    Returns:
        importance_dict_mapped (dict): {modality: {gwbh_id: importance}}
        gdf (GeoDataFrame): Result with importance fields added
    """
    # 1. Load attention weights
    data = np.load(att_path, allow_pickle=True)
    data = {key: data[key] for key in data.files}
    edge_index = data['edge_index']  # Remove and keep the rest as attention weights
    del data['edge_index']

    # Load node mapping
    node_id_to_idx = np.load(graph_file, allow_pickle=True).item()  # idx -> gwbh

    num_nodes = len(node_id_to_idx)
    
    # 2. Compute node importance for each modality (accumulate on target nodes)
    importance_dict = {}
    for modality, att_weights in data.items():
        att_weights = att_weights[:edge_index.shape[1]].flatten()  # Truncate extra parts to ensure correct length
        assert att_weights.shape[0] == edge_index.shape[1], f"Edge count mismatch: {modality}, {att_weights.shape[0]} vs {edge_index.shape[1]}"
        target_nodes = edge_index[1]
        imp = np.zeros(num_nodes, dtype=att_weights.dtype)
        np.add.at(imp, target_nodes, att_weights)
        importance_dict[modality] = imp

    # 3. Map back to original GWBH IDs
    importance_dict_mapped = {}
    idx_to_gwbh = node_id_to_idx
    for modality, imp_array in importance_dict.items():
        imp_map = {
            idx_to_gwbh.get(idx): float(value)
            for idx, value in enumerate(imp_array) if idx_to_gwbh.get(idx) is not None
        }
        importance_dict_mapped[modality] = imp_map

    # 4. Load GeoJSON and directly use the 'GWBH_500' column as node ID
    gdf = gpd.read_file(geojson_path)

    # Check if 'GWBH_500' column exists
    if 'GWBH_500' not in gdf.columns:
        raise KeyError("The GeoJSON is missing the 'GWBH_500' field. Please check if the column name is correct.")
    gdf['GWBH_500'] = gdf['GWBH_500'].apply(lambda x: int(x))

    # === Debug information: Check node ID mapping ===
    print("\n" + "="*50)
    print("Node Mapping Debug Information")
    print("="*50)
    
    # Check node IDs in graph data
    graph_gwbh_ids = set(node_id_to_idx.keys())
    print(f"Number of nodes in graph data: {len(graph_gwbh_ids)}")
    print(f"Example node IDs from graph data (first 10): {list(graph_gwbh_ids)[:10]}")
    print(f"Type of node IDs in graph data: {type(list(graph_gwbh_ids)[0])}")
    
    # Check node IDs in GeoJSON
    geojson_gwbh_ids = set(gdf['GWBH_500'].unique())
    print(f"Number of nodes in GeoJSON: {len(geojson_gwbh_ids)}")
    print(f"Example node IDs from GeoJSON (first 10): {list(geojson_gwbh_ids)[:10]}")
    print(f"Type of node IDs in GeoJSON: {type(list(geojson_gwbh_ids)[0])}")
    
    # Check common nodes
    common_ids = graph_gwbh_ids & geojson_gwbh_ids
    only_in_graph = graph_gwbh_ids - geojson_gwbh_ids
    only_in_geojson = geojson_gwbh_ids - graph_gwbh_ids
    
    print(f"\nNumber of common nodes: {len(common_ids)}")
    print(f"Nodes only in graph data: {len(only_in_graph)}")
    print(f"Nodes only in GeoJSON: {len(only_in_geojson)}")
    
    if len(common_ids) > 0:
        print(f"Example common nodes (first 10): {list(common_ids)[:10]}")
    else:
        print("❌ Warning: No common nodes found!")
        
    if len(only_in_graph) > 0:
        print(f"Example nodes only in graph data (first 10): {list(only_in_graph)[:10]}")
        
    if len(only_in_geojson) > 0:
        print(f"Example nodes only in GeoJSON (first 10): {list(only_in_geojson)[:10]}")
    
    # Check coverage of node importance dictionary
    print(f"\nImportance dictionary statistics:")
    for modality, imp_map in importance_dict_mapped.items():
        imp_gwbh_ids = set(imp_map.keys())
        common_with_geojson = imp_gwbh_ids & geojson_gwbh_ids
        print(f"  {modality}: {len(imp_map)} nodes have importance scores")
        print(f"    Nodes matching GeoJSON: {len(common_with_geojson)}")
    
    print("="*50 + "\n")
    # === End of debug information ===

    # 5. Add importance values for each modality
    for modality in importance_dict_mapped.keys():
        gdf[modality + '_importance'] = gdf['GWBH_500'].map(importance_dict_mapped[modality]).fillna(0.0)
        
        # New: Log mapping results for each modality
        mapped_values = gdf[modality + '_importance']
        zero_count = (mapped_values == 0).sum()
        non_zero_count = (mapped_values > 0).sum()
        print(f"Modality {modality}: {zero_count} zero-value nodes, {non_zero_count} non-zero nodes")

    # 6. Save result
    if output_geojson:
        gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"Saved GeoJSON with importance to: {output_geojson}")

    return importance_dict_mapped, gdf