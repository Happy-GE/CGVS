# kNN Graph Building
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors

from fignn.utils import encode_table, make_node_id_tensor


def build_knn_graph(
    table: pd.DataFrame, 
    table_name: str = "instance", 
    k: int = 10,
    pk: str = None,
):
    """
    Build a k-NN graph from a tabular dataset.

    Args:
        table (pd.DataFrame): Input table.
        table_name (str): Node type name.
        k (int): Number of nearest neighbors.
        pk (str, optional): Name of primary key column. If None, uses index.

    Returns:
        data (HeteroData): Heterogeneous graph with node features and k-NN edges.
        node_id_map (dict): Maps pk_value → node_index
    """
    data = HeteroData()

    # --- Step 1: Extract and map node IDs ---
    if pk is None:
        pk_values = table.index.tolist()
    else:
        pk_values = table[pk].tolist()
        table = table.set_index(pk)

    node_id_map = {pkv: idx for idx, pkv in enumerate(pk_values)}
    num_nodes = len(pk_values)
    data[table_name].node_ids = make_node_id_tensor(pk_values)
    data[table_name].raw_node_ids = list(pk_values)

    # --- Step 2: Encode features ---
    try:
        x = encode_table(table)
    except Exception as e:
        print(f"[build_knn_graph] Encoding failed: {e}")
        x = torch.ones((num_nodes, 1))

    data[table_name].x = x

    # --- Step 3: Compute k-NN edges ---
    features_np = x.numpy()
    nn = NearestNeighbors(n_neighbors=min(k + 1, num_nodes), metric="euclidean")
    knn_graph = nn.fit(features_np).kneighbors(return_distance=False)

    edge_index = []
    for src, neighbors in enumerate(knn_graph):
        for dst in neighbors:
            if src != dst:  # exclude self-loop
                edge_index.append([src, dst])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data[(table_name, "knn", table_name)].edge_index = edge_index

    # --- Step 4: Add reverse edges ---
    rev_index = edge_index[[1, 0]]
    data[(table_name, "rev_knn", table_name)].edge_index = rev_index

    return data, {table_name: node_id_map}
