import pandas as pd
import numpy as np
import hashlib
import os
import random
import torch
from scipy.stats import entropy
from collections import deque
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.utils import k_hop_subgraph


def make_node_id_tensor(pk_values):
    """
    Convert primary-key values into a tensor that is safe to store on a graph.

    PyTorch tensors cannot directly hold arbitrary Python strings, while some of
    our datasets use string primary keys such as ``d1`` or ``a_42``. For numeric
    keys we preserve the original values; otherwise we fall back to positional
    indices. The real PK -> node-index mapping still lives in ``node_id_map``.
    """
    series = pd.Series(pk_values)

    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
        return torch.tensor(series.tolist(), dtype=torch.long)

    if pd.api.types.is_float_dtype(series):
        return torch.tensor(series.tolist(), dtype=torch.float)

    return torch.arange(len(pk_values), dtype=torch.long)

def create_random_dict(n, num_classes, seed=42):
    random.seed(seed)  # Set the seed for reproducibility
    return {i: random.randint(0, num_classes-1) for i in range(1, n + 1)}

def add_reverse_edges(data):
    """
    Adds reverse edge types to a HeteroData graph in-place.
    Skips any edge whose relation name starts with 'rev_' to avoid double reversal.
    """
    new_edges = {}
    for (src, rel, dst) in data.edge_types:
        if rel.startswith('rev_'):
            continue  # Already reversed, skip

        edge_index = data[(src, rel, dst)].edge_index
        rev_edge_index = edge_index.flip(0)
        rev_rel = f"rev_{rel}"
        new_edges[(dst, rev_rel, src)] = rev_edge_index

    for (dst, rev_rel, src), rev_edge_index in new_edges.items():
        data[(dst, rev_rel, src)].edge_index = rev_edge_index

    return data

def join_path_tables(db, path):
    """
    Join tables along the given FK path in order.

    - The first table is used as the base and is NOT prefixed.
    - All other tables are prefixed, except for their join key column.
    - All FK-relevant columns are preserved across joins.

    Args:
        db (RelationalDatabase): schema and data
        path (list[str]): ordered list of table names (e.g. ['products', 'orders', 'customers'])

    Returns:
        pd.DataFrame: joined DataFrame with prefixed columns
    """
    assert len(path) >= 2, "Join path must contain at least two tables."

    joined = db.get_table(path[0]).copy()

    # Collect all FK columns that might be needed
    all_fk_keys = set()
    for src, src_col, dst, dst_col in db.foreign_keys:
        all_fk_keys.add(src_col)
        all_fk_keys.add(dst_col)

    for i in range(len(path) - 1):
        t1 = path[i]
        t2 = path[i + 1]

        # Find FK between t1 and t2
        fk = next(
            (fk for fk in db.foreign_keys if
             (fk[0] == t1 and fk[2] == t2) or
             (fk[0] == t2 and fk[2] == t1)),
            None
        )
        if fk is None:
            raise ValueError(f"No FK found between {t1} and {t2}")

        src_table, src_col, dst_table, dst_col = fk

        if t1 == src_table:
            left_key = src_col
            right_key = dst_col
            right_table = dst_table
        else:
            left_key = dst_col
            right_key = src_col
            right_table = src_table

        right_df = db.get_table(right_table).copy()

        # Rename non-key columns in right table
        right_prefix = f"{right_table}__"
        cols_to_prefix = [col for col in right_df.columns if col != right_key]
        right_renamed = right_df.rename(columns={col: right_prefix + col for col in cols_to_prefix})

        # Ensure join key is preserved
        if right_key not in right_renamed.columns:
            right_renamed[right_key] = right_df[right_key]

        # Merge
        joined = joined.merge(
            right_renamed,
            left_on=left_key,
            right_on=right_key,
            how="left"
        )

        # ✅ Preserve all FK-relevant columns for future joins
        for fk_col in all_fk_keys:
            if fk_col in right_df.columns and fk_col not in joined.columns:
                joined[fk_col] = right_df[fk_col]

    return joined


def get_label_entropies(graph, target_table, labels, node_id_map, k_hops):
    """
    Computes average label entropy for each node in the target table over its k-hop neighborhood.
    """
    if target_table not in graph.node_types:
        raise ValueError(f"Target node type '{target_table}' not found in graph.")

    if not graph.edge_types:
        # No edges at all — no neighborhood, so entropy is undefined
        return 0.0

    edge_index_dict = graph.edge_index_dict
    if not edge_index_dict:
        return 0.0

    index_to_label = {
        node_id_map[target_table][pk]: lbl
        for pk, lbl in labels.items()
        if pk in node_id_map[target_table]
    }

    entropies = []

    for center_node in index_to_label.keys():
        visited = set([center_node])
        frontier = deque([center_node])

        for _ in range(k_hops):
            next_frontier = set()
            for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
                # Skip empty edge tensors
                if edge_index.numel() == 0:
                    continue

                for direction in [(0, 1), (1, 0)]:
                    src, dst = edge_index[direction[0]], edge_index[direction[1]]
                    mask = np.isin(src.numpy(), list(frontier))
                    neighbors = dst[mask].tolist()
                    next_frontier.update(neighbors)

            frontier = next_frontier - visited
            if not frontier:
                break
            visited.update(frontier)

        # Collect neighbor labels
        neighbor_labels = [index_to_label[idx] for idx in visited if idx in index_to_label]
        if not neighbor_labels:
            continue

        counts = np.bincount(neighbor_labels)
        probs = counts / counts.sum()
        ent = entropy(probs, base=2)
        entropies.append(ent)

    return float(np.mean(entropies)) if entropies else 0.0


def assign_node_labels(graph: HeteroData,
                       labels: dict,
                       node_id_map: dict,
                       node_type: str,
                       label_key: str = 'y') -> None:
    """
    Assign labels to a node type in a HeteroData object.

    Args:
        graph (HeteroData): The heterogeneous graph.
        labels (dict): A mapping {primary_key: label}.
        node_id_map (dict): Maps {table_name: {primary_key: node_idx}}.
        node_type (str): The node type to assign labels to (e.g., 'users').
        label_key (str): The field to write the labels to. Defaults to 'y'.

    Returns:
        None. Modifies the `graph` in-place by setting `graph[node_type][label_key]`.
    """
    num_nodes = graph[node_type]['x'].shape[0]
    label_tensor = torch.full((num_nodes,), -1, dtype=torch.long)

    pk_to_idx = node_id_map.get(node_type, {})
    for pk, label in labels.items():
        if pk in pk_to_idx:
            idx = pk_to_idx[pk]
            label_tensor[idx] = label

    graph[node_type][label_key] = label_tensor


def split_node_labels(graph, node_type: str, label_key: str = 'y',
                      train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
                      seed=36):
    """
    Randomly splits labeled nodes into train/val/test masks.

    Modifies graph[node_type] in-place by setting train_mask, val_mask, test_mask.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    y = graph[node_type][label_key]
    # print('y',y)
    labeled_idx = (y >= 0).nonzero(as_tuple=False).view(-1)
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    perm = torch.randperm(len(labeled_idx))

    n_train = int(train_ratio * len(labeled_idx))
    n_val = int(val_ratio * len(labeled_idx))

    train_idx = labeled_idx[perm[:n_train]]
    val_idx = labeled_idx[perm[n_train:n_train + n_val]]
    test_idx = labeled_idx[perm[n_train + n_val:]]
    # print('train', train_idx)
    # print('val', val_idx)
    # print('test', test_idx)
    mask = torch.zeros(y.shape[0], dtype=torch.bool)

    graph[node_type]['train_mask'] = mask.clone()
    graph[node_type]['val_mask'] = mask.clone()
    graph[node_type]['test_mask'] = mask.clone()

    graph[node_type]['train_mask'][train_idx] = True


    graph[node_type]['val_mask'][val_idx] = True


    graph[node_type]['test_mask'][test_idx] = True

TEXT_MODEL_NAME = 'all-MiniLM-L6-v2'
_text_model = None


def get_text_model():
    global _text_model

    if _text_model is not None:
        return _text_model

    model_path = os.environ.get("AUGRAPH_TEXT_MODEL_PATH")
    model_cache = os.environ.get("AUGRAPH_TEXT_MODEL_CACHE") or os.environ.get("HF_HUB_CACHE")
    model_source = model_path or TEXT_MODEL_NAME
    model_kwargs = {"local_files_only": True}
    if model_cache and not model_path:
        model_kwargs["cache_folder"] = model_cache

    try:
        _text_model = SentenceTransformer(model_source, **model_kwargs)
    except TypeError:
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _text_model = SentenceTransformer(model_source)
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
    except Exception as exc:
        raise RuntimeError(
            f"Text column encoding needs cached HuggingFace model {model_source!r}. "
            "Pre-download it on a networked machine/server, or remove/shorten long text "
            "columns before graph construction."
        ) from exc

    return _text_model

def encode_table(df):
    """
    Encodes a DataFrame into a feature matrix with simple column-wise rules:
    - Numeric: z-score
    - Categorical: label encoding (as integers)
    - Text: average word embedding
    Returns: torch.FloatTensor of shape [num_rows, total_dim]
    """
    features = []
    scalers = {}
    label_encoders = {}
    
    for col in df.columns:
        col_data = df[col]
        if col_data.dtype == object or col_data.dtype.name == "category":
            if col_data.str.len().mean() > 20:
                # Probably a text field
                # Use sentence embeddings
                emb = get_text_model().encode(col_data.fillna("").tolist(), convert_to_tensor=True)
                features.append(emb)
            else:
                # Categorical, label encode
                le = LabelEncoder()
                encoded = le.fit_transform(col_data.fillna("MISSING"))
                features.append(torch.tensor(encoded, dtype=torch.float).unsqueeze(1))
                label_encoders[col] = le
        elif np.issubdtype(col_data.dtype, np.number):
            # Numeric, z-score
            scaler = StandardScaler()
            normed = scaler.fit_transform(col_data.fillna(0).values.reshape(-1, 1))
            features.append(torch.tensor(normed, dtype=torch.float))
            scalers[col] = scaler
        else:
            # Fallback: ignore
            continue

    if features:
        return torch.cat(features, dim=1)
    else:
        return torch.ones((len(df), 1))  # fallback dummy feature


def extract_local_subgraph(graph, target_table, target_pks, node_id_map, k_hops):
    """
    Extracts k-hop heterogeneous subgraph around target_table nodes.
    Includes all node types and all edge types reachable in k hops.
    """

    # Step 1: Find initial node indices
    target_indices = torch.tensor([node_id_map[target_table][pk] for pk in target_pks], dtype=torch.long)

    # Step 2: Collect neighbors
    all_nodes = {table: set() for table in graph.node_types}
    all_nodes[target_table].update(target_indices.tolist())

    frontier = {target_table: target_indices}
    for hop in range(k_hops):
        next_frontier = {table: set() for table in graph.node_types}
        for (src_type, rel_type, dst_type) in graph.edge_types:
            edges = graph[(src_type, rel_type, dst_type)].edge_index
            # Check if any src nodes are in the current frontier
            if src_type in frontier:
                src_nodes = frontier[src_type]
                mask = torch.isin(edges[0], src_nodes)
                neighbors = edges[1][mask]
                next_frontier[dst_type].update(neighbors.tolist())
            # Check if any dst nodes are in the current frontier (for reverse edges)
            if dst_type in frontier:
                dst_nodes = frontier[dst_type]
                mask = torch.isin(edges[1], dst_nodes)
                neighbors = edges[0][mask]
                next_frontier[src_type].update(neighbors.tolist())
        # Update frontier
        for table in graph.node_types:
            all_nodes[table].update(next_frontier[table])
        frontier = {table: torch.tensor(list(nodes)) for table, nodes in next_frontier.items() if nodes}

    # Step 3: Prepare subgraph nodes
    all_nodes = {table: list(nodes) for table, nodes in all_nodes.items() if nodes}  # sets -> lists

    all_nodes_mask = {}

    for node_type in graph.node_types:
        if node_type in all_nodes:
            num_nodes = graph[node_type].num_nodes
            mask = torch.zeros(num_nodes, dtype=torch.bool)

            selected_nodes = all_nodes[node_type]

            # Convert to tensor if not already
            if not isinstance(selected_nodes, torch.Tensor):
                selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)

            mask[selected_nodes] = True
            all_nodes_mask[node_type] = mask

    # Step 4: Extract subgraph
    subgraph = graph.subgraph(all_nodes_mask)

    return subgraph

def run_local_wl(graph, target_table, target_pks, node_id_map, num_iterations=2):
    """
    1-WL color refinement on heterogeneous graph.
    """

    # Initialize colors
    colors = {}
    for table in graph.node_types:
        num_nodes = graph[table].num_nodes
        for node_id in range(num_nodes):
            colors[(table, node_id)] = 0  # All nodes start with same color

    for _ in range(num_iterations):
        new_colors = {}
        for (table, node_id), color in colors.items():
            neighbor_colors = []
            for (src_type, rel_type, dst_type) in graph.edge_types:
                edges = graph[(src_type, rel_type, dst_type)].edge_index
                if src_type == table:
                    mask = (edges[0] == node_id)
                    neighbors = edges[1][mask]
                    neighbor_colors.extend([(dst_type, int(n.item())) for n in neighbors])
                if dst_type == table:
                    mask = (edges[1] == node_id)
                    neighbors = edges[0][mask]
                    neighbor_colors.extend([(src_type, int(n.item())) for n in neighbors])

            # Build color signature
            neighbor_colors_sorted = sorted([colors.get((n_type, n_id), -1) for n_type, n_id in neighbor_colors])
            combined = (color, tuple(neighbor_colors_sorted))
            combined_str = str(combined).encode('utf-8')
            color_hash = hashlib.md5(combined_str).hexdigest()
            new_colors[(table, node_id)] = color_hash

        # Normalize colors to integers
        unique_colors = sorted(set(new_colors.values()))
        color_to_id = {c: i for i, c in enumerate(unique_colors)}
        colors = {node: color_to_id[c] for node, c in new_colors.items()}

    # Return only target_table nodes
    target_node_indices = [node_id_map[target_table][pk] for pk in target_pks]
    return {idx: colors[(target_table, idx)] for idx in target_node_indices}


def build_node_id_map(graph):
    """
    Rebuild a node_id_map from the current graph structure.
    Maps node type -> {index: index}.
    """
    node_id_map = {}
    for node_type in graph.node_types:
        num_nodes = graph[node_type].num_nodes
        node_id_map[node_type] = {i+1: i for i in range(num_nodes)}
    return node_id_map

def ensure_node_features(graph):
    for node_type in graph.node_types:
        if 'x' not in graph[node_type]:
            num_nodes = graph[node_type].num_nodes
            graph[node_type].x = torch.zeros((num_nodes, 1))  # minimal dummy feature
    return graph

def log_promotion_step(
    logger,
    step,
    table,
    attribute,
    score,
    method,
    k_hops,
    graph,
    label_table
):
    """
    Appends promotion step info into logger.

    Args:
        logger: list of dicts (grows over steps)
        step: promotion step index (int)
        table: table name (str)
        attribute: attribute name (str)
        score: score value (float)
        method: scoring method name (str)
        k_hops: hops considered (int)
        graph: current graph (HeteroData)
        label_table: table where prediction happens

    Returns:
        None (modifies logger in-place)
    """
    log_entry = {
        "step": step,
        "table": table,
        "attribute": attribute,
        "score": score,
        "method": method,
        "k_hops": k_hops,
        "num_node_types": len(graph.node_types),
        "num_edge_types": len(graph.edge_types),
        "num_target_nodes": graph[label_table].num_nodes,
    }
    logger.append(log_entry)
