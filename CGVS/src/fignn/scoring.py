from sklearn.metrics import mutual_info_score as sklearn_mi
from scipy.stats import entropy as scipy_entropy
from fignn.gnn_scoring import gnn_gain_score
from fignn.utils import join_path_tables, get_label_entropies, run_local_wl, extract_local_subgraph, build_node_id_map
from fignn.graph_building import promote_attribute
import networkx as nx
import pandas as pd

def score_attribute(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    method: str,
    **kwargs
) -> float:
    """
    Unified scoring interface. Dispatches to the appropriate scoring method.

    Args:
        graph: Current graph object
        db: RelationalDatabase instance
        table: Table where the attribute is defined
        attribute: Attribute name to score
        labels: Dictionary mapping PKs to labels
        method: Scoring strategy name
        kwargs: Additional arguments for specific scorers

    Returns:
        float: Score for the attribute
    """
    scorers = {
        "mutual_info": mutual_info_score,
        "entropy_gain": label_entropy_gain,
        "gnn_gain": gnn_gain_score, 
        "wl_gain": wl_refinement_gain,
        "edge_disagreement": edge_disagreement
    }

    if method not in scorers:
        raise ValueError(f"Unknown scoring method: {method}")

    return scorers[method](
        graph=graph,
        db=db,
        table=table,
        attribute=attribute,
        labels=labels,
        **kwargs
    )

def mutual_info_score(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    **kwargs
) -> float:
    """
    Compute normalized mutual information between a categorical attribute (from `table`)
    and external labels defined over nodes in `target_table`. Returns 0.0 for continuous attributes.

    Args:
        table: Table where the attribute is defined
        attribute: Name of the attribute
        labels: Dict {primary_key → label} for nodes in `target_table`
        target_table: Table where labels are defined (i.e., the task target)

    Returns:
        Normalized Mutual Information (float), or 0.0 if attribute is not categorical or data is invalid.
    """
    label_col = "__temp_label__"
    pk_target = db.primary_keys[target_table]

    # Create temporary label DataFrame
    label_df = pd.DataFrame([{pk_target: pk, label_col: y} for pk, y in labels.items()])

    if table == target_table:
        df = db.get_table(table)[[attribute, pk_target]].copy()
        
        if not pd.api.types.is_categorical_dtype(df[attribute]) and not pd.api.types.is_object_dtype(df[attribute]):
            return 0.0

        df = df.merge(label_df, on=pk_target, how="inner")

        if df.empty:
            return 0.0

        # Drop rows with missing values in attribute or label
        df = df[[attribute, label_col]].dropna()
        if df.empty:
            return 0.0

        mi = sklearn_mi(df[attribute], df[label_col])
        ent_attr = scipy_entropy(df[attribute].value_counts(normalize=True))
        return mi / ent_attr if ent_attr > 0 else 0.0

    # Join path from table to target_table
    path = nx.shortest_path(db.schema_graph, source=table, target=target_table)
    if path[0] != table:
        path = list(reversed(path))

    joined = join_path_tables(db, path)

    attr_col = attribute if table == path[0] else f"{table}__{attribute}"
    target_pk_col = pk_target if target_table == path[0] else f"{target_table}__{pk_target}"

    if attr_col not in joined.columns or target_pk_col not in joined.columns:
        return 0.0

    if not pd.api.types.is_categorical_dtype(joined[attr_col]) and not pd.api.types.is_object_dtype(joined[attr_col]):
        return 0.0

    joined[label_col] = joined[target_pk_col].map(labels)
    df = joined[[attr_col, label_col]].dropna()
    if df.empty:
        return 0.0

    mi = sklearn_mi(df[attr_col], df[label_col])
    ent_attr = scipy_entropy(df[attr_col].value_counts(normalize=True))
    return mi / ent_attr if ent_attr > 0 else 0.0


#TODO: optimize this function for GPU use. 
def label_entropy_gain(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    node_id_map: dict,
    k_hops: int,
    **kwargs
) -> float:
    """
    Compute the reduction in average label entropy over k-hop neighborhoods
    of nodes in the target_table, after promoting a given attribute.

    Args:
        graph (HeteroData): Original FK-based graph.
        db (RelationalDatabase): Database for schema and data access.
        table (str): Table where the attribute is defined.
        attribute (str): Name of the attribute to promote.
        labels (dict): Mapping from primary key → label for target_table nodes.
        target_table (str): Table where the label nodes live.
        node_id_map (dict): Table → {pk → node index} mapping.
        k_hops (int): Neighborhood radius to evaluate entropy over (should be 2 * join depth).

    Returns:
        float: Average reduction in label entropy (positive is better).
    """
    # Measure before
    entropy_before = get_label_entropies(graph, target_table, labels, node_id_map, k_hops)

    # Promote attribute and measure again
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    entropy_after = get_label_entropies(graph_aug, target_table, labels, node_id_map, k_hops)

    return entropy_before - entropy_after

def wl_refinement_gain(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    node_id_map: dict,
    k_hops: int,
    **kwargs
) -> float:
    """
    Apply localized 1-WL color refinement around target nodes
    before and after promoting an attribute.
    Score is the increase in the number of distinct colors among target nodes.
    
    Positive = attribute helps distinguish target nodes.
    """
    # 1. Localize: extract k-hop neighborhoods around target nodes
    local_graph = extract_local_subgraph(graph, target_table, labels.keys(), node_id_map, k_hops)

    # 2. Run WL color refinement before promotion
    colors_before = run_local_wl(local_graph, target_table, labels.keys(), node_id_map)

    # 3. Promote the attribute (creates new nodes and edges)
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    new_node_id_map = build_node_id_map(graph_aug)

    # 4. Localize again after promotion (may have new structure)
    local_graph_aug = extract_local_subgraph(graph_aug, target_table, labels.keys(), new_node_id_map, k_hops)

    # 5. Run WL color refinement after promotion
    colors_after = run_local_wl(local_graph_aug, target_table, labels.keys(), new_node_id_map)

    # 6. Count distinct colors among target nodes
    distinct_colors_before = len(set(colors_before.values()))
    distinct_colors_after = len(set(colors_after.values()))

    # 7. Return the gain (positive is better)
    return distinct_colors_after - distinct_colors_before


def edge_disagreement(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    target_table: str,
    node_id_map: dict,
    **kwargs
) -> float:
    """
    Measure label disagreement over attribute-induced virtual paths between target nodes.
    Returns a score where HIGHER is better (more agreement among reachable nodes).
    """

    # Step 1: Promote the attribute
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    # Step 2: Build full undirected graph
    G = nx.Graph()
    for (src_type, rel_type, dst_type) in graph_aug.edge_types:
        edges = graph_aug[(src_type, rel_type, dst_type)].edge_index
        src_nodes = edges[0].tolist()
        dst_nodes = edges[1].tolist()
        src_nodes_named = [(src_type, int(x)) for x in src_nodes]
        dst_nodes_named = [(dst_type, int(x)) for x in dst_nodes]
        G.add_edges_from(zip(src_nodes_named, dst_nodes_named))

    # Step 3: Determine number of hops needed
    try:
        join_path = nx.shortest_path(db.schema_graph, source=table, target=target_table)
        if join_path[0] != table:
            join_path = list(reversed(join_path))
    except nx.NetworkXNoPath:
        return 0.0  # No join path means promotion is not meaningful

    path_length = len(join_path) - 1
    k_hops = 2 * (path_length + 1)

    # Step 4: Prepare target nodes
    all_target_nodes = [(target_table, node_id_map[target_table][pk]) for pk in labels.keys()]
    target_nodes = [node for node in all_target_nodes if G.has_node(node)]

    if len(target_nodes) == 0:
        return 0.0  # No nodes to evaluate

    # If too many target nodes are missing, I can return 0
    if len(target_nodes) < 0.5 * len(all_target_nodes):
        print(f"Warning: Only {len(target_nodes)} of {len(all_target_nodes)} target nodes exist in promoted graph.")
    #     return 0.0

    # Step 5: Precompute reverse mapping: node_id → primary key
    nid_to_pk = {nid: pk for pk, nid in node_id_map[target_table].items()}

    # Step 6: Build k-hop subgraph around target nodes
    nodes_to_keep = set()
    for node in target_nodes:
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=k_hops)
        nodes_to_keep.update(lengths.keys())

    G_sub = G.subgraph(nodes_to_keep).copy()

    # Step 7: Precompute reachable pairs
    reachable_pairs = set()
    for node in target_nodes:
        if not G_sub.has_node(node):
            continue
        lengths = nx.single_source_shortest_path_length(G_sub, node, cutoff=k_hops)
        for other_node in lengths.keys():
            if other_node in target_nodes and node < other_node:
                reachable_pairs.add((node, other_node))

    if len(reachable_pairs) == 0:
        return 0.0  # No reachable pairs

    # Step 8: Measure disagreement
    disagreement_count = 0
    for (u, v) in reachable_pairs:
        u_pk = nid_to_pk[u[1]]
        v_pk = nid_to_pk[v[1]]
        if labels[u_pk] != labels[v_pk]:
            disagreement_count += 1

    disagreement_rate = disagreement_count / len(reachable_pairs)

    # INVERT: higher = better
    agreement_score = 1.0 - disagreement_rate
    return agreement_score

if __name__ == "__main__":
    from src.fignn.data_loader import load_relational_data
    from src.fignn.graph_building import build_fk_graph
    import os

    scoring_method = "edge_disagreement"

    print(f"Scoring testing: metric {scoring_method}")

    # 1. Original graph
    db = load_relational_data("data/synthetic")
    graph, node_id_map = build_fk_graph(db)

    db.get_all_attributes() # Creates schema graph

    task = pd.read_csv(os.path.join("data/synthetic", "task.csv"))
    labels = dict(zip(task['user_id'], task['label']))

    # 2. Define which attribute will be promoted

    table = "vendors" # users: account_type, vendors: (vendor_type), products: category
    attr = "vendor_type"
    label_table = "users"
    k_hops = 3


    # 3. Calculate score when promoting such attribute
    score = score_attribute(
            graph,
            db,
            table=table,
            attribute=attr,
            labels=labels,
            method=scoring_method,
            task_table=label_table,
            #task_label=label_column,
            target_table=label_table,
            node_id_map=node_id_map,
            k_hops=k_hops
        )

    print(score)

    




    

