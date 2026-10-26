import copy
from itertools import combinations

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from fignn.utils import add_reverse_edges, encode_table, make_node_id_tensor


NODE_MODE = "node"
EDGE_MODE = "edge"


def get_table_foreign_keys(db, table):
    """Return all foreign keys whose source table is `table`."""
    return [fk for fk in db.foreign_keys if fk[0] == table]


def identify_intermediate_tables(db, target_table=None, min_outgoing_fks=2):
    """
    Identify relation/intermediate tables that can be represented as nodes or edges.

    The default rule treats a table as intermediate when it owns at least two
    outgoing foreign keys. This matches common bridge/event tables such as
    order_items(user_id, product_id) or visits(patient_id, doctor_id).
    """
    candidates = []
    for table in db.tables:
        if table not in db.primary_keys:
            continue
        if table == target_table:
            continue
        outgoing = get_table_foreign_keys(db, table)
        if len(outgoing) >= min_outgoing_fks:
            candidates.append(table)
    return candidates


def _key_columns(db, table):
    keys = {db.primary_keys[table]}
    for src_table, src_col, _, _ in db.foreign_keys:
        if src_table == table:
            keys.add(src_col)
    return keys


def _add_table_nodes(graph, db, table, node_id_maps, include_fk_features=False):
    df = db.get_table(table)
    pk = db.primary_keys[table]
    pk_values = df[pk].tolist()
    node_id_maps[table] = {pkv: idx for idx, pkv in enumerate(pk_values)}

    graph[table].node_ids = make_node_id_tensor(pk_values)
    graph[table].raw_node_ids = list(pk_values)

    if include_fk_features:
        feature_df = df.drop(columns=[pk], errors="ignore")
    else:
        feature_df = df.drop(columns=list(_key_columns(db, table)), errors="ignore")

    try:
        graph[table].x = encode_table(feature_df)
    except Exception as exc:
        print(f"[schema_graph] Encoding failed for {table}: {exc}")
        graph[table].x = torch.ones((len(df), 1))


def _add_fk_edges_for_node_table(graph, db, table, node_id_maps):
    src_df = db.get_table(table)
    src_pk = db.primary_keys[table]

    for src_table, src_col, dst_table, dst_col in get_table_foreign_keys(db, table):
        if src_table not in node_id_maps or dst_table not in node_id_maps:
            continue

        src_map = node_id_maps[src_table]
        dst_map = node_id_maps[dst_table]
        src_nodes = []
        dst_nodes = []

        for _, row in src_df.iterrows():
            if pd.isna(row[src_pk]) or pd.isna(row[src_col]):
                continue
            src_id = row[src_pk]
            dst_id = row[src_col]
            if src_id in src_map and dst_id in dst_map:
                src_nodes.append(src_map[src_id])
                dst_nodes.append(dst_map[dst_id])

        edge_type = (src_table, f"{src_col}_to_{dst_col}", dst_table)
        graph[edge_type].edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)


def _add_regular_fk_edges(graph, db, node_id_maps, edge_mode_tables):
    for src_table in sorted({fk[0] for fk in db.foreign_keys}):
        if src_table in edge_mode_tables:
            continue
        _add_fk_edges_for_node_table(graph, db, src_table, node_id_maps)


def _add_edgeized_table_edges(graph, db, table, node_id_maps):
    """
    Represent an intermediate table as direct edges between the entity tables it references.

    For a table with k outgoing FKs, this creates pairwise entity-entity edges for
    every FK pair. Each row contributes one edge per pair when both referenced
    entity IDs are present.
    """
    fks = get_table_foreign_keys(db, table)
    if len(fks) < 2:
        raise ValueError(f"Table '{table}' needs at least two outgoing FKs to be edgeized.")

    df = db.get_table(table)

    for left_fk, right_fk in combinations(fks, 2):
        _, left_col, left_dst_table, left_dst_col = left_fk
        _, right_col, right_dst_table, right_dst_col = right_fk

        if left_dst_table not in node_id_maps or right_dst_table not in node_id_maps:
            continue

        left_map = node_id_maps[left_dst_table]
        right_map = node_id_maps[right_dst_table]
        left_nodes = []
        right_nodes = []

        for _, row in df.iterrows():
            if pd.isna(row[left_col]) or pd.isna(row[right_col]):
                continue
            left_id = row[left_col]
            right_id = row[right_col]
            if left_id in left_map and right_id in right_map:
                left_nodes.append(left_map[left_id])
                right_nodes.append(right_map[right_id])

        rel = f"{table}_edge_{left_col}_to_{right_col}"
        graph[(left_dst_table, rel, right_dst_table)].edge_index = torch.tensor(
            [left_nodes, right_nodes],
            dtype=torch.long,
        )


def build_schema_aware_graph(
    db,
    table_modes=None,
    default_mode=NODE_MODE,
    target_table=None,
    include_fk_features=False,
):
    """
    Build a task-aware schema graph from table representation choices.

    Args:
        db: RelationalDatabase.
        table_modes: dict table -> "node" or "edge". Unspecified tables use default_mode.
        default_mode: representation for unspecified intermediate tables.
        target_table: target table is always kept as a node.
        include_fk_features: whether FK columns are kept in node features.

    Returns:
        graph: HeteroData.
        node_id_maps: dict table -> {primary_key: node index}. Edgeized tables are omitted.
    """
    table_modes = dict(table_modes or {})
    graph = HeteroData()
    node_id_maps = {}

    intermediate_tables = set(identify_intermediate_tables(db, target_table=target_table))
    edge_mode_tables = {
        table
        for table in intermediate_tables
        if table_modes.get(table, default_mode) == EDGE_MODE
    }

    if target_table in edge_mode_tables:
        edge_mode_tables.remove(target_table)

    for table in db.tables:
        if table not in db.primary_keys:
            continue
        if table in edge_mode_tables:
            continue
        _add_table_nodes(
            graph,
            db,
            table,
            node_id_maps,
            include_fk_features=include_fk_features,
        )

    _add_regular_fk_edges(graph, db, node_id_maps, edge_mode_tables)

    for table in edge_mode_tables:
        _add_edgeized_table_edges(graph, db, table, node_id_maps)

    graph = add_reverse_edges(graph)

    for node_type in graph.node_types:
        if "x" not in graph[node_type] or graph[node_type].x is None:
            graph[node_type].x = torch.ones((graph[node_type].num_nodes, 1))

    graph.schema_table_modes = copy.deepcopy(table_modes)
    graph.edge_mode_tables = sorted(edge_mode_tables)
    return graph, node_id_maps


def copy_task_tensors(reference_graph, candidate_graph, target_table):
    """Copy labels and split masks for the target table into a rebuilt graph."""
    for key in ("y", "train_mask", "val_mask", "test_mask"):
        if key in reference_graph[target_table]:
            candidate_graph[target_table][key] = reference_graph[target_table][key].clone()
    return candidate_graph
