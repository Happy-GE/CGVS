from fignn.gnn_scoring_schema import search_task_aware_schema_graph
from fignn.schema_graph_building import identify_intermediate_tables


def augment_schema_graph_with_gnn_gain(
    db,
    initial_graph,
    target_table,
    intermediate_tables=None,
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    device=None,
    seed=42,
    include_fk_features=False,
    verbose=True,
):
    """
    Build a task-aware schema graph by greedily fixing each intermediate table
    as either a node or an edge according to GNN validation gain.

    Returns:
        final_graph, final_node_id_map, selected_modes, history
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(
            db,
            target_table=target_table,
        )

    return search_task_aware_schema_graph(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        include_fk_features=include_fk_features,
        verbose=verbose,
    )
