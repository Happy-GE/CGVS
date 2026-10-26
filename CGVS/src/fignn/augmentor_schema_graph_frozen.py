from fignn.gnn_scoring_schema_frozen import search_frozen_forward_schema_graph
from fignn.schema_graph_building import identify_intermediate_tables


def augment_schema_graph_with_frozen_forward(
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
    retrain_each_round=True,
    include_fk_features=False,
    verbose=True,
):
    """
    Build a task-aware schema graph by freezing a trained GNN during candidate
    scoring and greedily fixing each intermediate table as node or edge.
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(
            db,
            target_table=target_table,
        )

    return search_frozen_forward_schema_graph(
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
        retrain_each_round=retrain_each_round,
        include_fk_features=include_fk_features,
        verbose=verbose,
    )
