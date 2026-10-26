from fignn.gnn_scoring_schema_light_weight import search_light_weight_schema_graph
from fignn.schema_graph_building import identify_intermediate_tables


def augment_schema_graph_light_weight(
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
    verbose=True,
):
    """
    Build a task-aware schema graph with one-shot lightweight mode selection.

    Each intermediate table is scored once as a node and once as an edge. The
    better mode for each table is selected independently, reducing candidate
    evaluation from O(m^2) to 2m.
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(
            db,
            target_table=target_table,
        )

    return search_light_weight_schema_graph(
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
        verbose=verbose,
    )
