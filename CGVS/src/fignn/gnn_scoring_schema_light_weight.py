from fignn.gnn_scoring_schema import (
    infer_num_classes,
    score_schema_candidate,
    train_and_score_schema_gnn,
)
from fignn.schema_graph_building import (
    EDGE_MODE,
    NODE_MODE,
    build_schema_aware_graph,
    copy_task_tensors,
    identify_intermediate_tables,
)


def search_light_weight_schema_graph(
    db,
    initial_graph,
    target_table,
    intermediate_tables=None,
    modes=(NODE_MODE, EDGE_MODE),
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    device=None,
    seed=42,
    verbose=True,
):
    """
    Lightweight one-shot schema search.

    This is the first round of greedy schema search only: each intermediate
    table is independently scored as a node and as an edge, then its better mode
    is fixed. The final schema is built once from all selected modes.

    Candidate evaluations: 2 * number_of_intermediate_tables.

    Returns:
        final_graph, final_node_id_map, selected_modes, history
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(db, target_table=target_table)

    if num_classes is None:
        num_classes = infer_num_classes(initial_graph, target_table)

    base_graph, _ = build_schema_aware_graph(
        db,
        table_modes={},
        target_table=target_table,
    )
    base_graph = copy_task_tensors(initial_graph, base_graph, target_table)
    base_score, _ = train_and_score_schema_gnn(
        base_graph,
        target_table=target_table,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        verbose=False,
    )

    selected_modes = {}
    history = []

    if verbose:
        print(
            f"\n[LightWeight Schema Search] One-shot scoring, "
            f"tables={list(intermediate_tables)}"
        )
        print(f"[LightWeight Schema Search] Base score={base_score:.4f}")

    for table in intermediate_tables:
        candidates = []
        for mode in modes:
            try:
                candidate = score_schema_candidate(
                    db=db,
                    reference_graph=initial_graph,
                    target_table=target_table,
                    table_modes={},
                    scoring_table=table,
                    scoring_mode=mode,
                    base_score=base_score,
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    epochs=epochs,
                    lr=lr,
                    device=device,
                    seed=seed,
                    verbose=False,
                )
                candidates.append(candidate)
                if verbose:
                    print(
                        f"  {table:20s} {mode:4s} "
                        f"score={candidate['score']:.4f} "
                        f"gain={candidate['gain']:.4f}"
                    )
            except Exception as exc:
                if verbose:
                    print(f"  skip {table}.{mode}: {exc}")

        if not candidates:
            continue

        best = max(candidates, key=lambda item: (item["score"], item["gain"]))
        selected_modes[table] = best["mode"]
        history.append(
            {
                "round": 1,
                "base_score": base_score,
                "selected_table": best["table"],
                "selected_mode": best["mode"],
                "selected_score": best["score"],
                "selected_gain": best["gain"],
                "fixed_modes": dict(selected_modes),
                "candidates": [
                    {
                        "table": item["table"],
                        "mode": item["mode"],
                        "score": item["score"],
                        "gain": item["gain"],
                    }
                    for item in candidates
                ],
            }
        )

        if verbose:
            print(
                f"[LightWeight Schema Search] Select {best['table']} as "
                f"{best['mode']} (score={best['score']:.4f}, "
                f"gain={best['gain']:.4f})"
            )

    final_graph, final_node_id_map = build_schema_aware_graph(
        db,
        table_modes=selected_modes,
        target_table=target_table,
    )
    final_graph = copy_task_tensors(initial_graph, final_graph, target_table)

    return final_graph, final_node_id_map, selected_modes, history
