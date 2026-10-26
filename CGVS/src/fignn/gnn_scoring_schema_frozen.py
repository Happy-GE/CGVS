import copy

import torch

from fignn.gnn_scoring_schema import infer_num_classes, set_schema_search_seed
from fignn.models import GNN
from fignn.schema_graph_building import (
    EDGE_MODE,
    NODE_MODE,
    build_schema_aware_graph,
    copy_task_tensors,
    identify_intermediate_tables,
)


def _make_schema_graph(db, reference_graph, target_table, table_modes, include_fk_features=False):
    graph, node_id_map = build_schema_aware_graph(
        db,
        table_modes=table_modes,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )
    graph = copy_task_tensors(reference_graph, graph, target_table)
    return graph, node_id_map


def _edgeized_relation_for(table, rel):
    return rel == f"{table}_edge" or rel.startswith(f"{table}_edge_")


def _edgeized_reverse_relation_for(table, rel):
    prefix = f"rev_{table}_edge"
    return rel == prefix or rel.startswith(f"{prefix}_")


def _table_node_relation(db, table, edge_type):
    src, rel, dst = edge_type
    if "_edge_" in rel:
        return False
    return any(
        src_table == table
        and (
            (src == src_table and rel == f"{src_col}_to_{dst_col}" and dst == dst_table)
            or (src == dst_table and rel == f"rev_{src_col}_to_{dst_col}" and dst == src_table)
        )
        for src_table, src_col, dst_table, dst_col in db.foreign_keys
    )


def _table_edge_relation(table, edge_type):
    _, rel, _ = edge_type
    return _edgeized_relation_for(table, rel) or _edgeized_reverse_relation_for(table, rel)


def _build_training_graph(
    db,
    reference_graph,
    target_table,
    fixed_modes,
    remaining,
    include_fk_features=False,
):
    """
    Build one shared-metadata graph for the current round.

    Remaining tables are kept as nodes so their node-mode relations are present.
    Edge-mode relations are then merged in, giving the frozen GNN parameters for
    both choices before validation-only candidate scoring.
    """
    node_modes = dict(fixed_modes)
    for table in remaining:
        node_modes[table] = NODE_MODE

    training_graph, node_id_map = build_schema_aware_graph(
        db,
        table_modes=node_modes,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )

    edge_modes = dict(fixed_modes)
    for table in remaining:
        edge_modes[table] = EDGE_MODE
    edge_graph, _ = build_schema_aware_graph(
        db,
        table_modes=edge_modes,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )

    for edge_type in edge_graph.edge_types:
        if "_edge_" in edge_type[1]:
            training_graph[edge_type].edge_index = edge_graph[edge_type].edge_index.clone()

    training_graph = copy_task_tensors(reference_graph, training_graph, target_table)
    return training_graph, node_id_map


def _mode_for_table(fixed_modes, scoring_table, scoring_mode, table):
    if table == scoring_table:
        return scoring_mode
    return fixed_modes.get(table, NODE_MODE)


def _edge_type_enabled(db, fixed_modes, remaining, scoring_table, scoring_mode, edge_type):
    all_tables = set(fixed_modes) | set(remaining)
    for table in all_tables:
        mode = _mode_for_table(fixed_modes, scoring_table, scoring_mode, table)
        if _table_node_relation(db, table, edge_type):
            return mode == NODE_MODE
        if _table_edge_relation(table, edge_type):
            return mode == EDGE_MODE
    return True


def _candidate_edge_index_dict(
    db,
    graph,
    fixed_modes,
    remaining,
    scoring_table,
    scoring_mode,
):
    edge_index_dict = {}
    for edge_type, edge_index in graph.edge_index_dict.items():
        if _edge_type_enabled(
            db,
            fixed_modes,
            remaining,
            scoring_table,
            scoring_mode,
            edge_type,
        ):
            edge_index_dict[edge_type] = edge_index
        else:
            edge_index_dict[edge_type] = edge_index.new_empty((2, 0))
    return edge_index_dict


@torch.no_grad()
def _evaluate_with_edge_index_dict(model, graph, edge_index_dict, target_table):
    model.eval()
    out = model(graph.x_dict, edge_index_dict)
    mask = graph[target_table].val_mask
    y_true = graph[target_table].y[mask]
    y_pred = out[mask].argmax(dim=-1)
    return (y_pred == y_true).float().mean().item()


def _train_current_schema_model(
    graph,
    edge_index_dict,
    target_table,
    num_classes,
    hidden_dim,
    epochs,
    lr,
    device,
    seed=None,
    verbose=False,
):
    set_schema_search_seed(seed)
    graph = graph.to(device)
    model = GNN(
        metadata=graph.metadata(),
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        target_node=target_table,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    y = graph[target_table].y
    train_mask = graph[target_table].train_mask
    val_mask = graph[target_table].val_mask
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x_dict, edge_index_dict)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(graph.x_dict, edge_index_dict)
            val_pred = val_out[val_mask].argmax(dim=-1)
            val_acc = (val_pred == y[val_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, graph


def _score_candidate_with_frozen_model(
    db,
    graph,
    target_table,
    table_modes,
    remaining,
    scoring_table,
    scoring_mode,
    source_model,
):
    candidate_modes = dict(table_modes)
    candidate_modes[scoring_table] = scoring_mode
    edge_index_dict = _candidate_edge_index_dict(
        db=db,
        graph=graph,
        fixed_modes=table_modes,
        remaining=remaining,
        scoring_table=scoring_table,
        scoring_mode=scoring_mode,
    )
    score = _evaluate_with_edge_index_dict(
        source_model,
        graph,
        edge_index_dict,
        target_table,
    )
    return {
        "table": scoring_table,
        "mode": scoring_mode,
        "score": score,
        "table_modes": candidate_modes,
    }


def search_frozen_forward_schema_graph(
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
    retrain_each_round=True,
    include_fk_features=False,
    verbose=True,
):
    """
    Greedily choose whether each intermediate table is a node or an edge.

    This follows an auGraph-like loop: train a GNN on the current schema using
    the training split, freeze it, score every remaining table/mode candidate by
    validation forward propagation only, fix the highest-scoring candidate, and
    continue to the next table.

    If ``retrain_each_round`` is False, the first frozen model is reused for all
    rounds. The default True retrains after a structure is fixed, matching the
    usual iterative graph-augmentation workflow.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(db, target_table=target_table)
    if num_classes is None:
        num_classes = infer_num_classes(initial_graph, target_table)

    fixed_modes = {}
    history = []
    remaining = list(intermediate_tables)
    current_graph, current_node_id_map = _make_schema_graph(
        db,
        initial_graph,
        target_table,
        fixed_modes,
        include_fk_features=include_fk_features,
    )

    frozen_model = None
    trained_graph = None
    round_id = 0

    while remaining:
        round_id += 1
        if verbose:
            print(f"\n[Frozen Forward Schema Search] Round {round_id}, remaining={remaining}")

        if frozen_model is None or retrain_each_round:
            training_graph, _ = _build_training_graph(
                db,
                reference_graph=initial_graph,
                target_table=target_table,
                fixed_modes=fixed_modes,
                remaining=remaining,
                include_fk_features=include_fk_features,
            )
            training_graph = training_graph.to(device)
            current_edge_index_dict = _candidate_edge_index_dict(
                db=db,
                graph=training_graph,
                fixed_modes=fixed_modes,
                remaining=remaining,
                scoring_table=None,
                scoring_mode=None,
            )
            frozen_model, trained_graph = _train_current_schema_model(
                training_graph,
                edge_index_dict=current_edge_index_dict,
                target_table=target_table,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed,
                verbose=False,
            )

        base_edge_index_dict = _candidate_edge_index_dict(
            db=db,
            graph=trained_graph,
            fixed_modes=fixed_modes,
            remaining=remaining,
            scoring_table=None,
            scoring_mode=None,
        )
        base_score = _evaluate_with_edge_index_dict(
            frozen_model,
            trained_graph,
            base_edge_index_dict,
            target_table,
        )

        candidates = []
        for table in remaining:
            for mode in modes:
                try:
                    candidate = _score_candidate_with_frozen_model(
                        db=db,
                        graph=trained_graph,
                        target_table=target_table,
                        table_modes=fixed_modes,
                        remaining=remaining,
                        scoring_table=table,
                        scoring_mode=mode,
                        source_model=frozen_model,
                    )
                    candidate["gain"] = candidate["score"] - base_score
                    candidates.append(candidate)
                    if verbose:
                        print(
                            f"  {table:20s} {mode:4s} "
                            f"val_acc={candidate['score']:.4f} "
                            f"gain={candidate['gain']:.4f}"
                        )
                except Exception as exc:
                    if verbose:
                        print(f"  skip {table}.{mode}: {exc}")

        if not candidates:
            break

        best = max(candidates, key=lambda item: (item["score"], item["gain"]))
        fixed_modes[best["table"]] = best["mode"]
        current_graph, current_node_id_map = _make_schema_graph(
            db,
            initial_graph,
            target_table,
            fixed_modes,
            include_fk_features=include_fk_features,
        )
        remaining = [table for table in remaining if table != best["table"]]

        step = {
            "round": round_id,
            "base_score": base_score,
            "selected_table": best["table"],
            "selected_mode": best["mode"],
            "selected_score": best["score"],
            "selected_gain": best["gain"],
            "fixed_modes": dict(fixed_modes),
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
        history.append(step)

        if verbose:
            print(
                f"[Frozen Forward Schema Search] Fix {best['table']} as {best['mode']} "
                f"(val_acc={best['score']:.4f}, gain={best['gain']:.4f})"
            )

    final_graph, final_node_id_map = _make_schema_graph(
        db,
        initial_graph,
        target_table,
        fixed_modes,
        include_fk_features=include_fk_features,
    )
    return final_graph, final_node_id_map, fixed_modes, history
