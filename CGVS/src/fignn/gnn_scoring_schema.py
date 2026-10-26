import copy
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from fignn.models import GNN
from fignn.schema_graph_building import (
    EDGE_MODE,
    NODE_MODE,
    build_schema_aware_graph,
    copy_task_tensors,
    identify_intermediate_tables,
)


def set_schema_search_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_num_classes(graph, target_table):
    y = graph[target_table].y
    labeled = y[y >= 0]
    if labeled.numel() == 0:
        raise ValueError(f"No labels found on target table '{target_table}'.")
    return int(labeled.max().item()) + 1


def train_and_score_schema_gnn(
    graph,
    target_table,
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    device=None,
    seed=None,
    selection_metric="accuracy",
    verbose=False,
):
    """
    Train a fresh GNN on one candidate schema graph and return a validation score.
    """
    set_schema_search_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_classes is None:
        num_classes = infer_num_classes(graph, target_table)

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

    if train_mask.sum().item() == 0:
        raise ValueError(f"No training labels available for target table '{target_table}'.")
    if val_mask.sum().item() == 0:
        raise ValueError(f"No validation labels available for target table '{target_table}'.")

    if selection_metric not in {"accuracy", "roc_auc"}:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    best_val = float("-inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(graph.x_dict, graph.edge_index_dict)
            val_metrics = {
                "accuracy": None,
                "roc_auc": None,
            }
            pred = val_out[val_mask].argmax(dim=1)
            val_acc = (pred == y[val_mask]).float().mean().item()
            val_metrics["accuracy"] = val_acc
            try:
                if val_out.shape[1] == 2:
                    probs = torch.softmax(val_out[val_mask], dim=1)[:, 1].detach().cpu().numpy()
                    val_metrics["roc_auc"] = roc_auc_score(y[val_mask].detach().cpu().numpy(), probs)
                else:
                    probs = torch.softmax(val_out[val_mask], dim=1).detach().cpu().numpy()
                    val_metrics["roc_auc"] = roc_auc_score(
                        y[val_mask].detach().cpu().numpy(),
                        probs,
                        multi_class="ovr",
                    )
            except ValueError:
                val_metrics["roc_auc"] = None

            if selection_metric == "roc_auc":
                val_score = val_metrics["roc_auc"]
                if val_score is None:
                    raise ValueError(
                        f"Validation ROC-AUC is unavailable for target_table='{target_table}'."
                    )
            else:
                val_score = val_metrics["accuracy"]

        if val_score > best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            roc_text = (
                f"{val_metrics['roc_auc']:.4f}"
                if val_metrics["roc_auc"] is not None
                else "N/A"
            )
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_roc_auc={roc_text}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val, model


def score_schema_candidate(
    db,
    reference_graph,
    target_table,
    table_modes,
    scoring_table,
    scoring_mode,
    base_score=None,
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    device=None,
    seed=None,
    include_fk_features=False,
    verbose=False,
):
    """
    Build and score one candidate schema graph where `scoring_table` uses
    `scoring_mode`.
    """
    candidate_modes = dict(table_modes)
    candidate_modes[scoring_table] = scoring_mode

    candidate_graph, candidate_node_id_map = build_schema_aware_graph(
        db,
        table_modes=candidate_modes,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )
    candidate_graph = copy_task_tensors(reference_graph, candidate_graph, target_table)

    score, _ = train_and_score_schema_gnn(
        candidate_graph,
        target_table=target_table,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    gain = None if base_score is None else score - base_score
    return {
        "table": scoring_table,
        "mode": scoring_mode,
        "score": score,
        "gain": gain,
        "graph": candidate_graph,
        "node_id_map": candidate_node_id_map,
        "table_modes": candidate_modes,
    }


def search_task_aware_schema_graph(
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
    include_fk_features=False,
    verbose=True,
):
    """
    Greedy schema-level GNN gain search.

    At each round, every unfixed intermediate table is tried as both a node and
    an edge. A fresh GNN is trained on every candidate graph, validation accuracy
    is used as the score, and the highest-gain candidate is fixed.

    Returns:
        final_graph, final_node_id_map, selected_modes, history
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(db, target_table=target_table)

    fixed_modes = {}
    history = []

    current_graph, current_node_id_map = build_schema_aware_graph(
        db,
        table_modes=fixed_modes,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )
    current_graph = copy_task_tensors(initial_graph, current_graph, target_table)

    if num_classes is None:
        num_classes = infer_num_classes(initial_graph, target_table)

    remaining = list(intermediate_tables)
    round_id = 0

    while remaining:
        round_id += 1
        if verbose:
            print(f"\n[Schema Search] Round {round_id}, remaining={remaining}")

        base_score, _ = train_and_score_schema_gnn(
            current_graph,
            target_table=target_table,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            device=device,
            seed=seed,
            verbose=False,
        )

        candidates = []
        for table in remaining:
            for mode in modes:
                try:
                    candidate = score_schema_candidate(
                        db=db,
                        reference_graph=initial_graph,
                        target_table=target_table,
                        table_modes=fixed_modes,
                        scoring_table=table,
                        scoring_mode=mode,
                        base_score=base_score,
                        num_classes=num_classes,
                        hidden_dim=hidden_dim,
                        epochs=epochs,
                        lr=lr,
                        device=device,
                        seed=seed,
                        include_fk_features=include_fk_features,
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
            break

        best = max(candidates, key=lambda item: (item["gain"], item["score"]))
        fixed_modes[best["table"]] = best["mode"]
        current_graph = best["graph"]
        current_node_id_map = best["node_id_map"]
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
                f"[Schema Search] Fix {best['table']} as {best['mode']} "
                f"(score={best['score']:.4f}, gain={best['gain']:.4f})"
            )

    return current_graph, current_node_id_map, fixed_modes, history
