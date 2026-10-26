import argparse
import copy
import math
import os
import time
from datetime import datetime
from itertools import product

import pandas as pd
import torch

import shared_method_utils as shared


DATASET = "Biodegradability"

REGRESSION_DATASET_CONFIGS = {
    "Biodegradability": {
        "data_dir": "Biodegradability",
        "task_table": "molecule",
        "task_pk": "molecule_id",
        "label_col": "activity",
        "split_ratio": (0.7, 0.15, 0.15),
        "split_seeds": [ 23, 24, 25 ],
    },
}

REGRESSION_SELECTION_METRICS = {
    "Biodegradability": {"gnn": "rmse", "schema": "rmse"},
}

ALL_METHODS = [
    "REG",
    "Attribute-aware",
    "Random",
    "Gate-edge-default",
    "Gate-node-default",
    "Gate-attribute-default",
    "Consensus-vote",
    "Consensus-Val",
    "All-edge",
    "Exhaustive-best",
]

METRICS = ("rmse", "mae", "r2")
LOWER_IS_BETTER = {"rmse", "mae"}


def _default_output_path(dataset_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = f"{dataset_name}_all_method_regression_{timestamp}"
    return shared._default_output_path(label)


def _format_schema(schema):
    return shared.base.format_schema(schema)


def _metric_sort_value(value, metric_name):
    if value is None or pd.isna(value):
        return float("-inf")
    if metric_name in LOWER_IS_BETTER:
        return -float(value)
    return float(value)


def _score_from_metrics(metrics, metric_name):
    if metric_name not in METRICS:
        raise ValueError(f"Unsupported regression metric: {metric_name}")
    return _metric_sort_value(metrics[metric_name], metric_name)


def _format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "-"
    if pd.isna(std_value):
        return f"{mean_value:.4f} +/- N/A"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _metrics_dict(metrics):
    return {metric: metrics.get(metric) for metric in METRICS}


def _metric_extra(gnn_selection_metric, schema_selection_metric, extra=None):
    payload = dict(extra or {})
    payload["gnn_selection_metric"] = gnn_selection_metric
    payload["schema_selection_metric"] = schema_selection_metric
    return payload


def _get_dataset_config(dataset_name):
    try:
        return dict(REGRESSION_DATASET_CONFIGS[dataset_name])
    except KeyError as exc:
        choices = ", ".join(REGRESSION_DATASET_CONFIGS)
        raise ValueError(f"Unknown regression dataset '{dataset_name}'. Choices: {choices}") from exc


def _selection_metrics_for_dataset(dataset_name):
    try:
        return dict(REGRESSION_SELECTION_METRICS[dataset_name])
    except KeyError as exc:
        choices = ", ".join(REGRESSION_SELECTION_METRICS)
        raise ValueError(
            f"Missing REGRESSION_SELECTION_METRICS entry for '{dataset_name}'. Choices: {choices}"
        ) from exc


def _get_split_seeds(dataset_name):
    return list(_get_dataset_config(dataset_name)["split_seeds"])


def _assign_regression_labels(graph, labels, node_id_map, node_type):
    num_nodes = graph[node_type]["x"].shape[0]
    y = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
    pk_to_idx = node_id_map.get(node_type, {})
    for pk, label in labels.items():
        if pk in pk_to_idx:
            y[pk_to_idx[pk]] = float(label)
    graph[node_type].y = y


def _split_regression_labels(graph, node_type, train_ratio, val_ratio, test_ratio, seed):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("Regression split ratios must sum to 1.0")

    y = graph[node_type].y
    labeled_idx = torch.isfinite(y).nonzero(as_tuple=False).view(-1)
    torch.manual_seed(seed)
    perm = torch.randperm(len(labeled_idx))

    n_train = int(train_ratio * len(labeled_idx))
    n_val = int(val_ratio * len(labeled_idx))
    train_idx = labeled_idx[perm[:n_train]]
    val_idx = labeled_idx[perm[n_train : n_train + n_val]]
    test_idx = labeled_idx[perm[n_train + n_val :]]

    mask = torch.zeros(y.shape[0], dtype=torch.bool)
    graph[node_type].train_mask = mask.clone()
    graph[node_type].val_mask = mask.clone()
    graph[node_type].test_mask = mask.clone()
    graph[node_type].train_mask[train_idx] = True
    graph[node_type].val_mask[val_idx] = True
    graph[node_type].test_mask[test_idx] = True


@torch.no_grad()
def _evaluate_regression(model, graph, target_table, split="test"):
    model.eval()
    mask = graph[target_table][f"{split}_mask"]
    if mask.sum().item() == 0:
        raise ValueError(f"No {split} labels available for target table '{target_table}'.")

    pred = model(graph.x_dict, graph.edge_index_dict).view(-1)[mask]
    y = graph[target_table].y[mask].float()
    diff = pred - y
    mse = torch.mean(diff * diff).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(diff)).item()
    ss_res = torch.sum(diff * diff).item()
    centered = y - torch.mean(y)
    ss_tot = torch.sum(centered * centered).item()
    r2 = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "target_mean": torch.mean(y).item(),
        "pred_mean": torch.mean(pred).item(),
    }


def _train_graph_regression(
    graph,
    target_table,
    device,
    seed,
    gnn_selection_metric,
    schema_selection_metric,
    epochs,
):
    exp = shared.base.exp
    shared.base.set_strict_seed(seed)

    graph = graph.to(device)
    model = shared.base.GNN(
        metadata=graph.metadata(),
        hidden_channels=exp.HIDDEN_DIM,
        out_channels=1,
        target_node=target_table,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp.LR,
        weight_decay=getattr(exp, "WEIGHT_DECAY", 0.0),
    )
    loss_fn = torch.nn.MSELoss()
    train_mask = graph[target_table].train_mask
    if train_mask.sum().item() == 0:
        raise ValueError(f"No training labels available for target table '{target_table}'.")

    y = graph[target_table].y.float()
    best_gnn_score = float("-inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict).view(-1)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = _evaluate_regression(model, graph, target_table, split="val")
        gnn_score = _score_from_metrics(val_metrics, gnn_selection_metric)
        if gnn_score > best_gnn_score:
            best_gnn_score = gnn_score
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = _evaluate_regression(model, graph, target_table, split="val")
    schema_score = _score_from_metrics(final_val_metrics, schema_selection_metric)
    return schema_score, model, final_val_metrics


def prepare_regression_context(seed, device, dataset_name=DATASET, split_seed=None):
    shared.base.exp.SEED = seed
    shared.base.set_strict_seed(seed)

    config = _get_dataset_config(dataset_name)
    if split_seed is None:
        split_seed = config["split_seeds"][0]

    data_path = os.path.join(shared.PROJECT_ROOT, "data", config["data_dir"])
    task_table = config["task_table"]
    task_pk = config["task_pk"]
    label_col = config["label_col"]

    db = shared.base.load_relational_data(data_path)
    initial_graph, initial_node_id_map = shared.base.build_fk_graph(db)

    task = pd.read_csv(os.path.join(data_path, "task.csv"))
    labels = dict(zip(task[task_pk], task[label_col].astype(float)))
    _assign_regression_labels(initial_graph, labels, initial_node_id_map, task_table)
    _split_regression_labels(
        initial_graph,
        task_table,
        *config["split_ratio"],
        seed=split_seed,
    )

    intermediate_tables = shared.base.identify_intermediate_tables(db, target_table=task_table)
    all_node_schema = {table: shared.base.NODE_MODE for table in intermediate_tables}
    all_edge_schema = {table: shared.base.EDGE_MODE for table in intermediate_tables}

    return {
        "db": db,
        "initial_graph": initial_graph,
        "task_table": task_table,
        "intermediate_tables": intermediate_tables,
        "all_node_schema": all_node_schema,
        "all_edge_schema": all_edge_schema,
        "device": device,
        "seed": seed,
        "split_seed": split_seed,
        "dataset": dataset_name,
    }


def _run_reg_training(ctx, device, seed, gnn_selection_metric, schema_selection_metric):
    shared.base.exp.SEED = seed
    shared.base.set_strict_seed(seed)
    start = time.perf_counter()
    graph = copy.deepcopy(ctx["initial_graph"])
    _, model, _ = _train_graph_regression(
        graph,
        target_table=ctx["task_table"],
        device=device,
        seed=seed,
        gnn_selection_metric=gnn_selection_metric,
        schema_selection_metric=schema_selection_metric,
        epochs=shared.base.exp.FINAL_EPOCHS,
    )
    metrics = _evaluate_regression(model, graph, ctx["task_table"], split="test")
    return {"metrics": metrics}, time.perf_counter() - start


def _run_final_training(
    method_name,
    schema,
    ctx,
    device,
    seed,
    gnn_selection_metric,
    schema_selection_metric,
):
    exp = shared.base.exp
    exp.SEED = seed
    shared.base.set_strict_seed(seed)
    include_fk_features = getattr(shared.base, "INCLUDE_FK_FEATURES", True)

    start = time.perf_counter()
    graph, node_id_map = exp.build_schema_aware_graph(
        ctx["db"],
        table_modes=schema,
        target_table=ctx["task_table"],
        include_fk_features=include_fk_features,
    )
    graph = exp.copy_task_tensors(ctx["initial_graph"], graph, ctx["task_table"])
    _, model, _ = _train_graph_regression(
        graph,
        target_table=ctx["task_table"],
        device=device,
        seed=seed,
        gnn_selection_metric=gnn_selection_metric,
        schema_selection_metric=schema_selection_metric,
        epochs=exp.FINAL_EPOCHS,
    )
    metrics = _evaluate_regression(model, graph, ctx["task_table"], split="test")
    runtime_sec = time.perf_counter() - start
    return {
        "method": method_name,
        "schema": schema,
        "metrics": metrics,
        "final_graph": graph,
        "final_node_id_map": node_id_map,
        "final_model": model,
    }, runtime_sec


def _score_all_possible_schemas(ctx, device, gnn_selection_metric, schema_selection_metric):
    exp = shared.base.exp
    exp.SEED = ctx["seed"]
    shared.base.set_strict_seed(ctx["seed"])
    include_fk_features = getattr(shared.base, "INCLUDE_FK_FEATURES", True)

    rows = []
    total = 2 ** len(ctx["intermediate_tables"])
    print("\n=== Regression exhaustive schema validation ===")
    for schema_id, modes in enumerate(
        product((shared.base.NODE_MODE, shared.base.EDGE_MODE), repeat=len(ctx["intermediate_tables"])),
        start=1,
    ):
        schema = dict(zip(ctx["intermediate_tables"], modes))
        print(f"[Regression-Exhaustive] Schema {schema_id}/{total}: {_format_schema(schema)}")

        start = time.perf_counter()
        graph, _ = exp.build_schema_aware_graph(
            ctx["db"],
            table_modes=schema,
            target_table=ctx["task_table"],
            include_fk_features=include_fk_features,
        )
        graph = exp.copy_task_tensors(ctx["initial_graph"], graph, ctx["task_table"])
        val_score, model, val_metrics = _train_graph_regression(
            graph,
            target_table=ctx["task_table"],
            device=device,
            seed=ctx["seed"],
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
            epochs=exp.VALIDATION_EPOCHS,
        )
        runtime_sec = time.perf_counter() - start
        test_metrics = _evaluate_regression(model, graph, ctx["task_table"], split="test")
        test_score = _score_from_metrics(test_metrics, schema_selection_metric)
        rows.append(
            {
                "run_id": ctx["run_id"],
                "seed": ctx["seed"],
                "split_seed": ctx["split_seed"],
                "schema_id": schema_id,
                "schema": schema,
                "schema_text": _format_schema(schema),
                "val_score": val_score,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_score": test_score,
                "runtime_sec": runtime_sec,
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
                "labels": "-",
            }
        )
    return rows


def _score_schema_for_selection(ctx, schema, device, gnn_selection_metric, schema_selection_metric):
    exp = shared.base.exp
    include_fk_features = getattr(shared.base, "INCLUDE_FK_FEATURES", True)

    start = time.perf_counter()
    graph, _ = exp.build_schema_aware_graph(
        ctx["db"],
        table_modes=schema,
        target_table=ctx["task_table"],
        include_fk_features=include_fk_features,
    )
    graph = exp.copy_task_tensors(ctx["initial_graph"], graph, ctx["task_table"])
    val_score, _, val_metrics = _train_graph_regression(
        graph,
        target_table=ctx["task_table"],
        device=device,
        seed=ctx["seed"],
        gnn_selection_metric=gnn_selection_metric,
        schema_selection_metric=schema_selection_metric,
        epochs=exp.VALIDATION_EPOCHS,
    )
    runtime_sec = time.perf_counter() - start
    return {
        "schema": schema,
        "val_score": val_score,
        "val_rmse": val_metrics["rmse"],
        "val_mae": val_metrics["mae"],
        "val_r2": val_metrics["r2"],
        "runtime_sec": runtime_sec,
    }


def _flip_mode(mode):
    return shared.base.NODE_MODE if mode == shared.base.EDGE_MODE else shared.base.EDGE_MODE


def _run_regression_gate_with_schema_background(
    ctx,
    background_schema,
    background_name,
    device,
    gnn_selection_metric,
    schema_selection_metric,
):
    print(f"\n=== Regression Gate validation: {background_name} ===")
    start = time.perf_counter()
    background_entry = _score_schema_for_selection(
        ctx,
        dict(background_schema),
        device,
        gnn_selection_metric,
        schema_selection_metric,
    )
    selected_schema = dict(background_schema)
    table_entries = {}
    probs = {}

    for table in ctx["intermediate_tables"]:
        candidate_schema = dict(background_schema)
        candidate_schema[table] = _flip_mode(background_schema[table])
        candidate_entry = _score_schema_for_selection(
            ctx,
            candidate_schema,
            device,
            gnn_selection_metric,
            schema_selection_metric,
        )
        table_entries[table] = {
            "background": background_entry,
            "candidate": candidate_entry,
        }
        if candidate_entry["val_score"] > background_entry["val_score"]:
            selected_schema[table] = candidate_schema[table]
            probs[table] = 1.0
        else:
            probs[table] = 0.0

    runtime_sec = time.perf_counter() - start
    print(f"{background_name} regression gate schema: {_format_schema(selected_schema)}")
    return {
        "background": background_name,
        "probs": probs,
        "schema": selected_schema,
        "history": table_entries,
        "runtime_sec": runtime_sec,
    }


def _run_regression_gate_with_background(
    ctx,
    background_mode,
    device,
    gnn_selection_metric,
    schema_selection_metric,
):
    background_name = "node-default" if background_mode == shared.base.NODE_MODE else "edge-default"
    background_schema = {table: background_mode for table in ctx["intermediate_tables"]}
    return _run_regression_gate_with_schema_background(
        ctx,
        background_schema,
        background_name,
        device,
        gnn_selection_metric,
        schema_selection_metric,
    )


def _vote_schema(schemas, intermediate_tables):
    voted = {}
    for table in intermediate_tables:
        node_votes = sum(schema[table] == shared.base.NODE_MODE for schema in schemas)
        edge_votes = len(schemas) - node_votes
        voted[table] = shared.base.NODE_MODE if node_votes > edge_votes else shared.base.EDGE_MODE
    return voted


def _split_agreed_and_conflicts(schemas, intermediate_tables):
    agreed = {}
    conflicts = []
    for table in intermediate_tables:
        modes = {schema[table] for schema in schemas}
        if len(modes) == 1:
            agreed[table] = next(iter(modes))
        else:
            conflicts.append(table)
    return agreed, conflicts


def _resolve_conflicts_by_regression_validation(
    ctx,
    device,
    agreed,
    conflicts,
    gnn_selection_metric,
    schema_selection_metric,
):
    if not conflicts:
        return dict(agreed), []

    candidates = []
    for schema_id, modes in enumerate(
        product((shared.base.NODE_MODE, shared.base.EDGE_MODE), repeat=len(conflicts)),
        start=1,
    ):
        schema = dict(agreed)
        schema.update(dict(zip(conflicts, modes)))
        entry = _score_schema_for_selection(
            ctx,
            schema,
            device,
            gnn_selection_metric,
            schema_selection_metric,
        )
        entry["schema_id"] = schema_id
        candidates.append(entry)

    best = max(
        candidates,
        key=lambda item: (
            item["val_score"],
            -item["schema_id"],
        ),
    )
    return best["schema"], candidates


def _best_schema_entry(entries):
    if not entries:
        raise ValueError("No exhaustive schema entries available.")
    return max(
        entries,
        key=lambda item: (
            _metric_sort_value(item["val_rmse"], "rmse")
            if item.get("val_rmse") is not None
            else item["val_score"],
            -item["schema_id"],
        ),
    )


def _make_method_row(
    run_id,
    seed,
    split_seed,
    method,
    schema,
    metrics,
    selection_time_sec,
    gate_time_sec,
    schema_time_sec,
    final_train_time_sec,
    method_total_time_sec,
    extra=None,
):
    row = {
        "run_id": run_id,
        "seed": seed,
        "split_seed": split_seed,
        "method": method,
        "selected_view": "",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics.get("r2"),
        "selection_time_sec": selection_time_sec,
        "gate_time_sec": gate_time_sec,
        "schema_time_sec": schema_time_sec,
        "final_train_time_sec": final_train_time_sec,
        "method_total_time_sec": method_total_time_sec,
        "schema": _format_schema(schema),
    }
    if extra:
        row.update(extra)
    return row


def _select_random_schema(ctx):
    start = time.perf_counter()
    random_seed = ctx["split_seed"] * 10 + ctx["seed"]
    import random

    rng = random.Random(random_seed)
    schema = {
        table: rng.choice((shared.base.NODE_MODE, shared.base.EDGE_MODE))
        for table in ctx["intermediate_tables"]
    }
    runtime_sec = time.perf_counter() - start
    return {
        "schema": schema,
        "selection_runtime_sec": runtime_sec,
        "gate_runtime_sec": 0.0,
        "runtime_sec": runtime_sec,
        "selected_view": "random-schema",
    }


def _select_attribute_aware(ctx):
    return shared.base.quiet_call(
        shared.base.exp.select_attribute_aware_view,
        db=ctx["db"],
        intermediate_tables=ctx["intermediate_tables"],
        rich_attr_threshold=shared.base.exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
    )


def _run_seed_methods(methods, run_id, seed, device, dataset_name, split_seed):
    ctx = prepare_regression_context(seed, device, dataset_name=dataset_name, split_seed=split_seed)
    ctx["run_id"] = run_id
    metric_config = _selection_metrics_for_dataset(dataset_name)
    gnn_selection_metric = metric_config["gnn"]
    schema_selection_metric = metric_config["schema"]

    method_rows = []
    exhaustive_rows = []
    row_by_method = {}

    attribute_aware_selection = None
    if any(
        method in methods
        for method in (
            "Attribute-aware",
            "Gate-attribute-default",
            "Consensus-vote",
            "Consensus-Val",
            "Exhaustive-best",
        )
    ):
        attribute_aware_selection = _select_attribute_aware(ctx)

    need_edge_gate = any(
        method in methods
        for method in ("Gate-edge-default", "Consensus-vote", "Consensus-Val")
    )
    need_node_gate = any(
        method in methods
        for method in ("Gate-node-default", "Consensus-vote", "Consensus-Val")
    )
    if "Gate-attribute-default" in methods and attribute_aware_selection is not None:
        attribute_schema = attribute_aware_selection["schema"]
        if attribute_schema == ctx["all_edge_schema"]:
            need_edge_gate = True
        elif attribute_schema == ctx["all_node_schema"]:
            need_node_gate = True

    gate_result = {}
    if need_edge_gate:
        gate_result["edge_result"] = _run_regression_gate_with_background(
            ctx,
            shared.base.EDGE_MODE,
            device,
            gnn_selection_metric,
            schema_selection_metric,
        )
    if need_node_gate:
        gate_result["node_result"] = _run_regression_gate_with_background(
            ctx,
            shared.base.NODE_MODE,
            device,
            gnn_selection_metric,
            schema_selection_metric,
        )

    gate_attribute_result = None
    gate_attribute_incremental_time = 0.0
    gate_attribute_source = None
    if any(method in methods for method in ("Gate-attribute-default", "Consensus-vote", "Consensus-Val")):
        attribute_schema = attribute_aware_selection["schema"]
        if attribute_schema == ctx["all_edge_schema"]:
            if "edge_result" in gate_result:
                gate_attribute_result = copy.deepcopy(gate_result["edge_result"])
                gate_attribute_source = "reuse-edge"
            else:
                gate_attribute_result = _run_regression_gate_with_background(
                    ctx,
                    shared.base.EDGE_MODE,
                    device,
                    gnn_selection_metric,
                    schema_selection_metric,
                )
                gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
                gate_attribute_source = "run-edge"
        elif attribute_schema == ctx["all_node_schema"]:
            if "node_result" in gate_result:
                gate_attribute_result = copy.deepcopy(gate_result["node_result"])
                gate_attribute_source = "reuse-node"
            else:
                gate_attribute_result = _run_regression_gate_with_background(
                    ctx,
                    shared.base.NODE_MODE,
                    device,
                    gnn_selection_metric,
                    schema_selection_metric,
                )
                gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
                gate_attribute_source = "run-node"
        else:
            gate_attribute_result = _run_regression_gate_with_schema_background(
                ctx,
                attribute_schema,
                "attribute-default",
                device,
                gnn_selection_metric,
                schema_selection_metric,
            )
            gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
            gate_attribute_source = "run-attribute"

    gate_schemas = []
    if "edge_result" in gate_result:
        gate_schemas.append(gate_result["edge_result"]["schema"])
    if "node_result" in gate_result:
        gate_schemas.append(gate_result["node_result"]["schema"])
    if gate_attribute_result is not None:
        gate_schemas.append(gate_attribute_result["schema"])

    consensus_vote_schema = None
    if "Consensus-vote" in methods:
        consensus_vote_schema = _vote_schema(gate_schemas, ctx["intermediate_tables"])

    consensus_val_schema = None
    consensus_conflict_count = None
    conflict_candidates = []
    if "Consensus-Val" in methods:
        agreed, conflicts = _split_agreed_and_conflicts(gate_schemas, ctx["intermediate_tables"])
        consensus_val_schema, conflict_candidates = _resolve_conflicts_by_regression_validation(
            ctx,
            device,
            agreed,
            conflicts,
            gnn_selection_metric,
            schema_selection_metric,
        )
        consensus_conflict_count = len(conflicts)

    if "REG" in methods:
        result, final_time = _run_reg_training(
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["REG"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "REG",
            ctx["all_node_schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=final_time,
            method_total_time_sec=final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "original-pk-fk-graph"},
            ),
        )

    if "All-edge" in methods:
        result, final_time = _run_final_training(
            "All-edge",
            ctx["all_edge_schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["All-edge"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "All-edge",
            ctx["all_edge_schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=final_time,
            method_total_time_sec=final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "fixed"},
            ),
        )

    if "Random" in methods:
        selection = _select_random_schema(ctx)
        result, final_time = _run_final_training(
            "Random",
            selection["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Random"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Random",
            selection["schema"],
            result["metrics"],
            selection_time_sec=selection["selection_runtime_sec"],
            gate_time_sec=0.0,
            schema_time_sec=selection["runtime_sec"],
            final_train_time_sec=final_time,
            method_total_time_sec=selection["runtime_sec"] + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": selection["selected_view"]},
            ),
        )

    if "Attribute-aware" in methods:
        selection = attribute_aware_selection
        result, final_time = _run_final_training(
            "Attribute-aware",
            selection["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Attribute-aware"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Attribute-aware",
            selection["schema"],
            result["metrics"],
            selection_time_sec=selection["selection_runtime_sec"],
            gate_time_sec=selection["gate_runtime_sec"],
            schema_time_sec=selection["runtime_sec"],
            final_train_time_sec=final_time,
            method_total_time_sec=selection["runtime_sec"] + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": selection["selected_view"]},
            ),
        )

    if "Gate-edge-default" in methods:
        result, final_time = _run_final_training(
            "Gate-edge-default",
            gate_result["edge_result"]["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Gate-edge-default"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Gate-edge-default",
            gate_result["edge_result"]["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=gate_result["edge_result"]["runtime_sec"],
            schema_time_sec=gate_result["edge_result"]["runtime_sec"],
            final_train_time_sec=final_time,
            method_total_time_sec=gate_result["edge_result"]["runtime_sec"] + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "edge-default-regression-gate"},
            ),
        )

    if "Gate-node-default" in methods:
        result, final_time = _run_final_training(
            "Gate-node-default",
            gate_result["node_result"]["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Gate-node-default"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Gate-node-default",
            gate_result["node_result"]["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=gate_result["node_result"]["runtime_sec"],
            schema_time_sec=gate_result["node_result"]["runtime_sec"],
            final_train_time_sec=final_time,
            method_total_time_sec=gate_result["node_result"]["runtime_sec"] + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "node-default-regression-gate"},
            ),
        )

    if "Gate-attribute-default" in methods:
        result, final_time = _run_final_training(
            "Gate-attribute-default",
            gate_attribute_result["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Gate-attribute-default"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Gate-attribute-default",
            gate_attribute_result["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=gate_attribute_result["runtime_sec"],
            schema_time_sec=gate_attribute_result["runtime_sec"],
            final_train_time_sec=final_time,
            method_total_time_sec=gate_attribute_result["runtime_sec"] + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {
                    "selected_view": "attribute-default-regression-gate",
                    "attribute_gate_source": gate_attribute_source,
                    "attribute_gate_incremental_time_sec": gate_attribute_incremental_time,
                },
            ),
        )

    if "Consensus-vote" in methods:
        vote_gate_time = (
            gate_result["edge_result"]["runtime_sec"]
            + gate_result["node_result"]["runtime_sec"]
            + gate_attribute_incremental_time
        )
        result, final_time = _run_final_training(
            "Consensus-vote",
            consensus_vote_schema,
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Consensus-vote"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Consensus-vote",
            consensus_vote_schema,
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=vote_gate_time,
            schema_time_sec=vote_gate_time,
            final_train_time_sec=final_time,
            method_total_time_sec=vote_gate_time + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {
                    "selected_view": "3gate-vote-regression",
                    "attribute_gate_source": gate_attribute_source,
                    "attribute_gate_incremental_time_sec": gate_attribute_incremental_time,
                },
            ),
        )

    if "Consensus-Val" in methods:
        conflict_time = sum(item["runtime_sec"] for item in conflict_candidates)
        gate_time = (
            gate_result["edge_result"]["runtime_sec"]
            + gate_result["node_result"]["runtime_sec"]
            + gate_attribute_incremental_time
        )
        schema_time = gate_time + conflict_time
        result, final_time = _run_final_training(
            "Consensus-Val",
            consensus_val_schema,
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Consensus-Val"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Consensus-Val",
            consensus_val_schema,
            result["metrics"],
            selection_time_sec=conflict_time,
            gate_time_sec=gate_time,
            schema_time_sec=schema_time,
            final_train_time_sec=final_time,
            method_total_time_sec=schema_time + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {
                    "selected_view": "3gate-conflict-val-regression",
                    "consensus_conflict_count": consensus_conflict_count,
                    "consensus_conflict_time_sec": conflict_time,
                    "attribute_gate_source": gate_attribute_source,
                    "attribute_gate_incremental_time_sec": gate_attribute_incremental_time,
                },
            ),
        )

    if "Exhaustive-best" in methods:
        all_schema_scores = _score_all_possible_schemas(
            ctx,
            device,
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
        )
        exhaustive_time = sum(item["runtime_sec"] for item in all_schema_scores)
        best = _best_schema_entry(all_schema_scores)
        best["labels"] = (
            "Exhaustive-best"
            if best["labels"] == "-"
            else f'{best["labels"]},Exhaustive-best'
        )
        result, final_time = _run_final_training(
            "Exhaustive-best",
            best["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Exhaustive-best"] = _make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Exhaustive-best",
            best["schema"],
            result["metrics"],
            selection_time_sec=exhaustive_time,
            gate_time_sec=0.0,
            schema_time_sec=exhaustive_time,
            final_train_time_sec=final_time,
            method_total_time_sec=exhaustive_time + final_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {
                    "selected_view": f"best-val-schema-{best['schema_id']}",
                    "exhaustive_val_rmse": best["val_rmse"],
                    "exhaustive_test_rmse": best["rmse"],
                },
            ),
        )
        for item in all_schema_scores:
            exhaustive_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "split_seed": ctx["split_seed"],
                    "schema_id": item["schema_id"],
                    "schema": item["schema_text"],
                    "val_score": item["val_score"],
                    "val_rmse": item["val_rmse"],
                    "val_mae": item["val_mae"],
                    "val_r2": item["val_r2"],
                    "test_score": item["test_score"],
                    "rmse": item["rmse"],
                    "mae": item["mae"],
                    "r2": item["r2"],
                    "runtime_sec": item["runtime_sec"],
                    "labels": item["labels"],
                }
            )

    for method in methods:
        print(
            f"Run {run_id} | split_seed={split_seed} | seed={seed} | "
            f"dataset={dataset_name} | method={method} | regression"
        )
        row = row_by_method.get(method)
        if row is None:
            raise ValueError(f"Method '{method}' was not prepared in regression runner.")
        method_rows.append(copy.deepcopy(row))

    method_rows.extend(_make_ceiling_rows(row_by_method))

    return method_rows, exhaustive_rows


def _best_metric_key(value, metric):
    if value is None or pd.isna(value):
        return float("-inf")
    return -float(value) if metric in LOWER_IS_BETTER else float(value)


def _make_ceiling_rows(row_by_method):
    if not row_by_method:
        return []

    rows = []
    for metric in METRICS:
        best_method, best_row = max(
            row_by_method.items(),
            key=lambda item: (
                _best_metric_key(item[1].get(metric), metric),
                _best_metric_key(item[1].get("rmse"), "rmse"),
                _best_metric_key(item[1].get("mae"), "mae"),
                _best_metric_key(item[1].get("r2"), "r2"),
            ),
        )
        ceiling_row = copy.deepcopy(best_row)
        ceiling_row["method"] = f"Ceiling-best-{metric}"
        ceiling_row["selected_view"] = f'ceiling-{metric}-from-{best_row.get("selected_view", "")}'
        ceiling_row["oracle_source_method"] = best_method
        ceiling_row["oracle_metric"] = metric
        rows.append(ceiling_row)
    return rows


def _summarize_method_rows(rows):
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, pd.DataFrame()

    aggregations = {
        "runs": ("run_id", "count"),
        "split_seeds": ("split_seed", lambda values: " | ".join(sorted(set(map(str, values))))),
        "selection_time_mean": ("selection_time_sec", "mean"),
        "selection_time_std": ("selection_time_sec", "std"),
        "selection_time_var": ("selection_time_sec", "var"),
        "gate_time_mean": ("gate_time_sec", "mean"),
        "gate_time_std": ("gate_time_sec", "std"),
        "gate_time_var": ("gate_time_sec", "var"),
        "schema_time_mean": ("schema_time_sec", "mean"),
        "schema_time_std": ("schema_time_sec", "std"),
        "schema_time_var": ("schema_time_sec", "var"),
        "final_train_time_mean": ("final_train_time_sec", "mean"),
        "final_train_time_std": ("final_train_time_sec", "std"),
        "final_train_time_var": ("final_train_time_sec", "var"),
        "method_total_time_mean": ("method_total_time_sec", "mean"),
        "method_total_time_std": ("method_total_time_sec", "std"),
        "method_total_time_var": ("method_total_time_sec", "var"),
        "schemas": ("schema", lambda values: " | ".join(sorted(set(map(str, values))))),
        "gnn_selection_metric": ("gnn_selection_metric", "first"),
        "schema_selection_metric": ("schema_selection_metric", "first"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
        aggregations[f"{metric}_var"] = (metric, "var")
    if "oracle_source_method" in raw_df.columns:
        aggregations["oracle_source_methods"] = (
            "oracle_source_method",
            lambda values: " | ".join(sorted(set(map(str, values)) - {"nan"})),
        )
    if "oracle_metric" in raw_df.columns:
        aggregations["oracle_metrics"] = (
            "oracle_metric",
            lambda values: " | ".join(sorted(set(map(str, values)) - {"nan"})),
        )

    summary_df = raw_df.groupby("method", as_index=False).agg(**aggregations)
    for metric in METRICS:
        summary_df[f"{metric}_mean_std"] = [
            _format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
            for _, row in summary_df.iterrows()
        ]
        ascending = metric in LOWER_IS_BETTER
        summary_df[f"{metric}_rank"] = (
            summary_df[f"{metric}_mean"].rank(method="min", ascending=ascending).astype("Int64")
        )
    return raw_df, summary_df


def _parse_schema_text(schema_text):
    if not schema_text or pd.isna(schema_text):
        return {}

    schema = {}
    for item in str(schema_text).split(", "):
        if ":" not in item:
            continue
        table, mode = item.split(":", 1)
        schema[table] = mode
    return schema


def _build_schema_bounds_rows(exhaustive_rows):
    if not exhaustive_rows:
        return []

    raw_df = pd.DataFrame(exhaustive_rows)
    required_cols = {"run_id", *METRICS}
    if not required_cols.issubset(raw_df.columns):
        return []

    rows = []
    for bound_name, is_upper in (("Schema-upper-bound", True), ("Schema-lower-bound", False)):
        row = {
            "method": bound_name,
            "runs": int(raw_df["run_id"].nunique()),
            "split_seeds": "-",
            "schemas": "-",
            "oracle_source_methods": "-",
            "oracle_metrics": "-",
        }
        for metric in METRICS:
            per_run = raw_df.groupby("run_id")[metric]
            if metric in LOWER_IS_BETTER:
                series = per_run.min() if is_upper else per_run.max()
            else:
                series = per_run.max() if is_upper else per_run.min()
            row[f"{metric}_mean"] = series.mean()
            row[f"{metric}_std"] = series.std()
            row[f"{metric}_var"] = series.var()
            row[f"{metric}_mean_std"] = _format_mean_std(series.mean(), series.std())
            row[f"{metric}_rank"] = pd.NA
        rows.append(row)
    return rows


def _build_table_mode_summary(method_raw_df):
    if method_raw_df.empty or "schema" not in method_raw_df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in method_raw_df.iterrows():
        schema = _parse_schema_text(row.get("schema"))
        if not schema:
            continue
        for table, mode in schema.items():
            rows.append(
                {
                    "method": row["method"],
                    "intermediate_table": table,
                    "mode": mode,
                }
            )

    if not rows:
        return pd.DataFrame()

    mode_df = pd.DataFrame(rows)
    summary_df = (
        mode_df.groupby(["method", "intermediate_table"], as_index=False)
        .agg(
            total_runs=("mode", "count"),
            node_count=("mode", lambda values: int((values == shared.base.NODE_MODE).sum())),
            edge_count=("mode", lambda values: int((values == shared.base.EDGE_MODE).sum())),
        )
    )
    summary_df["node_ratio"] = summary_df["node_count"] / summary_df["total_runs"]
    summary_df["edge_ratio"] = summary_df["edge_count"] / summary_df["total_runs"]
    return summary_df.sort_values(["method", "intermediate_table"], kind="stable").reset_index(
        drop=True
    )


def _relative_change_vs_reg(method_summary_df):
    baseline_row = method_summary_df[method_summary_df["method"] == "REG"]
    for metric in METRICS:
        col = f"{metric}_vs_reg_pct"
        if baseline_row.empty:
            method_summary_df[col] = pd.NA
            continue
        baseline_value = baseline_row.iloc[0][f"{metric}_mean"]
        if pd.isna(baseline_value) or baseline_value == 0:
            method_summary_df[col] = pd.NA
            continue
        if metric in LOWER_IS_BETTER:
            method_summary_df[col] = (
                (baseline_value - method_summary_df[f"{metric}_mean"]) / abs(baseline_value) * 100.0
            )
        else:
            method_summary_df[col] = (
                (method_summary_df[f"{metric}_mean"] - baseline_value) / abs(baseline_value) * 100.0
            )
    return method_summary_df


def _gap_to_exhaustive(method_summary_df):
    exhaustive_row = method_summary_df[method_summary_df["method"] == "Exhaustive-best"]
    for metric in METRICS:
        col = f"{metric}_gap_to_exhaustive_pct"
        if exhaustive_row.empty:
            method_summary_df[col] = pd.NA
            continue
        exhaustive_value = exhaustive_row.iloc[0][f"{metric}_mean"]
        if pd.isna(exhaustive_value) or exhaustive_value == 0:
            method_summary_df[col] = pd.NA
            continue
        if metric in LOWER_IS_BETTER:
            method_summary_df[col] = (
                (method_summary_df[f"{metric}_mean"] - exhaustive_value) / abs(exhaustive_value) * 100.0
            )
        else:
            method_summary_df[col] = (
                (exhaustive_value - method_summary_df[f"{metric}_mean"]) / abs(exhaustive_value) * 100.0
            )
    return method_summary_df


def _add_conflict_summary(method_raw_df, method_summary_df):
    if "consensus_conflict_count" in method_raw_df.columns:
        conflict_summary = (
            method_raw_df.groupby("method", as_index=False)
            .agg(
                consensus_conflict_count_mean=("consensus_conflict_count", "mean"),
                consensus_conflict_count_std=("consensus_conflict_count", "std"),
                consensus_conflict_count_var=("consensus_conflict_count", "var"),
            )
        )
        return method_summary_df.merge(conflict_summary, on="method", how="left")
    method_summary_df["consensus_conflict_count_mean"] = pd.NA
    method_summary_df["consensus_conflict_count_std"] = pd.NA
    method_summary_df["consensus_conflict_count_var"] = pd.NA
    return method_summary_df


def _reorder_method_summary_columns(df):
    if df.empty:
        return df

    df = df.drop(
        columns=[
            "final_train_time_mean",
            "final_train_time_std",
            "final_train_time_var",
        ],
        errors="ignore",
    )

    ordered = []
    seen = set()

    def add(col_name):
        if col_name in df.columns and col_name not in seen:
            ordered.append(col_name)
            seen.add(col_name)

    for col in ("method", "gnn_selection_metric", "schema_selection_metric", "runs", "split_seeds"):
        add(col)

    for metric in METRICS:
        for suffix in ("mean", "std", "var", "mean_std", "rank"):
            add(f"{metric}_{suffix}")
        add(f"{metric}_vs_reg_pct")

    for col in (
        "selection_time_mean",
        "selection_time_std",
        "selection_time_var",
        "gate_time_mean",
        "gate_time_std",
        "gate_time_var",
        "schema_time_mean",
        "schema_time_std",
        "schema_time_var",
        "method_total_time_mean",
        "method_total_time_std",
        "method_total_time_var",
        "schemas",
    ):
        add(col)

    for col in (
        "consensus_conflict_count_mean",
        "consensus_conflict_count_std",
        "consensus_conflict_count_var",
    ):
        add(col)

    for metric in METRICS:
        add(f"{metric}_gap_to_exhaustive_pct")

    for col in df.columns:
        add(col)

    return df[ordered]


def _summarize_exhaustive_rows(rows):
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, pd.DataFrame()

    summary_df = (
        raw_df.groupby("schema", as_index=False)
        .agg(
            runs=("run_id", "count"),
            val_rmse_mean=("val_rmse", "mean"),
            val_rmse_std=("val_rmse", "std"),
            test_rmse_mean=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            test_mae_mean=("mae", "mean"),
            test_mae_std=("mae", "std"),
            test_r2_mean=("r2", "mean"),
            test_r2_std=("r2", "std"),
            runtime_mean=("runtime_sec", "mean"),
            labels=("labels", lambda values: " | ".join(sorted(set(map(str, values))))),
        )
        .sort_values(["val_rmse_mean", "test_rmse_mean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return raw_df, summary_df


def _build_split_summary(method_raw_df):
    if method_raw_df.empty or "split_seed" not in method_raw_df.columns:
        return pd.DataFrame()
    return (
        method_raw_df.groupby(["split_seed", "method"], as_index=False)
        .agg(
            runs=("seed", "count"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            method_total_time_mean=("method_total_time_sec", "mean"),
            schemas=("schema", lambda values: " | ".join(sorted(set(map(str, values))))),
        )
    )


def run_selected_methods(methods, seeds, output_path, dataset_name=DATASET):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_method_rows = []
    all_exhaustive_rows = []
    split_seeds = _get_split_seeds(dataset_name)
    total_runs = len(split_seeds) * len(seeds)
    run_counter = 0

    for split_seed in split_seeds:
        for seed in seeds:
            run_counter += 1
            print(
                f"Regression progress {run_counter}/{total_runs} | "
                f"dataset={dataset_name} | split_seed={split_seed} | seed={seed}"
            )
            method_rows, exhaustive_rows = _run_seed_methods(
                methods,
                run_counter,
                seed,
                device,
                dataset_name,
                split_seed,
            )
            all_method_rows.extend(method_rows)
            all_exhaustive_rows.extend(exhaustive_rows)

    method_raw_df, method_summary_df = _summarize_method_rows(all_method_rows)
    exhaustive_raw_df, exhaustive_summary_df = _summarize_exhaustive_rows(all_exhaustive_rows)
    method_summary_df = _relative_change_vs_reg(method_summary_df)
    method_summary_df = _gap_to_exhaustive(method_summary_df)
    bounds_rows = _build_schema_bounds_rows(all_exhaustive_rows)
    method_summary_df = _add_conflict_summary(method_raw_df, method_summary_df)
    method_summary_df = _reorder_method_summary_columns(method_summary_df)

    for row in bounds_rows:
        for metric in METRICS:
            row[f"{metric}_vs_reg_pct"] = pd.NA
            row[f"{metric}_gap_to_exhaustive_pct"] = pd.NA

    split_summary_df = _build_split_summary(method_raw_df)
    table_mode_summary_df = _build_table_mode_summary(method_raw_df)
    method_summary_records = method_summary_df.to_dict("records")
    if bounds_rows:
        method_summary_records.extend(bounds_rows)

    sheets = {
        "method_raw": method_raw_df.to_dict("records"),
        "method_summary": method_summary_records,
        "split_summary": split_summary_df.to_dict("records"),
        "table_mode_summary": table_mode_summary_df.to_dict("records"),
        "exhaustive_raw": exhaustive_raw_df.to_dict("records"),
        "exhaustive_summary": exhaustive_summary_df.to_dict("records"),
    }
    shared.base.write_simple_xlsx(sheets, output_path)
    print(f"Saved results to: {output_path}")


def build_parser(default_dataset=DATASET):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        choices=sorted(REGRESSION_DATASET_CONFIGS),
        help=f"Regression dataset to run. Defaults to {default_dataset}.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of default seeds to run. Defaults to all 5 seeds.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Overrides the default seed list.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method list. Defaults to regression-safe methods.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .xlsx path.",
    )
    return parser


def _parse_seeds(args):
    if args.seeds:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    else:
        seeds = shared.DEFAULT_RUN_SEEDS
    if args.runs is not None:
        seeds = seeds[: args.runs]
    if not seeds:
        raise ValueError("No seeds selected. Use --runs >= 1 or provide --seeds.")
    return seeds


def _parse_methods(args):
    if not args.methods:
        return list(ALL_METHODS)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    valid = set(ALL_METHODS)
    unknown = sorted(set(methods) - valid)
    if unknown:
        raise ValueError(f"Unknown regression method(s): {unknown}. Choices: {sorted(valid)}")
    return methods


def main(default_dataset=DATASET):
    parser = build_parser(default_dataset=default_dataset)
    args = parser.parse_args()
    seeds = _parse_seeds(args)
    methods = _parse_methods(args)
    output_path = args.output or _default_output_path(args.dataset)
    run_selected_methods(methods, seeds, output_path, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
