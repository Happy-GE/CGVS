import argparse
import copy
import time
from datetime import datetime
from itertools import product

import pandas as pd
import torch

import shared_method_utils as shared

DATASET = "University"  # Default dataset for this script

# Per-dataset validation metrics used for:
# - gnn: selecting the best validation epoch inside each candidate GNN run
# - schema: selecting the best schema / conflict-resolution candidate
#
# Supported metrics in this file: "accuracy", "f1", "roc_auc"
DATASET_SELECTION_METRICS = {
    "University": {"gnn": "f1", "schema": "f1"},
    # "University": {"gnn": "accuracy", "schema": "accuracy"},
    "Carcinogenesis": {"gnn": "accuracy", "schema": "accuracy"},
    "MovieLens": {"gnn": "roc_auc", "schema": "roc_auc"},
    "UW-CSE": {"gnn": "f1", "schema": "f1"},
    "Hepatitis": {"gnn": "accuracy", "schema": "accuracy"},
    "SameGen": {"gnn": "roc_auc", "schema": "roc_auc"},
    "Elti": {"gnn": "roc_auc", "schema": "roc_auc"},
    # "Hepatitis": {"gnn": "f1", "schema": "f1"},
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
CEILING_METRICS = ("f1", "accuracy", "roc_auc")
METRICS = ("accuracy", "f1", "roc_auc")


def _selection_metrics_for_dataset(dataset_name):
    try:
        return dict(DATASET_SELECTION_METRICS[dataset_name])
    except KeyError as exc:
        choices = ", ".join(DATASET_SELECTION_METRICS)
        raise ValueError(
            f"Missing DATASET_SELECTION_METRICS entry for '{dataset_name}'. Choices: {choices}"
        ) from exc


def _score_from_metrics(metrics, metric_name):
    if metric_name == "roc_auc":
        score = metrics.get("roc_auc")
        if score is None:
            raise ValueError("Validation ROC-AUC is unavailable for this run.")
        return score
    if metric_name == "f1":
        return metrics["f1"]
    if metric_name == "accuracy":
        return metrics["accuracy"]
    raise ValueError(f"Unsupported metric: {metric_name}")


def _format_mean_std(mean_value, std_value):
    if pd.isna(mean_value):
        return "-"
    if pd.isna(std_value):
        return f"{mean_value:.4f} +/- N/A"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _metrics_dict(metrics):
    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics.get("roc_auc"),
    }


def _metric_extra(gnn_selection_metric, schema_selection_metric, extra=None):
    payload = dict(extra or {})
    payload["gnn_selection_metric"] = gnn_selection_metric
    payload["schema_selection_metric"] = schema_selection_metric
    return payload


def _train_graph_with_dataset_metrics(
    graph,
    target_table,
    num_classes,
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
        out_channels=num_classes,
        target_node=target_table,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp.LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    y = graph[target_table].y
    train_mask = graph[target_table].train_mask
    if train_mask.sum().item() == 0:
        raise ValueError(f"No training labels available for target table '{target_table}'.")

    best_gnn_score = float("-inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x_dict, graph.edge_index_dict)
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = shared.base.evaluate_model(
            model,
            graph,
            target_table,
            split="val",
            return_metrics=True,
        )
        gnn_score = _score_from_metrics(val_metrics, gnn_selection_metric)
        if gnn_score > best_gnn_score:
            best_gnn_score = gnn_score
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics = shared.base.evaluate_model(
        model,
        graph,
        target_table,
        split="val",
        return_metrics=True,
    )
    schema_score = _score_from_metrics(final_val_metrics, schema_selection_metric)
    return schema_score, model, final_val_metrics


def _run_final_training_with_dataset_metrics(
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
    _, model, _ = _train_graph_with_dataset_metrics(
        graph,
        target_table=ctx["task_table"],
        num_classes=ctx["num_classes"],
        device=device,
        seed=seed,
        gnn_selection_metric=gnn_selection_metric,
        schema_selection_metric=schema_selection_metric,
        epochs=exp.FINAL_EPOCHS,
    )
    metrics = shared.base.evaluate_model(
        model,
        graph,
        ctx["task_table"],
        split="test",
        return_metrics=True,
    )
    runtime_sec = time.perf_counter() - start
    return (
        {
            "method": method_name,
            "schema": schema,
            "metrics": metrics,
            "final_graph": graph,
            "final_node_id_map": node_id_map,
            "final_model": model,
        },
        runtime_sec,
    )


def _run_reg_training_with_dataset_metrics(
    ctx,
    device,
    seed,
    gnn_selection_metric,
    schema_selection_metric,
):
    exp = shared.base.exp
    exp.SEED = seed
    shared.base.set_strict_seed(seed)

    start = time.perf_counter()
    graph = ctx["initial_graph"]
    _, model, _ = _train_graph_with_dataset_metrics(
        graph,
        target_table=ctx["task_table"],
        num_classes=ctx["num_classes"],
        device=device,
        seed=seed,
        gnn_selection_metric=gnn_selection_metric,
        schema_selection_metric=schema_selection_metric,
        epochs=exp.FINAL_EPOCHS,
    )
    metrics = shared.base.evaluate_model(
        model,
        graph,
        ctx["task_table"],
        split="test",
        return_metrics=True,
    )
    runtime_sec = time.perf_counter() - start
    return {"metrics": metrics}, runtime_sec


def _score_all_possible_schemas_with_dataset_metrics(
    ctx,
    device,
    label_schemas,
    gnn_selection_metric,
    schema_selection_metric,
):
    exp = shared.base.exp
    exp.SEED = ctx["seed"]
    shared.base.set_strict_seed(ctx["seed"])
    include_fk_features = getattr(shared.base, "INCLUDE_FK_FEATURES", True)

    rows = []
    total = 2 ** len(ctx["intermediate_tables"])
    for schema_id, modes in enumerate(
        product((shared.base.NODE_MODE, shared.base.EDGE_MODE), repeat=len(ctx["intermediate_tables"])),
        start=1,
    ):
        schema = dict(zip(ctx["intermediate_tables"], modes))
        print(
            f"[Exhaustive] Schema {schema_id}/{total}: "
            f"{shared.base.format_schema(schema)}"
        )

        start = time.perf_counter()
        graph, _ = exp.build_schema_aware_graph(
            ctx["db"],
            table_modes=schema,
            target_table=ctx["task_table"],
            include_fk_features=include_fk_features,
        )
        graph = exp.copy_task_tensors(ctx["initial_graph"], graph, ctx["task_table"])
        val_score, model, val_metrics = _train_graph_with_dataset_metrics(
            graph,
            target_table=ctx["task_table"],
            num_classes=ctx["num_classes"],
            device=device,
            seed=ctx["seed"],
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
            epochs=exp.VALIDATION_EPOCHS,
        )
        runtime_sec = time.perf_counter() - start
        test_metrics = shared.base.evaluate_model(
            model,
            graph,
            ctx["task_table"],
            split="test",
            return_metrics=True,
        )
        test_score = _score_from_metrics(test_metrics, schema_selection_metric)
        labels = [
            label
            for label, label_schema in label_schemas.items()
            if tuple(sorted(schema.items())) == tuple(sorted(label_schema.items()))
        ]
        rows.append(
            {
                "schema_id": schema_id,
                "schema": schema,
                "val_score": val_score,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics.get("roc_auc"),
                "test_score": test_score,
                "runtime_sec": runtime_sec,
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1"],
                "roc_auc": test_metrics.get("roc_auc"),
                "labels": ",".join(labels) if labels else "-",
            }
        )
    return rows


def _resolve_conflicts_with_dataset_metrics(
    ctx,
    device,
    agreed,
    conflicts,
    gnn_selection_metric,
    schema_selection_metric,
):
    if not conflicts:
        return dict(agreed), []

    exp = shared.base.exp
    include_fk_features = getattr(shared.base, "INCLUDE_FK_FEATURES", True)
    candidates = []
    for schema_id, modes in enumerate(
        product((shared.base.NODE_MODE, shared.base.EDGE_MODE), repeat=len(conflicts)),
        start=1,
    ):
        schema = dict(agreed)
        schema.update(dict(zip(conflicts, modes)))

        start = time.perf_counter()
        graph, _ = exp.build_schema_aware_graph(
            ctx["db"],
            table_modes=schema,
            target_table=ctx["task_table"],
            include_fk_features=include_fk_features,
        )
        graph = exp.copy_task_tensors(ctx["initial_graph"], graph, ctx["task_table"])
        val_score, _, val_metrics = _train_graph_with_dataset_metrics(
            graph,
            target_table=ctx["task_table"],
            num_classes=ctx["num_classes"],
            device=device,
            seed=ctx["seed"],
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
            epochs=exp.VALIDATION_EPOCHS,
        )
        runtime_sec = time.perf_counter() - start
        candidates.append(
            {
                "schema_id": schema_id,
                "schema": schema,
                "val_score": val_score,
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics.get("roc_auc"),
                "runtime_sec": runtime_sec,
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            _metric_sort_value(item["val_score"]),
            -item["schema_id"],
        ),
    )
    return best["schema"], candidates


def _default_output_path(dataset_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    label = f"{dataset_name}_all_method_{timestamp}"
    return shared._default_output_path(label)


def _build_label_schemas(
    ctx,
    gate_result,
    attribute_aware_selection=None,
    gate_attribute_result=None,
    consensus_vote_schema=None,
    consensus_val_schema=None,
):
    if attribute_aware_selection is None:
        attribute_aware_selection = shared.base.quiet_call(
            shared.base.exp.select_attribute_aware_view,
            db=ctx["db"],
            intermediate_tables=ctx["intermediate_tables"],
            rich_attr_threshold=shared.base.exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
        )
    label_schemas = {
        "REG": ctx["all_node_schema"],
        "Attribute-aware": attribute_aware_selection["schema"],
        "All-node": ctx["all_node_schema"],
        "All-edge": ctx["all_edge_schema"],
    }
    if gate_result is not None:
        if "final_schema" in gate_result:
            label_schemas["Consensus"] = gate_result["final_schema"]
        label_schemas["Gate-edge-default"] = gate_result["edge_result"]["schema"]
        label_schemas["Gate-node-default"] = gate_result["node_result"]["schema"]
    if gate_attribute_result is not None:
        label_schemas["Gate-attribute-default"] = gate_attribute_result["schema"]
    if consensus_vote_schema is not None:
        label_schemas["Consensus-vote"] = consensus_vote_schema
    if consensus_val_schema is not None:
        label_schemas["Consensus-Val"] = consensus_val_schema
    return label_schemas


def _vote_schema(schemas, intermediate_tables):
    voted = {}
    for table in intermediate_tables:
        node_votes = sum(schema[table] == shared.base.NODE_MODE for schema in schemas)
        edge_votes = len(schemas) - node_votes
        voted[table] = (
            shared.base.NODE_MODE if node_votes > edge_votes else shared.base.EDGE_MODE
        )
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


def _make_attribute_aware_row(run_id, seed, split_seed, selection, selected_row):
    row = shared.base.make_method_row(
        run_id,
        seed,
        split_seed,
        "Attribute-aware",
        selection["schema"],
        {
            "accuracy": selected_row["accuracy"],
            "f1": selected_row["f1"],
            "roc_auc": selected_row["roc_auc"],
        },
        selection_time_sec=selection["selection_runtime_sec"],
        gate_time_sec=selection["gate_runtime_sec"],
        schema_time_sec=selection["runtime_sec"],
        final_train_time_sec=selected_row["final_train_time_sec"],
        method_total_time_sec=selection["runtime_sec"] + selected_row["final_train_time_sec"],
        extra=_metric_extra(
            selected_row["gnn_selection_metric"],
            selected_row["schema_selection_metric"],
            {"selected_view": selection["selected_view"]},
        ),
    )
    return row


def _make_random_row(run_id, seed, split_seed, selection, selected_row):
    row = shared.base.make_method_row(
        run_id,
        seed,
        split_seed,
        "Random",
        selection["schema"],
        {
            "accuracy": selected_row["accuracy"],
            "f1": selected_row["f1"],
            "roc_auc": selected_row["roc_auc"],
        },
        selection_time_sec=selection["selection_runtime_sec"],
        gate_time_sec=0.0,
        schema_time_sec=selection["runtime_sec"],
        final_train_time_sec=selected_row["final_train_time_sec"],
        method_total_time_sec=selection["runtime_sec"] + selected_row["final_train_time_sec"],
        extra=_metric_extra(
            selected_row["gnn_selection_metric"],
            selected_row["schema_selection_metric"],
            {"selected_view": selection["selected_view"]},
        ),
    )
    return row


def _metric_sort_value(value):
    if value is None:
        return float("-inf")
    return value


def _make_ceiling_rows(row_by_method):
    if not row_by_method:
        return []

    rows = []
    for metric in CEILING_METRICS:
        best_method, best_row = max(
            row_by_method.items(),
            key=lambda item: (
                _metric_sort_value(item[1].get(metric)),
                _metric_sort_value(item[1].get("f1")),
                _metric_sort_value(item[1].get("accuracy")),
                _metric_sort_value(item[1].get("roc_auc")),
            ),
        )
        ceiling_row = copy.deepcopy(best_row)
        ceiling_row["method"] = f"Ceiling-best-{metric}"
        ceiling_row["selected_view"] = f'ceiling-{metric}-from-{best_row.get("selected_view", "")}'
        ceiling_row["oracle_source_method"] = best_method
        ceiling_row["oracle_metric"] = metric
        rows.append(ceiling_row)
    return rows


def _build_split_summary(method_raw_df):
    if method_raw_df.empty or "split_seed" not in method_raw_df.columns:
        return pd.DataFrame()
    return (
        method_raw_df.groupby(["split_seed", "method"], as_index=False)
        .agg(
            runs=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            method_total_time_mean=("method_total_time_sec", "mean"),
            schemas=("schema", lambda values: " | ".join(sorted(set(map(str, values))))),
            oracle_source_methods=(
                "oracle_source_method",
                lambda values: " | ".join(sorted(set(map(str, values)) - {"nan"})),
            ),
            oracle_metrics=(
                "oracle_metric",
                lambda values: " | ".join(sorted(set(map(str, values)) - {"nan"})),
            ),
        )
    )


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


def _enrich_method_summary(method_summary_df, primary_metric):
    if method_summary_df.empty:
        return method_summary_df

    summary_df = method_summary_df.copy()
    for metric in METRICS:
        summary_df[f"{metric}_mean_std"] = [
            _format_mean_std(row[f"{metric}_mean"], row[f"{metric}_std"])
            for _, row in summary_df.iterrows()
        ]
        summary_df[f"{metric}_rank"] = (
            summary_df[f"{metric}_mean"].rank(method="min", ascending=False).astype("Int64")
        )
    summary_df = summary_df.sort_values(
        by=[f"{primary_metric}_mean", f"{primary_metric}_rank"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return summary_df


def _build_schema_bounds_rows(exhaustive_rows):
    if not exhaustive_rows:
        return []

    raw_df = pd.DataFrame(exhaustive_rows)
    required_cols = {"run_id", *METRICS}
    if not required_cols.issubset(raw_df.columns):
        return []

    rows = []
    for bound_name, reducer in (("Schema-upper-bound", "max"), ("Schema-lower-bound", "min")):
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
            series = per_run.max() if reducer == "max" else per_run.min()
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


def _run_seed_methods(methods, run_id, seed, device, dataset_name, split_seed):
    ctx = shared.prepare_context(
        seed,
        device,
        dataset_name=dataset_name,
        split_seed=split_seed,
    )
    method_rows = []
    exhaustive_rows = []
    row_by_method = {}
    metric_config = _selection_metrics_for_dataset(dataset_name)
    gnn_selection_metric = metric_config["gnn"]
    schema_selection_metric = metric_config["schema"]

    if "REG" in methods:
        result, final_train_time = _run_reg_training_with_dataset_metrics(
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["REG"] = shared.base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "REG",
            ctx["all_node_schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=final_train_time,
            method_total_time_sec=final_train_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "original-pk-fk-graph"},
            ),
        )

    if "All-edge" in methods:
        result, final_train_time = _run_final_training_with_dataset_metrics(
            "All-edge",
            ctx["all_edge_schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["All-edge"] = shared.base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "All-edge",
            result["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=final_train_time,
            method_total_time_sec=final_train_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "fixed"},
            ),
        )

    need_consensus_bundle = "Consensus" in methods
    need_gate_pair = any(
        method in methods
        for method in (
            "Gate-edge-default",
            "Gate-node-default",
            "Consensus-vote",
            "Consensus-Val",
            "Exhaustive-best",
        )
    )
    bundle = None
    gate_pair = None
    gate_result = None
    if need_consensus_bundle:
        bundle = shared.run_gate_consensus_bundle(ctx)
        gate_result = bundle["result"]
    elif need_gate_pair:
        gate_pair = shared.run_gate_pair_bundle(ctx)
        gate_result = {
            "edge_result": gate_pair["edge_result"],
            "node_result": gate_pair["node_result"],
        }

    if "Consensus" in methods:
        row_by_method["Consensus"] = shared.base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Consensus",
            gate_result["final_schema"],
            gate_result["metrics"],
            selection_time_sec=bundle["conflict_time"],
            gate_time_sec=bundle["gate_time"],
            schema_time_sec=bundle["schema_time"],
            final_train_time_sec=bundle["final_train_time"],
            method_total_time_sec=bundle["total_runtime"],
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": "edge+node+conflict-val"},
            ),
        )

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
        attribute_aware_selection = shared.base.quiet_call(
            shared.base.exp.select_attribute_aware_view,
            db=ctx["db"],
            intermediate_tables=ctx["intermediate_tables"],
            rich_attr_threshold=shared.base.exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
        )

    gate_attribute_result = None
    gate_attribute_incremental_time = 0.0
    gate_attribute_source = None
    if any(method in methods for method in ("Gate-attribute-default", "Consensus-vote", "Consensus-Val")):
        attribute_schema = attribute_aware_selection["schema"]
        if attribute_schema == ctx["all_edge_schema"]:
            if gate_result is not None:
                gate_attribute_result = copy.deepcopy(gate_result["edge_result"])
                gate_attribute_source = "reuse-edge"
            else:
                gate_attribute_result = shared.base.quiet_call(
                    shared.base.exp.run_schema_gate_with_background,
                    db=ctx["db"],
                    initial_graph=ctx["initial_graph"],
                    target_table=ctx["task_table"],
                    intermediate_tables=ctx["intermediate_tables"],
                    num_classes=ctx["num_classes"],
                    background_mode=shared.base.EDGE_MODE,
                    device=device,
                )
                gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
                gate_attribute_source = "run-edge"
        elif attribute_schema == ctx["all_node_schema"]:
            if gate_result is not None:
                gate_attribute_result = copy.deepcopy(gate_result["node_result"])
                gate_attribute_source = "reuse-node"
            else:
                gate_attribute_result = shared.base.quiet_call(
                    shared.base.exp.run_schema_gate_with_background,
                    db=ctx["db"],
                    initial_graph=ctx["initial_graph"],
                    target_table=ctx["task_table"],
                    intermediate_tables=ctx["intermediate_tables"],
                    num_classes=ctx["num_classes"],
                    background_mode=shared.base.NODE_MODE,
                    device=device,
                )
                gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
                gate_attribute_source = "run-node"
        else:
            gate_attribute_result = shared.base.quiet_call(
                shared.base.exp.run_schema_gate_with_schema_background,
                db=ctx["db"],
                initial_graph=ctx["initial_graph"],
                target_table=ctx["task_table"],
                intermediate_tables=ctx["intermediate_tables"],
                num_classes=ctx["num_classes"],
                background_schema=attribute_schema,
                background_name="attribute-default",
                device=device,
            )
            gate_attribute_incremental_time = gate_attribute_result["runtime_sec"]
            gate_attribute_source = "run-attribute"

    gate_schemas = []
    if gate_result is not None:
        gate_schemas.extend(
            [
                gate_result["edge_result"]["schema"],
                gate_result["node_result"]["schema"],
            ]
        )
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
        consensus_val_schema, conflict_candidates = _resolve_conflicts_with_dataset_metrics(
            ctx=ctx,
            device=device,
            agreed=agreed,
            conflicts=conflicts,
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
        )
        consensus_conflict_count = len(conflicts)

    need_gate_final_rows = any(
        method in methods
        for method in (
            "Gate-edge-default",
            "Gate-node-default",
            "Gate-attribute-default",
            "Consensus-vote",
            "Consensus-Val",
        )
    )
    if need_gate_final_rows:
        if "Gate-edge-default" in methods:
            edge_result, edge_final_train_time = _run_final_training_with_dataset_metrics(
                "Gate-edge-default",
                gate_result["edge_result"]["schema"],
                ctx,
                device,
                seed,
                gnn_selection_metric,
                schema_selection_metric,
            )
            row_by_method["Gate-edge-default"] = shared.base.make_method_row(
                run_id,
                seed,
                ctx["split_seed"],
                "Gate-edge-default",
                gate_result["edge_result"]["schema"],
                edge_result["metrics"],
                selection_time_sec=0.0,
                gate_time_sec=gate_result["edge_result"]["runtime_sec"],
                schema_time_sec=gate_result["edge_result"]["runtime_sec"],
                final_train_time_sec=edge_final_train_time,
                method_total_time_sec=gate_result["edge_result"]["runtime_sec"] + edge_final_train_time,
                extra=_metric_extra(
                    gnn_selection_metric,
                    schema_selection_metric,
                    {"selected_view": "edge-default"},
                ),
            )

        if "Gate-node-default" in methods:
            node_result, node_final_train_time = _run_final_training_with_dataset_metrics(
                "Gate-node-default",
                gate_result["node_result"]["schema"],
                ctx,
                device,
                seed,
                gnn_selection_metric,
                schema_selection_metric,
            )
            row_by_method["Gate-node-default"] = shared.base.make_method_row(
                run_id,
                seed,
                ctx["split_seed"],
                "Gate-node-default",
                gate_result["node_result"]["schema"],
                node_result["metrics"],
                selection_time_sec=0.0,
                gate_time_sec=gate_result["node_result"]["runtime_sec"],
                schema_time_sec=gate_result["node_result"]["runtime_sec"],
                final_train_time_sec=node_final_train_time,
                method_total_time_sec=gate_result["node_result"]["runtime_sec"] + node_final_train_time,
                extra=_metric_extra(
                    gnn_selection_metric,
                    schema_selection_metric,
                    {"selected_view": "node-default"},
                ),
            )

        if "Gate-attribute-default" in methods:
            gate_attr_result, gate_attr_final_train_time = _run_final_training_with_dataset_metrics(
                "Gate-attribute-default",
                gate_attribute_result["schema"],
                ctx,
                device,
                seed,
                gnn_selection_metric,
                schema_selection_metric,
            )
            row_by_method["Gate-attribute-default"] = shared.base.make_method_row(
                run_id,
                seed,
                ctx["split_seed"],
                "Gate-attribute-default",
                gate_attribute_result["schema"],
                gate_attr_result["metrics"],
                selection_time_sec=0.0,
                gate_time_sec=gate_attribute_result["runtime_sec"],
                schema_time_sec=gate_attribute_result["runtime_sec"],
                final_train_time_sec=gate_attr_final_train_time,
                method_total_time_sec=gate_attribute_result["runtime_sec"] + gate_attr_final_train_time,
                extra=_metric_extra(
                    gnn_selection_metric,
                    schema_selection_metric,
                    {
                        "selected_view": "attribute-default",
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
            consensus_vote_result, consensus_vote_final_train_time = _run_final_training_with_dataset_metrics(
                "Consensus-vote",
                consensus_vote_schema,
                ctx,
                device,
                seed,
                gnn_selection_metric,
                schema_selection_metric,
            )
            row_by_method["Consensus-vote"] = shared.base.make_method_row(
                run_id,
                seed,
                ctx["split_seed"],
                "Consensus-vote",
                consensus_vote_schema,
                consensus_vote_result["metrics"],
                selection_time_sec=0.0,
                gate_time_sec=vote_gate_time,
                schema_time_sec=vote_gate_time,
                final_train_time_sec=consensus_vote_final_train_time,
                method_total_time_sec=vote_gate_time + consensus_vote_final_train_time,
                extra=_metric_extra(
                    gnn_selection_metric,
                    schema_selection_metric,
                    {
                        "selected_view": "3gate-vote",
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
            consensus_val_result, consensus_val_final_train_time = _run_final_training_with_dataset_metrics(
                "Consensus-Val",
                consensus_val_schema,
                ctx,
                device,
                seed,
                gnn_selection_metric,
                schema_selection_metric,
            )
            row_by_method["Consensus-Val"] = shared.base.make_method_row(
                run_id,
                seed,
                ctx["split_seed"],
                "Consensus-Val",
                consensus_val_schema,
                consensus_val_result["metrics"],
                selection_time_sec=conflict_time,
                gate_time_sec=gate_time,
                schema_time_sec=schema_time,
                final_train_time_sec=consensus_val_final_train_time,
                method_total_time_sec=schema_time + consensus_val_final_train_time,
                extra=_metric_extra(
                    gnn_selection_metric,
                    schema_selection_metric,
                    {
                        "selected_view": "3gate-conflict-val",
                        "consensus_conflict_count": consensus_conflict_count,
                        "consensus_conflict_time_sec": conflict_time,
                        "attribute_gate_source": gate_attribute_source,
                        "attribute_gate_incremental_time_sec": gate_attribute_incremental_time,
                    },
                ),
            )

    if "Attribute-aware" in methods:
        attribute_aware_result, attribute_aware_final_train_time = _run_final_training_with_dataset_metrics(
            "Attribute-aware",
            attribute_aware_selection["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Attribute-aware"] = shared.base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Attribute-aware",
            attribute_aware_selection["schema"],
            attribute_aware_result["metrics"],
            selection_time_sec=attribute_aware_selection["selection_runtime_sec"],
            gate_time_sec=attribute_aware_selection["gate_runtime_sec"],
            schema_time_sec=attribute_aware_selection["runtime_sec"],
            final_train_time_sec=attribute_aware_final_train_time,
            method_total_time_sec=attribute_aware_selection["runtime_sec"] + attribute_aware_final_train_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {"selected_view": attribute_aware_selection["selected_view"]},
            ),
        )

    if "Random" in methods:
        random_selection = shared.select_random_schema(ctx)
        random_result, random_final_train_time = _run_final_training_with_dataset_metrics(
            "Random",
            random_selection["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Random"] = _make_random_row(
            run_id,
            seed,
            ctx["split_seed"],
            random_selection,
            {
                **random_result["metrics"],
                "final_train_time_sec": random_final_train_time,
                "gnn_selection_metric": gnn_selection_metric,
                "schema_selection_metric": schema_selection_metric,
            },
        )

    if "Exhaustive-best" in methods:
        label_schemas = _build_label_schemas(
            ctx,
            gate_result,
            attribute_aware_selection,
            gate_attribute_result=gate_attribute_result,
            consensus_vote_schema=consensus_vote_schema,
            consensus_val_schema=consensus_val_schema,
        )
        all_schema_scores = _score_all_possible_schemas_with_dataset_metrics(
            ctx=ctx,
            device=device,
            label_schemas=label_schemas,
            gnn_selection_metric=gnn_selection_metric,
            schema_selection_metric=schema_selection_metric,
        )
        exhaustive_schema_time = sum(item["runtime_sec"] for item in all_schema_scores)
        best = max(
            all_schema_scores,
            key=lambda item: (item["val_score"], -item["schema_id"]),
        )
        for item in all_schema_scores:
            if item["schema_id"] != best["schema_id"]:
                continue
            item["labels"] = (
                "Exhaustive-best"
                if item["labels"] == "-"
                else f'{item["labels"]},Exhaustive-best'
            )
            break
        result, final_train_time = _run_final_training_with_dataset_metrics(
            "Exhaustive-best",
            best["schema"],
            ctx,
            device,
            seed,
            gnn_selection_metric,
            schema_selection_metric,
        )
        row_by_method["Exhaustive-best"] = shared.base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Exhaustive-best",
            best["schema"],
            result["metrics"],
            selection_time_sec=exhaustive_schema_time,
            gate_time_sec=0.0,
            schema_time_sec=exhaustive_schema_time,
            final_train_time_sec=final_train_time,
            method_total_time_sec=exhaustive_schema_time + final_train_time,
            extra=_metric_extra(
                gnn_selection_metric,
                schema_selection_metric,
                {
                    "selected_view": f"best-val-schema-{best['schema_id']}",
                    "exhaustive_val_score": best["val_score"],
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
                    "schema": shared.base.format_schema(item["schema"]),
                    "val_score": item["val_score"],
                    "test_score": item["test_score"],
                    "runtime_sec": item["runtime_sec"],
                    "accuracy": item["accuracy"],
                    "f1": item["f1"],
                    "roc_auc": item["roc_auc"],
                    "labels": item["labels"],
                }
            )

    for method in methods:
        print(
            f"Run {run_id} | split_seed={split_seed} | seed={seed} | dataset={dataset_name} | method={method} | shared"
        )
        row = row_by_method.get(method)
        if row is None:
            raise ValueError(f"Method '{method}' was not prepared in shared runner.")
        method_rows.append(copy.deepcopy(row))

    method_rows.extend(_make_ceiling_rows(row_by_method))

    return method_rows, exhaustive_rows


def run_selected_methods(methods, seeds, output_path, dataset_name=DATASET):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_config = _selection_metrics_for_dataset(dataset_name)
    primary_metric = metric_config["schema"]
    all_method_rows = []
    all_exhaustive_rows = []
    split_seeds = shared.get_split_seeds(dataset_name)
    total_runs = len(split_seeds) * len(seeds)
    run_counter = 0
    if dataset_name == "University":
        progress_label = "University-MultiSplit"
    elif dataset_name == "IMDB_small":
        progress_label = "IMDB-MultiSplit"
    elif dataset_name == "MovieLens":
        progress_label = "MovieLens-MultiSplit"
    elif dataset_name == "UW-CSE":
        progress_label = "UW-CSE-MultiSplit"
    elif dataset_name == "Hepatitis":
        progress_label = "Hepatitis-MultiSplit"
    elif dataset_name == "Carcinogenesis":
        progress_label = "Carcinogenesis-MultiSplit"
    else:
        progress_label = "SingleSplit"

    for split_seed in split_seeds:
        for seed in seeds:
            run_counter += 1
            print(
                f"{progress_label} progress {run_counter}/{total_runs} | "
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

    method_raw_df, method_summary_df = shared.base.summarize_method_rows(all_method_rows)
    exhaustive_raw_df, exhaustive_summary_df = shared.base.summarize_exhaustive_rows(
        all_exhaustive_rows
    )
    if (
        not method_raw_df.empty
        and "gnn_selection_metric" in method_raw_df.columns
        and "schema_selection_metric" in method_raw_df.columns
    ):
        metric_summary = (
            method_raw_df.groupby("method", as_index=False)
            .agg(
                gnn_selection_metric=("gnn_selection_metric", "first"),
                schema_selection_metric=("schema_selection_metric", "first"),
            )
        )
        method_summary_df = method_summary_df.merge(metric_summary, on="method", how="left")
    method_summary_df = _enrich_method_summary(method_summary_df, primary_metric)

    baseline_row = method_summary_df[method_summary_df["method"] == "REG"]
    improvement_cols = [f"{metric}_vs_reg_pct" for metric in METRICS]
    if not baseline_row.empty:
        for metric in METRICS:
            improvement_col = f"{metric}_vs_reg_pct"
            baseline_value = baseline_row.iloc[0][f"{metric}_mean"]
            if pd.notna(baseline_value) and baseline_value != 0:
                method_summary_df[improvement_col] = (
                    (method_summary_df[f"{metric}_mean"] - baseline_value)
                    / baseline_value
                    * 100.0
                )
            else:
                method_summary_df[improvement_col] = pd.NA
    else:
        for improvement_col in improvement_cols:
            method_summary_df[improvement_col] = pd.NA

    exhaustive_row = method_summary_df[method_summary_df["method"] == "Exhaustive-best"]
    for metric in METRICS:
        gap_col = f"{metric}_gap_to_exhaustive_pct"
        if not exhaustive_row.empty:
            exhaustive_value = exhaustive_row.iloc[0][f"{metric}_mean"]
            if pd.notna(exhaustive_value) and exhaustive_value != 0:
                method_summary_df[gap_col] = (
                    (exhaustive_value - method_summary_df[f"{metric}_mean"])
                    / exhaustive_value
                    * 100.0
                )
            else:
                method_summary_df[gap_col] = pd.NA
        else:
            method_summary_df[gap_col] = pd.NA

    bounds_rows = _build_schema_bounds_rows(all_exhaustive_rows)

    if "consensus_conflict_count" in method_raw_df.columns:
        conflict_summary = (
            method_raw_df.groupby("method", as_index=False)
            .agg(
                consensus_conflict_count_mean=("consensus_conflict_count", "mean"),
                consensus_conflict_count_std=("consensus_conflict_count", "std"),
                consensus_conflict_count_var=("consensus_conflict_count", "var"),
            )
        )
        method_summary_df = method_summary_df.merge(conflict_summary, on="method", how="left")
    else:
        method_summary_df["consensus_conflict_count_mean"] = pd.NA
        method_summary_df["consensus_conflict_count_std"] = pd.NA
        method_summary_df["consensus_conflict_count_var"] = pd.NA

    for row in bounds_rows:
        for improvement_col in improvement_cols:
            row[improvement_col] = pd.NA
        for metric in METRICS:
            row[f"{metric}_gap_to_exhaustive_pct"] = pd.NA

    method_summary_df = _reorder_method_summary_columns(method_summary_df)
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
        choices=sorted(shared.DATASET_CONFIGS),
        help=f"Dataset to run. Defaults to {default_dataset}.",
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
        "--output",
        default=None,
        help="Output .xlsx path.",
    )
    return parser


def main(default_dataset=DATASET):
    parser = build_parser(default_dataset=default_dataset)
    args = parser.parse_args()
    seeds = shared._parse_seeds(args)
    output_path = args.output or _default_output_path(args.dataset)
    run_selected_methods(ALL_METHODS, seeds, output_path, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
