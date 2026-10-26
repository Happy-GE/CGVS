import itertools
import os
import sys
import time

import pandas as pd
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from fignn.data_loader import load_relational_data
from fignn.gnn_scoring_schema import train_and_score_schema_gnn
from fignn.graph_building import build_fk_graph
from fignn.schema_gate_joint import (
    build_schema_gate_channels,
    rebuild_and_retrain_discrete_schema,
    train_schema_gate_joint,
)
from fignn.schema_graph_building import (
    EDGE_MODE,
    NODE_MODE,
    build_schema_aware_graph,
    copy_task_tensors,
    identify_intermediate_tables,
)
from fignn.training import evaluate_model
from fignn.utils import assign_node_labels, split_node_labels


HIDDEN_DIM = 64
JOINT_EPOCHS = 30
FINAL_EPOCHS = 30
VALIDATION_EPOCHS = 30
LR = 0.01
WEIGHT_DECAY = 0.0
LAMBDA_DISC = 1e-3
SEED = 36
INCLUDE_FK_FEATURES = True
ADAPTIVE_RICH_ATTR_THRESHOLD = 1


def format_schema(schema):
    if not schema:
        return "-"
    return ", ".join(f"{table}:{mode}" for table, mode in sorted(schema.items()))


def print_final_metrics(metrics):
    print("\nFinal schema_gate_joint_consensus test metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    roc_auc = metrics.get("roc_auc")
    print(f"ROC AUC: {roc_auc if roc_auc is not None else 'N/A'}")


def print_method_comparison(rows):
    print("\n=== Method comparison ===")
    print(f"{'Method':<24} {'Accuracy':>10} {'F1':>10} {'ROC AUC':>10}  Schema")
    print("-" * 110)
    for row in rows:
        roc_auc = row["metrics"].get("roc_auc")
        roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        print(
            f"{row['method']:<24} "
            f"{row['metrics']['accuracy']:>10.4f} "
            f"{row['metrics']['f1']:>10.4f} "
            f"{roc_auc_text:>10}  "
            f"{format_schema(row['schema'])}"
        )


def schema_selection_metric(target_table):
    if target_table in {"canc", "users"}:
        return "roc_auc"
    return "accuracy"


def score_from_metrics(metrics, metric_name):
    if metric_name == "roc_auc":
        score = metrics.get("roc_auc")
        if score is None:
            raise ValueError("Validation ROC-AUC is unavailable for this schema.")
        return score
    return metrics["accuracy"]


def print_all_schema_score_comparison(schema_scores):
    print("\n=== All schema score comparison ===")
    print(
        f"{'Rank':>4} {'Schema ID':>9} {'Val':>8} {'Test':>8} "
        f"{'Time(s)':>8} {'Labels':<48} Schema"
    )
    print("-" * 148)
    ranked = sorted(
        schema_scores,
        key=lambda item: (item["val_score"], item["test_score"], -item["schema_id"]),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:>4} "
            f"{item['schema_id']:>9} "
            f"{item['val_score']:>8.4f} "
            f"{item['test_score']:>8.4f} "
            f"{item['runtime_sec']:>8.2f} "
            f"{item['labels']:<48} "
            f"{format_schema(item['schema'])}"
        )


def print_schema_time_comparison(rows, schema_scores):
    print("\n=== Schema acquisition time comparison ===")
    print(
        f"{'Method':<24} {'Selection(s)':>14} {'Selection(ms)':>14} "
        f"{'Gate(s)':>12} {'Total(s)':>12}  Schema"
    )
    print("-" * 158)
    for row in rows:
        selection_time = row.get("selection_time_sec", 0.0)
        gate_time = row.get("gate_time_sec", row["schema_time_sec"])
        print(
            f"{row['method']:<24} "
            f"{selection_time:>14.6f} "
            f"{selection_time * 1000.0:>14.3f} "
            f"{gate_time:>12.6f} "
            f"{row['schema_time_sec']:>12.6f}  "
            f"{format_schema(row['schema'])}"
        )

    exhaustive_time = sum(item["runtime_sec"] for item in schema_scores)
    print(
        f"{'Exhaustive-all-schemas':<24} "
        f"{0.0:>14.6f} "
        f"{0.0:>14.3f} "
        f"{0.0:>12.6f} "
        f"{exhaustive_time:>12.6f}  "
        f"{len(schema_scores)} complete schemas scored"
    )

    print("\n=== Exhaustive schema scoring time detail ===")
    print(f"{'Schema ID':>9} {'Time(s)':>8} {'Val':>8} {'Test':>8} {'Labels':<48} Schema")
    print("-" * 140)
    for item in sorted(schema_scores, key=lambda row: row["schema_id"]):
        print(
            f"{item['schema_id']:>9} "
            f"{item['runtime_sec']:>8.2f} "
            f"{item['val_score']:>8.4f} "
            f"{item['test_score']:>8.4f} "
            f"{item['labels']:<48} "
            f"{format_schema(item['schema'])}"
        )


def run_schema_gate_with_background(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    background_mode,
    device,
):
    background_name = "node-default" if background_mode == NODE_MODE else "edge-default"
    background_schema = {table: background_mode for table in intermediate_tables}
    return run_schema_gate_with_schema_background(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        background_schema=background_schema,
        background_name=background_name,
        device=device,
    )


def run_schema_gate_with_schema_background(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    background_schema,
    background_name,
    device,
):
    print(f"\n=== Gate learning: {background_name} ===")
    start = time.perf_counter()

    channel_graphs, intermediate_tables = build_schema_gate_channels(
        db=db,
        reference_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        background_schema=background_schema,
        include_fk_features=INCLUDE_FK_FEATURES,
    )

    _, _, learned_probs, discrete_schema, history = train_schema_gate_joint(
        channel_graphs=channel_graphs,
        intermediate_tables=intermediate_tables,
        target_table=target_table,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        epochs=JOINT_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        lambda_disc=LAMBDA_DISC,
        device=device,
        seed=SEED,
        log_every=10,
        verbose=True,
    )

    runtime_sec = time.perf_counter() - start
    print(f"{background_name} background schema: {format_schema(background_schema)}")
    print(f"\n{background_name} learned probabilities:")
    for table, prob in learned_probs.items():
        print(f"  {table}: {prob:.4f}")
    print(f"{background_name} schema: {format_schema(discrete_schema)}")
    print(f"### Gate runtime: {background_name} | {runtime_sec:.2f}s ###")

    return {
        "background": background_name,
        "probs": learned_probs,
        "schema": discrete_schema,
        "history": history,
        "runtime_sec": runtime_sec,
    }


def key_columns_for_table(db, table):
    keys = {db.primary_keys[table]}
    for src_table, src_col, _, _ in db.foreign_keys:
        if src_table == table:
            keys.add(src_col)
    return keys


def analyze_intermediate_attribute_richness(
    db,
    intermediate_tables,
    rich_attr_threshold=ADAPTIVE_RICH_ATTR_THRESHOLD,
):
    rows = []
    for table in intermediate_tables:
        df = db.get_table(table)
        key_cols = key_columns_for_table(db, table)
        non_key_cols = [
            col
            for col in df.columns
            if col not in key_cols
        ]
        non_key_attr_count = len(non_key_cols)
        is_rich = non_key_attr_count >= rich_attr_threshold
        rows.append(
            {
                "table": table,
                "key_cols": sorted(key_cols),
                "non_key_cols": non_key_cols,
                "non_key_attr_count": non_key_attr_count,
                "is_rich": is_rich,
            }
        )
    return rows


def select_attribute_aware_view(
    db,
    intermediate_tables,
    edge_result=None,
    node_result=None,
    rich_attr_threshold=ADAPTIVE_RICH_ATTR_THRESHOLD,
):
    start = time.perf_counter()
    analysis = analyze_intermediate_attribute_richness(
        db,
        intermediate_tables,
        rich_attr_threshold=rich_attr_threshold,
    )
    rich_count = sum(1 for row in analysis if row["is_rich"])
    pure_count = len(analysis) - rich_count
    total = len(analysis)
    selected_schema = {}
    for row in analysis:
        selected_schema[row["table"]] = EDGE_MODE if not row["is_rich"] else NODE_MODE

    selection_runtime_sec = time.perf_counter() - start
    gate_runtime_sec = 0.0
    runtime_sec = selection_runtime_sec

    print("\n=== Attribute-aware view selection ===")
    print(f"Rich attribute threshold: non-key attribute count >= {rich_attr_threshold}")
    for row in analysis:
        kind = "node" if row["is_rich"] else "edge"
        print(
            f"  {row['table']}: "
            f"non_key_attr_count={row['non_key_attr_count']} "
            f"kind={kind} "
            f"non_key_cols={row['non_key_cols']}"
        )
    print(
        f"Summary: pure_key_tables={pure_count}/{total}, "
        f"rich_attr_tables={rich_count}/{total}"
    )
    print("Selected attribute-aware view: per-table mode assignment")
    print(f"Attribute-aware schema: {format_schema(selected_schema)}")
    print(
        f"Attribute-aware selection runtime: {selection_runtime_sec:.6f}s | "
        f"selected gate runtime: {gate_runtime_sec:.4f}s | "
        f"total schema runtime: {runtime_sec:.4f}s"
    )

    return {
        "analysis": analysis,
        "selected_view": "per-table-attribute-aware",
        "schema": selected_schema,
        "selection_runtime_sec": selection_runtime_sec,
        "gate_runtime_sec": gate_runtime_sec,
        "runtime_sec": runtime_sec,
    }


def select_adaptive_view(
    db,
    intermediate_tables,
    edge_result=None,
    node_result=None,
    rich_attr_threshold=ADAPTIVE_RICH_ATTR_THRESHOLD,
):
    return select_attribute_aware_view(
        db=db,
        intermediate_tables=intermediate_tables,
        edge_result=edge_result,
        node_result=node_result,
        rich_attr_threshold=rich_attr_threshold,
    )


def split_consensus_and_conflicts(edge_schema, node_schema, intermediate_tables):
    agreed = {}
    conflicts = []
    for table in intermediate_tables:
        edge_mode = edge_schema[table]
        node_mode = node_schema[table]
        if edge_mode == node_mode:
            agreed[table] = edge_mode
        else:
            conflicts.append(table)
    return agreed, conflicts


def score_complete_schema(
    db,
    initial_graph,
    target_table,
    num_classes,
    schema,
    device,
    selection_metric="accuracy",
):
    graph, _ = build_schema_aware_graph(
        db,
        table_modes=schema,
        target_table=target_table,
        include_fk_features=INCLUDE_FK_FEATURES,
    )
    graph = copy_task_tensors(initial_graph, graph, target_table)
    score, _ = train_and_score_schema_gnn(
        graph,
        target_table=target_table,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        epochs=VALIDATION_EPOCHS,
        lr=LR,
        device=device,
        seed=SEED,
        selection_metric=selection_metric,
        verbose=False,
    )
    return score


def score_complete_schema_with_test_score(
    db,
    initial_graph,
    target_table,
    num_classes,
    schema,
    device,
    selection_metric="accuracy",
):
    graph, _ = build_schema_aware_graph(
        db,
        table_modes=schema,
        target_table=target_table,
        include_fk_features=INCLUDE_FK_FEATURES,
    )
    graph = copy_task_tensors(initial_graph, graph, target_table)
    val_score, model = train_and_score_schema_gnn(
        graph,
        target_table=target_table,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        epochs=VALIDATION_EPOCHS,
        lr=LR,
        device=device,
        seed=SEED,
        selection_metric=selection_metric,
        verbose=False,
    )
    test_metrics = evaluate_model(
        model,
        graph,
        target_table,
        split="test",
        return_metrics=True,
    )
    return val_score, score_from_metrics(test_metrics, selection_metric)


def resolve_conflicts_by_validation(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    agreed,
    conflicts,
    device,
    selection_metric="accuracy",
):
    if not conflicts:
        print("\nNo conflicts between edge-default and node-default schemas.")
        return dict(agreed), []

    print(f"\n=== Resolve conflicts by validation {selection_metric} ===")
    print("Agreed modes:", format_schema(agreed))
    print("Conflict tables:", conflicts)

    candidates = []
    total = 2 ** len(conflicts)
    for schema_id, modes in enumerate(
        itertools.product((NODE_MODE, EDGE_MODE), repeat=len(conflicts)),
        start=1,
    ):
        schema = dict(agreed)
        schema.update(dict(zip(conflicts, modes)))
        print(f"\n[Conflict-Val] Schema {schema_id}/{total}: {format_schema(schema)}")
        start = time.perf_counter()
        val_score = score_complete_schema(
            db=db,
            initial_graph=initial_graph,
            target_table=target_table,
            num_classes=num_classes,
            schema=schema,
            device=device,
            selection_metric=selection_metric,
        )
        runtime_sec = time.perf_counter() - start
        print(f"  val_{selection_metric}={val_score:.4f} runtime={runtime_sec:.2f}s")
        candidates.append(
            {
                "schema_id": schema_id,
                "schema": schema,
                f"val_{selection_metric}": val_score,
                "runtime_sec": runtime_sec,
            }
        )

    best = max(candidates, key=lambda item: (item[f"val_{selection_metric}"], -item["schema_id"]))
    print("\nSelected conflict-resolved schema:")
    print(f"  schema_id={best['schema_id']}")
    print(f"  val_{selection_metric}={best[f'val_{selection_metric}']:.4f}")
    print(f"  modes={format_schema(best['schema'])}")
    return best["schema"], candidates


def schema_key(schema):
    return tuple(sorted(schema.items()))


def labels_for_schema(schema, label_schemas):
    labels = [
        label
        for label, label_schema in label_schemas.items()
        if schema_key(schema) == schema_key(label_schema)
    ]
    return ",".join(labels) if labels else "-"


def score_all_possible_schemas(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    device,
    label_schemas=None,
    selection_metric="accuracy",
):
    print("\n=== All possible schema scores ===")
    candidates = []
    label_schemas = label_schemas or {}
    total = 2 ** len(intermediate_tables)

    for schema_id, modes in enumerate(
        itertools.product((NODE_MODE, EDGE_MODE), repeat=len(intermediate_tables)),
        start=1,
    ):
        schema = dict(zip(intermediate_tables, modes))
        print(f"\n[All-Schema-Val] Schema {schema_id}/{total}: {format_schema(schema)}")
        start = time.perf_counter()
        val_score, test_score = score_complete_schema_with_test_score(
            db=db,
            initial_graph=initial_graph,
            target_table=target_table,
            num_classes=num_classes,
            schema=schema,
            device=device,
            selection_metric=selection_metric,
        )
        runtime_sec = time.perf_counter() - start
        labels = labels_for_schema(schema, label_schemas)
        print(
            f"  val_{selection_metric}={val_score:.4f} "
            f"test_{selection_metric}={test_score:.4f} "
            f"runtime={runtime_sec:.2f}s "
            f"labels={labels}"
        )
        candidates.append(
            {
                "schema_id": schema_id,
                "schema": schema,
                "val_score": val_score,
                "test_score": test_score,
                "runtime_sec": runtime_sec,
                "labels": labels,
            }
        )

    print("\n=== All possible schema scores ranked ===")
    print(
        f"{'Rank':>4} {'Schema ID':>9} {'Val':>8} {'Test':>8} "
        f"{'Time(s)':>8} {'Labels':<48} Schema"
    )
    print("-" * 148)
    ranked = sorted(
        candidates,
        key=lambda item: (item["val_score"], item["test_score"], -item["schema_id"]),
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        print(
            f"{rank:>4} "
            f"{item['schema_id']:>9} "
            f"{item['val_score']:>8.4f} "
            f"{item['test_score']:>8.4f} "
            f"{item['runtime_sec']:>8.2f} "
            f"{item['labels']:<48} "
            f"{format_schema(item['schema'])}"
        )

    return candidates


def schema_gate_joint_consensus(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    device,
):
    selection_metric = schema_selection_metric(target_table)
    edge_result = run_schema_gate_with_background(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        background_mode=EDGE_MODE,
        device=device,
    )
    node_result = run_schema_gate_with_background(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        background_mode=NODE_MODE,
        device=device,
    )

    agreed, conflicts = split_consensus_and_conflicts(
        edge_result["schema"],
        node_result["schema"],
        intermediate_tables,
    )

    print("\n=== Gate consensus ===")
    print("Edge-default schema:", format_schema(edge_result["schema"]))
    print("Node-default schema:", format_schema(node_result["schema"]))
    print("Agreed modes:", format_schema(agreed))
    print("Conflicts:", conflicts if conflicts else "-")

    final_schema, conflict_candidates = resolve_conflicts_by_validation(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        agreed=agreed,
        conflicts=conflicts,
        device=device,
        selection_metric=selection_metric,
    )

    print("\n=== Final retraining with consensus schema ===")
    final_graph, final_node_id_map, final_model, metrics = rebuild_and_retrain_discrete_schema(
        db=db,
        reference_graph=initial_graph,
        target_table=target_table,
        discrete_schema=final_schema,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        epochs=FINAL_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
        seed=SEED,
        include_fk_features=INCLUDE_FK_FEATURES,
        verbose=True,
        selection_metric=selection_metric,
    )

    return {
        "edge_result": edge_result,
        "node_result": node_result,
        "agreed": agreed,
        "conflicts": conflicts,
        "conflict_candidates": conflict_candidates,
        "final_schema": final_schema,
        "final_graph": final_graph,
        "final_node_id_map": final_node_id_map,
        "final_model": final_model,
        "metrics": metrics,
    }


def run_fixed_schema_baseline(
    db,
    initial_graph,
    target_table,
    num_classes,
    device,
    schema,
    method_name,
    selection_metric=None,
):
    print(f"\n=== Final retraining with {method_name} schema ===")
    print(f"{method_name} schema:", format_schema(schema))
    if selection_metric is None:
        selection_metric = schema_selection_metric(target_table)
    final_graph, final_node_id_map, final_model, metrics = rebuild_and_retrain_discrete_schema(
        db=db,
        reference_graph=initial_graph,
        target_table=target_table,
        discrete_schema=schema,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        epochs=FINAL_EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        device=device,
        seed=SEED,
        include_fk_features=INCLUDE_FK_FEATURES,
        verbose=True,
        selection_metric=selection_metric,
    )
    return {
        "schema": schema,
        "final_graph": final_graph,
        "final_node_id_map": final_node_id_map,
        "final_model": final_model,
        "metrics": metrics,
    }


def run_fixed_mode_baseline(
    db,
    initial_graph,
    target_table,
    intermediate_tables,
    num_classes,
    device,
    mode,
    method_name,
):
    schema = {
        table: mode
        for table in intermediate_tables
    }
    return run_fixed_schema_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=target_table,
        num_classes=num_classes,
        device=device,
        schema=schema,
        method_name=method_name,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(project_root, "data", "Hepatitis")
    task_table = "dispat"
    task_pk = "m_id"
    label_col = "Type"
    num_classes = 2

    db = load_relational_data(data_path)
    initial_graph, initial_node_id_map = build_fk_graph(db)

    task = pd.read_csv(os.path.join(data_path, "task.csv"))
    labels = dict(zip(task[task_pk], task[label_col]))
    assign_node_labels(initial_graph, labels, initial_node_id_map, node_type=task_table)
    split_node_labels(initial_graph, node_type=task_table)

    intermediate_tables = identify_intermediate_tables(
        db,
        target_table=task_table,
    )
    print("Intermediate tables:", intermediate_tables)

    result = schema_gate_joint_consensus(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
    )

    adaptive_selection = select_attribute_aware_view(
        db=db,
        intermediate_tables=intermediate_tables,
        edge_result=result["edge_result"],
        node_result=result["node_result"],
        rich_attr_threshold=ADAPTIVE_RICH_ATTR_THRESHOLD,
    )

    adaptive_result = run_fixed_schema_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        num_classes=num_classes,
        device=device,
        schema=adaptive_selection["schema"],
        method_name="Attribute-aware",
    )

    edge_default_result = run_fixed_schema_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        num_classes=num_classes,
        device=device,
        schema=result["edge_result"]["schema"],
        method_name="Gate-edge-default",
    )

    node_default_result = run_fixed_schema_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        num_classes=num_classes,
        device=device,
        schema=result["node_result"]["schema"],
        method_name="Gate-node-default",
    )

    all_node_result = run_fixed_mode_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
        mode=NODE_MODE,
        method_name="All-node",
    )

    all_edge_result = run_fixed_mode_baseline(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
        mode=EDGE_MODE,
        method_name="All-edge",
    )

    all_schema_scores = score_all_possible_schemas(
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
        label_schemas={
            "Consensus": result["final_schema"],
            "Attribute-aware": adaptive_result["schema"],
            "Gate-edge-default": edge_default_result["schema"],
            "Gate-node-default": node_default_result["schema"],
            "All-node": all_node_result["schema"],
            "All-edge": all_edge_result["schema"],
        },
        selection_metric=schema_selection_metric(task_table),
    )

    print("\nFinal consensus schema:")
    print(format_schema(result["final_schema"]))
    print_final_metrics(result["metrics"])

    print_method_comparison(
        [
            {
                "method": "Consensus",
                "schema": result["final_schema"],
                "metrics": result["metrics"],
            },
            {
                "method": "Attribute-aware",
                "schema": adaptive_result["schema"],
                "metrics": adaptive_result["metrics"],
            },
            {
                "method": "Gate-edge-default",
                "schema": edge_default_result["schema"],
                "metrics": edge_default_result["metrics"],
            },
            {
                "method": "Gate-node-default",
                "schema": node_default_result["schema"],
                "metrics": node_default_result["metrics"],
            },
            {
                "method": "All-node",
                "schema": all_node_result["schema"],
                "metrics": all_node_result["metrics"],
            },
            {
                "method": "All-edge",
                "schema": all_edge_result["schema"],
                "metrics": all_edge_result["metrics"],
            },
        ]
    )
    print_all_schema_score_comparison(all_schema_scores)

    consensus_schema_time = (
        result["edge_result"]["runtime_sec"]
        + result["node_result"]["runtime_sec"]
        + sum(item["runtime_sec"] for item in result["conflict_candidates"])
    )
    consensus_gate_time = (
        result["edge_result"]["runtime_sec"]
        + result["node_result"]["runtime_sec"]
    )
    consensus_conflict_time = sum(
        item["runtime_sec"]
        for item in result["conflict_candidates"]
    )
    print_schema_time_comparison(
        [
            {
                "method": "Consensus",
                "schema": result["final_schema"],
                "selection_time_sec": consensus_conflict_time,
                "gate_time_sec": consensus_gate_time,
                "schema_time_sec": consensus_schema_time,
            },
            {
                "method": "Attribute-aware",
                "schema": adaptive_result["schema"],
                "selection_time_sec": adaptive_selection["selection_runtime_sec"],
                "gate_time_sec": adaptive_selection["gate_runtime_sec"],
                "schema_time_sec": adaptive_selection["runtime_sec"],
            },
            {
                "method": "Gate-edge-default",
                "schema": result["edge_result"]["schema"],
                "selection_time_sec": 0.0,
                "gate_time_sec": result["edge_result"]["runtime_sec"],
                "schema_time_sec": result["edge_result"]["runtime_sec"],
            },
            {
                "method": "Gate-node-default",
                "schema": result["node_result"]["schema"],
                "selection_time_sec": 0.0,
                "gate_time_sec": result["node_result"]["runtime_sec"],
                "schema_time_sec": result["node_result"]["runtime_sec"],
            },
            {
                "method": "All-node",
                "schema": all_node_result["schema"],
                "selection_time_sec": 0.0,
                "gate_time_sec": 0.0,
                "schema_time_sec": 0.0,
            },
            {
                "method": "All-edge",
                "schema": all_edge_result["schema"],
                "selection_time_sec": 0.0,
                "gate_time_sec": 0.0,
                "schema_time_sec": 0.0,
            },
        ],
        all_schema_scores,
    )
