import contextlib
import copy
import importlib.util
import os
import random
import sys
import time
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from fignn.data_loader import load_relational_data
from fignn.graph_building import build_fk_graph
from fignn.models import GNN
import fignn.schema_gate_joint as schema_gate_joint_module
import fignn.schema_graph_building as schema_graph_building_module
from fignn.schema_graph_building import EDGE_MODE, NODE_MODE, identify_intermediate_tables
from fignn.training import evaluate_model, train_model
from fignn.utils import add_reverse_edges, assign_node_labels, split_node_labels


NUM_RUNS = 5
RUN_SEEDS = [40, 41, 42, 43, 44]
SPLIT_SEED = 36
RESULTS_XLSX = os.path.join(project_root, "results", "6161.xlsx")
BASE_SCRIPT = os.path.join(current_dir, "schema_gate_consensus.py")
INCLUDE_FK_FEATURES = True

 
def load_consensus_module():
    spec = importlib.util.spec_from_file_location(
        "e3_schema_gate_joint_consensus",
        BASE_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp = load_consensus_module()
# Align node-mode features with REG/build_fk_graph: drop only the primary key
# and keep foreign-key columns as node features.
exp.INCLUDE_FK_FEATURES = INCLUDE_FK_FEATURES


def build_reg_aligned_schema_graph(
    db,
    table_modes=None,
    default_mode=NODE_MODE,
    target_table=None,
    include_fk_features=True,
):
    """Build schema graphs while matching REG's node and FK-edge handling."""
    table_modes = dict(table_modes or {})
    graph = HeteroData()
    node_id_maps = {}

    intermediate_tables = set(
        identify_intermediate_tables(db, target_table=target_table)
    )
    edge_mode_tables = {
        table
        for table in intermediate_tables
        if table_modes.get(table, default_mode) == EDGE_MODE
    }
    edge_mode_tables.discard(target_table)

    for table in db.tables:
        if table in edge_mode_tables:
            continue
        schema_graph_building_module._add_table_nodes(
            graph,
            db,
            table,
            node_id_maps,
            include_fk_features=include_fk_features,
        )

    # REG/build_fk_graph iterates db.foreign_keys directly. Keeping the same
    # order makes all-node metadata and GNN initialization match REG.
    for src_table, src_col, dst_table, dst_col in db.foreign_keys:
        if src_table in edge_mode_tables:
            continue
        if src_table not in node_id_maps or dst_table not in node_id_maps:
            continue

        src_df = db.get_table(src_table)
        src_map = node_id_maps[src_table]
        dst_map = node_id_maps[dst_table]
        src_pk = db.primary_keys[src_table]
        src_nodes = []
        dst_nodes = []

        for _, row in src_df.iterrows():
            if pd.isna(row[src_col]) or pd.isna(row[src_pk]):
                continue
            src_id = row[src_pk]
            dst_id = row[src_col]
            if src_id in src_map and dst_id in dst_map:
                src_nodes.append(src_map[src_id])
                dst_nodes.append(dst_map[dst_id])

        edge_type = (src_table, f"{src_col}_to_{dst_col}", dst_table)
        graph[edge_type].edge_index = torch.tensor(
            [src_nodes, dst_nodes],
            dtype=torch.long,
        )

    for table in db.tables:
        if table in edge_mode_tables:
            schema_graph_building_module._add_edgeized_table_edges(
                graph,
                db,
                table,
                node_id_maps,
            )

    graph = add_reverse_edges(graph)
    for node_type in graph.node_types:
        if "x" not in graph[node_type] or graph[node_type].x is None:
            graph[node_type].x = torch.ones((graph[node_type].num_nodes, 1))

    graph.schema_table_modes = copy.deepcopy(table_modes)
    graph.edge_mode_tables = sorted(edge_mode_tables)
    return graph, node_id_maps


# Limit REG alignment to this 621 experiment. Functions imported by the base
# consensus module and schema-gate module will both use this local builder.
exp.build_schema_aware_graph = build_reg_aligned_schema_graph
schema_gate_joint_module.build_schema_aware_graph = build_reg_aligned_schema_graph


def set_strict_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def format_schema(schema):
    if not schema:
        return "-"
    return ", ".join(f"{table}:{mode}" for table, mode in sorted(schema.items()))


def quiet_call(fn, *args, **kwargs):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            return fn(*args, **kwargs)


def run_reg_training(initial_graph, task_table, num_classes, device, seed):
    """Train REG on the original build_fk_graph graph with 621's final settings."""
    set_strict_seed(seed)
    start = time.perf_counter()
    selection_metric = exp.schema_selection_metric(task_table)
    graph = copy.deepcopy(initial_graph).to(device)
    model = GNN(
        metadata=graph.metadata(),
        hidden_channels=exp.HIDDEN_DIM,
        out_channels=num_classes,
        target_node=task_table,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp.LR,
        weight_decay=exp.WEIGHT_DECAY,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    quiet_call(
        train_model,
        model,
        graph,
        task_table,
        optimizer,
        loss_fn,
        epochs=exp.FINAL_EPOCHS,
        selection_metric=selection_metric,
        verbose=True,
    )
    metrics = evaluate_model(
        model,
        graph,
        task_table,
        split="test",
        return_metrics=True,
    )
    return {
        "final_graph": graph,
        "final_model": model,
        "metrics": metrics,
    }, time.perf_counter() - start


def run_final_training(
    method_name,
    schema,
    db,
    initial_graph,
    task_table,
    num_classes,
    device,
    seed,
    show_output=False,
):
    exp.SEED = seed
    set_strict_seed(seed)
    start = time.perf_counter()
    if show_output:
        result = exp.run_fixed_schema_baseline(
            db=db,
            initial_graph=initial_graph,
            target_table=task_table,
            num_classes=num_classes,
            device=device,
            schema=schema,
            method_name=method_name,
        )
    else:
        result = quiet_call(
            exp.run_fixed_schema_baseline,
            db=db,
            initial_graph=initial_graph,
            target_table=task_table,
            num_classes=num_classes,
            device=device,
            schema=schema,
            method_name=method_name,
        )
    runtime_sec = time.perf_counter() - start
    return result, runtime_sec


def run_fixed_mode(
    method_name,
    mode,
    db,
    initial_graph,
    task_table,
    intermediate_tables,
    num_classes,
    device,
    seed,
    show_output=False,
):
    schema = {
        table: mode
        for table in intermediate_tables
    }
    return run_final_training(
        method_name,
        schema,
        db,
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
        show_output=show_output,
    )


def run_once(run_id, seed, device, dataset_config=None):
    exp.SEED = seed
    set_strict_seed(seed)
    print(f"Run {run_id}/{NUM_RUNS} | seed={seed}")

    dataset_config = dataset_config or {
        "data_dir": "University",
        "task_table": "student",
        "task_pk": "student_id",
        "label_col": "intelligence",
    }
    data_path = os.path.join(project_root, "data", dataset_config["data_dir"])
    task_table = dataset_config["task_table"]
    task_pk = dataset_config["task_pk"]
    label_col = dataset_config["label_col"]

    db = load_relational_data(data_path)
    initial_graph, initial_node_id_map = build_fk_graph(db)

    task = pd.read_csv(os.path.join(data_path, "task.csv"))
    task[label_col] = pd.Categorical(task[label_col]).codes
    num_classes = int(task[label_col].nunique())
    labels = dict(zip(task[task_pk], task[label_col]))
    assign_node_labels(initial_graph, labels, initial_node_id_map, node_type=task_table)
    split_node_labels(initial_graph, node_type=task_table, seed=SPLIT_SEED)

    intermediate_tables = identify_intermediate_tables(
        db,
        target_table=task_table,
    )
    all_node_schema = {
        table: NODE_MODE
        for table in intermediate_tables
    }

    reg_result, reg_final_train_time = run_reg_training(
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
    )

    consensus_start = time.perf_counter()
    exp.SEED = seed
    set_strict_seed(seed)
    result = quiet_call(
        exp.schema_gate_joint_consensus,
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
    )
    consensus_total_runtime = time.perf_counter() - consensus_start

    consensus_conflict_time = sum(
        item["runtime_sec"]
        for item in result["conflict_candidates"]
    )
    consensus_gate_time = (
        result["edge_result"]["runtime_sec"]
        + result["node_result"]["runtime_sec"]
    )
    consensus_schema_time = consensus_gate_time + consensus_conflict_time
    consensus_final_train_time = max(0.0, consensus_total_runtime - consensus_schema_time)

    adaptive_selection = quiet_call(
        exp.select_attribute_aware_view,
        db=db,
        intermediate_tables=intermediate_tables,
        edge_result=result["edge_result"],
        node_result=result["node_result"],
        rich_attr_threshold=exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
    )

    adaptive_result, adaptive_final_train_time = run_final_training(
        "Attribute-aware",
        adaptive_selection["schema"],
        db,
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
    )
    edge_default_result, edge_default_final_train_time = run_final_training(
        "Gate-edge-default",
        result["edge_result"]["schema"],
        db,
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
    )
    node_default_result, node_default_final_train_time = run_final_training(
        "Gate-node-default",
        result["node_result"]["schema"],
        db,
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
    )
    all_node_result, all_node_final_train_time = run_fixed_mode(
        "All-node",
        NODE_MODE,
        db,
        initial_graph,
        task_table,
        intermediate_tables,
        num_classes,
        device,
        seed,
    )
    all_edge_result, all_edge_final_train_time = run_fixed_mode(
        "All-edge",
        EDGE_MODE,
        db,
        initial_graph,
        task_table,
        intermediate_tables,
        num_classes,
        device,
        seed,
    )

    exp.SEED = seed
    set_strict_seed(seed)
    all_schema_scores = quiet_call(
        exp.score_all_possible_schemas,
        db=db,
        initial_graph=initial_graph,
        target_table=task_table,
        intermediate_tables=intermediate_tables,
        num_classes=num_classes,
        device=device,
        label_schemas={
            "REG": all_node_schema,
            "Consensus": result["final_schema"],
            "Attribute-aware": adaptive_result["schema"],
            "Gate-edge-default": edge_default_result["schema"],
            "Gate-node-default": node_default_result["schema"],
            "All-node": all_node_result["schema"],
            "All-edge": all_edge_result["schema"],
        },
        selection_metric=exp.schema_selection_metric(task_table),
    )
    exhaustive_schema_time = sum(item["runtime_sec"] for item in all_schema_scores)
    best_exhaustive = max(
        all_schema_scores,
        key=lambda item: (item["val_score"], -item["schema_id"]),
    )
    exhaustive_best_result, exhaustive_best_final_train_time = run_final_training(
        "Exhaustive-best",
        best_exhaustive["schema"],
        db,
        initial_graph,
        task_table,
        num_classes,
        device,
        seed,
    )

    method_rows = []
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "REG",
            all_node_schema,
            reg_result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=reg_final_train_time,
            method_total_time_sec=reg_final_train_time,
            extra={"selected_view": "original-pk-fk-graph"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "Consensus",
            result["final_schema"],
            result["metrics"],
            selection_time_sec=consensus_conflict_time,
            gate_time_sec=consensus_gate_time,
            schema_time_sec=consensus_schema_time,
            final_train_time_sec=consensus_final_train_time,
            method_total_time_sec=consensus_total_runtime,
            extra={"selected_view": "edge+node+conflict-val"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "Attribute-aware",
            adaptive_result["schema"],
            adaptive_result["metrics"],
            selection_time_sec=adaptive_selection["selection_runtime_sec"],
            gate_time_sec=adaptive_selection["gate_runtime_sec"],
            schema_time_sec=adaptive_selection["runtime_sec"],
            final_train_time_sec=adaptive_final_train_time,
            method_total_time_sec=adaptive_selection["runtime_sec"] + adaptive_final_train_time,
            extra={"selected_view": adaptive_selection["selected_view"]},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "Gate-edge-default",
            result["edge_result"]["schema"],
            edge_default_result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=result["edge_result"]["runtime_sec"],
            schema_time_sec=result["edge_result"]["runtime_sec"],
            final_train_time_sec=edge_default_final_train_time,
            method_total_time_sec=result["edge_result"]["runtime_sec"] + edge_default_final_train_time,
            extra={"selected_view": "edge-default"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "Gate-node-default",
            result["node_result"]["schema"],
            node_default_result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=result["node_result"]["runtime_sec"],
            schema_time_sec=result["node_result"]["runtime_sec"],
            final_train_time_sec=node_default_final_train_time,
            method_total_time_sec=result["node_result"]["runtime_sec"] + node_default_final_train_time,
            extra={"selected_view": "node-default"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "All-node",
            all_node_result["schema"],
            all_node_result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=all_node_final_train_time,
            method_total_time_sec=all_node_final_train_time,
            extra={"selected_view": "fixed"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "All-edge",
            all_edge_result["schema"],
            all_edge_result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=all_edge_final_train_time,
            method_total_time_sec=all_edge_final_train_time,
            extra={"selected_view": "fixed"},
        )
    )
    method_rows.append(
        make_method_row(
            run_id,
            seed,
            SPLIT_SEED,
            "Exhaustive-best",
            best_exhaustive["schema"],
            exhaustive_best_result["metrics"],
            selection_time_sec=exhaustive_schema_time,
            gate_time_sec=0.0,
            schema_time_sec=exhaustive_schema_time,
            final_train_time_sec=exhaustive_best_final_train_time,
            method_total_time_sec=exhaustive_schema_time + exhaustive_best_final_train_time,
            extra={
                "selected_view": f"best-val-schema-{best_exhaustive['schema_id']}",
                "exhaustive_val_score": best_exhaustive["val_score"],
                "exhaustive_test_score": best_exhaustive["test_score"],
            },
        )
    )

    exhaustive_rows = []
    for item in all_schema_scores:
        exhaustive_rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "split_seed": SPLIT_SEED,
                "schema_id": item["schema_id"],
                "schema": format_schema(item["schema"]),
                "val_score": item["val_score"],
                "test_score": item["test_score"],
                "runtime_sec": item["runtime_sec"],
                "labels": item["labels"],
            }
        )

    return method_rows, exhaustive_rows


def make_method_row(
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
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics.get("roc_auc"),
        "selection_time_sec": selection_time_sec,
        "gate_time_sec": gate_time_sec,
        "schema_time_sec": schema_time_sec,
        "final_train_time_sec": final_train_time_sec,
        "method_total_time_sec": method_total_time_sec,
        "schema": format_schema(schema),
    }
    if extra:
        row.update(extra)
    return row


def summarize_method_rows(rows):
    raw_df = pd.DataFrame(rows)
    numeric_cols = [
        "accuracy",
        "f1",
        "roc_auc",
        "selection_time_sec",
        "gate_time_sec",
        "schema_time_sec",
        "final_train_time_sec",
        "method_total_time_sec",
    ]
    aggregations = {
        "runs": ("run_id", "count"),
        "split_seeds": ("split_seed", lambda values: " | ".join(sorted(set(map(str, values))))),
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "accuracy_var": ("accuracy", "var"),
        "f1_mean": ("f1", "mean"),
        "f1_std": ("f1", "std"),
        "f1_var": ("f1", "var"),
        "roc_auc_mean": ("roc_auc", "mean"),
        "roc_auc_std": ("roc_auc", "std"),
        "roc_auc_var": ("roc_auc", "var"),
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
    }
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
    for col in numeric_cols:
        if col in raw_df:
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    return raw_df, summary_df


def summarize_exhaustive_rows(rows):
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df, pd.DataFrame()
    return raw_df, (
        raw_df.groupby("schema_id", as_index=False)
        .agg(
            runs=("run_id", "count"),
            split_seeds=("split_seed", lambda values: " | ".join(sorted(set(map(str, values))))),
            val_score_mean=("val_score", "mean"),
            val_score_std=("val_score", "std"),
            val_score_var=("val_score", "var"),
            test_score_mean=("test_score", "mean"),
            test_score_std=("test_score", "std"),
            test_score_var=("test_score", "var"),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            runtime_sec_var=("runtime_sec", "var"),
            schemas=("schema", lambda values: " | ".join(sorted(set(map(str, values))))),
            labels=("labels", lambda values: " | ".join(sorted(set(map(str, values))))),
        )
        .sort_values(["val_score_mean", "test_score_mean"], ascending=False)
    )


def _excel_col_name(index):
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell_xml(row_idx, col_idx, value):
    ref = f"{_excel_col_name(col_idx)}{row_idx}"
    if value is None or pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(rows):
    if not rows:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData/></worksheet>'
        )

    headers = list(rows[0].keys())
    sheet_rows = []
    header_cells = [
        _xlsx_cell_xml(1, col_idx, header)
        for col_idx, header in enumerate(headers, start=1)
    ]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_idx, row in enumerate(rows, start=2):
        cells = [
            _xlsx_cell_xml(row_idx, col_idx, row.get(header))
            for col_idx, header in enumerate(headers, start=1)
        ]
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )


def write_simple_xlsx(sheets, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        workbook.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        workbook.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets>'
            + "".join(
                f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(sheets.keys(), start=1)
            )
            + "</sheets></workbook>",
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for i, rows in enumerate(sheets.values(), start=1):
            workbook.writestr(f"xl/worksheets/sheet{i}.xml", _xlsx_sheet_xml(rows))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_method_rows = []
    all_exhaustive_rows = []

    for run_id, seed in enumerate(RUN_SEEDS, start=1):
        method_rows, exhaustive_rows = run_once(run_id, seed, device)
        all_method_rows.extend(method_rows)
        all_exhaustive_rows.extend(exhaustive_rows)

    method_raw_df, method_summary_df = summarize_method_rows(all_method_rows)
    exhaustive_raw_df, exhaustive_summary_df = summarize_exhaustive_rows(all_exhaustive_rows)

    sheets = {
        "method_raw": method_raw_df.to_dict("records"),
        "method_summary": method_summary_df.to_dict("records"),
        "exhaustive_raw": exhaustive_raw_df.to_dict("records"),
        "exhaustive_summary": exhaustive_summary_df.to_dict("records"),
    }
    write_simple_xlsx(sheets, RESULTS_XLSX)
    print(f"Saved 5-run results to: {RESULTS_XLSX}")


if __name__ == "__main__":
    main()
