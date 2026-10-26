import argparse
import importlib.util
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_DIR)
BASE_621_SCRIPT = os.path.join(
    EXPERIMENTS_DIR,
    "schema_gate_base.py",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "method_results")

ALL_METHODS = [
    "REG",
    "Consensus",
    "Attribute-aware",
    "Random",
    "Gate-edge-default",
    "Gate-node-default",
    "All-edge",
    "Exhaustive-best",
]

# DEFAULT_RUN_SEEDS = [40]
DEFAULT_RUN_SEEDS = [40, 41, 42, 43, 44]
SPLIT_SEED = 33
UNIVERSITY_SPLIT_SEEDS = [12, 13, 14]
IMDB_SMALL_SPLIT_SEEDS = [33, 34, 35]
HEPATITIS_SPLIT_SEEDS = [22, 23, 24]
CARCINOGENESIS_SPLIT_SEEDS = [22, 23, 24]
MOVIELENS_SPLIT_SEEDS = [33, 34, 35]
UWCSE_SPLIT_SEEDS = [13, 14, 15]
SAMEGEN_SPLIT_SEEDS = [33, 34, 35]
ELTI_SPLIT_SEEDS = [4, 6, 8]
# UWCSE_SPLIT_SEEDS = [13, 14, 15]

DATASET_SPLIT_RATIOS = {
    "University": (0.5, 0.25, 0.25),
    "UW-CSE": (0.6, 0.2, 0.2),
    "Hepatitis": (0.5, 0.25, 0.25),
    "Carcinogenesis": (0.7, 0.15, 0.15),
    "MovieLens": (0.5, 0.25, 0.25),
    "IMDB_small": (0.5, 0.25, 0.25),
    "SameGen": (0.5, 0.25, 0.25),
    "Elti": (0.5, 0.25, 0.25),
}
DEFAULT_DATASET = "University"
DATASET_CONFIGS = {
    "University": {
        "data_dir": "University",
        "task_table": "student",
        "task_pk": "student_id",
        "label_col": "intelligence",
    },
    "Carcinogenesis": {
        "data_dir": "Carcinogenesis",
        "task_table": "canc",
        "task_pk": "drug_id",
        "label_col": "class",
    },
    "MovieLens": {
        "data_dir": "MovieLens",
        "task_table": "users",
        "task_pk": "userid",
        "label_col": "u_gender",
    },
    "UW-CSE": {
        "data_dir": "UW-CSE",
        "task_table": "person",
        "task_pk": "p_id",
        "label_col": "inPhase",
    },
    "Hepatitis": {
        "data_dir": "Hepatitis",
        "task_table": "dispat",
        "task_pk": "m_id",
        "label_col": "Type",
    },
    "IMDB_small": {
        "data_dir": "IMDB_small",
        "task_table": "actors",
        "task_pk": "actors_id",
        "label_col": "gender",
    },
    "SameGen": {
        "data_dir": "SameGen",
        "task_table": "target",
        "task_pk": "target_id",
        "label_col": "target",
    },
    "Elti": {
        "data_dir": "Elti",
        "task_table": "target",
        "task_pk": "target_id",
        "label_col": "is_elti",
    },
}


def load_base_module():
    spec = importlib.util.spec_from_file_location("e3_621_base", BASE_621_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def _parse_seeds(args):
    if args.seeds:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    else:
        seeds = DEFAULT_RUN_SEEDS
    if args.runs is not None:
        seeds = seeds[: args.runs]
    if not seeds:
        raise ValueError("No seeds selected. Use --runs >= 1 or provide --seeds.")
    return seeds


def _default_output_path(label):
    filename = label.lower().replace("-", "_").replace(" ", "_") + ".xlsx"
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def get_dataset_config(dataset_name):
    try:
        return dict(DATASET_CONFIGS[dataset_name])
    except KeyError as exc:
        choices = ", ".join(DATASET_CONFIGS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {choices}") from exc


def get_split_seeds(dataset_name):
    if dataset_name == "University":
        return list(UNIVERSITY_SPLIT_SEEDS)
    if dataset_name == "IMDB_small":
        return list(IMDB_SMALL_SPLIT_SEEDS)
    if dataset_name == "MovieLens":
        return list(MOVIELENS_SPLIT_SEEDS)
    if dataset_name == "UW-CSE":
        return list(UWCSE_SPLIT_SEEDS)
    if dataset_name == "Hepatitis":
        return list(HEPATITIS_SPLIT_SEEDS)
    if dataset_name == "Carcinogenesis":
        return list(CARCINOGENESIS_SPLIT_SEEDS)
    if dataset_name == "SameGen":
        return list(SAMEGEN_SPLIT_SEEDS)
    if dataset_name == "Elti":
        return list(ELTI_SPLIT_SEEDS)
    return [SPLIT_SEED]


def get_split_ratios(dataset_name):
    try:
        train_ratio, val_ratio, test_ratio = DATASET_SPLIT_RATIOS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset_name}' for split ratios.") from exc
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Split ratios for dataset '{dataset_name}' must sum to 1.0, "
            f"got {(train_ratio, val_ratio, test_ratio)}"
        )
    return train_ratio, val_ratio, test_ratio


def prepare_context(seed, device, dataset_name=DEFAULT_DATASET, split_seed=None):
    base.exp.SEED = seed
    base.set_strict_seed(seed)
    if split_seed is None:
        split_seed = SPLIT_SEED

    dataset_config = get_dataset_config(dataset_name)
    data_path = os.path.join(PROJECT_ROOT, "data", dataset_config["data_dir"])
    task_table = dataset_config["task_table"]
    task_pk = dataset_config["task_pk"]
    label_col = dataset_config["label_col"]

    db = base.load_relational_data(data_path)
    initial_graph, initial_node_id_map = base.build_fk_graph(db)

    task = pd.read_csv(os.path.join(data_path, "task.csv"))
    task[label_col] = pd.Categorical(task[label_col]).codes
    num_classes = int(task[label_col].nunique())
    labels = dict(zip(task[task_pk], task[label_col]))
    base.assign_node_labels(initial_graph, labels, initial_node_id_map, node_type=task_table)
    train_ratio, val_ratio, test_ratio = get_split_ratios(dataset_name)
    base.split_node_labels(
        initial_graph,
        node_type=task_table,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    intermediate_tables = base.identify_intermediate_tables(
        db,
        target_table=task_table,
    )
    all_node_schema = {table: base.NODE_MODE for table in intermediate_tables}
    all_edge_schema = {table: base.EDGE_MODE for table in intermediate_tables}

    return {
        "db": db,
        "initial_graph": initial_graph,
        "task_table": task_table,
        "num_classes": num_classes,
        "intermediate_tables": intermediate_tables,
        "all_node_schema": all_node_schema,
        "all_edge_schema": all_edge_schema,
        "device": device,
        "seed": seed,
        "split_seed": split_seed,
        "dataset": dataset_name,
    }


def run_gate_consensus_bundle(ctx):
    base.exp.SEED = ctx["seed"]
    base.set_strict_seed(ctx["seed"])
    start = time.perf_counter()
    result = base.quiet_call(
        base.exp.schema_gate_joint_consensus,
        db=ctx["db"],
        initial_graph=ctx["initial_graph"],
        target_table=ctx["task_table"],
        intermediate_tables=ctx["intermediate_tables"],
        num_classes=ctx["num_classes"],
        device=ctx["device"],
    )
    total_runtime = time.perf_counter() - start
    conflict_time = sum(item["runtime_sec"] for item in result["conflict_candidates"])
    gate_time = result["edge_result"]["runtime_sec"] + result["node_result"]["runtime_sec"]
    schema_time = gate_time + conflict_time
    final_train_time = max(0.0, total_runtime - schema_time)
    return {
        "result": result,
        "total_runtime": total_runtime,
        "conflict_time": conflict_time,
        "gate_time": gate_time,
        "schema_time": schema_time,
        "final_train_time": final_train_time,
    }


def run_gate_pair_bundle(ctx):
    """Run edge-default and node-default gate learners without final consensus retraining."""
    base.exp.SEED = ctx["seed"]
    base.set_strict_seed(ctx["seed"])
    start = time.perf_counter()
    edge_result = base.quiet_call(
        base.exp.run_schema_gate_with_background,
        db=ctx["db"],
        initial_graph=ctx["initial_graph"],
        target_table=ctx["task_table"],
        intermediate_tables=ctx["intermediate_tables"],
        num_classes=ctx["num_classes"],
        background_mode=base.EDGE_MODE,
        device=ctx["device"],
    )
    node_result = base.quiet_call(
        base.exp.run_schema_gate_with_background,
        db=ctx["db"],
        initial_graph=ctx["initial_graph"],
        target_table=ctx["task_table"],
        intermediate_tables=ctx["intermediate_tables"],
        num_classes=ctx["num_classes"],
        background_mode=base.NODE_MODE,
        device=ctx["device"],
    )
    return {
        "edge_result": edge_result,
        "node_result": node_result,
        "gate_time": edge_result["runtime_sec"] + node_result["runtime_sec"],
        "runtime_sec": time.perf_counter() - start,
    }


def _schema_key(schema):
    return tuple(sorted(schema.items()))


def build_exhaustive_label_schemas(ctx):
    """Collect method-name -> schema mappings used to annotate exhaustive rows."""
    bundle = run_gate_consensus_bundle(ctx)
    gate_result = bundle["result"]
    attribute_aware_selection = base.quiet_call(
        base.exp.select_attribute_aware_view,
        db=ctx["db"],
        intermediate_tables=ctx["intermediate_tables"],
        rich_attr_threshold=base.exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
    )
    return {
        "REG": ctx["all_node_schema"],
        "Consensus": gate_result["final_schema"],
        "Attribute-aware": attribute_aware_selection["schema"],
        "Gate-edge-default": gate_result["edge_result"]["schema"],
        "Gate-node-default": gate_result["node_result"]["schema"],
        "All-node": ctx["all_node_schema"],
        "All-edge": ctx["all_edge_schema"],
    }


def select_random_schema(ctx):
    """Pick a reproducible random schema by sampling node/edge modes per table."""
    start = time.perf_counter()
    random_seed = ctx["split_seed"] * 10 + ctx["seed"]
    rng = random.Random(random_seed)
    schema = {
        table: rng.choice((base.NODE_MODE, base.EDGE_MODE))
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


def select_best_gate_schema_by_validation(ctx, gate_schema_result, device):
    """Validate unique schemas seen during gate training and return the best one."""
    candidates = []
    seen = set()
    history = gate_schema_result.get("history", [])
    schemas = [record["schema"] for record in history]
    schemas.append(gate_schema_result["schema"])

    for schema in schemas:
        key = _schema_key(schema)
        if key in seen:
            continue
        seen.add(key)
        start = time.perf_counter()
        val_score = base.quiet_call(
            base.exp.score_complete_schema,
            db=ctx["db"],
            initial_graph=ctx["initial_graph"],
            target_table=ctx["task_table"],
            num_classes=ctx["num_classes"],
            schema=schema,
            device=device,
        )
        candidates.append(
            {
                "schema": schema,
                "val_score": val_score,
                "runtime_sec": time.perf_counter() - start,
            }
        )

    if not candidates:
        return {
            "schema": gate_schema_result["schema"],
            "val_score": None,
            "runtime_sec": 0.0,
            "num_candidates": 0,
        }

    best = max(
        candidates,
        key=lambda item: (item["val_score"], -len(candidates)),
    )
    return {
        "schema": best["schema"],
        "val_score": best["val_score"],
        "runtime_sec": sum(item["runtime_sec"] for item in candidates),
        "num_candidates": len(candidates),
    }


def run_method_once(
    method,
    run_id,
    seed,
    device,
    show_epochs=False,
    dataset_name=DEFAULT_DATASET,
    split_seed=None,
):
    ctx = prepare_context(
        seed,
        device,
        dataset_name=dataset_name,
        split_seed=split_seed,
    )
    exhaustive_rows = []

    if method == "REG":
        result, final_train_time = base.run_reg_training(
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
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
            extra={"selected_view": "original-pk-fk-graph"},
        )
        return [row], exhaustive_rows

    if method == "All-node":
        result, final_train_time = base.run_fixed_mode(
            "All-node",
            base.NODE_MODE,
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["intermediate_tables"],
            ctx["num_classes"],
            device,
            seed,
            show_output=show_epochs,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "All-node",
            result["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=0.0,
            schema_time_sec=0.0,
            final_train_time_sec=final_train_time,
            method_total_time_sec=final_train_time,
            extra={"selected_view": "fixed"},
        )
        return [row], exhaustive_rows

    if method == "All-edge":
        result, final_train_time = base.run_fixed_mode(
            "All-edge",
            base.EDGE_MODE,
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["intermediate_tables"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
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
            extra={"selected_view": "fixed"},
        )
        return [row], exhaustive_rows

    if method == "Exhaustive-best":
        label_schemas = build_exhaustive_label_schemas(ctx)
        all_schema_scores = base.quiet_call(
            base.exp.score_all_possible_schemas,
            db=ctx["db"],
            initial_graph=ctx["initial_graph"],
            target_table=ctx["task_table"],
            intermediate_tables=ctx["intermediate_tables"],
            num_classes=ctx["num_classes"],
            device=device,
            label_schemas=label_schemas,
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
        result, final_train_time = base.run_final_training(
            "Exhaustive-best",
            best["schema"],
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
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
            extra={
                "selected_view": f"best-val-schema-{best['schema_id']}",
                "exhaustive_val_score": best["val_score"],
                "exhaustive_test_score": best["test_score"],
            },
        )
        for item in all_schema_scores:
            exhaustive_rows.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "split_seed": ctx["split_seed"],
                    "schema_id": item["schema_id"],
                    "schema": base.format_schema(item["schema"]),
                    "val_score": item["val_score"],
                    "test_score": item["test_score"],
                    "runtime_sec": item["runtime_sec"],
                    "labels": item["labels"],
                }
            )
        return [row], exhaustive_rows

    bundle = run_gate_consensus_bundle(ctx)
    gate_result = bundle["result"]

    if method == "Consensus":
        row = base.make_method_row(
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
            extra={"selected_view": "edge+node+conflict-val"},
        )
        return [row], exhaustive_rows

    if method == "Consensus-val-defaults":
        gate_pair = run_gate_pair_bundle(ctx)
        edge_selection = select_best_gate_schema_by_validation(
            ctx,
            gate_pair["edge_result"],
            device,
        )
        node_selection = select_best_gate_schema_by_validation(
            ctx,
            gate_pair["node_result"],
            device,
        )
        agreed, conflicts = base.exp.split_consensus_and_conflicts(
            edge_selection["schema"],
            node_selection["schema"],
            ctx["intermediate_tables"],
        )
        final_schema, conflict_candidates = base.quiet_call(
            base.exp.resolve_conflicts_by_validation,
            db=ctx["db"],
            initial_graph=ctx["initial_graph"],
            target_table=ctx["task_table"],
            intermediate_tables=ctx["intermediate_tables"],
            num_classes=ctx["num_classes"],
            agreed=agreed,
            conflicts=conflicts,
            device=device,
        )
        conflict_time = sum(item["runtime_sec"] for item in conflict_candidates)
        selection_time = (
            edge_selection["runtime_sec"]
            + node_selection["runtime_sec"]
            + conflict_time
        )
        result, final_train_time = base.run_final_training(
            "Consensus-val-defaults",
            final_schema,
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Consensus-val-defaults",
            final_schema,
            result["metrics"],
            selection_time_sec=selection_time,
            gate_time_sec=gate_pair["gate_time"],
            schema_time_sec=gate_pair["gate_time"] + selection_time,
            final_train_time_sec=final_train_time,
            method_total_time_sec=gate_pair["gate_time"] + selection_time + final_train_time,
            extra={
                "selected_view": "val-selected-edge+node+conflict-val",
                "edge_gate_val_score": edge_selection["val_score"],
                "edge_gate_val_candidates": edge_selection["num_candidates"],
                "node_gate_val_score": node_selection["val_score"],
                "node_gate_val_candidates": node_selection["num_candidates"],
                "consensus_conflicts": ",".join(conflicts) if conflicts else "-",
            },
        )
        return [row], exhaustive_rows

    if method == "Attribute-aware":
        selection = base.quiet_call(
            base.exp.select_attribute_aware_view,
            db=ctx["db"],
            intermediate_tables=ctx["intermediate_tables"],
            rich_attr_threshold=base.exp.ADAPTIVE_RICH_ATTR_THRESHOLD,
        )
        result, final_train_time = base.run_final_training(
            "Attribute-aware",
            selection["schema"],
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Attribute-aware",
            result["schema"],
            result["metrics"],
            selection_time_sec=selection["selection_runtime_sec"],
            gate_time_sec=selection["gate_runtime_sec"],
            schema_time_sec=selection["runtime_sec"],
            final_train_time_sec=final_train_time,
            method_total_time_sec=selection["runtime_sec"] + final_train_time,
            extra={"selected_view": selection["selected_view"]},
        )
        return [row], exhaustive_rows

    if method == "Random":
        selection = select_random_schema(ctx, seed)
        result, final_train_time = base.run_final_training(
            "Random",
            selection["schema"],
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Random",
            result["schema"],
            result["metrics"],
            selection_time_sec=selection["selection_runtime_sec"],
            gate_time_sec=0.0,
            schema_time_sec=selection["runtime_sec"],
            final_train_time_sec=final_train_time,
            method_total_time_sec=selection["runtime_sec"] + final_train_time,
            extra={"selected_view": selection["selected_view"]},
        )
        return [row], exhaustive_rows

    if method == "Gate-edge-default":
        result, final_train_time = base.run_final_training(
            "Gate-edge-default",
            gate_result["edge_result"]["schema"],
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Gate-edge-default",
            gate_result["edge_result"]["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=gate_result["edge_result"]["runtime_sec"],
            schema_time_sec=gate_result["edge_result"]["runtime_sec"],
            final_train_time_sec=final_train_time,
            method_total_time_sec=gate_result["edge_result"]["runtime_sec"] + final_train_time,
            extra={"selected_view": "edge-default"},
        )
        return [row], exhaustive_rows

    if method == "Gate-node-default":
        result, final_train_time = base.run_final_training(
            "Gate-node-default",
            gate_result["node_result"]["schema"],
            ctx["db"],
            ctx["initial_graph"],
            ctx["task_table"],
            ctx["num_classes"],
            device,
            seed,
        )
        row = base.make_method_row(
            run_id,
            seed,
            ctx["split_seed"],
            "Gate-node-default",
            gate_result["node_result"]["schema"],
            result["metrics"],
            selection_time_sec=0.0,
            gate_time_sec=gate_result["node_result"]["runtime_sec"],
            schema_time_sec=gate_result["node_result"]["runtime_sec"],
            final_train_time_sec=final_train_time,
            method_total_time_sec=gate_result["node_result"]["runtime_sec"] + final_train_time,
            extra={"selected_view": "node-default"},
        )
        return [row], exhaustive_rows

    raise ValueError(f"Unknown method: {method}")


def run_selected_methods(methods, seeds, output_path, dataset_name=DEFAULT_DATASET):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_method_rows = []
    all_exhaustive_rows = []
    split_seeds = get_split_seeds(dataset_name)
    total_runs = len(split_seeds) * len(seeds)
    run_counter = 0

    for split_seed in split_seeds:
        for seed in seeds:
            run_counter += 1
            for method in methods:
                print(
                    f"Run {run_counter}/{total_runs} | split_seed={split_seed} | "
                    f"seed={seed} | dataset={dataset_name} | method={method}"
                )
                method_rows, exhaustive_rows = run_method_once(
                    method,
                    run_counter,
                    seed,
                    device,
                    dataset_name=dataset_name,
                    split_seed=split_seed,
                )
                all_method_rows.extend(method_rows)
                all_exhaustive_rows.extend(exhaustive_rows)

    method_raw_df, method_summary_df = base.summarize_method_rows(all_method_rows)
    exhaustive_raw_df, exhaustive_summary_df = base.summarize_exhaustive_rows(
        all_exhaustive_rows
    )

    sheets = {
        "method_raw": method_raw_df.to_dict("records"),
        "method_summary": method_summary_df.to_dict("records"),
        "exhaustive_raw": exhaustive_raw_df.to_dict("records"),
        "exhaustive_summary": exhaustive_summary_df.to_dict("records"),
    }
    base.write_simple_xlsx(sheets, output_path)
    print(f"Saved results to: {output_path}")


def build_parser(default_output, default_dataset=DEFAULT_DATASET):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        choices=sorted(DATASET_CONFIGS),
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
        default=default_output,
        help="Output .xlsx path.",
    )
    return parser


def main_single(method, default_dataset=DEFAULT_DATASET):
    parser = build_parser(_default_output_path(method), default_dataset=default_dataset)
    args = parser.parse_args()
    seeds = _parse_seeds(args)
    run_selected_methods([method], seeds, args.output, dataset_name=args.dataset)


def main_single_test(method, default_dataset=DEFAULT_DATASET):
    """Test mode: run once with seed=40, print results to console"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        choices=sorted(DATASET_CONFIGS),
        help=f"Dataset to run. Defaults to {default_dataset}.",
    )
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    print(f"\n{'='*60}")
    print(f"Testing method: {method}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Final Epochs: {base.exp.FINAL_EPOCHS}")
    print(f"{'='*60}\n")
    
    try:
        method_rows, exhaustive_rows = run_method_once(
            method,
            run_id=1,
            seed=seed,
            device=device,
            show_epochs=False,
            dataset_name=args.dataset,
        )
        
        # Print method results
        if method_rows:
            row = method_rows[0]
            print(f"Method: {row['method']}")
            print(f"Accuracy: {row['accuracy']:.4f}")
            print(f"F1 Score: {row['f1']:.4f}")
            if row['roc_auc'] is not None:
                print(f"ROC-AUC: {row['roc_auc']:.4f}")
            print(f"Selection Time: {row['selection_time_sec']:.4f}s")
            print(f"Gate Time: {row['gate_time_sec']:.4f}s")
            print(f"Schema Time: {row['schema_time_sec']:.4f}s")
            print(f"Final Train Time: {row['final_train_time_sec']:.4f}s")
            print(f"Total Time: {row['method_total_time_sec']:.4f}s")
            print(f"Selected View: {row.get('selected_view', 'N/A')}")
            print(f"Schema: {row.get('schema', 'N/A')}")
            
            if 'exhaustive_val_score' in row:
                print(f"Exhaustive Val Score: {row['exhaustive_val_score']:.4f}")
                print(f"Exhaustive Test Score: {row['exhaustive_test_score']:.4f}")
            if row.get('gate_val_score') is not None:
                print(f"Gate Val Score: {row['gate_val_score']:.4f}")
                print(f"Gate Val Candidates: {row.get('gate_val_candidates', 'N/A')}")
            if row.get('edge_gate_val_score') is not None:
                print(f"Edge Gate Val Score: {row['edge_gate_val_score']:.4f}")
                print(f"Edge Gate Val Candidates: {row.get('edge_gate_val_candidates', 'N/A')}")
            if row.get('node_gate_val_score') is not None:
                print(f"Node Gate Val Score: {row['node_gate_val_score']:.4f}")
                print(f"Node Gate Val Candidates: {row.get('node_gate_val_candidates', 'N/A')}")
            if row.get('consensus_conflicts') is not None:
                print(f"Consensus Conflicts: {row['consensus_conflicts']}")
        
        # Print exhaustive results if applicable
        if exhaustive_rows:
            print(f"\n--- Exhaustive Search Results ({len(exhaustive_rows)} schemas) ---")
            for i, item in enumerate(exhaustive_rows[:5], 1):  # Show first 5
                print(f"  Schema {item['schema_id']}: Val={item['val_score']:.4f}, "
                      f"Test={item['test_score']:.4f}, Time={item['runtime_sec']:.4f}s")
            if len(exhaustive_rows) > 5:
                print(f"  ... and {len(exhaustive_rows) - 5} more schemas")
        
        print(f"\n{'='*60}")
        print("✓ Test completed successfully")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Test failed: {str(e)}")
        print(f"{'='*60}\n")
        raise


def main_all(default_dataset=DEFAULT_DATASET):
    parser = build_parser(
        _default_output_path("all_methods"),
        default_dataset=default_dataset,
    )
    args = parser.parse_args()
    seeds = _parse_seeds(args)
    run_selected_methods(ALL_METHODS, seeds, args.output, dataset_name=args.dataset)
