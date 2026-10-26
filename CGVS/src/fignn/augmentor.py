from fignn.scoring import score_attribute
from fignn.utils import log_promotion_step
from fignn.graph_building import promote_attribute
from fignn.gnn_scoring import initialize_gnn_model, pretrain_gnn_model, make_eval_function

import pandas as pd
import os
from datetime import datetime
import torch

def batch_score_attributes(
    graph,
    db,
    labels,
    node_id_map,
    scoring_method: str,
    label_table: str,
    k_hops: int,
    max_depth: int = None,
    candidate_subset: list = None,
    **kwargs  # for the gnn
):
    """
    Scores all candidate attributes (or a given subset) using the selected method.

    Returns:
        List of (score, table, attribute) sorted descending.
    """
    if candidate_subset is None:
        candidates = db.get_all_attributes(
            exclude_keys=True,
            label_table=label_table,
            max_depth=max_depth
        )
    else:
        candidates = candidate_subset

    results = []
    for table, attr in candidates:
        score = score_attribute(
            graph=graph,
            db=db,
            table=table,
            attribute=attr,
            labels=labels,
            method=scoring_method,
            task_table=label_table,
            #target_table=label_table,
            node_id_map=node_id_map,
            k_hops=k_hops,
            **kwargs  # gnn args
        )
        results.append((score, table, attr))

    results.sort(reverse=True)
    return results


def augment_graph(
    db,
    initial_graph,
    labels,
    node_id_map,
    scoring_method: str,
    label_table: str,
    #label_column: str,
    k_hops: int = 2,
    max_attributes: int = 5,
    threshold: float = 0.0,
    max_depth: int = None,
    verbose: bool = True,
    log_path: str = None
):
    """
    Iteratively augment the graph by promoting informative attributes.

    Returns:
        final_graph: HeteroData with promoted attributes
        selected_attributes: List of (table, attribute, score) tuples
    """
    graph = initial_graph
    selected_attributes = []
    logger = []

    # Get initial candidate set
    candidates = db.get_all_attributes(
        exclude_keys=True,
        label_table=label_table,
        max_depth=max_depth
    )

        # initialize gnn if scoring_method is gnn
    if scoring_method == "gnn_gain":
        print("Initializing and pretraining GNN model for gnn_gain scoring...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get feature dimension and number of classes
        feature_dim = graph[label_table].x.size(-1)
        num_classes = len(set(labels.values()))

        # 1. Initialize GNN
        gnn_model = initialize_gnn_model(
            graph=graph,
            target_table=label_table,
            num_classes=num_classes,
            hidden_dim=64  # or whatever hidden size you want
        )

        # 2. Pretrain GNN
        gnn_model = pretrain_gnn_model(
            model=gnn_model,
            graph=graph,
            target_table=label_table,
            epochs=30,   # or tune it
            lr=0.01,
            verbose=True
        )

        # 3. Make evaluation function
        eval_fn = make_eval_function(target_table=label_table)

        # Move model to device (should already be there from pretraining, but safe)
        gnn_model = gnn_model.to(device)

    else:
        gnn_model = None
        eval_fn = None
        feature_dim = None
        device = None

    for _ in range(max_attributes):
        scored = batch_score_attributes(
        graph=graph,
        db=db,
        labels=labels,
        node_id_map=node_id_map,
        scoring_method=scoring_method,
        label_table=label_table,
        k_hops=k_hops,
        max_depth=max_depth,
        candidate_subset=candidates,
        gnn_model=gnn_model,
        eval_fn=eval_fn,
        feature_dim=feature_dim,
        device=device,
        target_table=label_table  # assuming label_table is your target table
    )

        if not scored or scored[0][0] < threshold:
            if verbose:
                print("Stopping: no attribute meets threshold.")
            break

        top_score, top_table, top_attr = scored[0]

        if verbose:
            print(f"Promoting {top_table}.{top_attr} (score = {top_score:.4f})")

        graph = promote_attribute(
            graph,
            db,
            table=top_table,
            attribute=top_attr,
            node_id_map=node_id_map,
            modify_db=False,
            inplace=False
        )

        selected_attributes.append((top_table, top_attr, top_score))
        candidates = [(t, a) for (t, a) in candidates if (t, a) != (top_table, top_attr)]

        log_promotion_step(
            logger=logger,
            step=len(selected_attributes),
            table=top_table,
            attribute=top_attr,
            score=top_score,
            method=scoring_method,
            k_hops=k_hops,
            graph=graph,
            label_table=label_table
        )

    
    if log_path:
        if log_path.endswith(".csv"):
            output_file = log_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(log_path, f"augment_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            output_file = os.path.join(log_dir, "run.csv")

        pd.DataFrame(logger).to_csv(output_file, index=False)
        if verbose:
            print(f"Log written to {output_file}")

    return graph, selected_attributes


if __name__ == "__main__":
    from src.fignn.data_loader import load_relational_data
    from src.fignn.graph_building import build_fk_graph
    from src.fignn.utils import assign_node_labels, split_node_labels
    from src.fignn.augmentor import augment_graph
    import torch
    import pandas as pd
    import os

    print("### Start GNN Gain Scoring Test with augment_graph ###")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load relational data ---
    db = load_relational_data("data/synthetic")

    # --- 2. Define task ---
    task_table = "users"
    num_classes = 3

    # --- 3. Load task labels ---
    task = pd.read_csv(os.path.join("data/synthetic", "task.csv"))
    labels = dict(zip(task['user_id'], task['label']))

    # --- 4. Build FK graph ---
    graph, node_id_map = build_fk_graph(db)

    # --- 5. Assign labels and split ---
    assign_node_labels(graph, labels, node_id_map, node_type=task_table)
    split_node_labels(graph, node_type=task_table)

    # --- 6. Filter labels for train+val (for gnn scoring) ---
    mask = graph[task_table].train_mask | graph[task_table].val_mask
    node_ids = graph[task_table]['node_ids'][mask]
    labels_trainval = {
        pk.item(): labels[pk.item()]
        for pk in node_ids
        if pk.item() in labels
    }

    # --- 7. Run augmentation (scoring with gnn_gain) ---
    print("\nRunning augment_graph with gnn_gain scoring...")

    aug_graph, selected_attributes = augment_graph(
        db=db,
        initial_graph=graph,
        labels=labels_trainval,   # only using train+val labels during augmentation
        node_id_map=node_id_map,
        scoring_method="gnn_gain",
        label_table=task_table,
        k_hops=3,
        max_attributes=5,         # number of attributes to promote
        threshold=0.00,
        max_depth=2,
        verbose=True,
        log_path="logs/augment_gnn_gain.csv"
    )

    # --- 8. Print selected attributes ---
    print("\nSelected attributes (via gnn_gain):")
    for table, attr, score in selected_attributes:
        print(f"Score: {score:.4f} | Table: {table} | Attribute: {attr}")