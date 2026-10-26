# training.py
import copy
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def train_model(
    model,
    graph,
    node_type,
    optimizer,
    loss_fn,
    epochs=30,
    verbose=True,
    selection_metric="accuracy",
):
    """
    Train a node classification model on a HeteroData graph, tracking the best validation score.

    Returns:
        model: the best model (according to validation selection metric)
    """
    if selection_metric not in {"accuracy", "roc_auc"}:
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    best_val_score = float("-inf")
    best_model_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(graph.x_dict, graph.edge_index_dict)
        # shuffled_indices = torch.randperm(graph[node_type].y.size(0))
        # graph[node_type].y = graph[node_type].y[shuffled_indices]
        y = graph[node_type].y
        # print('真实标签y',y)
        mask = graph[node_type].train_mask#训练集对应的true
        # print(out[mask], y[mask])

        loss = loss_fn(out[mask], y[mask])
        # print(out[mask].shape)
        loss.backward()
        optimizer.step()

        metrics = evaluate_model(model, graph, node_type, split="val", return_metrics=True)
        val_score = metrics["roc_auc"] if selection_metric == "roc_auc" else metrics["accuracy"]
        if val_score is None:
            raise ValueError(
                f"Validation {selection_metric} is unavailable for node_type '{node_type}'."
            )
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose:
            score_label = "Val ROC AUC" if selection_metric == "roc_auc" else "Val Acc"
            score_value = val_score
            roc_text = metrics["roc_auc"] if metrics["roc_auc"] is not None else "N/A"
            true_dist = ", ".join(
                f"{label}:{count}"
                for label, count in sorted(metrics["true_dist"].items())
            )
            pred_dist = ", ".join(
                f"{label}:{count}"
                for label, count in sorted(metrics["pred_dist"].items())
            )
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | "
                  f"{score_label}: {score_value:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"ROC AUC: {roc_text} | "
                  f"True Dist: [{true_dist}] | "
                  f"Pred Dist: [{pred_dist}]")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


@torch.no_grad()
def evaluate_model(model, graph, node_type, split="test", return_metrics=False):
    """
    Evaluate model accuracy (and optionally F1 and ROC AUC) on a given split.

    Args:
        model: the GNN model
        graph: HeteroData with masks and labels
        node_type: the node type (e.g., 'user')
        split: one of 'train', 'val', 'test'
        return_metrics: if True, return dict with all metrics

    Returns:
        accuracy (float) if return_metrics is False
        else dict with accuracy, f1, and roc_auc
    """
    model.eval()
    out = model(graph.x_dict, graph.edge_index_dict)
    y_true = graph[node_type].y

    mask = graph[node_type][f"{split}_mask"]

    y_true = y_true[mask]

    y_pred = out[mask]
    y_pred_labels = y_pred.argmax(dim=-1)
    # print(y_pred_labels ,y_true)
    acc = (y_pred_labels == y_true).sum().item() / mask.sum().item()

    if not return_metrics:
        return acc

    # Compute F1 (macro for multi-class)
    f1 = f1_score(y_true.cpu(), y_pred_labels.cpu(), average='macro')

    # Compute ROC AUC 
    try:
        y_true_np = y_true.cpu().numpy()
        if y_pred.shape[1] == 2:  # Binary classification with 2 logits
            y_probs = F.softmax(y_pred, dim=1)[:, 1].cpu().numpy()  # prob for class 1
            roc_auc = roc_auc_score(y_true_np, y_probs)
        else:  # Multi-class case
            y_probs = F.softmax(y_pred, dim=1).cpu().numpy()
            roc_auc = roc_auc_score(y_true_np, y_probs, multi_class='ovr')
    except ValueError:
        roc_auc = None

    true_labels, true_counts = torch.unique(y_true.cpu(), return_counts=True)
    pred_labels, pred_counts = torch.unique(y_pred_labels.cpu(), return_counts=True)

    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "true_dist": {
            int(label.item()): int(count.item())
            for label, count in zip(true_labels, true_counts)
        },
        "pred_dist": {
            int(label.item()): int(count.item())
            for label, count in zip(pred_labels, pred_counts)
        },
    }


def log_test_metrics(metrics, filename="log.txt"):
    """
    Logs the test evaluation metrics to a text file.

    Args:
        metrics (dict): Dictionary with keys 'accuracy', 'f1', 'roc_auc'
        filename (str): File name to write to (default: 'log.txt')
    """
    with open(filename, "a") as f:
        f.write("==== Final Test Evaluation ====\n")
        f.write(f"Test Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Test F1 Score : {metrics['f1']:.4f}\n")
        if metrics['roc_auc'] is not None:
            f.write(f"Test ROC AUC  : {metrics['roc_auc']:.4f}\n")
        else:
            f.write("Test ROC AUC  : N/A (not applicable for this task)\n")
        f.write("===============================\n\n")
