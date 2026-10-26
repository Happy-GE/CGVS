from fignn.data_loader import load_relational_data
from fignn.graph_building import build_fk_graph, promote_attribute
from fignn.augmentor import augment_graph
from fignn.utils import assign_node_labels, split_node_labels
from fignn.models import GNN
import pandas as pd
import os
from fignn.training import train_model, evaluate_model
import torch

print("### Start test script ###")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. Load relational data ---
db = load_relational_data("data/hepatitis")

# --- 2. Prediction task ---
task_table = "dispat"
num_classes = 2

# --- 2.1 load labels from task
task = pd.read_csv(os.path.join("data/hepatitis", "task.csv"))
labels = dict(zip(task['m_id'], task['Type']))

# --- 3. Build FK graph ---
graph, node_id_map = build_fk_graph(db)
feature_dim = graph[task_table].x.size(-1)

# --- 3b. Assign full labels to FK-graph for splitting ---
assign_node_labels(graph, labels, node_id_map, node_type=task_table)

# --- 4. Split into train/val/test ---
split_node_labels(graph, node_type=task_table)

# --- 5. Filter labels for train+val ---
mask = graph[task_table].train_mask | graph[task_table].val_mask
node_ids = graph[task_table]['node_ids'][mask]
labels_trainval = {
    pk.item(): labels[pk.item()]
    for pk in node_ids
    if pk.item() in labels
}

# --- 6. Run augmentation (train+val only) ---

aug_graph, selected = augment_graph(
    db=db,
    initial_graph=graph,
    labels=labels_trainval,
    node_id_map=node_id_map,
    scoring_method="entropy_gain",
    label_table=task_table,
    k_hops=3,
    max_attributes=2,
    threshold=0.00,
    max_depth=2,
    verbose=True,
    log_path="logs/augment_test.csv"
)


# --- 7. Assign full labels + masks to augmented graph ---
assign_node_labels(aug_graph, labels, node_id_map, node_type=task_table)
for split in ['train_mask', 'val_mask', 'test_mask']:
    aug_graph[task_table][split] = aug_graph[task_table][split].clone()

assign_node_labels(graph, labels, node_id_map, node_type=task_table)
for split in ['train_mask', 'val_mask', 'test_mask']:
    graph[task_table][split] = graph[task_table][split].clone()

# --- 8. Train and test GNN ---
aug_graph = aug_graph.to(device)

# Define model
model_og = GNN(
    metadata=graph.metadata(),
    hidden_channels=64,
    out_channels=num_classes,
    target_node=task_table
).to(device)

model_aug = GNN(
    metadata=aug_graph.metadata(),
    hidden_channels=64,
    out_channels=num_classes,
    target_node=task_table
).to(device)

# Optimizer and loss
optimizer_og = torch.optim.Adam(model_og.parameters(), lr=0.01)
optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Train & test, original graph
print("### Start training, original graph ###")
train_model(
    model=model_og,
    graph=graph,
    node_type=task_table,
    optimizer=optimizer_og,
    loss_fn=loss_fn,
    epochs=30,
    verbose=True
)

# Evaluate
print("### Final Test Evaluation, original graph ###")
test_acc = evaluate_model(model_og, graph, node_type=task_table, split="test")
print(f"Test Accuracy: {test_acc:.4f}")

# Train & test, original graph
print("### Start training, aug graph ###")
train_model(
    model=model_aug,
    graph=aug_graph,
    node_type=task_table,
    optimizer=optimizer_aug,
    loss_fn=loss_fn,
    epochs=30,
    verbose=True
)

# Evaluate
print("### Final Test Evaluation, aug graph ###")
test_acc = evaluate_model(model_aug, aug_graph, node_type=task_table, split="test")
print(f"Test Accuracy: {test_acc:.4f}")

