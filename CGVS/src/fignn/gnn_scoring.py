import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import copy
from fignn.models import GNN
from fignn.graph_building import promote_attribute

def initialize_gnn_model(graph, target_table, num_classes, hidden_dim=64):
    """
    Initialize a GNN model for relational graph.

    Args:
        graph: HeteroData graph object.
        target_table: Table for prediction task.
        num_classes: Number of output classes.
        hidden_dim: Hidden dimension size.

    Returns:
        GNN model (untrained)
    """
    model = GNN(
        metadata=graph.metadata(),
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        target_node=target_table
    )
    return model

def pretrain_gnn_model(
    model,
    graph,
    target_table,
    epochs=20,
    lr=0.01,
    verbose=True
):
    """
    Pre-train GNN model on original FK graph.

    Args:
        model: Initialized GNN model.
        graph: HeteroData graph.
        target_table: Table where labels live.
        epochs: Number of training epochs.
        lr: Learning rate.
        verbose: Whether to print training info.

    Returns:
        Pretrained, frozen GNN model (in eval mode).
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        #out = model(graph.x_dict, graph.edge_index_dict)
        #preds = out[target_table]
        out = model(graph.x_dict, graph.edge_index_dict)
        preds = out  # already predictions for target_table
        labels = graph[target_table].y
        mask = graph[target_table].train_mask

        loss = loss_fn(preds[mask], labels[mask])
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 5 == 0 or epoch == epochs-1):
            print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model

def make_eval_function(target_table):
    def eval_fn(model, graph):
        model.eval()
        with torch.no_grad():
            out = model(graph.x_dict, graph.edge_index_dict)
            preds = out
            labels = graph[target_table].y
            mask = graph[target_table].val_mask

            preds = preds[mask]
            labels = labels[mask]

            if preds.size(0) == 0:
                return 0.0

            acc = (preds.argmax(dim=1) == labels).float().mean().item()
        return acc  # Higher is better
    return eval_fn

class IdentityMessage(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, edge_index, *args, **kwargs):
        return x

def gnn_gain_score(
    graph,
    db,
    table: str,
    attribute: str,
    labels: dict,
    gnn_model,
    eval_fn,
    node_id_map,
    target_table,
    feature_dim,
    device=None,
    **kwargs
) -> float:

    if device is None:
        device = next(gnn_model.parameters()).device

    # 1. Evaluate on the original graph
    gnn_model.eval()
    with torch.no_grad():
        score_before = eval_fn(gnn_model, graph)

    # 2. Promote the attribute (augment the graph)
    graph_aug = promote_attribute(
        graph,
        db,
        table=table,
        attribute=attribute,
        node_id_map=node_id_map,
        modify_db=False,
        inplace=False
    )

    graph_aug = graph_aug.to(device)

    # 3. Make sure all nodes have features
    for node_type in graph_aug.node_types:
        if not hasattr(graph_aug[node_type], 'x') or graph_aug[node_type].x is None:
            num_nodes = graph_aug[node_type].num_nodes
            graph_aug[node_type].x = torch.zeros((num_nodes, feature_dim), device=device)

    # 4. 🛡 PATCH the GNN model to handle new edge types
    gnn_model_aug = copy.deepcopy(gnn_model)

    if hasattr(gnn_model_aug, "convs") and isinstance(gnn_model_aug.convs, nn.ModuleList):
        for edge_type in graph_aug.edge_types:
            if edge_type not in gnn_model_aug.convs[0].convs:
                print(f"Adding real SAGEConv for unseen edge type {edge_type}")
                for conv_layer in gnn_model_aug.convs:
                    conv_layer.convs[edge_type] = SAGEConv(
                        (-1, -1), 
                        conv_layer.convs[list(conv_layer.convs.keys())[0]].out_channels
                    )

    # 5. Evaluate on the augmented graph
    gnn_model_aug.eval()
    with torch.no_grad():
        score_after = eval_fn(gnn_model_aug, graph_aug)

    # 6. Return gain
    return score_after - score_before
