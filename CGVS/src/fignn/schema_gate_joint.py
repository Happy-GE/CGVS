import random

import numpy as np
import torch
from torch import nn

from fignn.models import GNN
from fignn.schema_graph_building import (
    EDGE_MODE,
    NODE_MODE,
    build_schema_aware_graph,
    copy_task_tensors,
    identify_intermediate_tables,
)
from fignn.training import evaluate_model, train_model


class SchemaGate(nn.Module):
    def __init__(self, table_names, init=0.0, temperature=1.0):
        super().__init__()
        self.table_names = list(table_names)
        self.temperature = temperature
        self.alpha = nn.Parameter(torch.full((len(table_names),), float(init)))

    def probs(self):
        return torch.sigmoid(self.alpha / self.temperature)

    def discreteness_loss(self):
        p = self.probs()
        return (p * (1.0 - p)).mean()

    def discrete_schema(self, threshold=0.5):
        p = self.probs().detach().cpu()
        return {
            name: ("node" if p[i].item() > threshold else "edge")
            for i, name in enumerate(self.table_names)
        }


class SchemaGateJointModel(nn.Module):
    """
    Joint schema-gate model over table-local node/edge channels.

    Each intermediate table gets two existing schema-aware graph channels:
    one where that table is represented as a node, and one where it is
    compressed as direct edges. The learned gate mixes the two target logits.
    """

    def __init__(self, table_names, channel_graphs, hidden_channels, out_channels, target_node):
        super().__init__()
        self.table_names = list(table_names)
        self.target_node = target_node
        self.node_models = nn.ModuleList()
        self.edge_models = nn.ModuleList()

        for channels in channel_graphs:
            self.node_models.append(
                GNN(
                    metadata=channels[NODE_MODE].metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    target_node=target_node,
                )
            )
            self.edge_models.append(
                GNN(
                    metadata=channels[EDGE_MODE].metadata(),
                    hidden_channels=hidden_channels,
                    out_channels=out_channels,
                    target_node=target_node,
                )
            )

    def forward(self, channel_graphs, schema_gate):
        probs = schema_gate.probs()
        mixed_logits = []

        for i, channels in enumerate(channel_graphs):
            node_graph = channels[NODE_MODE]
            edge_graph = channels[EDGE_MODE]

            node_logits = self.node_models[i](node_graph.x_dict, node_graph.edge_index_dict)
            edge_logits = self.edge_models[i](edge_graph.x_dict, edge_graph.edge_index_dict)

            p = probs[i]
            mixed_logits.append(p * node_logits + (1.0 - p) * edge_logits)

        return torch.stack(mixed_logits, dim=0).mean(dim=0)


def set_schema_gate_seed(seed):
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


def build_schema_gate_channels(
    db,
    reference_graph,
    target_table,
    intermediate_tables=None,
    default_mode=EDGE_MODE,
    background_schema=None,
    include_fk_features=False,
):
    """
    Build node/edge graph channels for each intermediate table.

    Other candidate intermediate tables are held at `default_mode` or the
    provided `background_schema`, so each gate learns the local preference of
    its own table while reusing the existing schema-aware graph builder.
    """
    if intermediate_tables is None:
        intermediate_tables = identify_intermediate_tables(db, target_table=target_table)

    intermediate_tables = list(intermediate_tables)
    channel_graphs = []

    for table in intermediate_tables:
        channels = {}
        for mode in (NODE_MODE, EDGE_MODE):
            if background_schema is None:
                table_modes = {
                    candidate_table: default_mode
                    for candidate_table in intermediate_tables
                }
            else:
                table_modes = {
                    candidate_table: background_schema.get(candidate_table, default_mode)
                    for candidate_table in intermediate_tables
                }
            table_modes[table] = mode

            graph, _ = build_schema_aware_graph(
                db,
                table_modes=table_modes,
                target_table=target_table,
                include_fk_features=include_fk_features,
            )
            graph = copy_task_tensors(reference_graph, graph, target_table)
            channels[mode] = graph

        channel_graphs.append(channels)

    return channel_graphs, intermediate_tables


def _move_channel_graphs(channel_graphs, device):
    return [
        {
            NODE_MODE: channels[NODE_MODE].to(device),
            EDGE_MODE: channels[EDGE_MODE].to(device),
        }
        for channels in channel_graphs
    ]


@torch.no_grad()
def _initialize_lazy_modules(model, channel_graphs, schema_gate):
    model.eval()
    model(channel_graphs, schema_gate)


def train_schema_gate_joint(
    channel_graphs,
    intermediate_tables,
    target_table,
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    weight_decay=0.0,
    lambda_disc=1e-3,
    init=0.0,
    temperature=1.0,
    device=None,
    seed=42,
    log_every=10,
    verbose=True,
):
    """
    Jointly train GNN channel models and schema gates with train loss only.
    """
    set_schema_gate_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not intermediate_tables:
        raise ValueError("No intermediate tables available for schema_gate_joint.")

    channel_graphs = _move_channel_graphs(channel_graphs, device)
    reference_graph = channel_graphs[0][NODE_MODE]

    if num_classes is None:
        num_classes = infer_num_classes(reference_graph, target_table)

    schema_gate = SchemaGate(
        intermediate_tables,
        init=init,
        temperature=temperature,
    ).to(device)
    model = SchemaGateJointModel(
        table_names=intermediate_tables,
        channel_graphs=channel_graphs,
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        target_node=target_table,
    ).to(device)

    _initialize_lazy_modules(model, channel_graphs, schema_gate)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(schema_gate.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    y = reference_graph[target_table].y
    train_idx = reference_graph[target_table].train_mask

    if train_idx.sum().item() == 0:
        raise ValueError(f"No training labels available for target table '{target_table}'.")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        schema_gate.train()
        optimizer.zero_grad()

        out = model(channel_graphs, schema_gate)
        task_loss = criterion(out[train_idx], y[train_idx])
        disc_loss = schema_gate.discreteness_loss()
        loss = task_loss + lambda_disc * disc_loss
        loss.backward()
        optimizer.step()

        epoch_record = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "task_loss": float(task_loss.item()),
            "disc_loss": float(disc_loss.item()),
            "probs": schema_gate.probs().detach().cpu().tolist(),
            "schema": schema_gate.discrete_schema(),
        }
        history.append(epoch_record)

        should_log = verbose and (epoch == 1 or epoch == epochs or epoch % log_every == 0)
        if should_log:
            prob_text = {
                table: round(prob, 4)
                for table, prob in zip(intermediate_tables, epoch_record["probs"])
            }
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"task={task_loss.item():.4f} | "
                f"disc={disc_loss.item():.4f}"
            )
            print(f"  gate_probs={prob_text}")
            print(f"  discrete_schema={epoch_record['schema']}")

    learned_probs = {
        table: prob
        for table, prob in zip(
            intermediate_tables,
            schema_gate.probs().detach().cpu().tolist(),
        )
    }
    discrete_schema = schema_gate.discrete_schema()

    return model, schema_gate, learned_probs, discrete_schema, history


def rebuild_and_retrain_discrete_schema(
    db,
    reference_graph,
    target_table,
    discrete_schema,
    num_classes=None,
    hidden_dim=64,
    epochs=30,
    lr=0.01,
    weight_decay=0.0,
    device=None,
    seed=42,
    include_fk_features=False,
    verbose=True,
    selection_metric="accuracy",
):
    set_schema_gate_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph, node_id_map = build_schema_aware_graph(
        db,
        table_modes=discrete_schema,
        target_table=target_table,
        include_fk_features=include_fk_features,
    )
    graph = copy_task_tensors(reference_graph, graph, target_table)
    graph = graph.to(device)

    if num_classes is None:
        num_classes = infer_num_classes(graph, target_table)

    model = GNN(
        metadata=graph.metadata(),
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        target_node=target_table,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    train_model(
        model,
        graph,
        target_table,
        optimizer,
        criterion,
        epochs=epochs,
        verbose=verbose,
        selection_metric=selection_metric,
    )
    metrics = evaluate_model(model, graph, target_table, split="test", return_metrics=True)

    return graph, node_id_map, model, metrics


def schema_gate_joint(
    db,
    initial_graph,
    target_table,
    intermediate_tables=None,
    num_classes=None,
    hidden_dim=64,
    joint_epochs=30,
    final_epochs=30,
    lr=0.01,
    weight_decay=0.0,
    lambda_disc=1e-3,
    init=0.0,
    temperature=1.0,
    device=None,
    seed=42,
    log_every=10,
    include_fk_features=False,
    verbose=True,
):
    """
    End-to-end simple joint schema gate method.

    This intentionally uses train-loss joint optimization only. After gate
    learning, it rebuilds the discrete schema graph and retrains a normal GNN
    from scratch with the existing pipeline.
    """
    channel_graphs, intermediate_tables = build_schema_gate_channels(
        db=db,
        reference_graph=initial_graph,
        target_table=target_table,
        intermediate_tables=intermediate_tables,
        include_fk_features=include_fk_features,
    )

    joint_model, schema_gate, learned_probs, discrete_schema, history = (
        train_schema_gate_joint(
            channel_graphs=channel_graphs,
            intermediate_tables=intermediate_tables,
            target_table=target_table,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            epochs=joint_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lambda_disc=lambda_disc,
            init=init,
            temperature=temperature,
            device=device,
            seed=seed,
            log_every=log_every,
            verbose=verbose,
        )
    )

    final_graph, final_node_id_map, final_model, metrics = (
        rebuild_and_retrain_discrete_schema(
            db=db,
            reference_graph=initial_graph,
            target_table=target_table,
            discrete_schema=discrete_schema,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            epochs=final_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
            include_fk_features=include_fk_features,
            verbose=verbose,
        )
    )

    return {
        "joint_model": joint_model,
        "schema_gate": schema_gate,
        "learned_probs": learned_probs,
        "discrete_schema": discrete_schema,
        "history": history,
        "final_graph": final_graph,
        "final_node_id_map": final_node_id_map,
        "final_model": final_model,
        "metrics": metrics,
    }
