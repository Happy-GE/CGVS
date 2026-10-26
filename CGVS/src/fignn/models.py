from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv

class GNN(nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, target_node):
        super().__init__()
        self.target_node = target_node

        self.convs = nn.ModuleList()
        for _ in range(2):  # 2 layers
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='mean')
            self.convs.append(conv)

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: x.relu() for k, x in x_dict.items()}
        return self.classifier(x_dict[self.target_node])