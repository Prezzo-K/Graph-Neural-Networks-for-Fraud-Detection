import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, to_hetero


# Attention-based alternative to SAGE: instead of averaging neighbours, each
# one gets a learned weight. A user with fraud_rate=0.8 gets far more attention
# than one with fraud_rate=0.0 — SAGE would average them equally.
#
# Layer 1: 4 heads × 16 dims → concat → 64 dims.
# Layer 2: 1 head, concat=False → 64 dims.
#
# add_self_loops=False is required: to_hetero creates bipartite subgraphs per
# relation, so a node cannot self-loop into a different node type's space.


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            (-1, -1), hidden_channels // heads,
            heads=heads, dropout=dropout, add_self_loops=False
        )
        self.conv2 = GATConv(
            (-1, -1), hidden_channels,
            heads=1, concat=False, dropout=dropout, add_self_loops=False
        )
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x)


def create_gat(data, hidden_channels=64, out_channels=1, heads=4, dropout=0.2):
    """
    Returns a to_hetero-wrapped GAT model.
    Forward signature: model(data.x_dict, data.edge_index_dict)['transaction']
    """
    model = GAT(hidden_channels, out_channels, heads=heads, dropout=dropout)
    return to_hetero(model, data.metadata(), aggr='sum')
