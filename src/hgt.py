import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


# HGTConv is natively heterogeneous: the attention function itself differs per
# (src_type, edge_type, dst_type) — not just the weights, but the computation.
#
#   SAGE: same aggregation for every relation (different learned weights)
#   GAT:  attention varies per neighbour, but the same attention function is
#         shared across relations when wrapped with to_hetero
#   HGT:  completely separate Q/K/V matrices per relation type
#
# Does not use to_hetero — takes x_dict and edge_index_dict directly.


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, heads=4, metadata=None, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # -1 triggers lazy initialisation: input size inferred from first forward pass
        self.conv1 = HGTConv(-1, hidden_channels, metadata, heads=heads)
        self.conv2 = HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                  for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return {'transaction': self.classifier(x_dict['transaction'])}


def create_hgt(data, hidden_channels=64, out_channels=1, heads=4, dropout=0.2):
    """
    Returns an HGT model (natively heterogeneous, no to_hetero wrapper needed).
    Forward signature: model(data.x_dict, data.edge_index_dict)['transaction']
    """
    return HGT(
        hidden_channels, out_channels,
        heads=heads, metadata=data.metadata(), dropout=dropout
    )
