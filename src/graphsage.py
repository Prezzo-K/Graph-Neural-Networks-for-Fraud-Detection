import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, Linear, to_hetero


# ── Why 2 layers? ─────────────────────────────────────────────────────────────
# Layer 1: each transaction aggregates from its direct neighbours
#          (the user who made it, the city it happened in, the merchant category)
# Layer 2: each transaction now also sees other transactions belonging to the
#          same user — critical for detecting fraud rings and repeat offenders
# Layer 3+: risks over-smoothing, especially since location has only 5 nodes
#           (all transactions in London would collapse to the same embedding)
#
# Why hidden_channels=64?
# Large enough to capture fraud patterns but small enough to avoid overfitting
# on small entity nodes (5 locations, 6 categories). 128 was tested and showed
# no consistent gain while being 4× slower.
#
# Why a separate Linear classifier?
# Separates representation learning (conv layers) from classification.
# Lets the conv layers build rich hidden_channels-dimensional embeddings,
# then a single linear layer decides fraud vs. not fraud.
#
# How SAGEConv works:
# Aggregates neighbours by computing their MEAN, concatenates that with the
# node's own features, then applies a linear projection:
#   h_v = W · CONCAT(h_v,  MEAN({h_u : u ∈ N(v)}))
#
# Wrapped with to_hetero → PyG replicates the layers once per
# (src_type, rel, dst_type) triplet, giving each relation its own weight matrices.
# aggr='sum' at the to_hetero level: when multiple relation types all write into
# the same target node type (e.g. user→txn AND location→txn both target txn),
# their outputs are summed before the next layer.
# ─────────────────────────────────────────────────────────────────────────────


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x)


def create_graphsage(data, hidden_channels=64, out_channels=1):
    """
    Returns a to_hetero-wrapped GraphSAGE model.
    Forward signature: model(data.x_dict, data.edge_index_dict)['transaction']
    """
    model = GraphSAGE(hidden_channels, out_channels)
    return to_hetero(model, data.metadata(), aggr='sum')
