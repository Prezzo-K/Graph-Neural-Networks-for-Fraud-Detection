import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, to_hetero


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
# on small entity nodes (5 locations, 6 categories).
#
# How GATConv works:
# Instead of a plain mean, each neighbour gets a learned attention weight —
# how much should this neighbour influence the current node?
#   α_vu = softmax(LeakyReLU(a^T [W·h_v || W·h_u]))
#   h_v  = σ( Σ_{u∈N(v)} α_vu · W · h_u )
#
# Why this beats SAGE for fraud: a user with fraud_rate=0.8 should get far more
# attention than a user with fraud_rate=0.0, even if both are neighbours.
# SAGE would average them; GAT learns to weight the suspicious one higher.
#
# heads=4: 4 independent attention functions running in parallel.
# Each head can specialise (e.g. one head for amount patterns, another for timing).
# hidden_channels // heads = 64 // 4 = 16 dims per head → concat → 64 total.
# Layer 2 uses heads=1, concat=False to collapse back to a single 64-d vector.
#
# add_self_loops=False: MANDATORY for heterogeneous graphs. to_hetero creates
# bipartite subgraphs for each relation (e.g. user → transaction). A user node
# cannot have a self-loop in the transaction node space — it doesn't exist there.
# Leaving this True would cause a runtime error.
#
# dropout=0.2 on attention coefficients: regularises which neighbours get
# attended to, preventing the model from always fixating on the same neighbours.
#
# Wrapped with to_hetero → each relation gets its own attention weight matrices.
# aggr='sum': outputs from all relations targeting the same node type are summed.
# ─────────────────────────────────────────────────────────────────────────────


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        # Layer 1: multi-head attention, concat=True (default)
        # output dim = (hidden_channels // heads) * heads = hidden_channels
        self.conv1 = GATConv(
            (-1, -1), hidden_channels // heads,
            heads=heads, dropout=dropout, add_self_loops=False
        )
        # Layer 2: single head, concat=False → output dim = hidden_channels
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
