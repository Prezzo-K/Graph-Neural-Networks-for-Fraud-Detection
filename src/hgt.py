import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


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
# How HGTConv works:
# Transformer-style attention where the attention mechanism explicitly conditions
# on BOTH the source node type AND the edge type. This means "user→transaction"
# attention is computed completely differently from "location→transaction"
# attention — different Q/K/V weight matrices per (src_type, edge_type, dst_type).
#
#   h_v = AGG_{τ} ( Σ_{u∈N_τ(v)} Attn(τ, u, v) · MSG(τ, h_u) )
#   where τ is the relation type
#
# Why this is the most expressive for our graph:
# SAGE:  same aggregation logic for every relation (just different weights)
# GAT:   attention varies per neighbour, but the attention function is the same
#        across all relations when wrapped with to_hetero
# HGT:   attention function itself changes per (src_type, edge_type, dst_type) —
#        a fundamentally different computation for "user performs transaction" vs
#        "location is_site_of transaction". Designed specifically for hetero graphs.
#
# Does NOT use to_hetero — HGTConv is natively heterogeneous and takes
# x_dict and edge_index_dict directly. metadata() provides the node/edge type
# lists so HGT can build the right weight matrices at init time.
#
# heads=4: 4 relation-aware attention heads running in parallel.
#
# Returns a dict {'transaction': logits} so train.py stays identical across
# all three models — always access output via ['transaction'].
# ─────────────────────────────────────────────────────────────────────────────


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, heads=4, metadata=None, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # -1 triggers lazy initialisation: input size inferred from first forward pass
        self.conv1 = HGTConv(-1, hidden_channels, metadata, heads=heads)
        self.conv2 = HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: v.relu() for k, v in x_dict.items()}
        x_dict = {k: self.dropout(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        # Return as dict so train.py can use ['transaction'] uniformly across all models
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
