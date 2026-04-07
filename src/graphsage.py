import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear, to_hetero


# Two-layer architecture: Layer 1 aggregates direct neighbours (user, location,
# merchant category). Layer 2 expands the receptive field so each transaction
# also sees other transactions by the same user — critical for detecting fraud
# rings. Three+ layers risk over-smoothing on small entity sets (5 locations).
#
# to_hetero replicates the layers once per (src, rel, dst) triplet, giving each
# relation its own weight matrices. aggr='sum' combines outputs from all
# relation types targeting the same node type.


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x)


def create_graphsage(data, hidden_channels=64, out_channels=1):
    """
    Returns a to_hetero-wrapped GraphSAGE model.
    Forward signature: model(data.x_dict, data.edge_index_dict)['transaction']
    """
    model = GraphSAGE(hidden_channels, out_channels)
    return to_hetero(model, data.metadata(), aggr='sum')
