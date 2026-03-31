from src.graphsage import create_graphsage
from src.gat import create_gat
from src.hgt import create_hgt


def create_model(model_type, data, hidden_channels=64, out_channels=1):
    """
    Factory function — returns the requested model ready to train.

    All three models expose the same forward interface:
        model(data.x_dict, data.edge_index_dict)['transaction']  →  (N, 1) logits

    Args:
        model_type:      'sage' | 'gat' | 'hgt'
        data:            HeteroData object (needed for metadata and lazy init)
        hidden_channels: embedding size (default 64)
        out_channels:    output size, 1 for binary classification
    """
    if model_type == 'sage':
        return create_graphsage(data, hidden_channels, out_channels)
    elif model_type == 'gat':
        return create_gat(data, hidden_channels, out_channels)
    elif model_type == 'hgt':
        return create_hgt(data, hidden_channels, out_channels)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose: 'sage', 'gat', 'hgt'")
