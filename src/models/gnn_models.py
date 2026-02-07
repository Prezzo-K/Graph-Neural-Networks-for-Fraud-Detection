"""Graph Neural Network model implementations using PyTorch Geometric."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from typing import Optional


class GCNModel(nn.Module):
    """
    Graph Convolutional Network (GCN) for fraud detection.
    
    Based on Kipf & Welling (2017): "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize GCN model.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        hidden_channels : int
            Hidden layer dimension.
        num_layers : int
            Number of GCN layers.
        dropout : float
            Dropout rate.
        edge_dim : int, optional
            Edge feature dimension (not used in basic GCN).
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Edge classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)  # Binary classification
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge features [num_edges, edge_dim].
        
        Returns
        -------
        out : torch.Tensor
            Edge predictions [num_edges, 2].
        """
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For edge classification, concatenate source and target node embeddings
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        
        # Classify edges
        out = self.edge_classifier(edge_embeddings)
        
        return out


class GATModel(nn.Module):
    """
    Graph Attention Network (GAT) for fraud detection.
    
    Based on Veličković et al. (2018): "Graph Attention Networks"
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize GAT model.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        hidden_channels : int
            Hidden layer dimension.
        num_layers : int
            Number of GAT layers.
        heads : int
            Number of attention heads.
        dropout : float
            Dropout rate.
        edge_dim : int, optional
            Edge feature dimension.
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim
            )
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )
        
        # Edge classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge features [num_edges, edge_dim].
        
        Returns
        -------
        out : torch.Tensor
            Edge predictions [num_edges, 2].
        """
        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For edge classification
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        out = self.edge_classifier(edge_embeddings)
        
        return out


class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for fraud detection.
    
    Based on Hamilton et al. (2017): "Inductive Representation Learning on Large Graphs"
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        edge_dim: Optional[int] = None
    ):
        """
        Initialize GraphSAGE model.
        
        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        hidden_channels : int
            Hidden layer dimension.
        num_layers : int
            Number of SAGE layers.
        dropout : float
            Dropout rate.
        edge_dim : int, optional
            Edge feature dimension (not used in GraphSAGE).
        """
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # Edge classification head
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices [2, num_edges].
        edge_attr : torch.Tensor, optional
            Edge features (not used).
        
        Returns
        -------
        out : torch.Tensor
            Edge predictions [num_edges, 2].
        """
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # For edge classification
        edge_embeddings = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        out = self.edge_classifier(edge_embeddings)
        
        return out
