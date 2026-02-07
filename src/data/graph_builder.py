"""Graph construction from transaction data."""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TransactionGraphBuilder:
    """
    Build a graph representation of financial transactions.
    
    This class constructs a heterogeneous graph with multiple node types:
    - Users (transaction senders)
    - Merchants (transaction receivers)
    - Transactions (individual transactions)
    
    Edges represent relationships between these entities.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.node_mappings = {}
        self.reverse_mappings = {}
        self.nx_graph = None
    
    def build_graph(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        use_transaction_nodes: bool = False
    ) -> Data:
        """
        Build a PyTorch Geometric graph from transaction data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction dataframe with sender, receiver, amount, etc.
        labels : pd.Series
            Binary fraud labels for transactions.
        use_transaction_nodes : bool, default=False
            If True, create transaction nodes in addition to user/merchant nodes.
            If False, create direct edges between users and merchants.
        
        Returns
        -------
        data : torch_geometric.data.Data
            PyTorch Geometric graph data object.
        """
        # Create node mappings
        self._create_node_mappings(df)
        
        if use_transaction_nodes:
            return self._build_heterogeneous_graph(df, labels)
        else:
            return self._build_homogeneous_graph(df, labels)
    
    def _create_node_mappings(self, df: pd.DataFrame):
        """Create mappings from entity IDs to node indices."""
        # Get unique entities
        unique_senders = df['sender'].unique()
        unique_receivers = df['receiver'].unique()
        all_entities = np.unique(np.concatenate([unique_senders, unique_receivers]))
        
        # Create node mappings
        self.node_mappings['entities'] = {entity: idx for idx, entity in enumerate(all_entities)}
        self.reverse_mappings['entities'] = {idx: entity for entity, idx in self.node_mappings['entities'].items()}
        
        print(f"Total nodes: {len(all_entities)}")
        print(f"  Unique senders: {len(unique_senders)}")
        print(f"  Unique receivers: {len(unique_receivers)}")
    
    def _build_homogeneous_graph(
        self,
        df: pd.DataFrame,
        labels: pd.Series
    ) -> Data:
        """
        Build a homogeneous graph where edges represent transactions.
        
        Nodes: All unique entities (senders + receivers)
        Edges: Transactions from sender to receiver
        Edge features: Transaction features
        Edge labels: Fraud labels
        """
        # Build edge list
        edge_list = []
        edge_features = []
        edge_labels = []
        
        for idx, row in df.iterrows():
            sender_id = self.node_mappings['entities'][row['sender']]
            receiver_id = self.node_mappings['entities'][row['receiver']]
            
            edge_list.append([sender_id, receiver_id])
            
            # Extract edge features
            features = self._extract_transaction_features(row)
            edge_features.append(features)
            edge_labels.append(labels.iloc[idx])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        edge_label = torch.tensor(edge_labels, dtype=torch.long)
        
        # Create node features (aggregate statistics for each entity)
        num_nodes = len(self.node_mappings['entities'])
        node_features = self._create_node_features(df, num_nodes)
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=edge_label,
            num_nodes=num_nodes
        )
        
        print(f"Graph created: {num_nodes} nodes, {len(edge_list)} edges")
        print(f"Node feature dimension: {x.shape[1]}")
        print(f"Edge feature dimension: {edge_attr.shape[1]}")
        
        return data
    
    def _build_heterogeneous_graph(
        self,
        df: pd.DataFrame,
        labels: pd.Series
    ) -> Data:
        """
        Build a heterogeneous graph with transaction nodes.
        
        This creates a more complex graph structure with three node types
        and multiple edge types.
        """
        # For simplicity, we'll use the homogeneous version
        # A full heterogeneous implementation would use torch_geometric.data.HeteroData
        return self._build_homogeneous_graph(df, labels)
    
    def _extract_transaction_features(self, row: pd.Series) -> List[float]:
        """Extract features from a transaction."""
        features = []
        
        # Amount
        if 'amount' in row:
            features.append(float(row['amount']))
        
        # Time features
        if 'timestamp' in row:
            features.append(float(row['timestamp']))
        
        # Balance features
        balance_cols = [
            'sender_balance_before', 'sender_balance_after',
            'receiver_balance_before', 'receiver_balance_after'
        ]
        for col in balance_cols:
            if col in row:
                features.append(float(row[col]))
        
        # Derived features
        if 'sender_balance_before' in row and 'amount' in row:
            # Fraction of balance spent
            if row['sender_balance_before'] > 0:
                features.append(float(row['amount']) / float(row['sender_balance_before']))
            else:
                features.append(0.0)
        
        # Ensure we have at least some features
        if not features:
            features = [1.0]
        
        return features
    
    def _create_node_features(
        self,
        df: pd.DataFrame,
        num_nodes: int
    ) -> np.ndarray:
        """
        Create node features by aggregating transaction statistics.
        
        For each entity (node), compute:
        - Total transaction count
        - Average transaction amount (sent/received)
        - Total amount sent/received
        - Standard deviation of amounts
        """
        # Initialize feature matrix
        features = np.zeros((num_nodes, 8))
        
        # Aggregate statistics for senders
        sender_stats = df.groupby('sender').agg({
            'amount': ['count', 'mean', 'sum', 'std']
        }).fillna(0)
        
        for sender, stats in sender_stats.iterrows():
            node_idx = self.node_mappings['entities'][sender]
            features[node_idx, 0] = stats[('amount', 'count')]
            features[node_idx, 1] = stats[('amount', 'mean')]
            features[node_idx, 2] = stats[('amount', 'sum')]
            features[node_idx, 3] = stats[('amount', 'std')]
        
        # Aggregate statistics for receivers
        receiver_stats = df.groupby('receiver').agg({
            'amount': ['count', 'mean', 'sum', 'std']
        }).fillna(0)
        
        for receiver, stats in receiver_stats.iterrows():
            node_idx = self.node_mappings['entities'][receiver]
            features[node_idx, 4] = stats[('amount', 'count')]
            features[node_idx, 5] = stats[('amount', 'mean')]
            features[node_idx, 6] = stats[('amount', 'sum')]
            features[node_idx, 7] = stats[('amount', 'std')]
        
        return features
    
    def to_networkx(self, data: Data) -> nx.DiGraph:
        """
        Convert PyTorch Geometric graph to NetworkX for visualization.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            PyTorch Geometric graph data.
        
        Returns
        -------
        G : nx.DiGraph
            NetworkX directed graph.
        """
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i, features=data.x[i].numpy())
        
        # Add edges
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(
                src, dst,
                features=data.edge_attr[i].numpy() if data.edge_attr is not None else None,
                label=data.y[i].item() if data.y is not None else None
            )
        
        self.nx_graph = G
        return G
