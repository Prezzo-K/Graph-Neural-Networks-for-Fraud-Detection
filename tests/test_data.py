"""Basic tests for data loading and graph construction."""

import sys
import pytest
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import generate_synthetic_data
from src.data.graph_builder import TransactionGraphBuilder


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    df, labels = generate_synthetic_data(
        n_transactions=100,
        fraud_rate=0.1,
        n_users=20,
        n_merchants=10,
        random_state=42
    )
    
    assert len(df) == 100
    assert len(labels) == 100
    assert 'sender' in df.columns
    assert 'receiver' in df.columns
    assert 'amount' in df.columns
    assert labels.sum() > 0  # Should have some fraud


def test_graph_construction():
    """Test graph building."""
    df, labels = generate_synthetic_data(
        n_transactions=50,
        n_users=10,
        n_merchants=5,
        random_state=42
    )
    
    builder = TransactionGraphBuilder()
    graph_data = builder.build_graph(df, labels)
    
    assert graph_data.num_nodes > 0
    assert graph_data.edge_index.shape[1] == 50
    assert graph_data.x is not None
    assert graph_data.y is not None
    assert graph_data.edge_attr is not None


def test_graph_features():
    """Test that graph features have correct shapes."""
    df, labels = generate_synthetic_data(
        n_transactions=30,
        n_users=15,
        n_merchants=8,
        random_state=42
    )
    
    builder = TransactionGraphBuilder()
    graph_data = builder.build_graph(df, labels)
    
    # Check dimensions
    assert graph_data.x.shape[0] == graph_data.num_nodes
    assert graph_data.x.shape[1] > 0  # Should have some features
    assert graph_data.edge_attr.shape[0] == graph_data.edge_index.shape[1]
    assert len(graph_data.y) == graph_data.edge_index.shape[1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
