"""Main experiment script comparing GNNs with baseline models for fraud detection."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

from src.data import load_transaction_data, TransactionGraphBuilder
from src.data.data_loader import generate_synthetic_data
from src.models import GCNModel, GATModel, GraphSAGEModel, BaselineMLModels
from src.utils import GNNTrainer, ModelEvaluator
from src.utils.visualization import (
    plot_results, plot_comparison, plot_confusion_matrices, plot_metrics_comparison
)


def prepare_baseline_features(df, graph_data):
    """
    Prepare features for baseline ML models.
    
    Combines edge features with aggregated node features.
    """
    # Get edge features
    edge_features = graph_data.edge_attr.numpy()
    
    # Get node features for source and target nodes
    edge_index = graph_data.edge_index.numpy()
    node_features = graph_data.x.numpy()
    
    src_node_features = node_features[edge_index[0]]
    dst_node_features = node_features[edge_index[1]]
    
    # Combine all features
    combined_features = np.hstack([
        edge_features,
        src_node_features,
        dst_node_features
    ])
    
    return combined_features


def split_data(graph_data, train_ratio=0.7, val_ratio=0.15, random_state=42):
    """
    Split graph data into train, validation, and test sets.
    """
    n_edges = graph_data.edge_index.shape[1]
    indices = np.arange(n_edges)
    
    # First split: train and temp (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_state,
        stratify=graph_data.y.numpy()
    )
    
    # Second split: val and test
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=random_state,
        stratify=graph_data.y[temp_idx].numpy()
    )
    
    # Create masks
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_mask = torch.zeros(n_edges, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask


def run_gnn_experiments(
    graph_data,
    train_mask,
    val_mask,
    test_mask,
    hidden_dim=64,
    num_layers=2,
    epochs=200,
    device='cpu'
):
    """
    Run experiments with different GNN architectures.
    """
    results = {}
    
    in_channels = graph_data.x.shape[1]
    edge_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else None
    
    gnn_models = {
        'GCN': GCNModel(in_channels, hidden_dim, num_layers),
        'GAT': GATModel(in_channels, hidden_dim, num_layers, edge_dim=edge_dim),
        'GraphSAGE': GraphSAGEModel(in_channels, hidden_dim, num_layers)
    }
    
    for model_name, model in gnn_models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Initialize trainer
        trainer = GNNTrainer(model, device=device, learning_rate=0.01)
        
        # Train
        history = trainer.fit(
            graph_data,
            train_mask,
            val_mask,
            epochs=epochs,
            early_stopping_patience=20,
            verbose=True
        )
        
        # Evaluate on test set
        _, _, y_pred, y_proba = trainer.evaluate(graph_data, test_mask)
        y_true = graph_data.y[test_mask].numpy()
        y_proba_positive = y_proba[:, 1]  # Probability of fraud class
        
        # Compute metrics
        metrics = ModelEvaluator.evaluate(y_true, y_pred, y_proba_positive)
        results[model_name] = metrics
        
        # Print results
        ModelEvaluator.print_evaluation(metrics, model_name)
    
    return results


def run_baseline_experiments(
    X_train, y_train,
    X_test, y_test
):
    """
    Run experiments with baseline ML models.
    """
    results = {}
    
    # Initialize baseline models
    baseline = BaselineMLModels()
    
    # Train all models
    print(f"\n{'='*60}")
    print("Training Baseline Models")
    print(f"{'='*60}")
    
    baseline.fit(X_train, y_train, model_name='all')
    
    # Evaluate each model
    for model_name in baseline.available_models():
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = baseline.predict(X_test, model_name)
        y_proba = baseline.predict_proba(X_test, model_name)[:, 1]
        
        # Compute metrics
        metrics = ModelEvaluator.evaluate(y_test, y_pred, y_proba)
        results[model_name] = metrics
        
        # Print results
        ModelEvaluator.print_evaluation(metrics, model_name)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Graph Neural Networks for Fraud Detection'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to transaction data CSV file'
    )
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of loading from file'
    )
    parser.add_argument(
        '--n-transactions',
        type=int,
        default=10000,
        help='Number of transactions for synthetic data'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension for GNN models'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of layers in GNN models'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load or generate data
    print("Loading data...")
    if args.use_synthetic or args.data_path is None:
        df, labels = generate_synthetic_data(n_transactions=args.n_transactions)
    else:
        df, labels = load_transaction_data(args.data_path)
    
    # Build graph
    print("\nBuilding transaction graph...")
    builder = TransactionGraphBuilder()
    graph_data = builder.build_graph(df, labels)
    
    # Split data
    print("\nSplitting data...")
    train_mask, val_mask, test_mask = split_data(graph_data)
    
    print(f"Train edges: {train_mask.sum().item()}")
    print(f"Val edges: {val_mask.sum().item()}")
    print(f"Test edges: {test_mask.sum().item()}")
    
    # Run GNN experiments
    gnn_results = run_gnn_experiments(
        graph_data,
        train_mask,
        val_mask,
        test_mask,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        device=args.device
    )
    
    # Prepare features for baseline models
    print("\nPreparing features for baseline models...")
    X = prepare_baseline_features(df, graph_data)
    y = graph_data.y.numpy()
    
    X_train = X[train_mask.numpy()]
    X_test = X[test_mask.numpy()]
    y_train = y[train_mask.numpy()]
    y_test = y[test_mask.numpy()]
    
    # Run baseline experiments
    baseline_results = run_baseline_experiments(X_train, y_train, X_test, y_test)
    
    # Combine all results
    all_results = {**gnn_results, **baseline_results}
    
    # Create comparison visualizations
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    comparison_df = ModelEvaluator.compare_models(all_results)
    print("\n", comparison_df.to_string(index=False))
    
    # Save results
    comparison_df.to_csv(output_dir / 'comparison_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'comparison_results.csv'}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison(comparison_df, save_path=output_dir / 'model_comparison.png')
    plot_confusion_matrices(all_results, save_path=output_dir / 'confusion_matrices.png')
    plot_metrics_comparison(all_results, save_path=output_dir / 'metrics_radar.png')
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print(f"Results and visualizations saved to {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
