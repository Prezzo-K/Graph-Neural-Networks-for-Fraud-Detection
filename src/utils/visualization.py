"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, List
from pathlib import Path


def plot_results(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Parameters
    ----------
    history : dict
        Training history with losses and accuracies.
    save_path : str, optional
        Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot model comparison.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with model comparison results.
    save_path : str, optional
        Path to save the plot.
    """
    # Select numerical columns (exclude 'Model' column)
    metrics = [col for col in results_df.columns if col != 'Model']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = width * (i - len(metrics) / 2)
        ax.bar(
            x + offset,
            results_df[metric],
            width,
            label=metric
        )
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
):
    """
    Plot confusion matrices for multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metrics.
    save_path : str, optional
        Path to save the plot.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, metrics) in zip(axes, results.items()):
        cm = np.array([
            [metrics['true_negatives'], metrics['false_positives']],
            [metrics['false_negatives'], metrics['true_positives']]
        ])
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud']
        )
        ax.set_title(f'{model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_graph(
    G: nx.Graph,
    node_colors: Optional[List] = None,
    edge_colors: Optional[List] = None,
    save_path: Optional[str] = None,
    max_nodes: int = 100
):
    """
    Visualize transaction graph.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph to visualize.
    node_colors : list, optional
        Colors for nodes.
    edge_colors : list, optional
        Colors for edges (e.g., red for fraudulent).
    save_path : str, optional
        Path to save the plot.
    max_nodes : int
        Maximum number of nodes to display (for performance).
    """
    # Sample if graph is too large
    if len(G.nodes()) > max_nodes:
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)
        print(f"Displaying subgraph with {max_nodes} nodes")
    
    plt.figure(figsize=(15, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    if node_colors is None:
        node_colors = 'lightblue'
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=300,
        alpha=0.7
    )
    
    # Draw edges
    if edge_colors is None:
        edge_colors = 'gray'
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        alpha=0.3,
        arrows=True,
        arrowsize=10
    )
    
    # Draw labels (optional, can be overwhelming for large graphs)
    if len(G.nodes()) <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Transaction Graph Visualization')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    save_path: Optional[str] = None
):
    """
    Create radar chart comparing multiple models across metrics.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model names to their metrics.
    metrics : list, optional
        List of metrics to compare. If None, use common metrics.
    save_path : str, optional
        Path to save the plot.
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Filter metrics that exist in all results
    available_metrics = [m for m in metrics if all(m in r for r in results.values())]
    
    if not available_metrics:
        print("No common metrics found for comparison")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model_name, model_results in results.items():
        values = [model_results[m] for m in available_metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.title('Model Performance Comparison', y=1.08, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
