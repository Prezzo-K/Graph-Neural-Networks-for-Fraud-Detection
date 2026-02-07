"""Training utilities for GNN models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm


class GNNTrainer:
    """
    Trainer for Graph Neural Network models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4
    ):
        """
        Initialize GNN trainer.
        
        Parameters
        ----------
        model : nn.Module
            GNN model to train.
        device : str
            Device to use ('cpu' or 'cuda').
        learning_rate : float
            Learning rate for optimizer.
        weight_decay : float
            Weight decay (L2 regularization).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Use weighted cross-entropy to handle class imbalance
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        data: Data,
        train_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data.
        train_mask : torch.Tensor, optional
            Boolean mask indicating training edges.
        
        Returns
        -------
        loss : float
            Training loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Forward pass
        out = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Compute loss
        if train_mask is not None:
            loss = self.criterion(out[train_mask], data.y[train_mask])
        else:
            loss = self.criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Data,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate model.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data.
        mask : torch.Tensor, optional
            Boolean mask indicating edges to evaluate.
        
        Returns
        -------
        loss : float
            Evaluation loss.
        accuracy : float
            Evaluation accuracy.
        predictions : np.ndarray
            Predicted labels.
        probabilities : np.ndarray
            Prediction probabilities.
        """
        self.model.eval()
        data = data.to(self.device)
        
        # Forward pass
        out = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Select relevant edges
        if mask is not None:
            out_masked = out[mask]
            y_masked = data.y[mask]
        else:
            out_masked = out
            y_masked = data.y
        
        # Compute loss
        loss = self.criterion(out_masked, y_masked).item()
        
        # Compute accuracy
        pred = out_masked.argmax(dim=1)
        accuracy = (pred == y_masked).float().mean().item()
        
        # Get probabilities
        probs = torch.softmax(out_masked, dim=1)
        
        return loss, accuracy, pred.cpu().numpy(), probs.cpu().numpy()
    
    def fit(
        self,
        data: Data,
        train_mask: torch.Tensor,
        val_mask: Optional[torch.Tensor] = None,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with optional validation.
        
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data.
        train_mask : torch.Tensor
            Boolean mask for training edges.
        val_mask : torch.Tensor, optional
            Boolean mask for validation edges.
        epochs : int
            Number of training epochs.
        early_stopping_patience : int
            Patience for early stopping based on validation loss.
        verbose : bool
            Whether to print progress.
        
        Returns
        -------
        history : dict
            Training history with losses and metrics.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        iterator = tqdm(range(epochs)) if verbose else range(epochs)
        
        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(data, train_mask)
            
            # Evaluate on training set
            _, train_acc, _, _ = self.evaluate(data, train_mask)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate if validation mask provided
            if val_mask is not None:
                val_loss, val_acc, _, _ = self.evaluate(data, val_mask)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose and epoch % 10 == 0:
                    iterator.set_description(
                        f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                        f"Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, "
                        f"Val Acc={val_acc:.4f}"
                    )
            else:
                if verbose and epoch % 10 == 0:
                    iterator.set_description(
                        f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                        f"Train Acc={train_acc:.4f}"
                    )
        
        # Restore best model if we did early stopping
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
