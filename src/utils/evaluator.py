"""Model evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple
import pandas as pd


class ModelEvaluator:
    """
    Comprehensive evaluation for fraud detection models.
    """
    
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        y_proba : np.ndarray, optional
            Prediction probabilities for positive class.
        
        Returns
        -------
        metrics : dict
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (if probabilities provided)
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        return metrics
    
    @staticmethod
    def print_evaluation(
        metrics: Dict[str, float],
        model_name: str = "Model"
    ):
        """
        Pretty print evaluation metrics.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of evaluation metrics.
        model_name : str
            Name of the model being evaluated.
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:       {metrics['accuracy']:.4f}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1 Score:       {metrics['f1_score']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC AUC:        {metrics['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:6d}  FP: {metrics['false_positives']:6d}")
        print(f"  FN: {metrics['false_negatives']:6d}  TP: {metrics['true_positives']:6d}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create a comparison table for multiple models.
        
        Parameters
        ----------
        results : dict
            Dictionary mapping model names to their evaluation metrics.
        
        Returns
        -------
        comparison_df : pd.DataFrame
            DataFrame with model comparison.
        """
        # Select key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        comparison_data = []
        for model_name, metrics in results.items():
            row = {'Model': model_name}
            for metric in key_metrics:
                if metric in metrics:
                    row[metric.replace('_', ' ').title()] = metrics[metric]
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Round numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        return df
    
    @staticmethod
    def create_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        
        Returns
        -------
        report : str
            Classification report string.
        """
        return classification_report(
            y_true, y_pred,
            target_names=['Legitimate', 'Fraud'],
            zero_division=0
        )
