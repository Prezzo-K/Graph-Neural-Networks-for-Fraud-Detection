"""Baseline machine learning models for comparison."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, Tuple
import xgboost as xgb


class BaselineMLModels:
    """
    Collection of traditional machine learning models for fraud detection.
    
    These serve as baselines to compare against GNN performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline models.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all baseline models."""
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            scale_pos_weight=10,  # Handle class imbalance
            n_jobs=-1
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_name: str = 'all'
    ):
        """
        Train baseline model(s).
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        model_name : str
            Name of model to train, or 'all' for all models.
        """
        if model_name == 'all':
            models_to_train = self.models.keys()
        elif model_name in self.models:
            models_to_train = [model_name]
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        for name in models_to_train:
            print(f"Training {name}...")
            self.models[name].fit(X_train, y_train)
            print(f"  {name} training complete")
    
    def predict(
        self,
        X_test: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        model_name : str
            Name of model to use for prediction.
        
        Returns
        -------
        predictions : np.ndarray
            Class predictions.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].predict(X_test)
    
    def predict_proba(
        self,
        X_test: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        model_name : str
            Name of model to use.
        
        Returns
        -------
        probabilities : np.ndarray
            Prediction probabilities [n_samples, n_classes].
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name].predict_proba(X_test)
    
    def get_model(self, model_name: str):
        """Get a specific model instance."""
        return self.models.get(model_name)
    
    def available_models(self):
        """Return list of available model names."""
        return list(self.models.keys())
