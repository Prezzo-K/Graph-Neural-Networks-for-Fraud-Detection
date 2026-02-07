"""Data loader for transaction datasets."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def load_transaction_data(
    data_path: str,
    sample_frac: Optional[float] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess transaction data from a CSV file.
    
    This function handles synthetic financial transaction datasets commonly found
    on Kaggle, such as the PaySim dataset or similar fraud detection datasets.
    
    Parameters
    ----------
    data_path : str
        Path to the CSV file containing transaction data.
    sample_frac : float, optional
        Fraction of data to sample (useful for large datasets). If None, use all data.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        Preprocessed transaction dataframe with features.
    labels : pd.Series
        Binary fraud labels (1 for fraud, 0 for legitimate).
    
    Examples
    --------
    >>> df, labels = load_transaction_data('data/transactions.csv', sample_frac=0.1)
    """
    # Load the data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Sample if requested
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    
    # Common column name mappings for various datasets
    fraud_col_candidates = ['isFraud', 'is_fraud', 'fraud', 'Class', 'label']
    fraud_col = None
    for col in fraud_col_candidates:
        if col in df.columns:
            fraud_col = col
            break
    
    if fraud_col is None:
        raise ValueError(
            f"Could not find fraud label column. Expected one of: {fraud_col_candidates}"
        )
    
    # Extract labels
    labels = df[fraud_col].astype(int)
    
    # Common preprocessing steps
    # Handle timestamp columns if present
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'step' in col.lower()]
    if time_cols:
        for col in time_cols:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_day'] = df[col].dt.dayofweek
                except:
                    pass  # Keep original if conversion fails
    
    # Ensure we have required columns for graph construction
    # Common patterns in transaction datasets
    if 'nameOrig' not in df.columns and 'sender' not in df.columns:
        # Create synthetic sender IDs if not present
        df['sender'] = df.index.map(lambda x: f'user_{x % 1000}')
    
    if 'nameDest' not in df.columns and 'receiver' not in df.columns:
        # Create synthetic receiver IDs if not present
        df['receiver'] = df.index.map(lambda x: f'merchant_{x % 500}')
    
    # Standardize column names
    col_mapping = {
        'nameOrig': 'sender',
        'nameDest': 'receiver',
        'oldbalanceOrg': 'sender_balance_before',
        'newbalanceOrig': 'sender_balance_after',
        'oldbalanceDest': 'receiver_balance_before',
        'newbalanceDest': 'receiver_balance_after'
    }
    
    for old_name, new_name in col_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    print(f"Loaded {len(df)} transactions")
    print(f"Fraud rate: {labels.mean():.4f} ({labels.sum()} fraudulent transactions)")
    print(f"Features: {df.columns.tolist()}")
    
    return df, labels


def generate_synthetic_data(
    n_transactions: int = 10000,
    fraud_rate: float = 0.1,
    n_users: int = 500,
    n_merchants: int = 200,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic transaction data for testing.
    
    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate.
    fraud_rate : float
        Proportion of fraudulent transactions.
    n_users : int
        Number of unique users.
    n_merchants : int
        Number of unique merchants.
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    df : pd.DataFrame
        Synthetic transaction dataframe.
    labels : pd.Series
        Fraud labels.
    """
    np.random.seed(random_state)
    
    # Generate transactions
    data = {
        'transaction_id': np.arange(n_transactions),
        'sender': [f'user_{np.random.randint(0, n_users)}' for _ in range(n_transactions)],
        'receiver': [f'merchant_{np.random.randint(0, n_merchants)}' for _ in range(n_transactions)],
        'amount': np.random.exponential(100, n_transactions),
        'timestamp': np.random.randint(0, 24*30, n_transactions)  # Hours in a month
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels
    labels = np.zeros(n_transactions, dtype=int)
    n_fraud = int(n_transactions * fraud_rate)
    fraud_indices = np.random.choice(n_transactions, n_fraud, replace=False)
    labels[fraud_indices] = 1
    
    # Make fraudulent transactions have distinctive patterns
    df.loc[fraud_indices, 'amount'] *= 2  # Larger amounts
    
    # Add balance features
    df['sender_balance_before'] = np.random.uniform(0, 10000, n_transactions)
    df['sender_balance_after'] = df['sender_balance_before'] - df['amount']
    df['receiver_balance_before'] = np.random.uniform(0, 50000, n_transactions)
    df['receiver_balance_after'] = df['receiver_balance_before'] + df['amount']
    
    print(f"Generated {n_transactions} synthetic transactions")
    print(f"Fraud rate: {labels.mean():.4f}")
    
    return df, pd.Series(labels)
