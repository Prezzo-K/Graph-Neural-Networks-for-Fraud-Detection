import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def preprocess_and_create_graph(csv_path, output_path=None):
    """
    Loads the Fraud Detection Transactions Dataset and converts it into a 
    PyTorch Geometric HeteroData object.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Temporal Feature Engineering and Sorting
    print("Engineering temporal features and sorting by timestamp...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # 2. Define Features
    # Numeric columns to scale
    # Failed_Transaction_Count_7d is excluded: it near-perfectly predicts fraud
    # and would dominate the model, making comparison with traditional ML unfair
    # (the traditional ML notebook drops it for the same reason)
    num_cols = [
        'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
        'Avg_Transaction_Amount_7d', 'Card_Age', 'Transaction_Distance',
        'Hour', 'Day', 'Month', 'DayOfWeek'
    ]
    
    # Categorical columns to One-Hot Encode (for Transaction nodes)
    cat_cols = [
        'Transaction_Type', 'Device_Type', 'Card_Type', 'Authentication_Method'
    ]
    
    # Binary columns (keep as is)
    bin_cols = ['IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']

    # 3. Scaling and Encoding
    print("Scaling numerical features and encoding categoricals...")
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # One-hot encoding for the categorical columns
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    
    # Drop columns that won't be used as features for the Transaction node
    # Risk_Score: data leakage (derived from fraud label)
    # Failed_Transaction_Count_7d: near-perfectly predicts fraud, excluded for fair comparison
    cols_to_drop = [
        'Transaction_ID', 'User_ID', 'Timestamp', 'Location', 'Merchant_Category',
        'Risk_Score', 'Failed_Transaction_Count_7d', 'Fraud_Label'
    ]
    feature_cols = [c for c in df_encoded.columns if c not in cols_to_drop]
    
    transaction_features = df_encoded[feature_cols].values.astype(np.float32)
    labels = df['Fraud_Label'].values.astype(np.int64)

    # 4. Entity Mapping (for nodes and edges)
    print("Mapping entities to indices...")
    
    def get_mapping(series):
        unique_values = series.unique()
        mapping = {val: i for i, val in enumerate(unique_values)}
        return mapping, unique_values

    user_map, unique_users = get_mapping(df['User_ID'])
    loc_map, unique_locs = get_mapping(df['Location'])
    cat_map, unique_cats = get_mapping(df['Merchant_Category'])

    # Create edge indices
    user_indices = df['User_ID'].map(user_map).values
    loc_indices = df['Location'].map(loc_map).values
    cat_indices = df['Merchant_Category'].map(cat_map).values
    txn_indices = np.arange(len(df))

    # 5. Build HeteroData Object and Create Temporal Masks
    print("Building HeteroData object and creating temporal masks...")
    num_transactions = len(df)
    train_end = int(0.7 * num_transactions)
    val_end = int(0.85 * num_transactions)

    train_mask = torch.zeros(num_transactions, dtype=torch.bool)
    val_mask = torch.zeros(num_transactions, dtype=torch.bool)
    test_mask = torch.zeros(num_transactions, dtype=torch.bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    # 5a. Compute aggregated node features from training data only (no leakage)
    # Using only train rows ensures val/test labels never influence entity representations.
    print("Computing entity node features from training data...")
    train_df = df.iloc[:train_end]

    # User node features: behaviour statistics per user
    user_agg = train_df.groupby('User_ID').agg(
        txn_count=('Transaction_Amount', 'count'),
        avg_amount=('Transaction_Amount', 'mean'),
        fraud_rate=('Fraud_Label', 'mean'),
        avg_balance=('Account_Balance', 'mean'),
        avg_failed=('Failed_Transaction_Count_7d', 'mean'),
    )
    # Reindex to match user_map order; cold-start users (test-only) get column means
    user_feature_matrix = (
        user_agg.reindex(unique_users)
        .fillna(user_agg.mean())
        .values.astype(np.float32)
    )
    user_feature_matrix = StandardScaler().fit_transform(user_feature_matrix).astype(np.float32)

    # Location node features: aggregate stats per city
    loc_agg = train_df.groupby('Location').agg(
        txn_count=('Transaction_Amount', 'count'),
        avg_amount=('Transaction_Amount', 'mean'),
        fraud_rate=('Fraud_Label', 'mean'),
    )
    loc_feature_matrix = (
        loc_agg.reindex(unique_locs)
        .fillna(loc_agg.mean())
        .values.astype(np.float32)
    )
    loc_feature_matrix = StandardScaler().fit_transform(loc_feature_matrix).astype(np.float32)

    # Merchant category node features: aggregate stats per category
    cat_agg = train_df.groupby('Merchant_Category').agg(
        txn_count=('Transaction_Amount', 'count'),
        avg_amount=('Transaction_Amount', 'mean'),
        fraud_rate=('Fraud_Label', 'mean'),
    )
    cat_feature_matrix = (
        cat_agg.reindex(unique_cats)
        .fillna(cat_agg.mean())
        .values.astype(np.float32)
    )
    cat_feature_matrix = StandardScaler().fit_transform(cat_feature_matrix).astype(np.float32)

    data = HeteroData()

    # Add Nodes
    data['transaction'].x = torch.from_numpy(transaction_features)
    data['transaction'].y = torch.from_numpy(labels)
    data['transaction'].train_mask = train_mask
    data['transaction'].val_mask = val_mask
    data['transaction'].test_mask = test_mask

    # Entity nodes now have real aggregated features instead of random/empty placeholders
    data['user'].x = torch.from_numpy(user_feature_matrix)
    data['location'].x = torch.from_numpy(loc_feature_matrix)
    data['merchant_category'].x = torch.from_numpy(cat_feature_matrix)

    # Add Edges (Bi-directional is usually better for GNNs)
    # User <-> Transaction
    data['user', 'performs', 'transaction'].edge_index = torch.stack([
        torch.from_numpy(user_indices), torch.from_numpy(txn_indices)
    ], dim=0).long()
    
    data['transaction', 'performed_by', 'user'].edge_index = torch.stack([
        torch.from_numpy(txn_indices), torch.from_numpy(user_indices)
    ], dim=0).long()

    # Transaction <-> Location
    data['transaction', 'at', 'location'].edge_index = torch.stack([
        torch.from_numpy(txn_indices), torch.from_numpy(loc_indices)
    ], dim=0).long()
    
    data['location', 'is_site_of', 'transaction'].edge_index = torch.stack([
        torch.from_numpy(loc_indices), torch.from_numpy(txn_indices)
    ], dim=0).long()

    # Transaction <-> Merchant Category
    data['transaction', 'belongs_to', 'merchant_category'].edge_index = torch.stack([
        torch.from_numpy(txn_indices), torch.from_numpy(cat_indices)
    ], dim=0).long()
    
    data['merchant_category', 'contains', 'transaction'].edge_index = torch.stack([
        torch.from_numpy(cat_indices), torch.from_numpy(txn_indices)
    ], dim=0).long()

    if output_path:
        print(f"Saving graph data to {output_path}...")
        torch.save(data, output_path)
    
    print("Graph construction complete!")
    print(data)
    return data

if __name__ == "__main__":
    import os
    input_csv = "data/Fraud Detection Transactions Dataset.csv"
    output_pt = "data/processed_graph.pt"
    
    if os.path.exists(input_csv):
        preprocess_and_create_graph(input_csv, output_pt)
    else:
        print(f"Error: Could not find {input_csv}")
