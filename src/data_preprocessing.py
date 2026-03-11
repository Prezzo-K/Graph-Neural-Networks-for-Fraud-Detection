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

    # 1. Temporal Feature Engineering
    print("Engineering temporal features...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    
    # 2. Define Features
    # Numeric columns to scale
    num_cols = [
        'Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
        'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d',
        'Card_Age', 'Transaction_Distance', 'Hour', 'DayOfWeek', 'Month'
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
    # Risk_Score is dropped to avoid data leakage as per previous experiments
    cols_to_drop = ['Transaction_ID', 'User_ID', 'Timestamp', 'Location', 'Merchant_Category', 'Risk_Score', 'Fraud_Label']
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

    # 5. Build HeteroData Object
    print("Building HeteroData object...")
    data = HeteroData()

    # Add Nodes
    data['transaction'].x = torch.from_numpy(transaction_features)
    data['transaction'].y = torch.from_numpy(labels)
    
    # Other nodes (initialized with identity or just count)
    # Using a simple range-based feature or identity if needed; 
    # many GNNs learn embeddings for these from scratch.
    data['user'].num_nodes = len(unique_users)
    data['location'].num_nodes = len(unique_locs)
    data['merchant_category'].num_nodes = len(unique_cats)

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
