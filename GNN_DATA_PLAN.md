# GNN Data Structure Plan

## Goal
Structure the CSV dataset into a PyTorch Geometric `HeteroData` object to enable Graph Neural Network modeling for fraud detection.

## Proposed Graph Schema (Heterogeneous)

### Node Types
1.  **`transaction`** (Nodes: 50,000)
    *   **Features**: Numerical (Amount, Balance, Distance, etc.) + One-Hot Encoded Categories (Type, Device, Card, etc.).
    *   **Target**: `Fraud_Label`.
2.  **`user`** (Nodes: ~8,963)
    *   **Features**: Identity matrix or learnable embeddings (captures user-level behavior).
3.  **`location`** (Nodes: Unique cities/regions)
4.  **`merchant_category`** (Nodes: Unique categories)

### Edge Types (Relations)
*   `(user, performs, transaction)`
*   `(transaction, at, location)`
*   `(transaction, belongs_to, merchant_category)`

---

## Implementation Steps

### 1. Data Preprocessing
*   Load `data/Fraud Detection Transactions Dataset.csv`.
*   Handle missing values (if any).
*   **Numerical Scaling**: Use `StandardScaler` for `Transaction_Amount`, `Account_Balance`, etc.
*   **Categorical Encoding**: One-Hot Encode `Transaction_Type`, `Device_Type`, `Card_Type`, and `Authentication_Method`.
*   **Temporal Features**: Extract `Hour`, `DayOfWeek`, and `Month` from the `Timestamp`.

### 2. Entity Mapping
*   Map unique `User_ID`, `Location`, and `Merchant_Category` values to zero-based integer indices $[0, N-1]$.
*   Store these mappings to reverse-lookup if needed.

### 3. Graph Construction (PyG)
*   Use `torch_geometric.data.HeteroData`.
*   Assign `transaction` node features from the preprocessed tabular data.
*   Construct `edge_index` tensors for each relation type.
*   Set the `Fraud_Label` as the `y` attribute for `transaction` nodes.

### 4. Validation
*   Verify node and edge counts match the dataset statistics.
*   Ensure no "orphan" transactions exist (every transaction must have a user).
*   Check for data leakage (exclude `Risk_Score` from input features).

---

## Risks & Considerations
*   **Class Imbalance**: Fraud cases are likely a minority; we may need weighted loss or oversampling in the GNN training loop.
*   **Memory**: While 50k nodes is small, the number of relations can grow; `HeteroData` is the efficient way to handle this.
*   **Temporal Splits**: Ensure the train/test split respects time (don't train on future data to predict the past).
