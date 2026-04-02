# Project Proposal

## Graph Neural Networks for Fraud Detection: A Comparative Study with Traditional Machine Learning Models

### Group Members
- Abdi
- Bhabin
- Aaron

## 1. Motivation
Financial fraud continues to evolve in complexity, often involving coordinated behavior across networks of users, accounts, and merchants. Traditional machine learning models treat transactions as isolated rows, which limits their ability to capture relational patterns such as shared devices, repeated merchant interactions, or suspicious clusters of accounts.

Graph Neural Networks (GNNs) offer a powerful alternative by modeling transactions as interconnected structures. This project explores whether GNNs can uncover hidden fraud patterns that conventional models miss.

## 2. Objectives
- Construct a graph representation of the transaction dataset, where:
  - Nodes represent customers, accounts, merchants and more.
  - Edges represent transactions.
- Apply Graph Neural Network architectures (e.g., GCN, GraphSAGE, GAT) to detect anomalous or fraudulent behavior.
- Compare performance of GNN models against traditional ML algorithms such as:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
- Evaluate whether GNNs capture complex relational fraud patterns more effectively than feature-based models.

## 3. Dataset
- **Source:** Synthetic Fraud Detection Dataset (Kaggle)

### Key Features
| Feature | Description |
| --- | --- |
| TransactionID | Unique identifier for each transaction |
| Timestamp/Date | Time of transaction (hour, day, temporal patterns) |
| Amount | Monetary value |
| CustomerID / AccountID | Entity initiating the transaction |
| MerchantID | Entity receiving the transaction |
| Category / Type | Retail, food, travel, etc. |
| Location | Branch or region |

More columns missing from above.
These features allow both tabular ML modeling and graph construction.

## 4. Tools & Technologies
- Python 3.14
- PyTorch Geometric - GNN model development
- NetworkX - Graph construction and preprocessing
- Pandas, NumPy - Data cleaning and transformation
- Scikit-Learn - Baseline ML models and evaluation metrics
- Matplotlib, Seaborn - Visualization and exploratory analysis

## 5. Methodology (Suggested Structure)
### Step 1: Data Preprocessing
- Clean missing values.
- Encode categorical features.
- Normalize numerical values.
- Extract temporal features (hour, weekday, etc.).

### Step 2: Graph Construction
- Create nodes for customers, accounts, and merchants.
- Create edges for each transaction.
- Assign edge attributes (amount, timestamp, category).
- Optionally create heterogeneous graphs (different node types).

### Step 3: Model Development
- Implement GNN architectures using PyTorch Geometric.
- Train models to classify transactions as fraudulent or legitimate.
- Train baseline ML models on the same dataset.

### Step 4: Evaluation
Use metrics such as:
- Accuracy
- Precision, Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Step 5: Comparative Analysis
- Compare GNN vs. traditional ML performance.
- Analyze cases where GNN detects fraud missed by ML models.
- Visualize graph structures and learned embeddings.

## 6. Expected Outcomes
- A fully trained GNN capable of identifying suspicious transaction patterns.
- A clear comparison demonstrating strengths and weaknesses of GNNs relative to traditional ML models.
- Insights into how graph-based modeling improves fraud detection in network-structured financial data.
