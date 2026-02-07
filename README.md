# Graph Neural Networks for Fraud Detection

A comparative study exploring the effectiveness of Graph Neural Networks (GNNs) for financial fraud detection. This project constructs a transaction graph from synthetic datasets, modeling users, accounts, merchants, and transactions, and evaluates GNN performance against traditional machine learning models.

*This is a group project in fulfillment of CSCI 3834, Winter 2026.*

## 🎯 Project Overview

Financial fraud detection is a critical challenge in the modern banking and payment industry. Traditional machine learning approaches treat transactions independently, missing important relational patterns. This project implements and compares:

- **Graph Neural Networks**: GCN, GAT, and GraphSAGE architectures that leverage transaction network structure
- **Baseline ML Models**: Logistic Regression, Random Forest, and XGBoost for performance comparison

## 🚀 Key Features

- **Graph Construction**: Builds heterogeneous transaction graphs modeling users, merchants, and transactions
- **Multiple GNN Architectures**: Implements state-of-the-art GNN models using PyTorch Geometric
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1-score, and AUC-ROC metrics
- **Visualization Tools**: Graph visualization, training curves, confusion matrices, and radar charts
- **Synthetic Data Generation**: Built-in synthetic data generator for testing and development
- **Modular Design**: Clean, extensible codebase for easy experimentation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NetworkX 3.0+
- Pandas 2.0+
- Scikit-learn 1.3+
- XGBoost 2.0+
- Matplotlib & Seaborn for visualization

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Prezzo-K/Graph-Neural-Networks-for-Fraud-Detection.git
cd Graph-Neural-Networks-for-Fraud-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Command Line Interface

Run experiments with synthetic data:
```bash
python main.py --use-synthetic --n-transactions 10000 --epochs 200
```

Run with your own dataset:
```bash
python main.py --data-path data/transactions.csv --epochs 200 --device cuda
```

Available arguments:
- `--data-path`: Path to CSV file with transaction data
- `--use-synthetic`: Generate synthetic data for testing
- `--n-transactions`: Number of synthetic transactions (default: 10000)
- `--hidden-dim`: Hidden dimension for GNN models (default: 64)
- `--num-layers`: Number of layers in GNN models (default: 2)
- `--epochs`: Training epochs (default: 200)
- `--device`: Device to use - 'cpu' or 'cuda' (default: cpu)
- `--output-dir`: Directory for results (default: results)

### Jupyter Notebook

For interactive exploration, use the example notebook:
```bash
jupyter notebook notebooks/fraud_detection_demo.ipynb
```

The notebook provides a complete walkthrough including:
1. Data loading and exploration
2. Graph construction and visualization
3. Training multiple GNN architectures
4. Training baseline ML models
5. Comprehensive performance comparison

## 📁 Project Structure

```
Graph-Neural-Networks-for-Fraud-Detection/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   └── graph_builder.py    # Transaction graph construction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_models.py       # GNN architectures (GCN, GAT, GraphSAGE)
│   │   └── baseline_models.py  # Traditional ML models
│   └── utils/
│       ├── __init__.py
│       ├── trainer.py          # GNN training utilities
│       ├── evaluator.py        # Model evaluation metrics
│       └── visualization.py    # Plotting and visualization
├── notebooks/
│   └── fraud_detection_demo.ipynb  # Interactive demo notebook
├── tests/                      # Unit tests (to be added)
├── data/                       # Data directory
├── results/                    # Output directory for results
├── main.py                     # Main experiment script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License information
```

## 🔬 Methodology

### 1. Graph Construction

The transaction data is transformed into a directed graph where:
- **Nodes**: Represent users and merchants
- **Edges**: Represent transactions with features (amount, timestamp, balances)
- **Labels**: Binary fraud labels on edges

### 2. Node Features

For each entity, we compute aggregate statistics:
- Transaction count (sent/received)
- Average transaction amount
- Total amount transacted
- Standard deviation of amounts

### 3. Edge Features

Transaction-specific features:
- Transaction amount
- Timestamp
- Account balances (before/after)
- Derived features (e.g., fraction of balance spent)

### 4. GNN Models

**Graph Convolutional Network (GCN)**:
- Based on Kipf & Welling (2017)
- Aggregates information from neighbor nodes
- Fast and simple architecture

**Graph Attention Network (GAT)**:
- Based on Veličković et al. (2018)
- Uses attention mechanism to weight neighbor importance
- Captures varying relationship strengths

**GraphSAGE**:
- Based on Hamilton et al. (2017)
- Inductive learning via neighbor sampling
- Scalable to large graphs

### 5. Baseline Models

- **Logistic Regression**: Linear baseline
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with advanced regularization

## 📈 Expected Results

GNN models typically show advantages when:
- Transaction networks exhibit clear structural patterns
- Fraud rings or coordinated attacks exist
- Relational features are more informative than node features alone

Traditional ML models may perform better when:
- Individual transaction features are highly discriminative
- Network structure is sparse or uninformative
- Interpretability and deployment simplicity are priorities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Geometric**: For excellent GNN implementations
- **NetworkX**: For graph manipulation and visualization
- **Kaggle**: For providing public fraud detection datasets

## 📚 References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
2. Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
3. Hamilton, W. L., et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
4. Weber, M., et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.

## 👥 Team

CSCI 3834, Winter 2026 Group Project

---

For questions or support, please open an issue on GitHub.
