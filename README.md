# Graph Neural Networks for Fraud Detection
Group project — CSCI 3834, Winter 2026.

Comparison of heterogeneous GNN architectures (GraphSAGE, GAT, HGT) against traditional ML baselines for node-level fraud classification on a 50K-transaction dataset.

---

## Results

All results are from the canonical clean run (`experiments/run_final_clean_baseline.py`) with `Risk_Score` and `Failed_Transaction_Count_7d` excluded to prevent data leakage. Saved to `results/`.

### Traditional ML (80/20 stratified split)

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.679 | 0.003 | 0.400 | 0.001 | 0.514 | 0.332 |
| Extra Trees | 0.675 | 0.025 | 0.333 | 0.013 | 0.504 | 0.327 |
| XGBoost | 0.539 | 0.354 | 0.322 | 0.393 | 0.498 | 0.320 |
| Logistic Regression | 0.490 | 0.381 | 0.312 | 0.488 | 0.491 | 0.316 |
| SVM (LinearSVC) | 0.490 | 0.381 | 0.312 | 0.488 | 0.491 | 0.316 |

### GNN Models (70/15/15 temporal split)

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| GraphSAGE | 0.369 | 0.317 | 0.441 | 0.490 | 0.315 |
| GAT | 0.390 | 0.318 | 0.502 | 0.488 | 0.317 |
| **HGT** | **0.416** | **0.325** | **0.576** | **0.494** | **0.315** |

HGT is the best model overall. All three GNNs outperform tree-based ensembles and match or exceed linear models in recall and F1.

To reproduce:
```bash
python experiments/run_final_clean_baseline.py
```

---

## Ablation Study: Data Leakage Analysis

Two features were identified as data leakage and excluded from all main models:
- `Risk_Score` — derived directly from `Fraud_Label`
- `Failed_Transaction_Count_7d` — rolling 7-day failure count that encodes near-future fraud signal

Three experiments re-introduced these features individually and together to quantify the artificial performance inflation they cause.

### F1-Score by Condition

| Model | Clean Baseline | + Risk_Score | + Failed_7d | + Both |
|---|---|---|---|---|
| Random Forest | 0.003 | 0.646 | 0.764 | **1.000** |
| Extra Trees | 0.025 | 0.634 | 0.764 | 0.982 |
| XGBoost | 0.354 | 0.622 | 0.744 | 0.999 |
| Logistic Regression | 0.381 | 0.549 | 0.629 | 0.706 |
| SVM | 0.381 | 0.549 | 0.629 | 0.707 |
| GraphSAGE | 0.369 | 0.491 | 0.079 | 0.491 |
| GAT | 0.390 | 0.402 | 0.465 | 0.451 |
| HGT | **0.416** | 0.438 | 0.540 | 0.627 |

Random Forest achieves perfect F1 = 1.000 with both leakage features — confirming these columns must be excluded. GNNs are substantially more robust to leakage (max F1 = 0.627 even with both features).

Scripts: `experiments/run_leakage_experiments.py`, `experiments/run_exp3_both.py`  
Full analysis: `experiments/ablation_study_report.md`

---

## Dataset

**File:** `data/Fraud Detection Transactions Dataset.csv`  
**Rows:** 50,000 transactions | **Columns:** 21 | **Fraud rate:** 32.1% (16,067 fraud / 33,933 legit)

| Column Group | Columns |
|---|---|
| IDs (dropped) | `Transaction_ID`, `User_ID` |
| Numerical | `Transaction_Amount`, `Account_Balance`, `Daily_Transaction_Count`, `Avg_Transaction_Amount_7d`, `Card_Age`, `Transaction_Distance` |
| Temporal | `Timestamp` → extracted to `Hour`, `Day`, `Month`, `DayOfWeek` |
| Categorical | `Transaction_Type`, `Device_Type`, `Location`, `Merchant_Category`, `Card_Type`, `Authentication_Method` |
| Binary | `IP_Address_Flag`, `Previous_Fraudulent_Activity`, `Is_Weekend` |
| Excluded (leakage) | `Risk_Score`, `Failed_Transaction_Count_7d` |
| Label | `Fraud_Label` (0 = legit, 1 = fraud) |

---

## Graph Schema

The dataset is modelled as a **heterogeneous graph** with 4 node types and 6 directed edge types.

**Node Types:**

| Node | Count | Features | How computed |
|---|---|---|---|
| `transaction` | 50,000 | ~28 (numerical + one-hot + temporal) | Direct from dataset columns |
| `user` | ~8,963 | 5 (`txn_count`, `avg_amount`, `fraud_rate`, `avg_balance`, `avg_failed`) | Aggregated from training rows only |
| `location` | 5 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |
| `merchant_category` | 6 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |

Entity node features are computed from training rows only — val/test labels never influence node representations.

**Edge Types (all bidirectional):**

| Relation | Direction |
|---|---|
| `user → performs → transaction` | user → txn |
| `transaction → performed_by → user` | txn → user |
| `transaction → at → location` | txn → loc |
| `location → is_site_of → transaction` | loc → txn |
| `transaction → belongs_to → merchant_category` | txn → cat |
| `merchant_category → contains → transaction` | cat → txn |

---

## Implementation

**Preprocessing** (`src/data_preprocessing.py`):
1. Temporal sort by `Timestamp`
2. Feature extraction: `Timestamp` → `Hour`, `Day`, `Month`, `DayOfWeek`
3. `StandardScaler` on numerical columns; one-hot encoding for categoricals
4. Temporal 70/15/15 train/val/test split (chronological, no shuffle)
5. Entity node features aggregated from training rows only
6. Graph saved to `data/processed_graph.pt` as `HeteroData`

**Models** (`src/graphsage.py`, `src/gat.py`, `src/hgt.py`):
- **GraphSAGE** — 2× `SAGEConv` with `to_hetero(aggr='sum')`, mean neighbourhood aggregation, Dropout(0.2)
- **GAT** — 2× `GATConv` with `to_hetero`, 4-head attention (Layer 1: 4×16→64; Layer 2: 1×64), `add_self_loops=False`, Dropout(0.2)
- **HGT** — 2× `HGTConv`, natively heterogeneous transformer attention per `(src_type, edge_type, dst_type)`, 4 heads, Dropout(0.2)

> All models use `nn.Dropout` (module) rather than `F.dropout` to avoid a PyG FX tracing bug where the functional form bakes `training=True` as a constant, preventing dropout from turning off at eval time.

**Training** (`src/train.py`): Adam lr=0.01, hidden_channels=64, 100 epochs, `BCEWithLogitsLoss` with `pos_weight=2.11`, best checkpoint by validation F1.

---

## Repo Structure

```
data/               # Raw CSV and processed graph (.pt)
notebooks/          # Jupyter notebooks (EDA, Traditional ML, GNN exploration)
experiments/        # Canonical run scripts and ablation study
results/            # Saved metric outputs (.txt)
src/                # Preprocessing, model definitions, training loop
models/             # Saved model checkpoints (.pth for GNNs, .joblib for ML)
```
