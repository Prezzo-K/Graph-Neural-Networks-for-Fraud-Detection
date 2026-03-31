# Graph-Neural-Networks-for-Fraud-Detection
Group project in fulfillment of CSCI 3834, Winter 2026.

---

### Team Progress Summary
| Milestone | Status | Key Deliverables Done |
|---|---|---|
| 1 — Project setup & dataset audit | ✅ Completed | Repo structure, EDA notebook, requirements.txt, metrics defined |
| 2 — Graph construction & baseline models | ✅ Completed | HeteroData graph, preprocessing script, Traditional ML baselines |
| 3 — GNN modeling & evaluation | ✅ Completed | GraphSAGE, GAT, HGT trained; metrics aligned with baselines |
| 4 — Analysis, visualization & final report | ⏳ Not Started | — |

---

## Dataset Overview

**File:** `data/Fraud Detection Transactions Dataset.csv`
**Rows:** 50,000 transactions | **Columns:** 21 | **Fraud rate:** 32.1% (16,067 fraud / 33,933 legit)

| Column Group | Columns |
|---|---|
| IDs (dropped) | `Transaction_ID`, `User_ID` |
| Numerical | `Transaction_Amount`, `Account_Balance`, `Daily_Transaction_Count`, `Avg_Transaction_Amount_7d`, `Card_Age`, `Transaction_Distance` |
| Temporal | `Timestamp` → extracted to `Hour`, `Day`, `Month`, `DayOfWeek` |
| Categorical | `Transaction_Type`, `Device_Type`, `Location`, `Merchant_Category`, `Card_Type`, `Authentication_Method` |
| Binary | `IP_Address_Flag`, `Previous_Fraudulent_Activity`, `Is_Weekend` |
| Excluded (leakage) | `Risk_Score` (derived from fraud label), `Failed_Transaction_Count_7d` (near-perfectly predicts fraud) |
| Label | `Fraud_Label` (0 = legit, 1 = fraud) |

---

## Milestone 1: Project Setup and Dataset Audit — ✅ Completed

### Scope & Metrics (Abdi)
- Defined the task as **node-level binary classification** on a heterogeneous transaction graph
- Selected evaluation metrics for imbalanced fraud detection:
  - **AUC-PR** — primary metric; measures ranking quality for the minority (fraud) class
  - **F1-Score** — harmonic mean of precision/recall at the decision threshold
  - **Precision / Recall** — decompose where the model is failing (false alarms vs. missed fraud)
  - **AUC-ROC** — standard ranking metric across all thresholds
  - **Accuracy** — included for completeness; less meaningful under class imbalance
  - **Confusion Matrix** — shows raw TP/TN/FP/FN counts; makes false negative cost visible
- Confirmed that `Risk_Score` and `Failed_Transaction_Count_7d` are excluded from all models to prevent data leakage and ensure fair comparison

### EDA (Bhabin)
**File:** `notebooks/Fraud_detection_dataset_EDA.ipynb`
- Verified no missing values across all 50,000 rows
- Fraud class imbalance: 32.1% fraud — significant but not extreme; handled via class weighting
- Location: 5 unique cities (Sydney, New York, London, Tokyo, Paris)
- Merchant category: 6 unique categories
- Users: ~8,963 unique User IDs across 50,000 transactions
- Temporal coverage: transactions span 2023, sorted chronologically for temporal splits

### Repo Structure (Aaron)
**File:** `requirements.txt`

```
data/               # Raw CSV and processed graph (.pt)
notebooks/          # Jupyter notebooks (EDA, Traditional ML, GNN exploration)
results/            # Saved metric outputs (traditional_ml_results.txt, gnn_results.txt)
src/                # Python source (preprocessing, model definitions, training)
models/             # Saved model checkpoints (.pth for GNNs, .joblib for ML)
tests/              # Test stubs
utils/              # Utility stubs
```

---

## Milestone 2: Graph Construction and Baseline Models — ✅ Completed

### Graph Schema Design (Abdi)
**File:** `GNN_DATA_PLAN.md`

The dataset is structured as a **heterogeneous graph** with 4 node types and 6 directed edge types:

**Node Types:**

| Node | Count | Features | How computed |
|---|---|---|---|
| `transaction` | 50,000 | ~28 (numerical + one-hot + temporal) | Direct from dataset columns |
| `user` | ~8,963 | 5 (`txn_count`, `avg_amount`, `fraud_rate`, `avg_balance`, `avg_failed`) | Aggregated from training rows only |
| `location` | 5 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |
| `merchant_category` | 6 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |

> Entity node features (`user`, `location`, `merchant_category`) are computed from **training rows only** — val/test labels never influence node representations.
> Cold-start users (appearing only in val/test) receive column means from training.

**Edge Types (all bidirectional):**

| Relation | Direction | Meaning |
|---|---|---|
| `user → performs → transaction` | user → txn | User initiated this transaction |
| `transaction → performed_by → user` | txn → user | Reverse; propagates transaction signals back to user |
| `transaction → at → location` | txn → loc | Transaction occurred in this city |
| `location → is_site_of → transaction` | loc → txn | Reverse; city context flows into transaction |
| `transaction → belongs_to → merchant_category` | txn → cat | Transaction is in this merchant category |
| `merchant_category → contains → transaction` | cat → txn | Reverse; category risk flows into transaction |

### Data Preprocessing (Bhabin)
**File:** `src/data_preprocessing.py`

Steps performed in order:
1. **Temporal sort** — data sorted by `Timestamp` ascending before any split
2. **Feature extraction** — `Timestamp` decomposed into `Hour`, `Day`, `Month`, `DayOfWeek`
3. **Scaling** — `StandardScaler` applied to 10 numerical columns
4. **One-hot encoding** — applied to `Transaction_Type`, `Device_Type`, `Card_Type`, `Authentication_Method`; `Location` and `Merchant_Category` are used for graph edges, not one-hot
5. **Temporal train/val/test split** — 70% / 15% / 15% by row order (chronological, no shuffle)
   - Train: 35,000 transactions | Val: 7,500 | Test: 7,500
   - Train class balance: ~23,731 legit / ~11,269 fraud → `pos_weight = 2.11`
6. **Entity node features** — aggregated statistics computed on training rows only
7. **Graph saved** to `data/processed_graph.pt` as a PyG `HeteroData` object

### Traditional ML Baselines (Aaron)
**File:** `notebooks/Traditional_ML.ipynb` | Results: `results/traditional_ml_results.txt`

**Setup:**
- 80/20 stratified train/test split (40,000 train / 10,000 test)
- Class imbalance handled via `class_weight='balanced'` (sklearn) or `scale_pos_weight=2.11` (XGBoost)
- Features: same exclusions as GNN (no `Risk_Score`, no `Failed_Transaction_Count_7d`)
- For AUC-ROC/AUC-PR: `predict_proba()[:,1]` for probabilistic models; `decision_function()` for LinearSVC (no `predict_proba`)

**Results (test set):**

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.6782 | 0.0006 | 0.1429 | 0.0003 | — | — |
| Extra Trees | 0.6740 | 0.0257 | 0.3233 | 0.0134 | — | — |
| XGBoost | 0.5341 | 0.3532 | 0.3188 | 0.3959 | — | — |
| Logistic Regression | 0.4962 | 0.3815 | 0.3150 | 0.4837 | — | — |
| SVM (LinearSVC) | 0.4958 | 0.3812 | 0.3147 | 0.4833 | — | — |

> AUC-ROC and AUC-PR will populate after re-running the notebook — these metrics were added in the latest update alongside confusion matrix output.

**Key observations:**
- Random Forest and Extra Trees collapse to near-zero recall despite `class_weight='balanced'` — their tree structure biases toward the majority class even with reweighting
- XGBoost, Logistic Regression, and SVM are more competitive with ~0.38–0.40 F1
- High accuracy for tree models is misleading — they mostly predict "legit" and get credit for the 68% majority class

---

## Milestone 3: GNN Modeling and Evaluation — ✅ Completed

### Model Implementation (Abdi)
**Files:** `src/graphsage.py`, `src/gat.py`, `src/hgt.py`, `src/model.py`, `src/train.py`

Three heterogeneous GNN architectures implemented and trained:

**GraphSAGE** (`src/graphsage.py`)
- 2× `SAGEConv` layers wrapped with `to_hetero(aggr='sum')`
- Aggregation: `h_v = W · CONCAT(h_v, MEAN({h_u : u ∈ N(v)}))`
- Treats all neighbours equally — baseline for attention-based models
- `nn.Dropout(p=0.2)` between layers

**GAT** (`src/gat.py`)
- 2× `GATConv` layers wrapped with `to_hetero(aggr='sum')`
- Layer 1: 4 attention heads × 16 dims → concat → 64 dims; Layer 2: 1 head, 64 dims
- Attention: `α_vu = softmax(LeakyReLU(a^T [W·h_v || W·h_u]))`
- `add_self_loops=False` (mandatory for bipartite hetero subgraphs)
- `nn.Dropout(p=0.2)` on input and between layers

**HGT** (`src/hgt.py`)
- 2× `HGTConv` layers — natively heterogeneous, no `to_hetero` needed
- Transformer-style attention conditioned on both source node type and edge type
- Each `(src_type, edge_type, dst_type)` triplet has its own Q/K/V weight matrices
- 4 attention heads; `nn.Dropout(p=0.2)` between layers

> **Dropout fix:** All three models use `nn.Dropout` (module) instead of `F.dropout(..., training=self.training)`. The functional form bakes `training=True` as a constant during FX tracing by `to_hetero`, so dropout would never turn off during evaluation. The module form is handled correctly by PyTorch's FX tracer.

**Training setup (`src/train.py`):**
- Optimizer: Adam, `lr=0.001`, `hidden_channels=128`, 200 epochs
- Loss: `BCEWithLogitsLoss` with `pos_weight = num_neg / num_pos` per model run
- Model selection: best checkpoint by validation F1
- Full-graph training (no mini-batching — 50k nodes fits in memory)

### GNN Results (Test Set)

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| GraphSAGE | — | 0.4030 | 0.3234 | 0.5344 | 0.4932 | 0.3150 |
| GAT | — | 0.4179 | 0.3242 | 0.5877 | 0.4896 | 0.3143 |
| HGT | — | 0.4007 | 0.3227 | 0.5283 | 0.4963 | 0.3191 |

> Accuracy and Confusion Matrix will populate after re-running `src/train.py` — these were added in the latest metrics alignment update.

**Key observations:**
- All three GNNs achieve ~0.40 F1 and ~0.31–0.32 AUC-PR on test, similar to the best traditional ML models (XGBoost/LR/SVM)
- GNNs show substantially higher recall (~0.53–0.59) compared to tree-based models (RF: 0.0003, ET: 0.01)
- AUC-ROC near 0.49 suggests the current models are only marginally better than random ranking — likely due to training instability or the need for hyperparameter tuning
- GAT edges out slightly on F1 and recall; HGT is marginally best on AUC-PR

### Graph Exploration (Bhabin)
**File:** `notebooks/GNN_Graph_Exploration.ipynb`
- Visual inspection of the constructed `HeteroData` graph
- Node/edge count verification across all 4 node types and 6 edge types
- Feature shape validation per node type

---

## Future Metric Exploration

- **Matthews Correlation Coefficient (MCC):** A single score (−1 to 1) that accounts for all four cells of the confusion matrix (TP, TN, FP, FN). More robust than F1 under class imbalance because it penalises models that ignore either class. Worth adding to the final comparison table.

---

## Milestone 4: Analysis, Visualization, and Final Report — ⏳ Not Started

- **Lead (Abdi):** Lead comparative analysis and finalize conclusions.
  - *How:* Synthesize GNN vs. traditional ML comparisons, highlight where graph structure helps, and document limitations.
  - *Deliverable:* Final analysis narrative and conclusions section.
- **Bhabin:** Build visualizations (fraud subgraphs, metrics plots).
  - *How:* Use seaborn/matplotlib for metric bar charts and NetworkX for fraud subgraph visuals.
  - *Deliverable:* Final figure set saved for the report.
- **Aaron:** Draft the final report and presentation materials.
  - *How:* Assemble report outline, integrate figures, and iterate with team review.
  - *Deliverable:* Report draft and presentation deck.
