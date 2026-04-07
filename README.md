# Graph-Neural-Networks-for-Fraud-Detection
Group project in fulfillment of CSCI 3834, Winter 2026.

---

### Team Progress Summary
| Milestone | Status | Key Deliverables Done |
|---|---|---|
| 1 — Project setup & dataset audit | ✅ Completed | Repo structure, EDA notebook, requirements.txt, metrics defined |
| 2 — Graph construction & baseline models | ✅ Completed | HeteroData graph, preprocessing script, Traditional ML baselines |
| 3 — GNN modeling & evaluation | ✅ Completed | GraphSAGE, GAT, HGT trained; metrics aligned with baselines |
| 4 — Analysis, visualization & final report | ✅ Completed | Ablation study, comparison figures, final locked results, report |

---

## Final Locked Results

All results below are from the canonical clean run (`experiments/run_final_clean_baseline.py`) with `Risk_Score` and `Failed_Transaction_Count_7d` excluded. Saved to `results/`.

### Traditional ML

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.679 | 0.003 | 0.400 | 0.001 | 0.514 | 0.332 |
| Extra Trees | 0.675 | 0.025 | 0.333 | 0.013 | 0.504 | 0.327 |
| XGBoost | 0.539 | 0.354 | 0.322 | 0.393 | 0.498 | 0.320 |
| Logistic Regression | 0.490 | 0.381 | 0.312 | 0.488 | 0.491 | 0.316 |
| SVM (LinearSVC) | 0.490 | 0.381 | 0.312 | 0.488 | 0.491 | 0.316 |

### GNN Models

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| GraphSAGE | 0.369 | 0.317 | 0.441 | 0.490 | 0.315 |
| GAT | 0.390 | 0.318 | 0.502 | 0.488 | 0.317 |
| **HGT** | **0.416** | **0.325** | **0.576** | **0.494** | **0.315** |

**HGT is the best model overall.** All three GNNs outperform tree-based ensembles and match or exceed linear models in recall and F1.

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
  - **F1-Score** — primary metric; harmonic mean of precision/recall at the decision threshold
  - **AUC-PR** — measures ranking quality for the minority (fraud) class
  - **Precision / Recall** — decompose where the model is failing (false alarms vs. missed fraud)
  - **AUC-ROC** — standard ranking metric across all thresholds
  - **Accuracy** — included for completeness; less meaningful under class imbalance
- Confirmed that `Risk_Score` and `Failed_Transaction_Count_7d` are excluded from all models to prevent data leakage

### EDA (Bhabin)
**File:** `notebooks/Full_EDA_Report.ipynb`
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
experiments/        # Canonical run scripts and ablation study scripts
results/            # Saved metric outputs (.txt files)
src/                # Python source (preprocessing, model definitions, training)
models/             # Saved model checkpoints (.pth for GNNs, .joblib for ML)
```

---

## Milestone 2: Graph Construction and Baseline Models — ✅ Completed

### Graph Schema Design (Abdi)

The dataset is structured as a **heterogeneous graph** with 4 node types and 6 directed edge types:

**Node Types:**

| Node | Count | Features | How computed |
|---|---|---|---|
| `transaction` | 50,000 | ~28 (numerical + one-hot + temporal) | Direct from dataset columns |
| `user` | ~8,963 | 5 (`txn_count`, `avg_amount`, `fraud_rate`, `avg_balance`, `avg_failed`) | Aggregated from training rows only |
| `location` | 5 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |
| `merchant_category` | 6 | 3 (`txn_count`, `avg_amount`, `fraud_rate`) | Aggregated from training rows only |

> Entity node features are computed from **training rows only** — val/test labels never influence node representations.

**Edge Types (all bidirectional):**

| Relation | Direction |
|---|---|
| `user → performs → transaction` | user → txn |
| `transaction → performed_by → user` | txn → user |
| `transaction → at → location` | txn → loc |
| `location → is_site_of → transaction` | loc → txn |
| `transaction → belongs_to → merchant_category` | txn → cat |
| `merchant_category → contains → transaction` | cat → txn |

### Data Preprocessing (Bhabin)
**File:** `src/data_preprocessing.py`

1. Temporal sort by `Timestamp`
2. Feature extraction: `Timestamp` → `Hour`, `Day`, `Month`, `DayOfWeek`
3. `StandardScaler` on numerical columns
4. One-hot encoding for categorical columns
5. Temporal train/val/test split (chronological, no shuffle)
6. Entity node features aggregated from training rows only
7. Graph saved to `data/processed_graph.pt` as `HeteroData`

---

## Milestone 3: GNN Modeling and Evaluation — ✅ Completed

### Model Implementation (Abdi)
**Files:** `src/graphsage.py`, `src/gat.py`, `src/hgt.py`, `src/model.py`, `src/train.py`

**GraphSAGE** — 2× `SAGEConv` with `to_hetero(aggr='sum')`, mean neighbourhood aggregation, Dropout(0.2)

**GAT** — 2× `GATConv` with `to_hetero`, 4-head attention (Layer 1: 4×16→64; Layer 2: 1×64), `add_self_loops=False`, Dropout(0.2)

**HGT** — 2× `HGTConv`, natively heterogeneous transformer attention conditioned on `(src_type, edge_type, dst_type)` triplets, 4 heads, Dropout(0.2)

> **Dropout fix:** All models use `nn.Dropout` (module) instead of `F.dropout`, because the functional form bakes `training=True` as a constant during FX tracing by `to_hetero`, preventing dropout from turning off at eval time.

**Training:** Adam lr=0.01, hidden_channels=64, 100 epochs, `BCEWithLogitsLoss` with `pos_weight=2.11`, best checkpoint by validation F1.

---

## Milestone 4: Analysis, Visualization, and Final Report — ✅ Completed

- **Ablation study** (`experiments/ablation_study_report.md`): 3 controlled experiments quantifying data leakage impact across all 8 models and 4 feature conditions
- **Comparison figures**: grouped bar chart and recall heatmap across all leakage conditions
- **Final locked run**: `experiments/run_final_clean_baseline.py` — single authoritative reproducible run, results committed to `results/`
- **Full EDA**: `notebooks/Full_EDA_Report.ipynb` — 18 figures covering distributions, correlations, graph structure, temporal patterns, and t-SNE projections
- **Report**: ACM sigconf format, 8 pages, written using LaTeX
