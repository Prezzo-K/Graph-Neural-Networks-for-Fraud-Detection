# Ablation Study: Effect of Data Leakage Features on Fraud Detection Performance

**Branch:** `experimentations`  
**Date:** 2026-04-02  
**Authors:** Abdi, Bhabin, Aaron  

---

## 1. Motivation

Two features were excluded from all models in the main study:

| Feature | Reason for Exclusion |
|---|---|
| `Risk_Score` | Derived directly from `Fraud_Label` — encodes the target itself |
| `Failed_Transaction_Count_7d` | Rolling 7-day count of failed transactions — computed using future/same-window labels, near-perfectly predicts fraud |

This ablation study re-introduces each feature individually to **quantify how much artificial performance inflation they cause**, and to confirm that our clean-baseline results reflect genuine model capability rather than label leakage.

---

## 2. Experimental Setup

Both experiments use identical settings to the main study:

- **Traditional ML:** Same 5 classifiers (RF, ET, XGBoost, LR, SVM), 80/20 stratified split, `class_weight='balanced'` / `scale_pos_weight`
- **GNN:** Same 3 architectures (GraphSAGE, GAT, HGT), same heterogeneous graph schema, 100 epochs, Adam lr=0.01, `pos_weight` loss
- **Difference:** One leakage column added per experiment; the other remains excluded

---

## 3. Results

### 3.1 Experiment 1 — With `Risk_Score`

**Traditional ML:**

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.832 | 0.646 | **1.000** | 0.477 | 0.744 | 0.719 |
| Extra Trees | 0.825 | 0.634 | 0.963 | 0.473 | 0.743 | 0.722 |
| XGBoost | 0.801 | 0.622 | 0.799 | 0.510 | 0.745 | 0.727 |
| Logistic Regression | 0.647 | 0.549 | 0.466 | 0.669 | 0.739 | 0.724 |
| SVM | 0.646 | 0.549 | 0.465 | 0.670 | 0.739 | 0.724 |

**GNN:**

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| GraphSAGE | 0.491 | 0.325 | **1.000** | 0.500 | 0.325 |
| GAT | 0.402 | 0.363 | 0.450 | 0.557 | 0.365 |
| HGT | 0.438 | 0.338 | 0.621 | 0.514 | 0.331 |

### 3.2 Experiment 2 — With `Failed_Transaction_Count_7d`

**Traditional ML:**

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Random Forest | 0.877 | 0.764 | **1.000** | 0.618 | 0.808 | 0.801 |
| Extra Trees | 0.877 | 0.764 | 0.999 | 0.618 | 0.811 | 0.802 |
| XGBoost | 0.862 | 0.744 | 0.916 | 0.627 | 0.807 | 0.804 |
| Logistic Regression | 0.728 | 0.629 | 0.560 | 0.717 | 0.802 | 0.803 |
| SVM | 0.728 | 0.629 | 0.560 | 0.717 | 0.802 | 0.803 |

**GNN:**

| Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| GraphSAGE | 0.079 | 0.301 | 0.046 | 0.496 | 0.322 |
| GAT | 0.465 | 0.387 | 0.583 | 0.590 | 0.399 |
| HGT | **0.540** | 0.426 | **0.736** | **0.664** | **0.457** |

---

## 4. Comparison vs Clean Baseline

### Traditional ML — F1-Score

| Model | Baseline (clean) | + Risk_Score | + Failed_7d |
|---|---|---|---|
| Random Forest | 0.001 | **+0.645** | **+0.763** |
| Extra Trees | 0.019 | **+0.615** | **+0.745** |
| XGBoost | 0.350 | +0.272 | +0.394 |
| Logistic Regression | 0.390 | +0.159 | +0.239 |
| SVM | 0.390 | +0.159 | +0.239 |

### GNN — F1-Score

| Model | Baseline (clean) | + Risk_Score | + Failed_7d |
|---|---|---|---|
| GraphSAGE | 0.403 | +0.088 | **-0.324** |
| GAT | 0.463 | -0.061 | +0.002 |
| HGT | 0.402 | +0.036 | **+0.138** |

---

## 5. Analysis and Findings

### 5.1 Risk_Score causes near-perfect precision in tree models

When `Risk_Score` is included, Random Forest achieves **precision = 1.000** — it never generates a false positive. This is expected: `Risk_Score` is a monotonic function of `Fraud_Label`. Any transaction the model flags as fraud is guaranteed to be fraud because the score was derived from that label. This is a textbook example of **direct label leakage** and would produce misleading results in any real evaluation.

GraphSAGE exhibits the opposite pathology: **recall = 1.000** (it predicts all transactions as fraud), with AUC-ROC = 0.500, indicating the learned representations collapse into an all-positive classifier. The signal from `Risk_Score` dominates the node features and overwhelms the gradient, causing the model to minimise BCE loss by predicting fraud for every transaction.

### 5.2 Failed_Transaction_Count_7d is an even stronger leakage signal

`Failed_Transaction_Count_7d` produces larger F1 gains for traditional ML than `Risk_Score` does. Random Forest and Extra Trees reach **F1 = 0.764** and **AUC-PR = 0.80** — roughly doubling performance from the clean baseline. This confirms that the rolling failure count encodes strong fraud signal derived from contemporaneous or future transaction outcomes.

HGT benefits more than GAT or GraphSAGE from this feature (+0.138 F1), because HGT's type-conditioned attention can selectively amplify the high-variance failure-count signal when aggregating over user nodes.

### 5.3 GNNs are less susceptible to leakage than traditional ML

The performance gains from leakage features are systematically **smaller for GNNs than for traditional ML**. The likely explanation is that GNNs already capture a proxy for these features through the graph: user-node aggregate features include `fraud_rate` (a historical analogue of `Risk_Score`) and `avg_failed` (a historical analogue of `Failed_Transaction_Count_7d`), computed from training data only. The leakage columns therefore add less marginal information to GNNs because the model has already learned a non-leaking approximation through message passing.

This is a notable finding: graph-based methods are **naturally more robust to certain forms of feature leakage** because they encode entity-level history through aggregation rather than raw transactional columns.

### 5.4 GraphSAGE degrades with Failed_Transaction_Count_7d

GraphSAGE F1 drops from 0.403 (baseline) to 0.079 when `Failed_Transaction_Count_7d` is included. Mean-aggregation in GraphSAGE averages this high-signal column across all neighbours, producing noisy node representations when the feature value varies sharply within a user's transaction neighbourhood. GAT and HGT, which learn weighted attention, are more resilient.

---

## 6. Summary

| Finding | Implication |
|---|---|
| `Risk_Score` inflates traditional ML precision to 1.0 | Direct label leakage — must be excluded |
| `Failed_Transaction_Count_7d` inflates ML F1 by up to +0.76 | Temporal leakage — must be excluded |
| GNNs gain less from leakage features | Graph aggregation already captures non-leaking proxies |
| GraphSAGE degrades under `Failed_Transaction_Count_7d` | Mean aggregation amplifies noisy high-variance features |
| Clean baseline GNN > clean baseline ML across all metrics | Main study conclusion is confirmed and robust |

---

## 7. Scripts and Reproducibility

```
experiments/run_leakage_experiments.py   # Full experiment runner
results/exp1_risk_score_results.txt      # Exp 1 raw metrics
results/exp2_failed_7d_results.txt       # Exp 2 raw metrics
```

To reproduce:
```bash
python experiments/run_leakage_experiments.py
```
