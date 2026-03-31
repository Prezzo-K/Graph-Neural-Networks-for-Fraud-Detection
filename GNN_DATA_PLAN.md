# GNN Data Structure Plan

## Goal
Structure the CSV dataset into a PyTorch Geometric `HeteroData` object to enable Graph Neural Network modeling for fraud detection.

---

## Why This Graph is Heterogeneous

A graph is **homogeneous** when all nodes are the same type (e.g. all users) and all edges mean the same thing.
A graph is **heterogeneous** when it has **multiple node types and/or multiple edge types with different semantic meanings**.

This dataset is heterogeneous because:

| Dimension | Homogeneous would be... | Ours is... |
|-----------|------------------------|------------|
| Node types | One type (e.g. just "transaction") | 4 types: `transaction`, `user`, `location`, `merchant_category` |
| Edge types | One type (e.g. "connected to") | 6 types (3 relations + their reverses) |
| Node features | Same shape for all nodes | Different: transactions have ~28 features, users have 5, location/category have 3 |
| Node semantics | All nodes mean the same thing | Each type means something different — a user is not a transaction |

**Why it matters for fraud detection:**
- A `user` node carries **who** did the transaction (behaviour history, fraud rate)
- A `location` node carries **where** (city-level fraud patterns)
- A `merchant_category` node carries **what** was bought (category risk)
- The `transaction` node carries **how** (amount, device, authentication, time)

A homogeneous GNN would collapse all of this into one node type and lose semantic distinctions. A heterogeneous GNN keeps them separate and learns different transformation weights per relation type — so "user → transaction" aggregation is learned differently from "location → transaction" aggregation.

---

## Graph Schema

### Node Types

| Node | Count | Features | Source Columns |
|------|-------|----------|----------------|
| `transaction` | 50,000 | ~28 (numerical + one-hot + temporal) | Amount, Balance, Device, Card, Auth, Hour, etc. |
| `user` | ~8,963 | 5 (aggregated from train) | txn_count, avg_amount, fraud_rate, avg_balance, avg_failed |
| `location` | 5 | 3 (aggregated from train) | txn_count, avg_amount, fraud_rate |
| `merchant_category` | 6 | 3 (aggregated from train) | txn_count, avg_amount, fraud_rate |

> **Note:** Entity node features (user, location, merchant_category) are computed from **training rows only** to prevent label leakage into val/test.

### Edge Types (Bidirectional)

| Relation | Direction | Semantic Meaning |
|----------|-----------|-----------------|
| `(user, performs, transaction)` | user → txn | A user initiated this transaction |
| `(transaction, performed_by, user)` | txn → user | Reverse: lets transaction info flow back to user |
| `(transaction, at, location)` | txn → loc | Transaction happened in this city |
| `(location, is_site_of, transaction)` | loc → txn | Reverse: city context flows into transaction |
| `(transaction, belongs_to, merchant_category)` | txn → cat | Transaction is in this merchant category |
| `(merchant_category, contains, transaction)` | cat → txn | Reverse: category context flows into transaction |

Bidirectional edges are important: without the reverse direction, information would only flow one way and the GNN cannot aggregate neighbour context from both sides during message passing.

---

## GNN Architecture Plan (Systematic)

The approach is to start simple and increase model complexity, keeping all other variables fixed (same data, same splits, same evaluation). This lets us isolate the effect of each architectural change.

### Why use `to_hetero`?
PyG's `to_hetero` wrapper takes a homogeneous GNN and replicates its layers once per node/edge type, giving each relation its own learnable weights. This is the standard way to adapt standard GNN layers (SAGEConv, GATConv) for heterogeneous graphs without writing custom hetero layers.

The exception is **HGT**, which is natively heterogeneous and doesn't need `to_hetero`.

---

### Model 1 — GraphSAGE (Baseline)
**Layer:** `SAGEConv`
**Aggregation:** Mean of neighbour features + own features (concatenated, then projected)

```
h_v = W · CONCAT(h_v, MEAN({h_u : u ∈ N(v)}))
```

**Why start here:**
- Simple and well-understood
- Inductive: works on unseen nodes
- Already implemented — serves as the baseline to beat
- `to_hetero(..., aggr='sum')` applies separate SAGEConv weights per relation

**Weakness for this task:** Treats all neighbours equally — a fraudulent user and a legitimate user contribute the same amount to a transaction's representation.

---

### Model 2 — GAT (Graph Attention Network)
**Layer:** `GATConv`
**Aggregation:** Weighted sum of neighbours, where weights are learned by an attention mechanism

```
h_v = CONCAT_k( σ( Σ_{u∈N(v)} α^k_{vu} · W^k · h_u ) )

where α_{vu} = softmax( LeakyReLU( a^T [W·h_v || W·h_u] ) )
```

**Why add attention:**
- Not all neighbours are equally informative for fraud detection
- A user with high `fraud_rate` should get higher attention weight when aggregated into a transaction
- The model **learns** which neighbours to trust rather than averaging blindly
- `to_hetero` gives each relation its own attention weights

**Weakness:** Multi-head attention increases parameter count and can overfit on small graphs; not a concern here with 50k nodes.

---

### Model 3 — HGT (Heterogeneous Graph Transformer)
**Layer:** `HGTConv` (natively heterogeneous, no `to_hetero` needed)
**Aggregation:** Transformer-style attention that conditions on **both the source node type AND the edge type**

```
h_v = AGG_{τ∈T_src} ( Σ_{u∈N_τ(v)} Attn(τ, φ(u,v)) · MSG(τ, h_u) )
```
where `τ` is the relation type and `φ(u,v)` is the edge type.

**Why this is the most expressive choice:**
- GraphSAGE + `to_hetero`: separate weights per relation, no attention
- GAT + `to_hetero`: attention per relation, but attention only conditions on node features
- **HGT**: attention conditions on **node type + edge type** simultaneously — a fundamentally different aggregation for each (source_type, edge_type, target_type) triplet
- Purpose-built for heterogeneous graphs — this is the architecture designed exactly for our setup

**Weakness:** More parameters, slower to train. With only 5 location nodes and 6 category nodes, the relation-specific weights for those may overfit — watch val metrics carefully.

---

### Comparison Strategy

| Model | Architecture | Key Differentiator |
|-------|-------------|-------------------|
| GraphSAGE | `SAGEConv` + `to_hetero` | Baseline mean aggregation |
| GAT | `GATConv` + `to_hetero` | Learned attention weights |
| HGT | `HGTConv` (native hetero) | Type-conditioned attention |

**Fixed across all models:**
- Same `HeteroData` graph (same nodes, edges, features)
- Same temporal train/val/test splits (70/15/15)
- Same weighted BCE loss (`pos_weight = neg/pos`)
- Same optimizer (Adam, lr=0.01)
- Same hidden dimension (64)
- Same evaluation metrics: F1, Precision, Recall, AUC-ROC, AUC-PR

**Expected performance order:** GraphSAGE < GAT < HGT
(Though GAT and HGT may be close — the dataset may not be complex enough to show a big gap.)

---

## Implementation Status

| Step | Status | File |
|------|--------|------|
| Data preprocessing + graph construction | ✅ Done | `src/data_preprocessing.py` |
| Entity node features (aggregated, no leakage) | ✅ Done | `src/data_preprocessing.py` |
| GraphSAGE model | ✅ Done | `src/model.py`, `src/train.py` |
| GAT model | ⏳ Next | `src/model.py` |
| HGT model | ⏳ Next | `src/model.py` |
| Graph exploration notebook | ✅ Done | `notebooks/GNN_Graph_Exploration.ipynb` |

---

## Risks & Considerations

- **Class Imbalance**: Handled via `pos_weight` in BCE loss (`neg/pos` ratio).
- **Temporal Splits**: Data sorted by `Timestamp` before splitting — no future leakage.
- **Label Leakage**: Entity node features computed on `train_df` only.
- **Cold-start users**: Users appearing only in test get global mean features (not zero, not random).
- **Memory**: 50k nodes is small — no mini-batching needed, full-graph training works.
