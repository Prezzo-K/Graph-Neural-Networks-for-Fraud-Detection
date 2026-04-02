# =============================================================================
# GNN Heterogeneous Graph EDA — Sections 11–19
# Fraud Detection Project
# =============================================================================
# Each section begins with a # CELL comment, marking a separate notebook cell.
# Copy each block into its own Jupyter cell.
# =============================================================================

# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 11: Graph Setup & Loading
# ─────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ── Aesthetic defaults ────────────────────────────────────────────────────────
FRAUD_COLOR  = "#E74C3C"
LEGIT_COLOR  = "#2ECC71"
ACCENT_COLOR = "#3498DB"
PALETTE      = [LEGIT_COLOR, FRAUD_COLOR]

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.titlesize":  16,
    "figure.titleweight":"bold",
})
sns.set_style("whitegrid")

# ── Load graph ────────────────────────────────────────────────────────────────
DATA_PATH = "../data/processed_graph.pt"

try:
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
except TypeError:
    # Older PyTorch versions do not accept weights_only
    data = torch.load(DATA_PATH, map_location="cpu")

print("=" * 60)
print("  HETEROGENEOUS GRAPH — FULL SUMMARY")
print("=" * 60)

# Node types
print("\n── Node Types ──────────────────────────────────────────────")
for ntype in data.node_types:
    n_nodes = data[ntype].num_nodes
    x = data[ntype].get("x", None)
    feat_dim = x.shape[1] if x is not None else "N/A"
    print(f"  {ntype:<25}  nodes={n_nodes:>7,}   feat_dim={feat_dim}")

# Edge types
print("\n── Edge Types ──────────────────────────────────────────────")
for etype in data.edge_types:
    n_edges = data[etype].edge_index.shape[1]
    print(f"  {str(etype):<55}  edges={n_edges:>7,}")

# Label distribution
labels = data["transaction"].y.numpy()
n_legit = int((labels == 0).sum())
n_fraud = int((labels == 1).sum())
pct_fraud = n_fraud / len(labels) * 100

print("\n── Transaction Label Distribution ─────────────────────────")
print(f"  Total transactions : {len(labels):>8,}")
print(f"  Legit  (label=0)   : {n_legit:>8,}  ({100 - pct_fraud:.2f}%)")
print(f"  Fraud  (label=1)   : {n_fraud:>8,}  ({pct_fraud:.2f}%)")
print(f"  Imbalance ratio    : {n_legit / n_fraud:.1f}:1")

# Mask sizes
print("\n── Split Masks ─────────────────────────────────────────────")
for mask_name in ("train_mask", "val_mask", "test_mask"):
    mask = data["transaction"].get(mask_name, None)
    if mask is not None:
        count = int(mask.sum())
        pct   = count / len(labels) * 100
        print(f"  {mask_name:<15}: {count:>7,}  ({pct:.1f}%)")
    else:
        print(f"  {mask_name:<15}: NOT FOUND")

print("\n" + "=" * 60)
print("  PyTorch Geometric raw repr:")
print("=" * 60)
print(data)


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 12: Graph Structure Summary Table
# ─────────────────────────────────────────────────────────────────────────────

# ── Node summary ──────────────────────────────────────────────────────────────
node_rows = []
for ntype in data.node_types:
    x = data[ntype].get("x", None)
    feat_dim = x.shape[1] if x is not None else "—"
    extra = ""
    if ntype == "transaction":
        extra = " (+ y, train/val/test masks)"
    node_rows.append({
        "Node Type":         ntype,
        "Node Count":        f"{data[ntype].num_nodes:,}",
        "Feature Dim":       feat_dim,
        "Additional Attrs":  extra.strip(),
    })

df_nodes = pd.DataFrame(node_rows)

# ── Edge summary ──────────────────────────────────────────────────────────────
edge_rows = []
for etype in data.edge_types:
    src, rel, dst = etype
    n_edges = data[etype].edge_index.shape[1]
    edge_rows.append({
        "Source":       src,
        "Relation":     rel,
        "Destination":  dst,
        "Edge Count":   f"{n_edges:,}",
    })

df_edges = pd.DataFrame(edge_rows)

# ── Styled display ────────────────────────────────────────────────────────────
from IPython.display import display, HTML

header_props = [
    ("background-color", "#2C3E50"),
    ("color",            "white"),
    ("font-weight",      "bold"),
    ("font-size",        "13px"),
    ("padding",          "8px 14px"),
    ("text-align",       "center"),
]
cell_props = [
    ("font-size",   "12px"),
    ("padding",     "7px 14px"),
    ("text-align",  "center"),
]

def style_df(df, caption):
    return (
        df.style
          .set_caption(f"<b>{caption}</b>")
          .set_table_styles([
              {"selector": "th",  "props": header_props},
              {"selector": "td",  "props": cell_props},
              {"selector": "tr:nth-child(even)",
               "props": [("background-color", "#F2F3F4")]},
              {"selector": "caption",
               "props": [("font-size", "15px"), ("font-weight", "bold"),
                         ("color", "#2C3E50"), ("padding-bottom", "8px")]},
          ])
          .set_properties(**{"border": "1px solid #ddd"})
          .hide(axis="index")
    )

print("Node Type Summary")
display(style_df(df_nodes, "Node Type Summary — Heterogeneous Fraud Graph"))

print("\nEdge Type Summary")
display(style_df(df_edges, "Edge Type Summary — Heterogeneous Fraud Graph"))


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 13: Meta-Graph (Schema) Visualization
# ─────────────────────────────────────────────────────────────────────────────

# Edge counts (forward direction only for clarity)
forward_edges = {
    ("user",         "performs",   "transaction"):    data["user",        "performs",   "transaction"].edge_index.shape[1],
    ("transaction",  "at",         "location"):       data["transaction", "at",         "location"].edge_index.shape[1],
    ("transaction",  "belongs_to", "merchant_category"): data["transaction", "belongs_to", "merchant_category"].edge_index.shape[1],
}

NODE_META = {
    "transaction":       {"color": "#E8A838", "size": 4200, "label": "transaction\n(50,000 nodes\n28 features)"},
    "user":              {"color": "#3498DB", "size": 3600, "label": "user\n(8,963 nodes\n5 features)"},
    "location":          {"color": "#2ECC71", "size": 3000, "label": "location\n(5 nodes\n3 features)"},
    "merchant_category": {"color": "#9B59B6", "size": 3000, "label": "merchant_category\n(5 nodes\n3 features)"},
}

# Manual positions for a clean layout
POS = {
    "transaction":       (0.5,  0.5),
    "user":              (0.0,  0.5),
    "location":          (0.5,  0.0),
    "merchant_category": (1.0,  0.5),
}

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.35, 1.05)
ax.axis("off")

# Draw directed edges with curved arrows
edge_styles = [
    ("user",        "transaction",       "performs\n(50,000 edges)",       -0.20),
    ("transaction", "location",          "at\n(50,000 edges)",              0.22),
    ("transaction", "merchant_category", "belongs_to\n(50,000 edges)",      0.22),
]

for src, dst, label, rad in edge_styles:
    x0, y0 = POS[src]
    x1, y1 = POS[dst]
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#555555",
            lw=2.0,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=22,
        ),
    )
    # Label at midpoint with slight offset
    mx = (x0 + x1) / 2 + (0.08 if rad > 0 else -0.08)
    my = (y0 + y1) / 2 + (0.06 if rad != 0 else 0)
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.8))

# Draw nodes
for ntype, meta in NODE_META.items():
    x, y = POS[ntype]
    circle = plt.Circle((x, y), 0.115, color=meta["color"],
                         zorder=3, ec="white", lw=3.0)
    ax.add_patch(circle)
    ax.text(x, y, meta["label"],
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white",
            zorder=4, linespacing=1.4)

# Legend patches
legend_patches = [
    mpatches.Patch(color=m["color"], label=ntype)
    for ntype, m in NODE_META.items()
]
ax.legend(handles=legend_patches, loc="lower center",
          ncol=4, frameon=True, fontsize=10,
          bbox_to_anchor=(0.5, -0.12))

ax.set_title("Heterogeneous Graph Schema — Fraud Detection\n"
             "(forward relations shown; all edges are bidirectional)",
             fontsize=15, fontweight="bold", pad=20)

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 14: Node Degree Distribution (User → Transaction)
# ─────────────────────────────────────────────────────────────────────────────

edge_index = data["user", "performs", "transaction"].edge_index  # [2, E]
user_ids   = edge_index[0].numpy()

n_users = data["user"].num_nodes
degree  = np.bincount(user_ids, minlength=n_users)  # degree per user

mean_deg   = degree.mean()
median_deg = np.median(degree)
max_deg    = degree.max()

print(f"User degree stats  (# transactions per user)")
print(f"  Mean   : {mean_deg:.2f}")
print(f"  Median : {median_deg:.0f}")
print(f"  Max    : {max_deg}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Section 14 — User Node Degree Distribution", y=1.01)

# ── Left: histogram + KDE ────────────────────────────────────────────────────
ax = axes[0]
sns.histplot(degree[degree > 0], bins=40, kde=True, color=ACCENT_COLOR,
             edgecolor="white", linewidth=0.5, ax=ax)
ax.axvline(mean_deg,   color="#E74C3C", lw=2, ls="--", label=f"Mean   = {mean_deg:.1f}")
ax.axvline(median_deg, color="#F39C12", lw=2, ls=":",  label=f"Median = {median_deg:.0f}")
ax.set_title("Distribution of Transactions per User")
ax.set_xlabel("Number of Transactions (degree)")
ax.set_ylabel("Number of Users")
ax.legend()

# ── Right: top-10 most active users ──────────────────────────────────────────
ax = axes[1]
top10_idx  = np.argsort(degree)[-10:][::-1]
top10_deg  = degree[top10_idx]
top10_labels = [f"User {i}" for i in top10_idx]

bars = ax.bar(top10_labels, top10_deg, color=ACCENT_COLOR,
              edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, top10_deg):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Top-10 Most Active Users by Transaction Count")
ax.set_xlabel("User ID")
ax.set_ylabel("Transaction Count")
ax.tick_params(axis="x", rotation=40)

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 15: Transaction Node Feature Analysis (PCA + t-SNE)
# ─────────────────────────────────────────────────────────────────────────────

features = data["transaction"].x.numpy()
labels   = data["transaction"].y.numpy()

n_legit_all = int((labels == 0).sum())
n_fraud_all = int((labels == 1).sum())

# ── PCA (full dataset) ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ── t-SNE (5,000 random samples) ─────────────────────────────────────────────
N_TSNE = 5_000
rng = np.random.default_rng(42)
idx_sample = rng.choice(len(labels), size=N_TSNE, replace=False)

X_tsne_in = X_scaled[idx_sample]
y_tsne    = labels[idx_sample]

print(f"Running t-SNE on {N_TSNE:,} samples (perplexity=30, n_iter=1000)…")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            random_state=42, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_tsne_in)
print("t-SNE complete.")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Section 15 — Transaction Feature Space Projection", y=1.01)

scatter_kw = dict(alpha=0.30, s=8, edgecolors="none")

for ax, X_2d, y_2d, title, subtitle in [
    (axes[0], X_pca, labels,
     "PCA  (2 Components)",
     f"All {len(labels):,} transactions  |  "
     f"Var. explained: {pca.explained_variance_ratio_.sum()*100:.1f}%"),
    (axes[1], X_tsne, y_tsne,
     "t-SNE  (perplexity=30, iter=1000)",
     f"Subsample: {N_TSNE:,} transactions"),
]:
    for cls, color, clabel in [
        (0, LEGIT_COLOR, f"Legit  (n={n_legit_all:,})"),
        (1, FRAUD_COLOR, f"Fraud  (n={n_fraud_all:,})"),
    ]:
        mask = y_2d == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=color, label=clabel, **scatter_kw)
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=3, frameon=True)

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 16: User Node Feature Analysis
# ─────────────────────────────────────────────────────────────────────────────

USER_FEATURE_NAMES = ["txn_count", "avg_amount", "fraud_rate",
                      "avg_balance", "avg_failed"]

user_x = data["user"].x.numpy()          # (8963, 5)
df_user = pd.DataFrame(user_x, columns=USER_FEATURE_NAMES)

HIGH_RISK_THRESH = 0.7
df_high_risk = df_user[df_user["fraud_rate"] > HIGH_RISK_THRESH]

print(f"High-risk users (fraud_rate > {HIGH_RISK_THRESH}): {len(df_high_risk):,}")
print("\nHigh-risk user feature stats:")
display(df_high_risk.describe().round(4))

# ── Top figure: distribution of each feature ──────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Section 16 — User Node Feature Distributions")

axes_flat = axes.flatten()

for i, feat in enumerate(USER_FEATURE_NAMES):
    ax = axes_flat[i]
    sns.histplot(df_user[feat], bins=50, kde=True,
                 color=ACCENT_COLOR, edgecolor="white",
                 linewidth=0.4, ax=ax)
    ax.axvline(df_user[feat].mean(),   color=FRAUD_COLOR, lw=1.8,
               ls="--", label=f"Mean={df_user[feat].mean():.3f}")
    ax.axvline(df_user[feat].median(), color="#F39C12", lw=1.8,
               ls=":",  label=f"Median={df_user[feat].median():.3f}")
    ax.set_title(f"Distribution of `{feat}`")
    ax.set_xlabel(feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

# Last subplot: fraud_rate vs txn_count scatter
ax = axes_flat[5]
sc = ax.scatter(df_user["txn_count"], df_user["fraud_rate"],
                c=df_user["fraud_rate"], cmap="RdYlGn_r",
                alpha=0.5, s=12, edgecolors="none")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("fraud_rate", fontsize=10)
ax.axhline(HIGH_RISK_THRESH, color=FRAUD_COLOR, lw=1.5, ls="--",
           label=f"High-risk threshold ({HIGH_RISK_THRESH})")
ax.set_title("Fraud Rate vs. Transaction Count per User")
ax.set_xlabel("txn_count")
ax.set_ylabel("fraud_rate")
ax.legend(fontsize=9)

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 17: Location & Merchant Category Analysis
# ─────────────────────────────────────────────────────────────────────────────

LOCATION_NAMES  = ["Sydney", "New York", "Mumbai", "Tokyo", "London"]
CATEGORY_NAMES  = ["Travel", "Clothing", "Restaurants", "Electronics", "Groceries"]

LOC_FEAT_NAMES  = ["txn_count", "avg_amount", "fraud_rate"]
CAT_FEAT_NAMES  = ["txn_count", "avg_amount", "fraud_rate"]

loc_x = data["location"].x.numpy()            # (5, 3)
cat_x = data["merchant_category"].x.numpy()   # (5 or 6, 3)

n_cats = cat_x.shape[0]
cat_names_used = CATEGORY_NAMES[:n_cats]

df_loc = pd.DataFrame(loc_x, columns=LOC_FEAT_NAMES, index=LOCATION_NAMES)
df_cat = pd.DataFrame(cat_x, columns=CAT_FEAT_NAMES, index=cat_names_used)

def _hbar_with_labels(ax, series, color, title, xlabel):
    """Horizontal bar chart with inline value labels."""
    bars = ax.barh(series.index, series.values, color=color,
                   edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, series.values):
        ax.text(bar.get_width() + series.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=10)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, series.values.max() * 1.18)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Section 17 — Location & Merchant Category Analysis")

# Row 0, Col 0 — fraud_rate per location
_hbar_with_labels(axes[0, 0], df_loc["fraud_rate"],
                  color=FRAUD_COLOR,
                  title="Fraud Rate by Location",
                  xlabel="Fraud Rate")

# Row 0, Col 1 — fraud_rate per category
_hbar_with_labels(axes[0, 1], df_cat["fraud_rate"],
                  color="#9B59B6",
                  title="Fraud Rate by Merchant Category",
                  xlabel="Fraud Rate")

# Row 1, Col 0 — avg_amount per location vs category (grouped)
x_pos   = np.arange(max(len(LOCATION_NAMES), n_cats))
bar_w   = 0.38

ax = axes[1, 0]
ax.bar(np.arange(len(LOCATION_NAMES)) - bar_w / 2,
       df_loc["avg_amount"], width=bar_w,
       color=ACCENT_COLOR, label="Location", edgecolor="white")
ax.bar(np.arange(n_cats) + bar_w / 2,
       df_cat["avg_amount"], width=bar_w,
       color="#F39C12", label="Merchant Cat.", edgecolor="white")

tick_positions = np.arange(max(len(LOCATION_NAMES), n_cats))
loc_labels_padded = LOCATION_NAMES + [""] * (n_cats - len(LOCATION_NAMES)) \
    if n_cats > len(LOCATION_NAMES) else LOCATION_NAMES
ax.set_xticks(tick_positions)
ax.set_xticklabels(loc_labels_padded, rotation=30, ha="right", fontsize=9)
ax.set_title("Average Transaction Amount\nby Location vs. Merchant Category")
ax.set_ylabel("Average Amount")
ax.legend()

# Row 1, Col 1 — txn_count per location vs category
ax = axes[1, 1]
ax.bar(np.arange(len(LOCATION_NAMES)) - bar_w / 2,
       df_loc["txn_count"], width=bar_w,
       color=LEGIT_COLOR, label="Location", edgecolor="white")
ax.bar(np.arange(n_cats) + bar_w / 2,
       df_cat["txn_count"], width=bar_w,
       color="#E67E22", label="Merchant Cat.", edgecolor="white")
ax.set_xticks(np.arange(max(len(LOCATION_NAMES), n_cats)))
ax.set_xticklabels(loc_labels_padded, rotation=30, ha="right", fontsize=9)
ax.set_title("Transaction Count\nby Location vs. Merchant Category")
ax.set_ylabel("Transaction Count")
ax.legend()

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 18: Graph Connectivity Insights
# ─────────────────────────────────────────────────────────────────────────────

connectivity_rows = []
for etype in data.edge_types:
    src_type, rel, dst_type = etype
    ei       = data[etype].edge_index
    n_edges  = ei.shape[1]
    n_src    = data[src_type].num_nodes
    avg_deg  = n_edges / n_src
    connectivity_rows.append({
        "Edge Type":           f"({src_type}, {rel}, {dst_type})",
        "Source Type":         src_type,
        "Relation":            rel,
        "Destination Type":    dst_type,
        "Total Edges":         n_edges,
        "Source Nodes":        n_src,
        "Avg Out-Degree":      round(avg_deg, 3),
    })

df_conn = pd.DataFrame(connectivity_rows)
print("Graph Connectivity Summary:")
display(df_conn[["Edge Type", "Total Edges", "Source Nodes", "Avg Out-Degree"]])

# ── Figure 1: avg degree bar chart ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Section 18 — Graph Connectivity Insights")

ax = axes[0]
short_labels = [f"({r['Source Type'][:4]}…,\n {r['Relation']},\n …{r['Destination Type'][:4]})"
                for _, r in df_conn.iterrows()]
bar_colors = plt.cm.tab10(np.linspace(0, 0.8, len(df_conn)))
bars = ax.bar(range(len(df_conn)), df_conn["Avg Out-Degree"],
              color=bar_colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, df_conn["Avg Out-Degree"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(df_conn)))
ax.set_xticklabels(short_labels, fontsize=8, linespacing=1.3)
ax.set_title("Average Out-Degree per Edge Type")
ax.set_ylabel("Avg Out-Degree (edges / source nodes)")

# ── Figure 2: transaction out-degree histogram (transaction → user edge) ──────
ax = axes[1]
txn_ei     = data["transaction", "performed_by", "user"].edge_index
txn_src    = txn_ei[0].numpy()
n_txn      = data["transaction"].num_nodes
txn_degree = np.bincount(txn_src, minlength=n_txn)

sns.histplot(txn_degree, bins=30, color=ACCENT_COLOR,
             edgecolor="white", linewidth=0.4, ax=ax, kde=True)
ax.axvline(txn_degree.mean(), color=FRAUD_COLOR, lw=2, ls="--",
           label=f"Mean = {txn_degree.mean():.2f}")
ax.set_title("Transaction Out-Degree Distribution\n"
             "(transaction → performed_by → user)")
ax.set_xlabel("Out-Degree (edges per transaction node)")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.show()


# CELL
# ─────────────────────────────────────────────────────────────────────────────
# Section 19: Train / Val / Test Split Visualization
# ─────────────────────────────────────────────────────────────────────────────

labels = data["transaction"].y.numpy()

split_info = {}
for split in ("train_mask", "val_mask", "test_mask"):
    mask = data["transaction"].get(split, None)
    if mask is not None:
        m       = mask.numpy().astype(bool)
        n_total = int(m.sum())
        n_fraud = int((labels[m] == 1).sum())
        n_legit = n_total - n_fraud
        split_info[split.replace("_mask", "")] = {
            "total": n_total,
            "fraud": n_fraud,
            "legit": n_legit,
            "fraud_pct": n_fraud / n_total * 100 if n_total else 0,
        }

split_labels = list(split_info.keys())
totals       = [split_info[s]["total"] for s in split_labels]
frauds       = [split_info[s]["fraud"] for s in split_labels]
legits       = [split_info[s]["legit"] for s in split_labels]

print("Train / Val / Test split summary:")
for s in split_labels:
    si = split_info[s]
    print(f"  {s:>6}: {si['total']:>6,} total  "
          f"| legit={si['legit']:>5,}  fraud={si['fraud']:>5,}  "
          f"({si['fraud_pct']:.2f}% fraud)")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Section 19 — Train / Val / Test Split Analysis")

# ── Left: pie chart of split sizes ────────────────────────────────────────────
ax = axes[0]
pie_colors = ["#3498DB", "#F39C12", "#9B59B6"]
explode    = [0.03] * len(split_labels)
wedges, texts, autotexts = ax.pie(
    totals,
    labels=[f"{s.capitalize()}\n({t:,})" for s, t in zip(split_labels, totals)],
    autopct="%1.1f%%",
    colors=pie_colors,
    explode=explode,
    startangle=90,
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax.set_title("Overall Split Proportions\n(number of transaction nodes)")

# ── Right: stacked bar — fraud vs legit in each split ────────────────────────
ax = axes[1]
x      = np.arange(len(split_labels))
bar_w  = 0.50

p1 = ax.bar(x, legits,  width=bar_w, label="Legit",
            color=LEGIT_COLOR, edgecolor="white")
p2 = ax.bar(x, frauds,  width=bar_w, bottom=legits, label="Fraud",
            color=FRAUD_COLOR, edgecolor="white")

# Value labels inside bars
for xi, (leg, frau, tot) in enumerate(zip(legits, frauds, totals)):
    ax.text(xi, leg / 2,         f"{leg:,}\n({100-split_info[split_labels[xi]]['fraud_pct']:.1f}%)",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(xi, leg + frau / 2,  f"{frau:,}\n({split_info[split_labels[xi]]['fraud_pct']:.1f}%)",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in split_labels], fontsize=12)
ax.set_title("Fraud vs. Legit Composition per Split\n(validates representative class ratio)")
ax.set_ylabel("Number of Transactions")
ax.legend(loc="upper right")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}"))

plt.tight_layout()
plt.show()

print("\nAll EDA sections (11–19) complete.")
