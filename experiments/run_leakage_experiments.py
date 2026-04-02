"""
Ablation Study: Effect of Data Leakage Features on Fraud Detection Performance
===============================================================================
This script runs two controlled experiments to quantify the performance gain
when historically excluded "leakage" features are re-introduced:

  Experiment 1 — Add Risk_Score
  Experiment 2 — Add Failed_Transaction_Count_7d

For each experiment, both Traditional ML baselines and GNN architectures
(GraphSAGE, GAT, HGT) are retrained from scratch using identical settings
to the clean baseline, with only the leakage feature added.

Results are saved to:
  results/exp1_risk_score_results.txt
  results/exp2_failed_7d_results.txt
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, roc_auc_score,
                             average_precision_score)
from xgboost import XGBClassifier
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero

from src.model import create_model

CSV_PATH = 'data/Fraud Detection Transactions Dataset.csv'
RANDOM_STATE = 42
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_tabular(extra_cols: list[str]):
    """Return (X_train, X_test, y_train, y_test) with extra_cols included."""
    df = pd.read_csv(CSV_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour']      = df['Timestamp'].dt.hour
    df['Day']       = df['Timestamp'].dt.day
    df['Month']     = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    always_drop = ['Transaction_ID', 'User_ID', 'Timestamp',
                   'Location', 'Merchant_Category', 'Fraud_Label']
    # Remove extra_cols from the drop list so they ARE included
    drop_cols = [c for c in always_drop if c not in extra_cols]

    cat_cols = ['Transaction_Type', 'Device_Type', 'Card_Type',
                'Authentication_Method']
    df_enc = pd.get_dummies(df, columns=cat_cols)

    feature_cols = [c for c in df_enc.columns if c not in drop_cols
                    and c not in ['Fraud_Label']]
    X = df_enc[feature_cols]
    y = df['Fraud_Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def run_traditional_ml(X_train, X_test, y_train, y_test):
    """Train and evaluate 5 traditional ML models. Returns dict of metrics."""
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos

    models = {
        'Random Forest':       RandomForestClassifier(
                                   n_estimators=100, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1),
        'Extra Trees':         ExtraTreesClassifier(
                                   n_estimators=100, class_weight='balanced',
                                   random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost':             XGBClassifier(
                                   n_estimators=100, scale_pos_weight=scale,
                                   random_state=RANDOM_STATE,
                                   eval_metric='logloss', verbosity=0),
        'Logistic Regression': LogisticRegression(
                                   class_weight='balanced', max_iter=1000,
                                   random_state=RANDOM_STATE),
        'SVM':                 LinearSVC(
                                   class_weight='balanced', max_iter=2000,
                                   random_state=RANDOM_STATE),
    }

    results = {}
    for name, clf in models.items():
        print(f'  Training {name}...', flush=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_prob = clf.decision_function(X_test)
        results[name] = {
            'accuracy':  accuracy_score(y_test, y_pred),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'auc_roc':   roc_auc_score(y_test, y_prob),
            'auc_pr':    average_precision_score(y_test, y_prob),
        }
    return results


def build_hetero_graph(extra_col: str | None):
    """
    Build a HeteroData graph identical to data_preprocessing.py but with
    one optional leakage column added to the transaction feature matrix.
    """
    from sklearn.preprocessing import StandardScaler as SS
    df = pd.read_csv(CSV_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['Hour']      = df['Timestamp'].dt.hour
    df['Day']       = df['Timestamp'].dt.day
    df['Month']     = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    num_cols = ['Transaction_Amount', 'Account_Balance',
                'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d',
                'Card_Age', 'Transaction_Distance',
                'Hour', 'Day', 'Month', 'DayOfWeek']
    if extra_col:
        num_cols.append(extra_col)

    cat_cols = ['Transaction_Type', 'Device_Type', 'Card_Type',
                'Authentication_Method']
    bin_cols = ['IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Is_Weekend']

    scaler = SS()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    df_enc = pd.get_dummies(df, columns=cat_cols)

    always_drop = ['Transaction_ID', 'User_ID', 'Timestamp',
                   'Location', 'Merchant_Category', 'Fraud_Label',
                   'Risk_Score', 'Failed_Transaction_Count_7d']
    if extra_col:
        always_drop = [c for c in always_drop if c != extra_col]

    feature_cols = [c for c in df_enc.columns if c not in always_drop]
    txn_feats = df_enc[feature_cols].values.astype(np.float32)
    labels    = df['Fraud_Label'].values.astype(np.int64)

    def mapping(series):
        u = series.unique()
        return {v: i for i, v in enumerate(u)}

    user_map = mapping(df['User_ID'])
    loc_map  = mapping(df['Location'])
    cat_map  = mapping(df['Merchant_Category'])

    user_idx = df['User_ID'].map(user_map).values
    loc_idx  = df['Location'].map(loc_map).values
    cat_idx  = df['Merchant_Category'].map(cat_map).values
    txn_idx  = np.arange(len(df))

    n = len(df)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[:train_end]       = True
    val_mask[train_end:val_end]  = True
    test_mask[val_end:]          = True

    train_df = df.iloc[:train_end]

    def entity_feats(group_col, keys):
        stats = train_df.groupby(group_col).agg(
            txn_count=('Fraud_Label', 'count'),
            avg_amount=('Transaction_Amount', 'mean'),
            fraud_rate=('Fraud_Label', 'mean'),
        ).reindex(keys, fill_value=0).values.astype(np.float32)
        return torch.tensor(stats)

    unique_users = list(user_map.keys())
    unique_locs  = list(loc_map.keys())
    unique_cats  = list(cat_map.keys())

    user_stats = train_df.groupby('User_ID').agg(
        txn_count=('Fraud_Label','count'),
        avg_amount=('Transaction_Amount','mean'),
        fraud_rate=('Fraud_Label','mean'),
        avg_balance=('Account_Balance','mean'),
    ).reindex(unique_users, fill_value=0).values.astype(np.float32)

    loc_stats = entity_feats('Location', unique_locs)
    cat_stats = entity_feats('Merchant_Category', unique_cats)

    data = HeteroData()
    data['transaction'].x           = torch.tensor(txn_feats)
    data['transaction'].y           = torch.tensor(labels)
    data['transaction'].train_mask  = train_mask
    data['transaction'].val_mask    = val_mask
    data['transaction'].test_mask   = test_mask
    data['user'].x                  = torch.tensor(user_stats)
    data['location'].x              = loc_stats
    data['merchant_category'].x     = cat_stats

    data['user', 'performs', 'transaction'].edge_index = torch.tensor(
        [user_idx, txn_idx], dtype=torch.long)
    data['transaction', 'performed_by', 'user'].edge_index = torch.tensor(
        [txn_idx, user_idx], dtype=torch.long)
    data['transaction', 'at', 'location'].edge_index = torch.tensor(
        [txn_idx, loc_idx], dtype=torch.long)
    data['location', 'is_site_of', 'transaction'].edge_index = torch.tensor(
        [loc_idx, txn_idx], dtype=torch.long)
    data['transaction', 'belongs_to', 'merchant_category'].edge_index = torch.tensor(
        [txn_idx, cat_idx], dtype=torch.long)
    data['merchant_category', 'contains', 'transaction'].edge_index = torch.tensor(
        [cat_idx, txn_idx], dtype=torch.long)
    return data


def run_gnn(data, model_type: str, epochs: int = 100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data   = data.to(device)

    train_labels = data['transaction'].y[data['transaction'].train_mask]
    num_pos   = (train_labels == 1).sum().item()
    num_neg   = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)

    model = create_model(model_type, data, hidden_channels=64, out_channels=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def evaluate(mask):
        model.eval()
        with torch.inference_mode():
            out   = model(data.x_dict, data.edge_index_dict)['transaction']
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            pred  = (out > 0).cpu().numpy().flatten().astype(float)
            labs  = data['transaction'].y.cpu().numpy()
            m     = mask.cpu().numpy()
            return {
                'f1':      f1_score(labs[m], pred[m], zero_division=0),
                'prec':    precision_score(labs[m], pred[m], zero_division=0),
                'rec':     recall_score(labs[m], pred[m], zero_division=0),
                'auc_roc': roc_auc_score(labs[m], probs[m]),
                'auc_pr':  average_precision_score(labs[m], probs[m]),
            }

    best_val_f1, best_metrics = 0.0, {}
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict)['transaction']
        loss = F.binary_cross_entropy_with_logits(
            out[data['transaction'].train_mask],
            data['transaction'].y[data['transaction'].train_mask].float().view(-1,1),
            pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            vm = evaluate(data['transaction'].val_mask)
            print(f'    Epoch {epoch:03d} | loss {loss.item():.4f} | '
                  f'val F1 {vm["f1"]:.4f} | val Rec {vm["rec"]:.4f}', flush=True)
            if vm['f1'] > best_val_f1:
                best_val_f1 = vm['f1']
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = evaluate(data['transaction'].test_mask)

    if not best_metrics:
        best_metrics = evaluate(data['transaction'].test_mask)
    return best_metrics


def save_results(path: str, exp_name: str, ml_results: dict, gnn_results: dict):
    with open(path, 'w') as f:
        f.write(f'Experiment: {exp_name}\n')
        f.write('=' * 60 + '\n\n')

        f.write('Traditional ML Models\n')
        f.write('-' * 40 + '\n')
        for name, m in ml_results.items():
            f.write(f'Model: {name}\n')
            f.write(f'  Accuracy:  {m["accuracy"]:.4f}\n')
            f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
            f.write(f'  Precision: {m["precision"]:.4f}\n')
            f.write(f'  Recall:    {m["recall"]:.4f}\n')
            f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
            f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
            f.write('-' * 40 + '\n')

        f.write('\nGNN Models\n')
        f.write('-' * 40 + '\n')
        for name, m in gnn_results.items():
            f.write(f'Model: {name.upper()}\n')
            f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
            f.write(f'  Precision: {m["prec"]:.4f}\n')
            f.write(f'  Recall:    {m["rec"]:.4f}\n')
            f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
            f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
            f.write('-' * 40 + '\n')
    print(f'\nResults saved to {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        'name':        'Exp 1 — With Risk_Score',
        'leakage_col': 'Risk_Score',
        'results_file': os.path.join(RESULTS_DIR, 'exp1_risk_score_results.txt'),
    },
    {
        'name':        'Exp 2 — With Failed_Transaction_Count_7d',
        'leakage_col': 'Failed_Transaction_Count_7d',
        'results_file': os.path.join(RESULTS_DIR, 'exp2_failed_7d_results.txt'),
    },
]

if __name__ == '__main__':
    for exp in EXPERIMENTS:
        col   = exp['leakage_col']
        name  = exp['name']
        fpath = exp['results_file']
        print(f'\n{"="*60}')
        print(f'  {name}')
        print(f'{"="*60}')

        # ── Traditional ML ────────────────────────────────────────
        print('\n[Traditional ML]')
        # For tabular, always_drop is handled inside build_tabular
        always_drop_base = ['Transaction_ID', 'User_ID', 'Timestamp',
                            'Location', 'Merchant_Category']
        df = pd.read_csv(CSV_PATH)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour']      = df['Timestamp'].dt.hour
        df['Day']       = df['Timestamp'].dt.day
        df['Month']     = df['Timestamp'].dt.month
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        cat_cols = ['Transaction_Type','Device_Type','Card_Type',
                    'Authentication_Method']
        df_enc   = pd.get_dummies(df, columns=cat_cols)
        # Drop the *other* leakage col, keep this exp's col
        other_leak = ('Failed_Transaction_Count_7d'
                      if col == 'Risk_Score' else 'Risk_Score')
        drop_list = always_drop_base + [other_leak]
        feat_cols = [c for c in df_enc.columns
                     if c not in drop_list and c != 'Fraud_Label']
        X = df_enc[feat_cols].values
        y = df['Fraud_Label'].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        ml_res = run_traditional_ml(X_tr, X_te, y_tr, y_te)

        # ── GNN ───────────────────────────────────────────────────
        print('\n[GNN]')
        graph = build_hetero_graph(extra_col=col)
        gnn_res = {}
        for arch in ['sage', 'gat', 'hgt']:
            print(f'  Training {arch.upper()}...', flush=True)
            gnn_res[arch] = run_gnn(graph, arch, epochs=100)
            m = gnn_res[arch]
            print(f'  {arch.upper()} test | F1 {m["f1"]:.4f}  '
                  f'Rec {m["rec"]:.4f}  AUC-PR {m["auc_pr"]:.4f}')

        save_results(fpath, name, ml_res, gnn_res)
        print(f'  Done: {fpath}')

    print('\nAll experiments complete.')
