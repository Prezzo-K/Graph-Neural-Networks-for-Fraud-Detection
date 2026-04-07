"""Experiment 3: Both Risk_Score AND Failed_Transaction_Count_7d included."""
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
from src.model import create_model

CSV_PATH     = 'data/Fraud Detection Transactions Dataset.csv'
RANDOM_STATE = 42
RESULTS_FILE = 'results/exp3_both_leakage_results.txt'
EXTRA_COLS   = ['Risk_Score', 'Failed_Transaction_Count_7d']

def add_temporal(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour']      = df['Timestamp'].dt.hour
    df['Day']       = df['Timestamp'].dt.day
    df['Month']     = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    return df

print('[Traditional ML]')
df = add_temporal(pd.read_csv(CSV_PATH))
cat_cols = ['Transaction_Type', 'Device_Type', 'Card_Type', 'Authentication_Method']
df_enc   = pd.get_dummies(df, columns=cat_cols)
drop     = ['Transaction_ID', 'User_ID', 'Timestamp',
            'Location', 'Merchant_Category', 'Fraud_Label']
feat_cols = [c for c in df_enc.columns if c not in drop and c != 'Fraud_Label']
# Both leakage cols are kept (not in drop list)
X = df_enc[feat_cols].values
y = df['Fraud_Label'].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)
pos   = (y_tr == 1).sum()
neg   = (y_tr == 0).sum()
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
ml_res = {}
for name, clf in models.items():
    print(f'  Training {name}...', flush=True)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    try:    y_prob = clf.predict_proba(X_te)[:, 1]
    except: y_prob = clf.decision_function(X_te)
    ml_res[name] = {
        'accuracy':  accuracy_score(y_te, y_pred),
        'f1':        f1_score(y_te, y_pred, zero_division=0),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall':    recall_score(y_te, y_pred, zero_division=0),
        'auc_roc':   roc_auc_score(y_te, y_prob),
        'auc_pr':    average_precision_score(y_te, y_prob),
    }

print('\n[Building graph with both leakage features]')
df2 = add_temporal(pd.read_csv(CSV_PATH))
df2 = df2.sort_values('Timestamp').reset_index(drop=True)

num_cols = ['Transaction_Amount', 'Account_Balance', 'Daily_Transaction_Count',
            'Avg_Transaction_Amount_7d', 'Card_Age', 'Transaction_Distance',
            'Hour', 'Day', 'Month', 'DayOfWeek',
            'Risk_Score', 'Failed_Transaction_Count_7d']   # both included
cat_cols2 = ['Transaction_Type', 'Device_Type', 'Card_Type', 'Authentication_Method']

sc2 = StandardScaler()
df2[num_cols] = sc2.fit_transform(df2[num_cols])
df_enc2 = pd.get_dummies(df2, columns=cat_cols2)

always_drop = ['Transaction_ID', 'User_ID', 'Timestamp',
               'Location', 'Merchant_Category', 'Fraud_Label']
feat_cols2 = [c for c in df_enc2.columns if c not in always_drop]
txn_feats  = df_enc2[feat_cols2].values.astype(np.float32)
labels     = df2['Fraud_Label'].values.astype(np.int64)

def mapping(s):
    return {v: i for i, v in enumerate(s.unique())}

user_map = mapping(df2['User_ID'])
loc_map  = mapping(df2['Location'])
cat_map  = mapping(df2['Merchant_Category'])
user_idx = df2['User_ID'].map(user_map).values
loc_idx  = df2['Location'].map(loc_map).values
cat_idx  = df2['Merchant_Category'].map(cat_map).values
txn_idx  = np.arange(len(df2))

n      = len(df2)
t_end  = int(0.70 * n)
v_end  = int(0.85 * n)
tr_mask = torch.zeros(n, dtype=torch.bool)
va_mask = torch.zeros(n, dtype=torch.bool)
te_mask = torch.zeros(n, dtype=torch.bool)
tr_mask[:t_end]     = True
va_mask[t_end:v_end] = True
te_mask[v_end:]     = True

train_df2 = df2.iloc[:t_end]

def entity_stats(group_col, keys):
    s = (train_df2.groupby(group_col)
         .agg(txn_count=('Fraud_Label', 'count'),
              avg_amount=('Transaction_Amount', 'mean'),
              fraud_rate=('Fraud_Label', 'mean'))
         .reindex(keys, fill_value=0).values.astype(np.float32))
    return torch.tensor(s)

unique_users = list(user_map.keys())
user_stats   = (train_df2.groupby('User_ID')
                .agg(txn_count=('Fraud_Label', 'count'),
                     avg_amount=('Transaction_Amount', 'mean'),
                     fraud_rate=('Fraud_Label', 'mean'),
                     avg_balance=('Account_Balance', 'mean'))
                .reindex(unique_users, fill_value=0).values.astype(np.float32))

data = HeteroData()
data['transaction'].x          = torch.tensor(txn_feats)
data['transaction'].y          = torch.tensor(labels)
data['transaction'].train_mask = tr_mask
data['transaction'].val_mask   = va_mask
data['transaction'].test_mask  = te_mask
data['user'].x                 = torch.tensor(user_stats)
data['location'].x             = entity_stats('Location',          list(loc_map.keys()))
data['merchant_category'].x    = entity_stats('Merchant_Category', list(cat_map.keys()))

data['user', 'performs', 'transaction'].edge_index          = torch.tensor([user_idx, txn_idx], dtype=torch.long)
data['transaction', 'performed_by', 'user'].edge_index      = torch.tensor([txn_idx, user_idx], dtype=torch.long)
data['transaction', 'at', 'location'].edge_index            = torch.tensor([txn_idx, loc_idx],  dtype=torch.long)
data['location', 'is_site_of', 'transaction'].edge_index    = torch.tensor([loc_idx, txn_idx],  dtype=torch.long)
data['transaction', 'belongs_to', 'merchant_category'].edge_index = torch.tensor([txn_idx, cat_idx], dtype=torch.long)
data['merchant_category', 'contains', 'transaction'].edge_index   = torch.tensor([cat_idx, txn_idx], dtype=torch.long)

print('\n[GNN]')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data   = data.to(device)
train_labels = data['transaction'].y[data['transaction'].train_mask]
pos_weight   = torch.tensor(
    [(train_labels == 0).sum().item() / (train_labels == 1).sum().item()],
    device=device)

gnn_res = {}
for arch in ['sage', 'gat', 'hgt']:
    print(f'  Training {arch.upper()}...', flush=True)
    model = create_model(arch, data, hidden_channels=64, out_channels=1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

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

    best_f1, best_m = 0.0, {}
    for ep in range(1, 101):
        model.train()
        opt.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict)['transaction']
        loss = F.binary_cross_entropy_with_logits(
            out[data['transaction'].train_mask],
            data['transaction'].y[data['transaction'].train_mask].float().view(-1, 1),
            pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            vm = evaluate(data['transaction'].val_mask)
            print(f'    Epoch {ep:03d} | loss {loss.item():.4f} | '
                  f'val F1 {vm["f1"]:.4f} | val Rec {vm["rec"]:.4f}', flush=True)
            if vm['f1'] > best_f1:
                best_f1 = vm['f1']
                best_m  = evaluate(data['transaction'].test_mask)
    if not best_m:
        best_m = evaluate(data['transaction'].test_mask)
    gnn_res[arch] = best_m
    print(f'  {arch.upper()} test | F1 {best_m["f1"]:.4f}  '
          f'Rec {best_m["rec"]:.4f}  AUC-PR {best_m["auc_pr"]:.4f}')

os.makedirs('results', exist_ok=True)
with open(RESULTS_FILE, 'w') as f:
    f.write('Experiment: Exp 3 -- Risk_Score AND Failed_Transaction_Count_7d\n')
    f.write('=' * 60 + '\n\n')
    f.write('Traditional ML Models\n' + '-' * 40 + '\n')
    for name, m in ml_res.items():
        f.write(f'Model: {name}\n')
        f.write(f'  Accuracy:  {m["accuracy"]:.4f}\n')
        f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
        f.write(f'  Precision: {m["precision"]:.4f}\n')
        f.write(f'  Recall:    {m["recall"]:.4f}\n')
        f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
        f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
        f.write('-' * 40 + '\n')
    f.write('\nGNN Models\n' + '-' * 40 + '\n')
    for name, m in gnn_res.items():
        f.write(f'Model: {name.upper()}\n')
        f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
        f.write(f'  Precision: {m["prec"]:.4f}\n')
        f.write(f'  Recall:    {m["rec"]:.4f}\n')
        f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
        f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
        f.write('-' * 40 + '\n')

print(f'\nSaved to {RESULTS_FILE}')
