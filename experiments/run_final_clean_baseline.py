"""
Final Clean Baseline Run — LOCKED RESULTS
==========================================
This is the single authoritative run for the project.
No leakage features (Risk_Score, Failed_Transaction_Count_7d excluded).
Results saved to:
  results/traditional_ml_results.txt   (overwrite)
  results/gnn_results.txt              (overwrite)
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
from src.model import create_model
from src.data_preprocessing import preprocess_and_create_graph

CSV_PATH     = 'data/Fraud Detection Transactions Dataset.csv'
GRAPH_PATH   = 'data/processed_graph.pt'
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

print('='*60)
print('  FINAL CLEAN BASELINE — Traditional ML')
print('='*60)

df = pd.read_csv(CSV_PATH)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
for feat, fn in [('Hour','hour'),('Day','day'),('Month','month'),('DayOfWeek','dayofweek')]:
    df[feat] = getattr(df['Timestamp'].dt, fn)

cat_cols = ['Transaction_Type','Device_Type','Card_Type','Authentication_Method']
df_enc   = pd.get_dummies(df, columns=cat_cols)

# Strictly exclude both leakage columns
drop = ['Transaction_ID','User_ID','Timestamp','Location','Merchant_Category',
        'Fraud_Label','Risk_Score','Failed_Transaction_Count_7d']
feat_cols = [c for c in df_enc.columns if c not in drop]
X = df_enc[feat_cols].values
y = df['Fraud_Label'].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
sc   = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

pos   = (y_tr == 1).sum()
neg   = (y_tr == 0).sum()
scale = neg / pos
print(f'Train: {len(y_tr)} | Test: {len(y_te)} | pos_weight: {scale:.2f}')

models = {
    'Random Forest Classifier':       RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1),
    'Extra Trees Classifier':         ExtraTreesClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost Classifier':             XGBClassifier(
        n_estimators=100, scale_pos_weight=scale,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'Logistic Regression':            LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
    'SVM Classifier':                 LinearSVC(
        class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE),
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
    m = ml_res[name]
    print(f'    F1 {m["f1"]:.4f} | Prec {m["precision"]:.4f} | '
          f'Rec {m["recall"]:.4f} | AUC-ROC {m["auc_roc"]:.4f}')

os.makedirs('results', exist_ok=True)
with open('results/traditional_ml_results.txt', 'w') as f:
    f.write('Traditional ML Models Performance\n')
    f.write('=================================\n\n')
    for name, m in ml_res.items():
        f.write(f'Model: {name}\n')
        f.write(f'  Accuracy:  {m["accuracy"]:.4f}\n')
        f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
        f.write(f'  Precision: {m["precision"]:.4f}\n')
        f.write(f'  Recall:    {m["recall"]:.4f}\n')
        f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
        f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
        f.write('-' * 35 + '\n')
print('\nSaved: results/traditional_ml_results.txt')

print('\n' + '='*60)
print('  FINAL CLEAN BASELINE — GNN')
print('='*60)

data = torch.load(GRAPH_PATH, weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
data = data.to(device)

train_labels = data['transaction'].y[data['transaction'].train_mask]
pos_weight   = torch.tensor(
    [(train_labels == 0).sum().item() / (train_labels == 1).sum().item()],
    device=device)
print(f'Graph pos_weight: {pos_weight.item():.2f}')

def evaluate_gnn(model, mask):
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

gnn_res = {}
for arch in ['sage', 'gat', 'hgt']:
    print(f'\n  Training {arch.upper()} (100 epochs)...', flush=True)
    torch.manual_seed(RANDOM_STATE)
    model = create_model(arch, data, hidden_channels=64, out_channels=1).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_f1, best_state = 0.0, None
    for ep in range(1, 101):
        model.train()
        opt.zero_grad()
        out  = model(data.x_dict, data.edge_index_dict)['transaction']
        loss = F.binary_cross_entropy_with_logits(
            out[data['transaction'].train_mask],
            data['transaction'].y[data['transaction'].train_mask].float().view(-1,1),
            pos_weight=pos_weight)
        loss.backward()
        opt.step()
        if ep % 20 == 0:
            vm = evaluate_gnn(model, data['transaction'].val_mask)
            print(f'    Epoch {ep:03d} | loss {loss.item():.4f} | '
                  f'val F1 {vm["f1"]:.4f} | val Rec {vm["rec"]:.4f}', flush=True)
            if vm['f1'] > best_val_f1:
                best_val_f1 = vm['f1']
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    test_m = evaluate_gnn(model, data['transaction'].test_mask)
    gnn_res[arch] = test_m
    print(f'  {arch.upper()} FINAL | F1 {test_m["f1"]:.4f} | '
          f'Prec {test_m["prec"]:.4f} | Rec {test_m["rec"]:.4f} | '
          f'AUC-ROC {test_m["auc_roc"]:.4f} | AUC-PR {test_m["auc_pr"]:.4f}')

with open('results/gnn_results.txt', 'w') as f:
    f.write('GNN Models Performance\n')
    f.write('======================\n\n')
    for arch, m in gnn_res.items():
        label = 'GraphSAGE' if arch == 'sage' else arch.upper()
        f.write(f'Model: {label}\n')
        f.write(f'  F1-Score:  {m["f1"]:.4f}\n')
        f.write(f'  Precision: {m["prec"]:.4f}\n')
        f.write(f'  Recall:    {m["rec"]:.4f}\n')
        f.write(f'  AUC-ROC:   {m["auc_roc"]:.4f}\n')
        f.write(f'  AUC-PR:    {m["auc_pr"]:.4f}\n')
        f.write('-' * 35 + '\n')
print('\nSaved: results/gnn_results.txt')

print('\n' + '='*65)
print(f'{"Model":<28} {"F1":>6} {"Prec":>6} {"Recall":>7} {"AUC-ROC":>8}')
print('-'*65)
for name, m in ml_res.items():
    short = name.replace(' Classifier','').replace(' Regression',' Reg.')
    print(f'{short:<28} {m["f1"]:>6.4f} {m["precision"]:>6.4f} '
          f'{m["recall"]:>7.4f} {m["auc_roc"]:>8.4f}')
print('-'*65)
for arch, m in gnn_res.items():
    label = {'sage':'GraphSAGE','gat':'GAT','hgt':'HGT'}[arch]
    print(f'{label:<28} {m["f1"]:>6.4f} {m["prec"]:>6.4f} '
          f'{m["rec"]:>7.4f} {m["auc_roc"]:>8.4f}')
print('='*65)
print('\nFINAL RESULTS LOCKED.')
