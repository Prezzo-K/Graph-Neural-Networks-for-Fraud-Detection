import sys
import os

# Allow running as either `python src/train.py` or `python -m src.train`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from src.model import create_model


def train(model_type='sage', hidden_channels=64, epochs=100, lr=0.01):
    """
    Train a heterogeneous GNN for fraud detection.

    Args:
        model_type:      'sage' | 'gat' | 'hgt'
        hidden_channels: size of hidden node embeddings (default 64)
        epochs:          number of training epochs (default 100)
        lr:              learning rate (default 0.01)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"  Model: {model_type.upper()}  |  hidden={hidden_channels}  |  lr={lr}")
    print(f"  Device: {device}")
    print(f"{'='*50}")

    # ── Load graph ────────────────────────────────────────────────────────────
    data_path = 'data/processed_graph.pt'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run src/data_preprocessing.py first.")
        return

    data = torch.load(data_path, weights_only=False)
    data = data.to(device)

    # ── Class imbalance weight ────────────────────────────────────────────────
    # pos_weight = neg / pos tells BCE to penalise missing a fraud case
    # proportionally more than flagging a legitimate transaction as fraud.
    train_labels = data['transaction'].y[data['transaction'].train_mask]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    print(f"  Train — Legit: {num_neg}, Fraud: {num_pos}, pos_weight: {pos_weight.item():.2f}\n")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = create_model(model_type, data, hidden_channels=hidden_channels, out_channels=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Evaluation helper ─────────────────────────────────────────────────────
    def evaluate(mask):
        model.eval()
        with torch.inference_mode():
            out    = model(data.x_dict, data.edge_index_dict)['transaction']
            probs  = torch.sigmoid(out).cpu().numpy().flatten()
            pred   = (out > 0).cpu().numpy().flatten().astype(float)
            labels = data['transaction'].y.cpu().numpy()

            mask_np  = mask.cpu().numpy()
            m_labels = labels[mask_np]
            m_pred   = pred[mask_np]
            m_probs  = probs[mask_np]

            return {
                'f1':      f1_score(m_labels, m_pred, zero_division=0),
                'prec':    precision_score(m_labels, m_pred, zero_division=0),
                'rec':     recall_score(m_labels, m_pred, zero_division=0),
                'auc_roc': roc_auc_score(m_labels, m_probs),
                'auc_pr':  average_precision_score(m_labels, m_probs),
            }

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    model_path = f'models/{model_type}_gnn.pt'

    best_val_f1 = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out  = model(data.x_dict, data.edge_index_dict)['transaction']
        loss = F.binary_cross_entropy_with_logits(
            out[data['transaction'].train_mask],
            data['transaction'].y[data['transaction'].train_mask].float().view(-1, 1),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            m = evaluate(data['transaction'].val_mask)
            print(
                f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                f"Val F1 {m['f1']:.4f} | Val AUC-PR {m['auc_pr']:.4f} | "
                f"P {m['prec']:.4f} | R {m['rec']:.4f}"
            )
            if m['f1'] > best_val_f1:
                best_val_f1 = m['f1']
                torch.save(model.state_dict(), model_path)

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n--- {model_type.upper()} Test Results ---")
    model.load_state_dict(torch.load(model_path, weights_only=False))
    m = evaluate(data['transaction'].test_mask)
    print(f"  F1:      {m['f1']:.4f}")
    print(f"  Prec:    {m['prec']:.4f}")
    print(f"  Recall:  {m['rec']:.4f}")
    print(f"  AUC-ROC: {m['auc_roc']:.4f}")
    print(f"  AUC-PR:  {m['auc_pr']:.4f}")

    return m


def save_results(results):
    """Save all model metrics to results/gnn_results.txt (mirrors traditional ML format)."""
    results_path = 'results/gnn_results.txt'
    with open(results_path, 'w') as f:
        f.write("GNN Models Performance\n")
        f.write("======================\n\n")
        for model_name, m in results.items():
            f.write(f"Model: {model_name.upper()} (GraphSAGE)\n" if model_name == 'sage'
                    else f"Model: {model_name.upper()}\n")
            f.write(f"  F1-Score:  {m['f1']:.4f}\n")
            f.write(f"  Precision: {m['prec']:.4f}\n")
            f.write(f"  Recall:    {m['rec']:.4f}\n")
            f.write(f"  AUC-ROC:   {m['auc_roc']:.4f}\n")
            f.write(f"  AUC-PR:    {m['auc_pr']:.4f}\n")
            f.write("-" * 35 + "\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    results = {}

    # Run all three models with identical settings for fair comparison
    for model_type in ['sage', 'gat', 'hgt']:
        results[model_type] = train(
            model_type=model_type,
            hidden_channels=64,
            epochs=100,
            lr=0.01,
        )

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>9} {'AUC-PR':>8}")
    print(f"{'-'*65}")
    for name, m in results.items():
        print(
            f"{name.upper():<8} {m['f1']:>8.4f} {m['prec']:>10.4f} "
            f"{m['rec']:>8.4f} {m['auc_roc']:>9.4f} {m['auc_pr']:>8.4f}"
        )
    print(f"{'='*65}")

    # ── Save metrics + models already saved per-epoch (best val F1) ───────────
    save_results(results)
    print("\nModels saved to models/sage_gnn.pt, models/gat_gnn.pt, models/hgt_gnn.pt")
