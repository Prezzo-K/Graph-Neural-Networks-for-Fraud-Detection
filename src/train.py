import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from src.model import create_hetero_model
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = 'data/processed_graph.pt'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run src/data_preprocessing.py first.")
        return

    data = torch.load(data_path, weights_only=False)
    
    # Initialize features for nodes without them (user, location, merchant_category)
    # We'll use random features or learnable embeddings. For simplicity, identity-like or random.
    for node_type in data.node_types:
        if 'x' not in data[node_type]:
            # Initialize with random features if no features are present
            data[node_type].x = torch.randn(data[node_type].num_nodes, 16)
    
    data = data.to(device)

    # Calculate class weights for imbalance handling
    train_labels = data['transaction'].y[data['transaction'].train_mask]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    print(f"Positive samples: {num_pos}, Negative samples: {num_neg}, pos_weight: {pos_weight.item():.2f}")

    # Model, Optimizer
    model = create_hetero_model(data, hidden_channels=64, out_channels=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def evaluate(mask):
        model.eval()
        with torch.inference_mode():
            out = model(data.x_dict, data.edge_index_dict)['transaction']
            pred = (out > 0).float().cpu().numpy()
            probs = torch.sigmoid(out).cpu().numpy()
            labels = data['transaction'].y.cpu().numpy()
            
            mask_np = mask.cpu().numpy()
            m_labels = labels[mask_np]
            m_pred = pred[mask_np]
            m_probs = probs[mask_np]
            
            f1 = f1_score(m_labels, m_pred)
            precision = precision_score(m_labels, m_pred, zero_division=0)
            recall = recall_score(m_labels, m_pred, zero_division=0)
            auc_roc = roc_auc_score(m_labels, m_probs)
            auc_pr = average_precision_score(m_labels, m_probs)
            
            return f1, precision, recall, auc_roc, auc_pr

    # Training loop
    best_val_f1 = 0
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x_dict, data.edge_index_dict)['transaction']
        loss = F.binary_cross_entropy_with_logits(
            out[data['transaction'].train_mask], 
            data['transaction'].y[data['transaction'].train_mask].float().view(-1, 1),
            pos_weight=pos_weight
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            val_f1, val_p, val_r, val_auc, val_pr = evaluate(data['transaction'].val_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val F1: {val_f1:.4f}, Val AUC-PR: {val_pr:.4f}')
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'results/best_gnn_model.pt')

    # Final Test Evaluation
    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load('results/best_gnn_model.pt', weights_only=False))
    test_f1, test_p, test_r, test_auc, test_pr = evaluate(data['transaction'].test_mask)
    print(f'Test F1: {test_f1:.4f}')
    print(f'Test Precision: {test_p:.4f}')
    print(f'Test Recall: {test_r:.4f}')
    print(f'Test AUC-ROC: {test_auc:.4f}')
    print(f'Test AUC-PR: {test_pr:.4f}')

if __name__ == "__main__":
    train()
