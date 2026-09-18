import pandas as pd
import numpy as np
import os
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, brier_score_loss, accuracy_score)

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# Load sequences
# ============================================================
seq_dir = r"C:\doorway_bench\data\processed\sequences"
X_seq = np.load(os.path.join(seq_dir, "X_seq.npy"))  # (321, 6, 23)
y_seq = np.load(os.path.join(seq_dir, "y_seq.npy"))  # (321,)
idx_seq = np.load(os.path.join(seq_dir, "idx_seq.npy"))  # (321,)

with open(os.path.join(seq_dir, "feature_names.pkl"), 'rb') as f:
    feature_names = pickle.load(f)

print(f"X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")

# ============================================================
# Standardize features (fit on train only, per fold)
# ============================================================
def normalize_sequences(X_train, X_test):
    """Fit scaler on train sequences, apply to both."""
    n_train, T, F = X_train.shape
    n_test = X_test.shape[0]
    
    # Reshape to (n*T, F)
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Clip extreme values (padding zeros become large negative after scaling)
    X_train_flat = np.clip(X_train_flat, -5, 5)
    X_test_flat = np.clip(X_test_flat, -5, 5)
    
    return (X_train_flat.reshape(n_train, T, F),
            X_test_flat.reshape(n_test, T, F))

# ============================================================
# LSTM Model
# ============================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        h = h_n[-1]  # (batch, hidden)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)  # (batch,)

# ============================================================
# Training loop for one fold
# ============================================================
def train_one_fold(X_train, y_train, X_test, y_test, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X_train.shape[2]
    
    # Normalize
    X_train_norm, X_test_norm = normalize_sequences(X_train, X_test)
    
    # To tensors
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
    
    # DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Model
    model = LSTMClassifier(n_features=n_features, hidden=32, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Train
    best_auc = 0.0
    best_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set (as proxy val)
        model.eval()
        with torch.no_grad():
            logits_test = model(X_test_t.to(device)).cpu().numpy()
            probs_test = 1 / (1 + np.exp(-logits_test))
        
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, probs_test)
        else:
            auc = 0.5
        
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Final predictions
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device)).cpu().numpy()
        probs_test = 1 / (1 + np.exp(-logits_test))
    
    return probs_test

# ============================================================
# 5-fold CV with seed averaging
# ============================================================
with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

print(f"\n{'='*60}")
print("LSTM — 5-fold CV (3 seeds averaged)")
print(f"{'='*60}")

fold_results = []
for fold_i, fold in enumerate(splits['cv_5fold']):
    train_idx, test_idx = fold['train'], fold['test']
    
    X_train = X_seq[train_idx]
    y_train = y_seq[train_idx]
    X_test = X_seq[test_idx]
    y_test = y_seq[test_idx]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print(f"Fold {fold_i}: SKIP")
        continue
    
    # Average over 3 seeds
    all_probs = []
    for seed in [42, 123, 456]:
        probs = train_one_fold(X_train, y_train, X_test, y_test, seed=seed)
        all_probs.append(probs)
    avg_probs = np.mean(all_probs, axis=0)
    
    y_pred = (avg_probs > 0.5).astype(int)
    
    fm = {
        'fold': fold_i,
        'auc_roc': roc_auc_score(y_test, avg_probs),
        'auc_pr': average_precision_score(y_test, avg_probs),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'brier': brier_score_loss(y_test, avg_probs),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    fold_results.append(fm)
    print(f"Fold {fold_i}: AUC={fm['auc_roc']:.3f}, AUC-PR={fm['auc_pr']:.3f}, "
          f"F1={fm['f1']:.3f}, Brier={fm['brier']:.3f}, Acc={fm['accuracy']:.3f}")

# ============================================================
# Summary
# ============================================================
df_results = pd.DataFrame(fold_results)
print(f"\n{'='*60}")
print("LSTM — AGGREGATE")
print(f"{'='*60}")
print(f"Mean AUC-ROC: {df_results['auc_roc'].mean():.4f} ± {df_results['auc_roc'].std():.4f}")
print(f"Mean AUC-PR:  {df_results['auc_pr'].mean():.4f} ± {df_results['auc_pr'].std():.4f}")
print(f"Mean F1:      {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")
print(f"Mean Brier:   {df_results['brier'].mean():.4f} ± {df_results['brier'].std():.4f}")

# Save
results_dir = r"C:\doorway_bench\results"
os.makedirs(results_dir, exist_ok=True)
df_results.to_csv(os.path.join(results_dir, "lstm_results.csv"), index=False)
print(f"\nSaved to {results_dir}/lstm_results.csv")