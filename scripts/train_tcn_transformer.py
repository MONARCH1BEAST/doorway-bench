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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load sequences
seq_dir = r"C:\doorway_bench\data\processed\sequences"
X_seq = np.load(os.path.join(seq_dir, "X_seq.npy"))
y_seq = np.load(os.path.join(seq_dir, "y_seq.npy"))

# ============================================================
# Normalization
# ============================================================
def normalize_sequences(X_train, X_test):
    n_train, T, F = X_train.shape
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    X_train_flat = np.clip(X_train_flat, -5, 5)
    X_test_flat = np.clip(X_test_flat, -5, 5)
    return X_train_flat.reshape(n_train, T, F), X_test_flat.reshape(n_test, T, F)

# ============================================================
# TCN Model
# ============================================================
class TCNClassifier(nn.Module):
    def __init__(self, n_features, channels=32, kernel_size=3, dropout=0.3):
        super().__init__()
        # Two dilated causal conv layers
        self.conv1 = nn.Conv1d(n_features, channels, kernel_size, padding=kernel_size-1, dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)*2, dilation=2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features) -> (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        # Global max pooling
        x = x.max(dim=2)[0]  # (batch, channels)
        return self.fc(x).squeeze(-1)

# ============================================================
# Transformer Model
# ============================================================
class TransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model=32, nhead=2, num_layers=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=64
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        # Mean pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)

# ============================================================
# Training loop
# ============================================================
def train_one_fold(model_class, X_train, y_train, X_test, y_test, seed=42, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X_train.shape[2]
    X_train_norm, X_test_norm = normalize_sequences(X_train, X_test)
    
    X_train_t = torch.tensor(X_train_norm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model = model_class(n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_auc = 0.0
    best_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
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
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t.to(device)).cpu().numpy()
        probs_test = 1 / (1 + np.exp(-logits_test))
    
    return probs_test

# ============================================================
# 5-fold CV for both models
# ============================================================
with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

results_dir = r"C:\doorway_bench\results"
os.makedirs(results_dir, exist_ok=True)

for model_name, model_class in [("TCN", TCNClassifier), ("Transformer", TransformerClassifier)]:
    print(f"\n{'='*60}")
    print(f"{model_name} — 5-fold CV (3 seeds averaged)")
    print(f"{'='*60}")
    
    fold_results = []
    for fold_i, fold in enumerate(splits['cv_5fold']):
        train_idx, test_idx = fold['train'], fold['test']
        X_train = X_seq[train_idx]; y_train = y_seq[train_idx]
        X_test = X_seq[test_idx];   y_test = y_seq[test_idx]
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Fold {fold_i}: SKIP")
            continue
        
        all_probs = []
        for seed in [42, 123, 456]:
            probs = train_one_fold(model_class, X_train, y_train, X_test, y_test, seed=seed)
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
        print(f"Fold {fold_i}: AUC={fm['auc_roc']:.3f}, F1={fm['f1']:.3f}, Brier={fm['brier']:.3f}")
    
    df_results = pd.DataFrame(fold_results)
    print(f"\n{model_name} Mean AUC-ROC: {df_results['auc_roc'].mean():.4f} ± {df_results['auc_roc'].std():.4f}")
    print(f"{model_name} Mean AUC-PR:  {df_results['auc_pr'].mean():.4f} ± {df_results['auc_pr'].std():.4f}")
    print(f"{model_name} Mean F1:      {df_results['f1'].mean():.4f} ± {df_results['f1'].std():.4f}")
    print(f"{model_name} Mean Brier:   {df_results['brier'].mean():.4f} ± {df_results['brier'].std():.4f}")
    
    df_results.to_csv(os.path.join(results_dir, f"{model_name.lower()}_results.csv"), index=False)

print(f"\nSaved results to {results_dir}")