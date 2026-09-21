import pandas as pd
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import json

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cpu')
print(f"Device: {device}")

# ============================================================
# Load universal features
# ============================================================
universal_dir = r"C:\doorway_bench\data\processed\universal"
rlkwic = pd.read_parquet(os.path.join(universal_dir, "rlkwic_universal.parquet"))
mindful = pd.read_parquet(os.path.join(universal_dir, "mindful_universal.parquet"))

FEATURES = [
    'time_since_last_event', 'log_time_since_last',
    'event_rate_1min', 'event_rate_5min', 'event_rate_30min',
    'inter_event_mean', 'inter_event_std', 'inter_event_cv',
    'burstiness', 'fano_factor',
    'hour_sin', 'hour_cos',
    'session_position',
    'event_count_so_far', 'log_event_count',
]

# ============================================================
# Per-domain standardization
# ============================================================
def standardize_per_domain(df, features, name):
    df = df.copy()
    # Log transform skewed features
    for c in ['time_since_last_event', 'inter_event_mean', 'inter_event_std',
              'event_count_so_far']:
        df[c] = np.log1p(df[c].clip(lower=0))
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    # Clip to avoid extreme values from standardization
    df[features] = df[features].clip(-5, 5)
    print(f"  {name}: standardized {len(df)} events")
    return df

print("\nStandardizing features per domain...")
rlkwic_std = standardize_per_domain(rlkwic, FEATURES, "RLKWiC")
mindful_std = standardize_per_domain(mindful, FEATURES, "Mindful")

# Add domain label
rlkwic_std['domain'] = 0  # 0 = desktop
mindful_std['domain'] = 1  # 1 = mobile

# ============================================================
# Build combined dataset
# ============================================================
combined = pd.concat([
    rlkwic_std[FEATURES + ['forgetting_proxy', 'domain']],
    mindful_std[FEATURES + ['forgetting_proxy', 'domain']]
], ignore_index=True)

print(f"\nCombined dataset: {len(combined)} events")
print(f"  RLKWiC events: {len(rlkwic_std)}")
print(f"  Mindful events: {len(mindful_std)}")
print(f"  Positive rate: {combined['forgetting_proxy'].mean():.2%}")

# ============================================================
# Contrastive Dataset
# ============================================================
class ContrastiveDataset(Dataset):
    """
    Each __getitem__ returns (anchor, positive, negative) triplets.
    Positive: same label, different event.
    Negative: different label.
    """
    def __init__(self, X, y, domain):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.domain = torch.tensor(domain, dtype=torch.long)
        
        # Pre-index by label
        self.pos_idx = [np.where(y == 1)[0], np.where(y == 0)[0]]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        anchor_x = self.X[idx]
        anchor_y = self.y[idx]
        
        # Positive: same label, different event
        pos_pool = self.pos_idx[anchor_y]
        pos_idx = np.random.choice(pos_pool)
        while pos_idx == idx and len(pos_pool) > 1:
            pos_idx = np.random.choice(pos_pool)
        positive_x = self.X[pos_idx]
        
        # Negative: different label
        neg_pool = self.pos_idx[1 - anchor_y]
        neg_idx = np.random.choice(neg_pool)
        negative_x = self.X[neg_idx]
        
        return anchor_x, positive_x, negative_x

# ============================================================
# Encoder + Projection Head (SimCLR style)
# ============================================================
class Encoder(nn.Module):
    def __init__(self, input_dim=15, hidden=64, embed_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=16, proj_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

# ============================================================
# NT-Xent Loss
# ============================================================
def nt_xent_loss(anchor_z, pos_z, neg_z, temperature=0.5):
    """
    Contrastive loss: anchor should be closer to positive than to negative.
    """
    # Similarity of anchor to positive and negative
    sim_pos = F.cosine_similarity(anchor_z, pos_z, dim=-1) / temperature
    sim_neg = F.cosine_similarity(anchor_z, neg_z, dim=-1) / temperature
    
    # Softmax over {positive, negative} for each anchor
    logits = torch.stack([sim_pos, sim_neg], dim=1)  # (batch, 2)
    labels = torch.zeros(len(anchor_z), dtype=torch.long, device=anchor_z.device)
    
    return F.cross_entropy(logits, labels)

# ============================================================
# Train the encoder
# ============================================================
print(f"\n{'='*60}")
print("TRAINING CONTRASTIVE ENCODER")
print(f"{'='*60}")

X_all = combined[FEATURES].values
y_all = combined['forgetting_proxy'].values
domain_all = combined['domain'].values

dataset = ContrastiveDataset(X_all, y_all, domain_all)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

encoder = Encoder(input_dim=len(FEATURES)).to(device)
projection = ProjectionHead().to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(projection.parameters()),
    lr=1e-3, weight_decay=1e-4
)

EPOCHS = 30
for epoch in range(EPOCHS):
    encoder.train()
    projection.train()
    total_loss = 0.0
    n_batches = 0
    
    for anchor_x, pos_x, neg_x in loader:
        anchor_x = anchor_x.to(device)
        pos_x = pos_x.to(device)
        neg_x = neg_x.to(device)
        
        # Forward
        anchor_e = encoder(anchor_x)
        pos_e = encoder(pos_x)
        neg_e = encoder(neg_x)
        
        anchor_z = projection(anchor_e)
        pos_z = projection(pos_e)
        neg_z = projection(neg_e)
        
        loss = nt_xent_loss(anchor_z, pos_z, neg_z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}: loss = {total_loss/n_batches:.4f}")

# ============================================================
# Extract embeddings
# ============================================================
print(f"\n{'='*60}")
print("LINEAR PROBE EVALUATION ON RLKWIC")
print(f"{'='*60}")

encoder.eval()
with torch.no_grad():
    rlkwic_X = torch.tensor(rlkwic_std[FEATURES].values, dtype=torch.float32).to(device)
    rlkwic_embeddings = encoder(rlkwic_X).cpu().numpy()
    
    mindful_X = torch.tensor(mindful_std[FEATURES].values, dtype=torch.float32).to(device)
    mindful_embeddings = encoder(mindful_X).cpu().numpy()

y_rlkwic = rlkwic_std['forgetting_proxy'].values

print(f"RLKWiC embeddings: {rlkwic_embeddings.shape}")
print(f"Mindful embeddings: {mindful_embeddings.shape}")

# Load splits
with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

# ============================================================
# Linear probe on RLKWiC embeddings
# ============================================================
fold_results = []
for fold_i, fold in enumerate(splits['cv_5fold']):
    train_idx, test_idx = fold['train'], fold['test']
    
    X_train = rlkwic_embeddings[train_idx]
    y_train = y_rlkwic[train_idx]
    X_test = rlkwic_embeddings[test_idx]
    y_test = y_rlkwic[test_idx]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)
    fold_results.append({'fold': fold_i, 'auc_roc': auc, 'auc_pr': auc_pr})
    print(f"Fold {fold_i}: AUC={auc:.3f}, AUC-PR={auc_pr:.3f}")

df_folds = pd.DataFrame(fold_results)
print(f"\nLinear Probe on Contrastive Embeddings:")
print(f"  Mean AUC-ROC: {df_folds['auc_roc'].mean():.4f} ± {df_folds['auc_roc'].std():.4f}")
print(f"  Mean AUC-PR:  {df_folds['auc_pr'].mean():.4f} ± {df_folds['auc_pr'].std():.4f}")

# ============================================================
# Compare to baseline (raw features, no contrastive)
# ============================================================
print(f"\n{'='*60}")
print("BASELINE: Linear Probe on Raw Features")
print(f"{'='*60}")

X_raw = rlkwic_std[FEATURES].values
baseline_results = []
for fold_i, fold in enumerate(splits['cv_5fold']):
    train_idx, test_idx = fold['train'], fold['test']
    X_train = X_raw[train_idx]; y_train = y_rlkwic[train_idx]
    X_test = X_raw[test_idx];   y_test = y_rlkwic[test_idx]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    baseline_results.append({
        'fold': fold_i,
        'auc_roc': roc_auc_score(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
    })
    print(f"Fold {fold_i}: AUC={baseline_results[-1]['auc_roc']:.3f}")

df_baseline = pd.DataFrame(baseline_results)
print(f"\nBaseline (raw features):")
print(f"  Mean AUC-ROC: {df_baseline['auc_roc'].mean():.4f} ± {df_baseline['auc_roc'].std():.4f}")

# ============================================================
# Save
# ============================================================
results_dir = r"C:\doorway_bench\results"
os.makedirs(results_dir, exist_ok=True)
df_folds.to_csv(os.path.join(results_dir, "contrastive_probe.csv"), index=False)
df_baseline.to_csv(os.path.join(results_dir, "baseline_probe.csv"), index=False)

# Save encoder
torch.save(encoder.state_dict(), os.path.join(results_dir, "contrastive_encoder.pt"))
np.save(os.path.join(results_dir, "rlkwic_embeddings.npy"), rlkwic_embeddings)
np.save(os.path.join(results_dir, "mindful_embeddings.npy"), mindful_embeddings)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("CONTRASTIVE LEARNING — FINAL SUMMARY")
print(f"{'='*60}")
print(f"\nContrastive Probe AUC-ROC: {df_folds['auc_roc'].mean():.4f} ± {df_folds['auc_roc'].std():.4f}")
print(f"Baseline Probe AUC-ROC:    {df_baseline['auc_roc'].mean():.4f} ± {df_baseline['auc_roc'].std():.4f}")
print(f"Improvement:               {df_folds['auc_roc'].mean() - df_baseline['auc_roc'].mean():+.4f}")
print(f"\nPrevious XGBoost (raw features): 0.7001")
print(f"Contrastive + Linear Probe:      {df_folds['auc_roc'].mean():.4f}")