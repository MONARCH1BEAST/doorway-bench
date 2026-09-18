import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from scipy import stats

# ============================================================
# Load data
# ============================================================
processed = r"C:\doorway_bench\data\processed"
X = pd.read_parquet(os.path.join(processed, "X_features.parquet"))
y = pd.read_parquet(os.path.join(processed, "y_labels.parquet"))['forgetting_proxy']
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

# ============================================================
# DeLong test implementation
# ============================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def delong_auc(y_true, y_score):
    """Compute AUC and its variance using DeLong method."""
    order = np.argsort(-y_score)
    y_true = np.array(y_true)[order]
    y_score = np.array(y_score)[order]
    
    n1 = int(np.sum(y_true))
    n0 = len(y_true) - n1
    
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan
    
    # Simplified DeLong for single AUC (bootstrap for CI)
    auc = roc_auc_score(y_true, y_score)
    return auc, np.nan  # We'll use bootstrap for CI

def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05):
    """Bootstrap CI for AUC."""
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    scores = np.array(scores)
    return np.mean(scores), np.percentile(scores, 100*alpha/2), np.percentile(scores, 100*(1-alpha/2))

# ============================================================
# Collect out-of-fold predictions for each model
# ============================================================
models = {
    'LogisticRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ]),
    'RandomForest': RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=3, random_state=42, eval_metric='logloss'
    ),
}

oof_predictions = {}  # out-of-fold predictions

for model_name, model in models.items():
    oof_probs = np.full(len(X), np.nan)
    for fold in splits['cv_5fold']:
        train_idx, test_idx = fold['train'], fold['test']
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        
        if y_train.nunique() < 2:
            continue
        
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train)
        oof_probs[test_idx] = m.predict_proba(X_test)[:, 1]
    
    oof_predictions[model_name] = oof_probs

# Add neural network OOF results if we had them saved — for now use feature models only
# (We'll compare feature vs neural in a different way)

# ============================================================
# Bootstrap CIs for all models
# ============================================================
print(f"\n{'='*70}")
print("BOOTSTRAP 95% CONFIDENCE INTERVALS FOR AUC-ROC")
print(f"{'='*70}")

valid_mask = ~np.isnan(oof_predictions['XGBoost'])
y_valid = y.values[valid_mask]

for model_name, probs in oof_predictions.items():
    if np.isnan(probs).any():
        probs = probs[valid_mask]
    
    mean_auc, ci_low, ci_high = bootstrap_auc_ci(y_valid, probs)
    print(f"{model_name:22s}: AUC = {mean_auc:.4f}  [95% CI: {ci_low:.4f}, {ci_high:.4f}]")

# ============================================================
# Pairwise comparison: bootstrap difference between models
# ============================================================
print(f"\n{'='*70}")
print("PAIRWISE AUC DIFFERENCES (Bootstrap)")
print(f"{'='*70}")

model_names = list(oof_predictions.keys())
n_boot = 2000
rng = np.random.default_rng(42)

for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        p1, p2 = oof_predictions[m1][valid_mask], oof_predictions[m2][valid_mask]
        
        diffs = []
        n = len(y_valid)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_valid[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y_valid[idx], p1[idx]) -
                         roc_auc_score(y_valid[idx], p2[idx]))
        
        diffs = np.array(diffs)
        ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
        p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{m1:22s} vs {m2:22s}: ΔAUC={diffs.mean():+.4f} "
              f"[95% CI: {ci_low:+.4f}, {ci_high:+.4f}] p={p_value:.4f} {sig}")

print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")