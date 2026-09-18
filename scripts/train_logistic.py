import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             brier_score_loss, accuracy_score)
from sklearn.pipeline import Pipeline

# ============================================================
# 1. Load data
# ============================================================
processed = r"C:\doorway_bench\data\processed"
X = pd.read_parquet(os.path.join(processed, "X_features.parquet"))
y = pd.read_parquet(os.path.join(processed, "y_labels.parquet"))['forgetting_proxy']

# Fill remaining NaNs (dwell_time_after)
for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

print(f"X shape: {X.shape}")
print(f"Missing values in X: {X.isnull().sum().sum()}")

with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

# ============================================================
# 2. Helper: safe AUC
# ============================================================
def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

# ============================================================
# 3. Per-user evaluation
# ============================================================
print(f"\n{'='*60}")
print("PER-USER RESULTS (Within-Temporal Split)")
print(f"{'='*60}")

all_metrics = []

for user in sorted(splits['within_temporal'].keys()):
    s = splits['within_temporal'][user]
    train_idx, val_idx, test_idx = s['train'], s['val'], s['test']
    
    X_train = X.iloc[train_idx]; y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx];  y_test = y.iloc[test_idx]
    
    # Skip if training has only one class
    if y_train.nunique() < 2:
        print(f"{user}: SKIP (train has only class {y_train.iloc[0]})")
        continue
    # Skip if test has only one class (can't compute AUC)
    if y_test.nunique() < 2:
        print(f"{user}: SKIP (test has only class {y_test.iloc[0]})")
        continue
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = {
        'user': user,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'auc_roc': safe_auc(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'brier': brier_score_loss(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    all_metrics.append(metrics)
    print(f"{user}: AUC={metrics['auc_roc']:.3f}, F1={metrics['f1']:.3f}, "
          f"Brier={metrics['brier']:.3f}, Acc={metrics['accuracy']:.3f}")

metrics_df = pd.DataFrame(all_metrics)
if len(metrics_df) > 0:
    print(f"\nMean AUC-ROC: {metrics_df['auc_roc'].mean():.4f} ± {metrics_df['auc_roc'].std():.4f}")
    print(f"Mean AUC-PR:  {metrics_df['auc_pr'].mean():.4f} ± {metrics_df['auc_pr'].std():.4f}")
    print(f"Mean F1:      {metrics_df['f1'].mean():.4f} ± {metrics_df['f1'].std():.4f}")

# ============================================================
# 4. POOLED 5-fold CV evaluation (more stable)
# ============================================================
print(f"\n{'='*60}")
print("POOLED 5-FOLD CV (Stable)")
print(f"{'='*60}")

pooled_metrics = []
for fold_i, fold in enumerate(splits['cv_5fold']):
    train_idx, test_idx = fold['train'], fold['test']
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    if y_train.nunique() < 2:
        print(f"Fold {fold_i}: SKIP (train one class)")
        continue
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    m = {
        'fold': fold_i,
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'auc_roc': safe_auc(y_test, y_prob),
        'auc_pr': average_precision_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'brier': brier_score_loss(y_test, y_prob),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    pooled_metrics.append(m)
    print(f"Fold {fold_i}: AUC={m['auc_roc']:.3f}, F1={m['f1']:.3f}, "
          f"Brier={m['brier']:.3f}, Acc={m['accuracy']:.3f}")

pooled_df = pd.DataFrame(pooled_metrics)
print(f"\nPooled Mean AUC-ROC: {pooled_df['auc_roc'].mean():.4f} ± {pooled_df['auc_roc'].std():.4f}")
print(f"Pooled Mean AUC-PR:  {pooled_df['auc_pr'].mean():.4f} ± {pooled_df['auc_pr'].std():.4f}")
print(f"Pooled Mean F1:      {pooled_df['f1'].mean():.4f} ± {pooled_df['f1'].std():.4f}")

# ============================================================
# 5. Save
# ============================================================
results_dir = r"C:\doorway_bench\results"
os.makedirs(results_dir, exist_ok=True)
metrics_df.to_csv(os.path.join(results_dir, "logistic_per_user.csv"), index=False)
pooled_df.to_csv(os.path.join(results_dir, "logistic_pooled_cv.csv"), index=False)
print(f"\nSaved results to {results_dir}")

# ============================================================
# 6. Feature importance (train on full data)
# ============================================================
print(f"\n{'='*60}")
print("FEATURE IMPORTANCE")
print(f"{'='*60}")
model_full = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
model_full.fit(X, y)
coefs = model_full.named_steps['clf'].coef_[0]
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'coefficient': coefs,
    'abs_coefficient': np.abs(coefs)
}).sort_values('abs_coefficient', ascending=False)
print(feat_imp.to_string(index=False))
feat_imp.to_csv(os.path.join(results_dir, "logistic_coefficients.csv"), index=False)