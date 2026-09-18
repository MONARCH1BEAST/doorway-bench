import pandas as pd
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             brier_score_loss, accuracy_score)
from xgboost import XGBClassifier

processed = r"C:\doorway_bench\data\processed"
X = pd.read_parquet(os.path.join(processed, "X_features.parquet"))
y = pd.read_parquet(os.path.join(processed, "y_labels.parquet"))['forgetting_proxy']

for col in X.columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

with open(r"C:\doorway_bench\data\splits\splits.json", 'r') as f:
    splits = json.load(f)

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)

# ============================================================
# Pooled 5-fold CV for RF and XGBoost
# ============================================================
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=3, random_state=42, eval_metric='logloss',
        use_label_encoder=False
    ),
}

results = {}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"{model_name} — 5-fold CV")
    print(f"{'='*60}")
    
    fold_metrics = []
    for fold_i, fold in enumerate(splits['cv_5fold']):
        train_idx, test_idx = fold['train'], fold['test']
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        if y_train.nunique() < 2:
            continue
        
        # Clone model for each fold
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train)
        
        y_prob = m.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        fm = {
            'fold': fold_i,
            'auc_roc': safe_auc(y_test, y_prob),
            'auc_pr': average_precision_score(y_test, y_prob),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'brier': brier_score_loss(y_test, y_prob),
            'accuracy': accuracy_score(y_test, y_pred),
        }
        fold_metrics.append(fm)
        print(f"  Fold {fold_i}: AUC={fm['auc_roc']:.3f}, "
              f"F1={fm['f1']:.3f}, Brier={fm['brier']:.3f}")
    
    df_folds = pd.DataFrame(fold_metrics)
    results[model_name] = df_folds
    print(f"\n  Mean AUC-ROC: {df_folds['auc_roc'].mean():.4f} ± {df_folds['auc_roc'].std():.4f}")
    print(f"  Mean AUC-PR:  {df_folds['auc_pr'].mean():.4f} ± {df_folds['auc_pr'].std():.4f}")
    print(f"  Mean F1:      {df_folds['f1'].mean():.4f} ± {df_folds['f1'].std():.4f}")

# ============================================================
# Compare all three models
# ============================================================
print(f"\n{'='*60}")
print("MODEL COMPARISON (Pooled 5-fold CV)")
print(f"{'='*60}")

# Load logistic results
log_df = pd.read_csv(r"C:\doorway_bench\results\logistic_pooled_cv.csv")

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'AUC-ROC': [
        log_df['auc_roc'].mean(),
        results['RandomForest']['auc_roc'].mean(),
        results['XGBoost']['auc_roc'].mean(),
    ],
    'AUC-PR': [
        log_df['auc_pr'].mean(),
        results['RandomForest']['auc_pr'].mean(),
        results['XGBoost']['auc_pr'].mean(),
    ],
    'F1': [
        log_df['f1'].mean(),
        results['RandomForest']['f1'].mean(),
        results['XGBoost']['f1'].mean(),
    ],
    'Brier': [
        log_df['brier'].mean(),
        results['RandomForest']['brier'].mean(),
        results['XGBoost']['brier'].mean(),
    ],
})
print(comparison.to_string(index=False))

# ============================================================
# Feature importance (XGBoost on full data)
# ============================================================
print(f"\n{'='*60}")
print("XGBOOST FEATURE IMPORTANCE")
print(f"{'='*60}")
xgb_full = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=3, random_state=42, eval_metric='logloss',
    use_label_encoder=False
)
xgb_full.fit(X, y)
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_full.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp.to_string(index=False))

# Save
results_dir = r"C:\doorway_bench\results"
os.makedirs(results_dir, exist_ok=True)
comparison.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
feat_imp.to_csv(os.path.join(results_dir, "xgboost_importance.csv"), index=False)
print(f"\nSaved results to {results_dir}")