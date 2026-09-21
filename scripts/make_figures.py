import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

results_dir = r"C:\doorway_bench\results"
figures_dir = r"C:\doorway_bench\figures"
os.makedirs(figures_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# ============================================================
# Figure 1: Model comparison
# ============================================================
models = ['XGBoost', 'Random Forest', 'TCN', 'LSTM', 'Transformer',
          'Contrastive Probe', 'Logistic Regression']
aucs = [0.700, 0.697, 0.675, 0.655, 0.650, 0.635, 0.634]
errors = [0.101, 0.097, 0.094, 0.090, 0.074, 0.105, 0.093]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#2ecc71' if a == max(aucs) else '#3498db' for a in aucs]
bars = ax.barh(models, aucs, xerr=errors, color=colors, capsize=4)
ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('AUC-ROC (5-fold CV)')
ax.set_title('Model Comparison on DoorwayBench')
ax.set_xlim(0.4, 0.85)
ax.legend(loc='lower right')
for bar, auc in zip(bars, aucs):
    ax.text(auc + 0.005, bar.get_y() + bar.get_height()/2,
            f'{auc:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'model_comparison.png'), bbox_inches='tight')
plt.close()
print("Saved model_comparison.png")

# ============================================================
# Figure 2: Feature importance
# ============================================================
xgb_path = os.path.join(results_dir, "xgboost_importance.csv")
if os.path.exists(xgb_path):
    xgb_imp = pd.read_csv(xgb_path)
    top10 = xgb_imp.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top10['feature'][::-1], top10['importance'][::-1], color='#3498db')
    ax.set_xlabel('XGBoost Importance')
    ax.set_title('Top 10 Features for Context-Boundary Forgetting')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'feature_importance.png'), bbox_inches='tight')
    plt.close()
    print("Saved feature_importance.png")

# ============================================================
# Figure 3: Power curve
# ============================================================
def compute_power(auc1, auc2, n):
    delta = abs(auc1 - auc2)
    avg_auc = (auc1 + auc2) / 2
    var_auc = avg_auc * (1 - avg_auc) / 0.25
    z_alpha = stats.norm.ppf(0.975)
    z_beta = (delta * np.sqrt(n / (2 * var_auc))) - z_alpha
    return stats.norm.cdf(z_beta)

ns = np.logspace(2, 5, 50).astype(int)
deltas = [0.05, 0.10, 0.15, 0.20]

fig, ax = plt.subplots(figsize=(8, 5))
for d in deltas:
    powers = [compute_power(0.65, 0.65 + d, n) for n in ns]
    ax.plot(ns, powers, label=f'ΔAUC = {d:.2f}')
ax.axhline(0.80, color='red', linestyle='--', alpha=0.5, label='80% power')
ax.axvline(321, color='green', linestyle='--', alpha=0.5, label='Current n=321')
ax.set_xscale('log')
ax.set_xlabel('Sample size (log scale)')
ax.set_ylabel('Statistical power')
ax.set_title('Power Analysis: Required Sample Size')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'power_curve.png'), bbox_inches='tight')
plt.close()
print("Saved power_curve.png")

# ============================================================
# Figure 4: Domain distributions
# ============================================================
rlkwic_path = r"C:\doorway_bench\data\processed\universal\rlkwic_universal.parquet"
mindful_path = r"C:\doorway_bench\data\processed\universal\mindful_universal.parquet"

if os.path.exists(rlkwic_path) and os.path.exists(mindful_path):
    rlkwic = pd.read_parquet(rlkwic_path)
    mindful = pd.read_parquet(mindful_path)

    features_to_plot = ['log_time_since_last', 'event_rate_5min', 'burstiness', 'fano_factor']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flatten(), features_to_plot):
        ax.hist(rlkwic[feat], bins=30, alpha=0.6, label='RLKWiC (desktop)',
                color='#e74c3c', density=True)
        ax.hist(mindful[feat], bins=30, alpha=0.6, label='Mindful (mobile)',
                color='#3498db', density=True)
        ax.set_title(feat)
        ax.legend()
    plt.suptitle('Domain Gap: Feature Distributions Differ Dramatically', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'domain_distributions.png'), bbox_inches='tight')
    plt.close()
    print("Saved domain_distributions.png")

print(f"\nAll figures saved to {figures_dir}")