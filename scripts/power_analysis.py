import numpy as np
from scipy import stats

# ============================================================
# Power analysis: How many samples to detect AUC differences?
# ============================================================
# Use the formula for comparing two AUCs (Hanley & McNeil 1982)

def auc_sample_size(auc1, auc2, alpha=0.05, power=0.80, ratio=1.0):
    """
    Estimate required sample size to detect a difference between two AUCs.
    Based on Hanley & McNeil (1982).
    """
    # Variance approximation for AUC
    # V(AUC) ≈ AUC*(1-AUC) / (n_pos * n_neg / (n_pos + n_neg))
    # Simplified: use the variance of the difference
    
    # Effect size (Cohen's h-like for AUC)
    delta = abs(auc1 - auc2)
    
    # Average AUC for variance estimation
    avg_auc = (auc1 + auc2) / 2
    var_auc = avg_auc * (1 - avg_auc) / 0.25  # rough approximation
    
    # Z values
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size per group (simplified)
    n = 2 * ((z_alpha + z_beta) ** 2) * var_auc / (delta ** 2)
    
    return int(np.ceil(n))


print("=" * 70)
print("POWER ANALYSIS: Required sample size to detect AUC differences")
print("=" * 70)

# Current sample
n_current = 321
n_pos_current = 70
n_neg_current = 251

print(f"\nCurrent benchmark: n = {n_current} (pos = {n_pos_current}, neg = {n_neg_current})")
print(f"Positive rate: {n_pos_current/n_current:.1%}")

# Scenarios
scenarios = [
    (0.65, 0.70, "Detect LR → XGBoost (0.05 difference)"),
    (0.65, 0.75, "Detect a 0.10 difference"),
    (0.65, 0.80, "Detect a 0.15 difference"),
    (0.70, 0.75, "Detect RF → better model (0.05)"),
    (0.60, 0.75, "Detect a strong improvement (0.15)"),
]

print(f"\n{'Scenario':<45} {'Required n':>12} {'Multiplier':>12}")
print("-" * 70)
for auc1, auc2, label in scenarios:
    n_req = auc_sample_size(auc1, auc2)
    multiplier = n_req / n_current
    print(f"{label:<45} {n_req:>12,} {multiplier:>11.1f}x")

# ============================================================
# Power at current sample size
# ============================================================
print(f"\n{'='*70}")
print("POWER AT CURRENT SAMPLE SIZE (n=321)")
print(f"{'='*70}")

def compute_power(auc1, auc2, n):
    delta = abs(auc1 - auc2)
    avg_auc = (auc1 + auc2) / 2
    var_auc = avg_auc * (1 - avg_auc) / 0.25
    z_alpha = stats.norm.ppf(0.975)
    z_beta = (delta * np.sqrt(n / (2 * var_auc))) - z_alpha
    power = stats.norm.cdf(z_beta)
    return power

differences = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
print(f"\n{'ΔAUC':>8} {'Power at n=321':>18}")
print("-" * 30)
for d in differences:
    power = compute_power(0.65, 0.65 + d, n_current)
    print(f"{d:>8.3f} {power:>18.3f}")

# ============================================================
# What can we detect at n=321?
# ============================================================
print(f"\n{'='*70}")
print("MINIMUM DETECTABLE EFFECT (80% power, n=321)")
print(f"{'='*70}")

# Solve for delta given n and power=0.80
from scipy.optimize import brentq

def power_at_delta(delta, n=321):
    return compute_power(0.65, 0.65 + delta, n) - 0.80

delta_min = brentq(power_at_delta, 0.01, 0.50)
print(f"\nWith n=321 and 80% power, we can detect ΔAUC ≥ {delta_min:.3f}")
print(f"Any smaller difference would not reach statistical significance.")