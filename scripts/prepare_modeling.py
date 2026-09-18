import pandas as pd
import numpy as np
import os

# Load full feature matrix
input_path = r"C:\doorway_bench\data\processed\rlkwic_full_features.parquet"
df = pd.read_parquet(input_path)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# 1. Define which columns are features
# ============================================================
# Non-feature columns: identifiers, labels, metadata
non_features = [
    'user_id', 'timestamp', 'event_type',
    'source_context', 'target_context',
    'device', 'label_source', 'confidence', 'dataset',
    'forgetting_proxy',  # this is the label
]

feature_cols = [c for c in df.columns if c not in non_features]
print(f"\nFeature columns ({len(feature_cols)}):")
for c in feature_cols:
    print(f"  {c}")

# ============================================================
# 2. Handle missing values
# ============================================================
# time_since_last_switch has 8 NaNs (first switch per user).
# Fill with the median (or a large value meaning "very long time")
df['time_since_last_switch'] = df['time_since_last_switch'].fillna(
    df['time_since_last_switch'].median()
)

print(f"\nMissing values after fill:")
print(df[feature_cols].isnull().sum().sum())

# ============================================================
# 3. Create X and y
# ============================================================
X = df[feature_cols].copy()
y = df['forgetting_proxy'].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y distribution:")
print(y.value_counts())
print(f"Positive rate: {y.mean():.2%}")

# ============================================================
# 4. Save
# ============================================================
output_dir = r"C:\doorway_bench\data\processed"
os.makedirs(output_dir, exist_ok=True)

X.to_parquet(os.path.join(output_dir, "X_features.parquet"), index=False)
y.to_frame('forgetting_proxy').to_parquet(
    os.path.join(output_dir, "y_labels.parquet"), index=False
)

# Save feature names
with open(os.path.join(output_dir, "feature_names.txt"), 'w') as f:
    for c in feature_cols:
        f.write(c + '\n')

# Save the full dataframe too (for reference)
df.to_parquet(os.path.join(output_dir, "rlkwic_modeling_ready.parquet"), index=False)

print(f"\nSaved X to {output_dir}/X_features.parquet")
print(f"Saved y to {output_dir}/y_labels.parquet")
print(f"Saved feature names to {output_dir}/feature_names.txt")

# ============================================================
# 5. Report feature statistics
# ============================================================
print(f"\n{'='*60}")
print("FEATURE STATISTICS")
print(f"{'='*60}")
print(X.describe().T.to_string())