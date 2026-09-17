import pandas as pd
import os

input_path = r"C:\doorway_bench\data\raw\kaggle\mental_health_digital_behavior_data.csv"
output_path = r"C:\doorway_bench\data\processed\kaggle_reference.parquet"

df = pd.read_csv(input_path)

# Add a synthetic row ID (each row is an anonymous daily summary)
df['row_id'] = range(len(df))

# Add our standard dataset column
df['dataset'] = 'kaggle_reference'

# Create a forgetting proxy for validation purposes only
# focus_score < 4 AND num_app_switches > 50 = forgetting day
df['forgetting_proxy'] = ((df['focus_score'] < 4) & (df['num_app_switches'] > 50)).astype(int)

# Save
try:
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\kaggle_reference.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

print(f"\nTotal rows: {len(df)}")
print(f"Forgetting proxy distribution:")
print(df['forgetting_proxy'].value_counts())
print(f"\nCorrelation between num_app_switches and focus_score:")
print(df[['num_app_switches', 'focus_score']].corr())
print(f"\nFirst 5 rows:")
print(df.head())