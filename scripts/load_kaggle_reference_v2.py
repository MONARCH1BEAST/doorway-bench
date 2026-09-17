import pandas as pd
import os

input_path = r"C:\doorway_bench\data\raw\kaggle\mental_health_digital_behavior_data.csv"
output_path = r"C:\doorway_bench\data\processed\kaggle_reference.parquet"

df = pd.read_csv(input_path)
df['row_id'] = range(len(df))
df['dataset'] = 'kaggle_reference'

# Create a composite "cognitive strain" score
# High switches + low focus + low mood + high anxiety = forgetting risk
df['strain_score'] = (
    (df['num_app_switches'] - df['num_app_switches'].min()) / 
    (df['num_app_switches'].max() - df['num_app_switches'].min())
    - (df['focus_score'] - df['focus_score'].min()) / 
    (df['focus_score'].max() - df['focus_score'].min())
    + (df['anxiety_level'] - df['anxiety_level'].min()) / 
    (df['anxiety_level'].max() - df['anxiety_level'].min())
    - (df['mood_score'] - df['mood_score'].min()) / 
    (df['mood_score'].max() - df['mood_score'].min())
)

# Label top 20% strain as forgetting
threshold = df['strain_score'].quantile(0.80)
df['forgetting_proxy'] = (df['strain_score'] > threshold).astype(int)

print(f"Strain threshold (80th percentile): {threshold:.4f}")
print(f"Forgetting proxy distribution:")
print(df['forgetting_proxy'].value_counts())
print(f"\nMean features by forgetting label:")
print(df.groupby('forgetting_proxy')[['num_app_switches', 'focus_score', 
                                       'anxiety_level', 'mood_score']].mean())

# Save
try:
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
except:
    output_path = r"C:\doorway_bench\data\processed\kaggle_reference.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

print(f"\nCorrelations:")
print(df[['num_app_switches', 'focus_score', 'anxiety_level', 
          'mood_score', 'strain_score']].corr()['strain_score'])