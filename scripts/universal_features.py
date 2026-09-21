import pandas as pd
import numpy as np
import os

# ============================================================
# Universal feature computation
# ============================================================
def compute_universal_features(df, session_cols, time_col='timestamp', label_col='forgetting_proxy'):
    """
    Compute 15 universal temporal features from any timestamped event sequence.
    
    session_cols: list of columns that define a session (e.g., ['user_id'])
    """
    df = df.sort_values(session_cols + [time_col]).reset_index(drop=True)
    
    print(f"  Computing features for {len(df)} events across {df.groupby(session_cols).ngroups} sessions")
    
    # 1. time_since_last_event
    df['time_since_last_event'] = df.groupby(session_cols)[time_col].diff()
    df['time_since_last_event'] = df['time_since_last_event'].fillna(
        df['time_since_last_event'].median()
    )
    df['log_time_since_last'] = np.log1p(df['time_since_last_event'].clip(lower=0))
    
    # 2-4. Event rates (rolling counts within window)
    for window_name, window_seconds in [('1min', 60), ('5min', 300), ('30min', 1800)]:
        rates = []
        for _, group in df.groupby(session_cols):
            ts = group[time_col].values
            counts = np.zeros(len(ts))
            for i in range(len(ts)):
                counts[i] = ((ts >= ts[i] - window_seconds) & (ts <= ts[i])).sum() - 1
            rates.extend(counts.tolist())
        df[f'event_rate_{window_name}'] = rates
    
    # 5-7. Inter-event interval statistics (last 10 events)
    inter_mean, inter_std, inter_cv = [], [], []
    for _, group in df.groupby(session_cols):
        ts = group[time_col].values
        intervals = np.diff(ts)
        for i in range(len(ts)):
            if i == 0:
                recent = np.array([60.0])
            else:
                start = max(0, i - 10)
                recent = intervals[start:i] if i > 0 else np.array([60.0])
                if len(recent) == 0:
                    recent = np.array([60.0])
            m = np.mean(recent)
            s = np.std(recent)
            inter_mean.append(m)
            inter_std.append(s)
            inter_cv.append(s / (m + 1e-8))
    df['inter_event_mean'] = inter_mean
    df['inter_event_std'] = inter_std
    df['inter_event_cv'] = inter_cv
    
    # 8. Burstiness: (std - mean) / (std + mean)
    df['burstiness'] = (df['inter_event_std'] - df['inter_event_mean']) / (
        df['inter_event_std'] + df['inter_event_mean'] + 1e-8
    )
    
    # 9. Fano factor: variance / mean of event counts per minute
    fano = []
    for _, group in df.groupby(session_cols):
        ts = group[time_col].values
        if len(ts) < 2:
            fano.extend([0.0] * len(ts))
            continue
        bins = np.arange(ts.min(), ts.max() + 60, 60)
        counts, _ = np.histogram(ts, bins=bins)
        f = np.var(counts) / (np.mean(counts) + 1e-8)
        fano.extend([f] * len(ts))
    df['fano_factor'] = fano
    
    # 10-11. Hour sin/cos
    df['hour'] = (df[time_col] % 86400) / 3600.0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 12. Session position
    positions = []
    for _, group in df.groupby(session_cols):
        n = len(group)
        positions.extend([i / max(n - 1, 1) for i in range(n)])
    df['session_position'] = positions
    
    # 13-14. Event count so far
    df['event_count_so_far'] = df.groupby(session_cols).cumcount()
    df['log_event_count'] = np.log1p(df['event_count_so_far'])
    
    return df


# ============================================================
# Load and process RLKWiC
# ============================================================
print("=" * 60)
print("PROCESSING RLKWiC")
print("=" * 60)

rlkwic = pd.read_parquet(r"C:\doorway_bench\data\processed\rlkwic_full_features.parquet")
print(f"Loaded {len(rlkwic)} RLKWiC events")

rlkwic_univ = compute_universal_features(
    rlkwic, session_cols=['user_id'], time_col='timestamp', label_col='forgetting_proxy'
)

# ============================================================
# Load and process Mindful
# ============================================================
print("\n" + "=" * 60)
print("PROCESSING MINDFUL")
print("=" * 60)

mindful = pd.read_parquet(r"C:\doorway_bench\data\processed\mindful_secondary.parquet")
print(f"Loaded {len(mindful)} Mindful events")

# For Mindful, each user has two conditions. Create a session column.
mindful['session_id'] = mindful['user_id'] + "_" + mindful['forgetting_proxy'].astype(str)

mindful_univ = compute_universal_features(
    mindful, session_cols=['session_id'], time_col='timestamp', label_col='forgetting_proxy'
)

# ============================================================
# Define the 15 universal feature columns
# ============================================================
UNIVERSAL_FEATURES = [
    'time_since_last_event', 'log_time_since_last',
    'event_rate_1min', 'event_rate_5min', 'event_rate_30min',
    'inter_event_mean', 'inter_event_std', 'inter_event_cv',
    'burstiness', 'fano_factor',
    'hour_sin', 'hour_cos',
    'session_position',
    'event_count_so_far', 'log_event_count',
]

# ============================================================
# Save
# ============================================================
output_dir = r"C:\doorway_bench\data\processed\universal"
os.makedirs(output_dir, exist_ok=True)

rlkwic_univ[['user_id', 'timestamp', 'forgetting_proxy'] + UNIVERSAL_FEATURES].to_parquet(
    os.path.join(output_dir, "rlkwic_universal.parquet"), index=False
)
mindful_univ[['user_id', 'timestamp', 'forgetting_proxy'] + UNIVERSAL_FEATURES].to_parquet(
    os.path.join(output_dir, "mindful_universal.parquet"), index=False
)

# ============================================================
# Report
# ============================================================
print(f"\n{'='*60}")
print("UNIVERSAL FEATURES SUMMARY")
print(f"{'='*60}")
print(f"\nRLKWiC: {rlkwic_univ.shape[0]} events, {len(UNIVERSAL_FEATURES)} features")
print(f"Mindful: {mindful_univ.shape[0]} events, {len(UNIVERSAL_FEATURES)} features")
print(f"Total: {rlkwic_univ.shape[0] + mindful_univ.shape[0]} events")

print(f"\nRLKWiC feature statistics:")
print(rlkwic_univ[UNIVERSAL_FEATURES].describe().T[['mean', 'std', 'min', 'max']].to_string())

print(f"\nMindful feature statistics:")
print(mindful_univ[UNIVERSAL_FEATURES].describe().T[['mean', 'std', 'min', 'max']].to_string())

print(f"\nRLKWiC label distribution:")
print(rlkwic_univ['forgetting_proxy'].value_counts())
print(f"\nMindful label distribution:")
print(mindful_univ['forgetting_proxy'].value_counts())

print(f"\nSaved to {output_dir}")
print(f"  rlkwic_universal.parquet")
print(f"  mindful_universal.parquet")