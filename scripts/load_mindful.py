import pandas as pd
import os

# Path to the master files
master_folder = r"C:\doorway_bench\data\raw\mindful\master"

# Find the exact filenames
files = os.listdir(master_folder)
print("Files in master folder:")
for f in files:
    print(f"  {f}")

# Load the Master TouchLog
touch_path = os.path.join(master_folder, "Master_TouchLog_All")
if not os.path.exists(touch_path):
    # Try with .csv extension
    touch_path = os.path.join(master_folder, "Master_TouchLog_All.csv")

print(f"\nLoading touch log from: {touch_path}")
touch_df = pd.read_csv(touch_path)

print("\n--- TOUCH LOG INFO ---")
print(f"Shape: {touch_df.shape}")
print(f"Columns: {touch_df.columns.tolist()}")
print("\nFirst 5 rows:")
print(touch_df.head())
print("\nParticipants:", touch_df['Participant_ID'].nunique())
print("Conditions:", touch_df['Condition'].unique())
print("\nRows per condition:")
print(touch_df.groupby('Condition').size())

# Load the Master SensorLog
sensor_path = os.path.join(master_folder, "Master_SensorLog_All")
if not os.path.exists(sensor_path):
    sensor_path = os.path.join(master_folder, "Master_SensorLog_All.csv")

print(f"\n\nLoading sensor log from: {sensor_path}")
sensor_df = pd.read_csv(sensor_path)

print("\n--- SENSOR LOG INFO ---")
print(f"Shape: {sensor_df.shape}")
print(f"Columns: {sensor_df.columns.tolist()}")
print("\nFirst 5 rows:")
print(sensor_df.head())
print("\nRows per sensor type:")
print(sensor_df.groupby('Sensor_Type').size())