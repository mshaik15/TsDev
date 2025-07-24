from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#================================================================
# 1. Infer the time interval from the timestamps in the DataFrame
#================================================================
def infer_time_interval(df) -> int:
    default = 1 # Set a default time interval in seconds
    if len(df) < 2:
        print("Not enough timestamps to make an interval")
        return default
    
    df = df.copy() # Make a copy
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    intervals = df["timestamp"].diff().dropna() # Create a list of intervals
    intervals_seconds = [int(i.total_seconds()) for i in intervals] # Convert all intervals into seconds

    interval_count = Counter(intervals_seconds)
    t_rec, _ = interval_count.most_common(1)[0]

    response = input(f"Given time interval {t_rec}. Continue? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        return t_rec
    else:
        try:
            user_input = int(input("Input time interval in seconds: ").strip())
            print(f"Using user-defined time {user_input}")
            return user_input
        except ValueError:
            print("Invalid input. Using default time interval.")
            return default

#================================================================
# 2. Create full time index
#================================================================
def create_full_time_index (df, t_rec) -> pd.DatetimeIndex:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    start = df["timestamp"].min()
    end = df["timestamp"].max()

    full_index = pd.date_range(start=start, end=end, freq=f"{t_rec}s")
    return full_index

#================================================================
# 3. Reindex and find gaps
#================================================================
def reindex_and_find_gaps(df, full_index) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    reindexed_df = df.reindex(full_index)
    missing = reindexed_df[reindexed_df.isnull().any(axis=1)].index
    return missing

#================================================================
# 4. Fill in missing stats
#================================================================
def get_gap_stats(missing, t_rec) -> dict:
    print(f"Found {len(missing)} missing timestamps")
#================================================================
# 5. Plot missing stats
#================================================================
def plot_missing_timestamps(missing):
    plt.figure(figsize=(10, 5))
    plt.plot(missing, [1] * len(missing), 'ro', markersize=2)
    plt.title('Missing Timestamps')
    plt.xlabel('Timestamp')
    plt.ylabel('Missing Indicator')
    plt.yticks([])
    plt.grid()
    plt.show()
#================================================================
# 6. Print summary stats
#================================================================
def print_summary_stats(df):
    print("Summary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
    print("\nFirst 5 Rows:")
    print(df.head())