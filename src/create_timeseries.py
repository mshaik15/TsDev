from collections import Counter
import pandas as pd
import numpy as np

#================================================================
# 1. Infer the time interval from the timestamps in the DataFrame
#================================================================
def infer_time_interval(df) -> int:
    default = 1  # Default time interval in seconds
    if len(df) < 2:
        print("Not enough timestamps to infer interval.")
        return default

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    deltas = df["timestamp"].diff().dropna()
    deltas_in_seconds = [int(delta.total_seconds()) for delta in deltas]

    delta_counts = Counter(deltas_in_seconds)
    t_rec, _ = delta_counts.most_common(1)[0]

    response = input(f"Time interval inferred: {t_rec} seconds. Continue with this value? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        print(f"Using time interval: {t_rec} seconds.")
        return t_rec
    else:
        try:
            user_input = int(input("Please enter the time interval in seconds: ").strip())
            print(f"Using user-defined time interval: {user_input} seconds.")
            return user_input
        except ValueError:
            print("Invalid input. Using default time interval.")
            return default
