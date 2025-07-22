from collections import Counter
import pandas as pd
import numpy as np

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
        