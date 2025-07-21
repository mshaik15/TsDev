import pandas as pd
import numpy as np

def infer_time_interval(df):
    deltas = []
    for i in range(1, len(df)):
        delta = df.timestamp[i] - df.timestamp[i - 1]
        deltas.append(delta)
    t_rec = np.mode(deltas)

    response = input(f"Time interval inferred: {t_rec}, continue with this value? (y/n).")
    if response.lower() == 'y':
        print(f"Using time interval: {t_rec}")
        return t_rec
    else:
        t_rec = input("Please enter the time interval in seconds:")
        try:
            t_rec = int(t_rec)
            print(f"Using user-defined time interval: {t_rec}")
            return t_rec
        except ValueError:
            print("Invalid input. Using default time interval of 1 second.")
            return 1
        
