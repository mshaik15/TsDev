import pandas as pd
import numpy as np
import os

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        # Timestamp candidates
        "time": "timestamp",
        "timestamp": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "created_at": "timestamp",
        "logged_at": "timestamp",
        "recorded_at": "timestamp",
        "ts": "timestamp",
        "event_time": "timestamp",
        "collected_at": "timestamp",
        "start_time": "timestamp",
        "end_time": "timestamp",
        "observed_at": "timestamp",
        "t": "timestamp",
    }

    df.columns = [col.strip().lower() for col in df.columns]

    rename_columns = {}
    for col in df.columns:
        try:
            rename_columns[col] = column_map[col]
            
        except KeyError:
            print(f"Warning: No mapping found for column '{col}', leaving it unchanged.")

    df = df.rename(columns=rename_columns)
    return df

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["timestamp"])  
    return df

def clean_data(df: pd.DataFrame, file_path:str = None) -> pd.DataFrame:
    df = df.copy()
    df = standardize_columns(df) 
    df.drop_duplicates(inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = remove_missing_values(df)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if file_path:
        base = os.path.basename(file_path)            
        name, _ = os.path.splitext(base)                    
        new_filename = f"{name}_cleaned.csv"                      
        output_path = os.path.join("../data/cleaned", new_filename)
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned file to: {output_path}")

    return df
