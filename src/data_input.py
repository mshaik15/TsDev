import pandas as pd
import numpy as np

def load_data_csv(file_path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_data_json(file_path) -> pd.DataFrame:
    try:
        df = pd.read_json(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def load_data_excel(file_path) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None
