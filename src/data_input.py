import pandas as pd
import numpy as np

def load_data_csv(file_path) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df, file_path
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

def load_data_json(file_path) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_json(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df, file_path
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

def load_data_excel(file_path) -> tuple[pd.DataFrame, str]:
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df, file_path
    
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None
