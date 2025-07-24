from typing import Literal, Optional
import pandas as pd
import numpy as np

#============================================================================
# Used types
#============================================================================
possible_freq = Literal["D", "H", "T", "30T", "W", "M"]
possible_agg = Literal["mean", "sum", "first", "last", "max", "min"]
possible_interp = Literal["linear", "time", "spline", "ffill", "bfill", None]

#============================================================================
# Validate inputs, check if we can construct a valid time series
#============================================================================
def validate_inputs(df: pd.DataFrame, dependent_var: str):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if "timestamp" not in df.columns:
        raise ValueError("Data must have a timestamp column")
    if dependent_var not in df.columns: # Show usable y-axis variables
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{dependent_var}' doesn't exist in data. Available columns: {available_cols}")
    if not pd.api.types.is_numeric_dtype(df[dependent_var]):
        raise ValueError(f"Dependent variable '{dependent_var}' must be numeric")
    
#============================================================================
# Prepare dataframe, convert timestamp -> datetime and set as index
#============================================================================
def prepare_data(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    if inplace:
        df.set_index("timestamp", inplace=True)
        return df
    else:
        return df.set_index("timestamp")

#============================================================================
# Resample timeseries to specific aggregation and frequency
#============================================================================
def resample_series(df: pd.DataFrame, dependent_var: str, freq: possible_freq, agg: possible_agg) -> pd.Series:
    try:
        return getattr(df[dependent_var].resample(freq), agg)()
    except AttributeError:
        raise ValueError(f"Unsupported aggregation method: {agg}")

#============================================================================
# Fill in missing data, using statistical methods
#============================================================================
def interpolate_series(ts: pd.Series, interpolation: possible_interp, spline_order: Optional[int]) -> pd.Series:
    try:
        if interpolation == "spline":
            if spline_order is None:
                raise ValueError("spline_order required for spline interpolation")
            return ts.interpolate(method="spline", order=spline_order)
        elif interpolation in {"linear", "time", "ffill", "bfill"}:
            return ts.interpolate(method=interpolation)
        return ts
    except Exception as e:
        raise ValueError(f"Interpolation failed with method '{interpolation}': {str(e)}")

#============================================================================
# Check end values
#============================================================================
def fill_end_values(ts: pd.Series) -> pd.Series:
    return ts.ffill().bfill()

#============================================================================
# Infer Frequency, Function to help the user
#============================================================================
def infer_frequency(df: pd.DataFrame) -> str:
    timestamps = df["timestamp"].sort_values()
    if len(timestamps) > 1000:
        timestamps = timestamps.head(1000)
    deltas = timestamps.diff().dropna()
    if deltas.empty:
        return "Unknown"
        
    mode_delta = deltas.mode()[0] if not deltas.mode().empty else deltas.iloc[0]
    seconds = mode_delta.total_seconds()
    
    print(f"Inferred time delta: {seconds} seconds ({mode_delta})")

    if seconds <= 60:
        print("Suggested freq: 'T' (minutely) or '30T' (30 mins)")
    elif seconds <= 3600:
        print("Suggested freq: 'H' (hourly)")
    elif seconds <= 86400:
        print("Suggested freq: 'D' (daily)")
    else:
        print("Suggested freq: 'W' (weekly) or 'M' (monthly)")
    return pd.infer_freq(df["timestamp"].sort_values()) or "Unknown"

#============================================================================
# Interpolation Methods, Function to help the user
#============================================================================
def interpolation_help():
    print("Available options:")
    print("- 'linear': Fills missing values with a straight-line")
    print("- 'time': Fills missing values with time-based indexing")
    print("- 'spline': Fills missing values with smooth curve")
    print("- 'ffill': Forward fill using previous known value")
    print("- 'bfill': Backward fill using next known value\n")

#============================================================================
# Final check
#============================================================================
def run_checks(ts: pd.Series):
    if not ts.index.is_monotonic_increasing:
        raise ValueError("Time series index is not monotonic increasing.")
    if ts.isnull().sum() > 0:
        print(f"Warning: Time series still contains {ts.isnull().sum()} missing values.")

#============================================================================
# Main Function
#============================================================================
def construct_time_series(
    df: pd.DataFrame,
    dependent_var: str = "value",
    freq: possible_freq = "D",
    agg: possible_agg = "mean",
    interpolation: possible_interp = "linear",
    spline_order: Optional[int] = 2,
    fill_extremes: bool = True) -> pd.Series:

    validate_inputs(df, dependent_var)
    df = prepare_data(df)
    ts = resample_series(df, dependent_var, freq, agg)
    ts = interpolate_series(ts, interpolation, spline_order)
    if fill_extremes:
        ts = fill_end_values(ts)
    run_checks(ts)
    return ts
