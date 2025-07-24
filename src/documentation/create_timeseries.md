
# `construct_time_series` Pipeline Documentation

## Overview

This module transforms a raw timestamped dataset into a clean, regularly spaced time series.
It supports **resampling**, **aggregation**, and **interpolation**, making it ideal for analysis of time-based data (e.g. sensor readings, financial data, etc.).

---

## Workflow Summary

```text
Step 1: Input raw data (must contain a timestamp column and a numeric variable)
Step 2: Choose desired frequency (e.g., daily, hourly)
Step 3: Resample data using an aggregation method (e.g., average, sum)
Step 4: Fill missing values using interpolation (e.g., linear, time-based, spline)
Step 5: Optionally fill beginning and end NaNs using forward/backward fill
Step 6: Return the final, clean time series
```

---

## Function Reference

### 1. `validate_inputs(df, dependent_var)`

* **Purpose**: Ensure input DataFrame is usable
* **Checks**:

  * DataFrame is non-empty
  * Column `timestamp` exists
  * Column `dependent_var` exists and is numeric

---

### 2. `prepare_data(df, inplace=False)`

* **Purpose**: Converts `'timestamp'` to `datetime` type and sets it as index
* **Returns**: DataFrame indexed by `timestamp`

---

### 3. `infer_frequency(df)`

* **Purpose**: Prints and returns the most common time intervals between timestamps. This is useful to understand the data and how to resample it if needed
* **Useful for**: Choosing the appropriate frequency (`freq`) for resampling (if needed)

---

### 4. `resample_series(df, dependent_var, freq, agg)`

* **Purpose**: Resamples time series using the specified frequency and aggregation method
* **Example**:
  * If you want to analyze the data in a different frequency, you can resample it using resample_series
  * This function Changes the frequency of the data by applying a summary method like avg or sum 
  * For example, it can convert minute data into hourly data by taking the avg or sum of values within each hour
  * Converting the data into a time series in mins to a time series in hours with `.resample('H').mean()`

---

### 5. `interpolate_series(ts, interpolation, spline_order)`

* **Purpose**: Fills missing values using interpolation.
* **Methods Supported**:

  * `"linear"`: Straight line between points
  * `"time"`: Interpolation based on datetime index
  * `"spline"`: Smooth polynomial interpolation (requires `spline_order`)
  * `"ffill"`: Forward fill
  * `"bfill"`: Backward fill

---

### 6. `fill_end_values(ts)`

* **Purpose**: Applies both forward and backward fill to fill missing values at the beginning or end

---

### 7. `interpolation_help()`

* **Purpose**: Displays supported interpolation methods and when to use them

---

### 8. `run_checks(ts)`

* **Purpose**: Performs a final validation check on the time series

  * Index must be monotonic increasing
  * Warns if null values still remain

---

### 9. `construct_time_series(...)`

* **Purpose**: Main function to build a clean time series.
* **Parameters**:

| Parameter       | Type                    | Description                                   |
| --------------- | ----------------------- | --------------------------------------------- |
| `df`            | `pd.DataFrame`          | Raw dataset with a `'timestamp'` column       |
| `dependent_var` | `str`                   | The variable to analyze (default = `"value"`) |
| `freq`          | `"D"`, `"H"`, etc.      | Resampling frequency (e.g., daily, hourly)    |
| `agg`           | `"mean"`, `"sum"`, etc. | Aggregation strategy for downsampling         |
| `interpolation` | See above               | Interpolation method to fill missing values   |
| `spline_order`  | `int` or `None`         | Spline order (required if using `"spline"`)   |
| `fill_extremes` | `bool`                  | Fill values at start/end using ffill/bfill    |

* **Returns**: A clean, indexed `pd.Series` ready for analysis.

---

## Suggested Frequencies (based on time delta)

| Time Delta | Suggested `freq` |
| ---------- | ---------------- |
| < 60 sec   | `'T'` or `'30T'` |
| < 1 hour   | `'H'`            |
| < 1 day    | `'D'`            |
| ≥ 1 week   | `'W'` or `'M'`   |

Use `infer_frequency(df)` to guide your selection.

---

## Example Usage

```python
ts = construct_time_series(
    df=my_data,
    dependent_var="temperature",
    freq="H",
    agg="mean",
    interpolation="linear"
)
ts.plot()
```

---

## Notes

* The `timestamp` column **must be present** in your dataset.
* The `dependent_var` column must contain **numeric values**.
* Interpolation is optional but recommended if data has gaps.
* Frequency labels follow [pandas offset aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).

