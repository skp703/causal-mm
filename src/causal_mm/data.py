from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import LagConfig


@dataclass
class TimeSeriesData:
    """
    Container for time-series aligned to concept IDs.

    index: sequence of time points (e.g., years)
    data: DataFrame with concept ID columns (strings) and numeric observations
    """

    index: np.ndarray  # shape (T,)
    data: pd.DataFrame  # shape (T, n_concepts), columns = concept IDs (strings)

    def validate(self) -> None:
        """
        Validate shape and column uniqueness.
        """

        if len(self.index) != len(self.data):
            raise ValueError("index length must match number of rows in data")
        if self.data.columns.duplicated().any():
            raise ValueError("duplicate columns in data are not allowed")
        if self.data.isna().all(axis=None):
            raise ValueError("all values are NA; need numeric observations")


def build_lagged_design(
    ts: TimeSeriesData, lag_config: LagConfig
) -> Tuple[pd.DataFrame, TimeSeriesData, Dict[str, List[str]]]:
    """
    Construct a lagged design matrix from TimeSeriesData.

    Returns
    -------
    X_lagged : DataFrame
        Columns correspond to lagged variables, e.g. 'C1_lag1', 'C1_lag2'.
    trimmed_ts : TimeSeriesData
        TimeSeriesData aligned to X_lagged (rows dropped if drop_initial_na).
    lag_metadata : dict
        Mapping concept_id -> list of created column names for its lags.
    """

    ts.validate()
    max_lag = lag_config.max_lag
    cols = []
    lag_meta: Dict[str, List[str]] = {}

    for concept in ts.data.columns:
        lag_cols: List[str] = []
        for lag in range(1, max_lag + 1):
            col_name = f"{concept}_lag{lag}"
            ts_col = ts.data[concept].shift(lag)
            cols.append(ts_col.rename(col_name))
            lag_cols.append(col_name)
        lag_meta[concept] = lag_cols

    X_lagged = pd.concat(cols, axis=1)

    if lag_config.drop_initial_na:
        # Drop rows with any NaNs created by lagging (first max_lag rows).
        keep_mask = ~X_lagged.isna().any(axis=1)
        X_lagged = X_lagged.loc[keep_mask]
        trimmed_index = ts.index[keep_mask]
        trimmed_data = ts.data.loc[keep_mask]
    else:
        trimmed_index = ts.index
        trimmed_data = ts.data

    # Ensure types align
    X_lagged = X_lagged.astype(float)
    trimmed_data = trimmed_data.astype(float)
    trimmed_ts = TimeSeriesData(index=np.asarray(trimmed_index), data=trimmed_data)
    return X_lagged, trimmed_ts, lag_meta


def standardize_timeseries(ts: TimeSeriesData) -> TimeSeriesData:
    """
    Z-score each concept column to zero mean and unit variance.

    This ensures that DML causal effect estimates (tau) are in units of
    standard deviations, making them comparable across concepts with
    different natural scales.  Columns with zero variance are left unchanged
    (a warning is logged).

    Returns a new TimeSeriesData; the original is not modified.
    """

    data = ts.data.copy()
    means = data.mean()
    stds = data.std(ddof=1)
    zero_var = stds[stds == 0].index.tolist()
    if zero_var:
        import logging
        logging.getLogger(__name__).warning(
            "Zero-variance columns will not be standardized: %s", zero_var
        )
    safe_stds = stds.replace(0, 1)  # avoid division by zero
    data = (data - means) / safe_stds
    return TimeSeriesData(index=ts.index.copy(), data=data)


def detrend_timeseries(
    ts: TimeSeriesData,
    method: str = "linear",
    columns: List[str] | None = None,
) -> TimeSeriesData:
    """
    Remove deterministic trends from concept time series.

    Parameters
    ----------
    ts : TimeSeriesData
        Input time series.
    method : str
        Detrending method:
        - "linear": subtract OLS-fitted linear trend from each column.
        - "first_diff": replace each column with first differences (loses one obs).
    columns : list[str] or None
        If provided, only detrend these columns; others are left unchanged.
        If None, detrend all columns.

    Returns a new TimeSeriesData; the original is not modified.
    """

    if method not in ("linear", "first_diff"):
        raise ValueError(f"Unknown detrend method '{method}'; use 'linear' or 'first_diff'")

    data = ts.data.copy()
    target_cols = columns if columns is not None else list(data.columns)

    if method == "linear":
        # Use integer time index as regressor
        t = np.arange(len(data), dtype=float)
        for col in target_cols:
            if col not in data.columns:
                import logging
                logging.getLogger(__name__).warning(
                    "Detrend: column '%s' not found in data, skipping", col
                )
                continue
            y = data[col].values.astype(float)
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                continue
            # Fit OLS: y = a + b*t
            t_valid = t[mask]
            y_valid = y[mask]
            b = np.cov(t_valid, y_valid, ddof=0)[0, 1] / np.var(t_valid, ddof=0) if np.var(t_valid, ddof=0) > 0 else 0.0
            a = np.mean(y_valid) - b * np.mean(t_valid)
            data[col] = y - (a + b * t)

        return TimeSeriesData(index=ts.index.copy(), data=data)

    else:  # first_diff
        diffed = data.copy()
        for col in target_cols:
            if col not in data.columns:
                continue
            diffed[col] = data[col].diff()
        # Drop the first row (NaN from differencing)
        diffed = diffed.iloc[1:].reset_index(drop=True)
        new_index = ts.index[1:] if hasattr(ts.index, '__getitem__') else np.asarray(ts.index)[1:]
        return TimeSeriesData(index=np.asarray(new_index), data=diffed)


def align_fcm_and_timeseries(fcm: "FCMGraph", ts: TimeSeriesData) -> TimeSeriesData:
    """
    Ensure that FCM concept IDs are present as columns in ts.data.
    Reorder ts.data columns to match FCM concept order.
    """

    concept_ids = [str(c.id) for c in fcm.concepts]
    missing = [c for c in concept_ids if c not in ts.data.columns]
    if missing:
        raise ValueError(f"timeseries missing concept IDs: {missing}")

    aligned_data = ts.data.loc[:, concept_ids]
    return TimeSeriesData(index=ts.index, data=aligned_data)
