from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import BootstrapConfig, DMLConfig, LagConfig
from .data import TimeSeriesData, align_fcm_and_timeseries
from .fcm import FCMGraph, EdgeEstimate
from .dml import estimate_edge_dml, _prepare_matrices, estimate_tau_from_matrices

logger = logging.getLogger(__name__)


def generate_block_indices(T: int, block_size: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Construct a bootstrap sample of indices 0..T-1 using block bootstrap.

    Why: time-series observations are dependent, so we resample contiguous blocks
    of length `block_size` instead of individual points. This keeps local
    autocorrelation intact while generating a same-length pseudo-series.
    """

    rng = np.random.default_rng(random_state)
    blocks: List[int] = []
    starts = rng.integers(0, max(1, T - block_size + 1), size=max(1, int(np.ceil(T / block_size))))
    for start in starts:
        blocks.extend(range(start, min(T, start + block_size)))
        if len(blocks) >= T:
            break
    return np.asarray(blocks[:T])


def _resample_timeseries(ts: TimeSeriesData, indices: np.ndarray) -> TimeSeriesData:
    """
    Create a new TimeSeriesData resampled along the time dimension.

    We keep the index values aligned with the sampled rows. The actual order
    matches the bootstrap draw (contiguous blocks stitched together).
    """

    resampled_index = ts.index[indices]
    resampled_data = ts.data.iloc[indices].reset_index(drop=True)
    return TimeSeriesData(index=resampled_index, data=resampled_data)


def _infer_expected_step(ts_index: np.ndarray) -> Optional[object]:
    """
    Infer the typical step size of the time index (numeric or datetime-like).
    Returns None if it cannot be inferred.
    """

    if len(ts_index) < 2:
        return None
    try:
        diffs = pd.Index(ts_index).to_series().diff().dropna()
        if diffs.empty:
            return None
        step = diffs.median()
        if pd.isna(step) or step == 0:
            return None
        return step
    except Exception:
        return None


def _compute_run_lengths(indices: np.ndarray, resampled_index: np.ndarray, expected_step: Optional[object]) -> np.ndarray:
    """
    Compute run lengths of contiguous time within each resampled block.

    A boundary is flagged when the time gap differs from the expected step
    (if available). If step inference fails, fall back to index contiguity.
    """

    run = np.ones(len(indices), dtype=int)
    if len(indices) == 0:
        return run

    contiguous = np.zeros(len(indices) - 1, dtype=bool)
    used_expected = False

    if expected_step is not None:
        try:
            diffs = pd.Series(resampled_index).diff().iloc[1:]
            contiguous = diffs.eq(expected_step).to_numpy()
            used_expected = True
        except Exception:
            used_expected = False

    if not used_expected:
        # Fallback: contiguous if original sampled positions are adjacent
        contiguous = indices[1:] == (indices[:-1] + 1)

    for i in range(1, len(run)):
        run[i] = run[i - 1] + 1 if contiguous[i - 1] else 1
    return run


def _prepare_bootstrap_matrices(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    source: str,
    target: str,
    dml_config: DMLConfig,
    indices: np.ndarray,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Build Y, T, W for a bootstrap draw, breaking lag continuity at block boundaries.
    """

    resampled_ts = _resample_timeseries(ts, indices)
    aligned = align_fcm_and_timeseries(fcm, resampled_ts)

    lag_cfg: LagConfig = dml_config.lag_config
    max_lag = lag_cfg.max_lag

    # Build lags without dropping so we can mask block boundaries first.
    cols: List[pd.Series] = []
    for concept in aligned.data.columns:
        for lag in range(1, max_lag + 1):
            col_name = f"{concept}_lag{lag}"
            ts_col = aligned.data[concept].shift(lag)
            cols.append(ts_col.rename(col_name))
    X_lagged = pd.concat(cols, axis=1)

    # Mask lags that would cross a bootstrap block boundary.
    expected_step = _infer_expected_step(aligned.index)
    run_lengths = _compute_run_lengths(indices, aligned.index, expected_step)
    run_series = pd.Series(run_lengths, index=X_lagged.index)
    for col in X_lagged.columns:
        try:
            lag_k = int(col.split("_lag")[1])
        except Exception:
            continue
        boundary_mask = run_series <= lag_k
        if boundary_mask.any():
            X_lagged.loc[boundary_mask, col] = np.nan

    # Drop rows with any NaNs if requested (default), otherwise keep and clean later.
    if lag_cfg.drop_initial_na:
        keep_mask = ~X_lagged.isna().any(axis=1)
        X_lagged = X_lagged.loc[keep_mask]
        trimmed_data = aligned.data.loc[keep_mask]
        trimmed_index = aligned.index[keep_mask]
    else:
        trimmed_data = aligned.data
        trimmed_index = aligned.index

    X_lagged = X_lagged.astype(float)
    trimmed_data = trimmed_data.astype(float)

    Y = pd.Series(trimmed_data[target].astype(float).values, index=trimmed_index)
    T = pd.Series(trimmed_data[source].astype(float).values, index=trimmed_index)
    W = X_lagged.copy()
    if not lag_cfg.include_self_lags:
        drop_cols = [c for c in W.columns if c.startswith(f"{target}_lag")]
        W = W.drop(columns=drop_cols)

    # Ensure no NaNs propagate into model fitting.
    if W.isna().any(axis=None):
        valid_mask = ~W.isna().any(axis=1)
        W = W.loc[valid_mask]
        Y = Y.loc[valid_mask]
        T = T.loc[valid_mask]

    return Y.reset_index(drop=True), T.reset_index(drop=True), W.reset_index(drop=True)


def bootstrap_edge(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    source: str,
    target: str,
    dml_config: DMLConfig,
    bs_config: BootstrapConfig,
) -> Dict[str, Optional[float]]:
    """
    Run block bootstrap for a single edge.

    For each bootstrap replicate:
      1) Draw block indices.
      2) Rebuild Y, T, W on the resampled sequence, breaking lags at block boundaries.
      3) Re-estimate tau_raw using the resampled matrices.
    """

    tau_samples: List[float] = []

    for b in range(bs_config.n_bootstrap):
        try:
            idx = generate_block_indices(len(ts.data), bs_config.block_size, random_state=(bs_config.random_state or 0) + b)
            Y_resampled, T_resampled, W_resampled = _prepare_bootstrap_matrices(
                ts, fcm, source, target, dml_config, idx
            )
            if len(Y_resampled) == 0:
                continue
            tau = estimate_tau_from_matrices(Y_resampled, T_resampled, W_resampled, dml_config)
            tau_samples.append(tau)
        except Exception:
            # Skip failed bootstrap iterations
            continue


    if not tau_samples:
        return {"tau_se": None, "ci_low": None, "ci_high": None, "sign_stability": None}

    arr = np.asarray(tau_samples)
    tau_se = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))

    base_est = fcm.estimates.get((source, target))
    if base_est and base_est.tau_raw is not None:
        base_sign = np.sign(base_est.tau_raw)
        sign_stability = float(np.mean(np.sign(arr) == base_sign))
    else:
        sign_stability = None

    return {
        "tau_se": tau_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sign_stability": sign_stability,
    }


def _bootstrap_edge_safe(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    edge: Tuple[str, str],
    dml_config: DMLConfig,
    bs_config: BootstrapConfig,
) -> Tuple[Tuple[str, str], Dict[str, Optional[float]]]:
    """
    Wrapper to catch exceptions so parallel jobs don't crash the whole run.
    """
    try:
        return edge, bootstrap_edge(ts, fcm, edge[0], edge[1], dml_config, bs_config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Bootstrap failed for edge %s->%s", edge[0], edge[1])
        return edge, {"tau_se": None, "ci_low": None, "ci_high": None, "sign_stability": None, "error": str(exc)}


def bootstrap_all_edges(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    dml_config: DMLConfig,
    bs_config: BootstrapConfig,
) -> FCMGraph:
    """
    Run bootstrap for all edges and update estimates.

    Parallelism: controlled by bs_config.n_jobs. Uses joblib when n_jobs != 1.
    """

    edges = [(e.source, e.target) for e in fcm.edges]
    if bs_config.n_jobs and bs_config.n_jobs != 1:
        results = Parallel(n_jobs=bs_config.n_jobs)(
            delayed(_bootstrap_edge_safe)(ts, fcm, edge, dml_config, bs_config) for edge in edges
        )
    else:
        results = [_bootstrap_edge_safe(ts, fcm, edge, dml_config, bs_config) for edge in edges]

    for (source, target), summary in results:
        est = fcm.estimates.get((source, target))
        if est is None:
            est = EdgeEstimate(source=source, target=target)
            fcm.estimates[(source, target)] = est
        est.tau_se = summary.get("tau_se")
        est.ci_low = summary.get("ci_low")
        est.ci_high = summary.get("ci_high")
        est.sign_stability = summary.get("sign_stability")
        est.method = "bootstrap-dml"
        if summary.get("error"):
            est.status = "failed"
            est.error_message = summary["error"]
    return fcm
