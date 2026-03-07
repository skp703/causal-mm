from __future__ import annotations

import logging
from datetime import datetime, timezone
import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression

from .config import DMLConfig
from .data import TimeSeriesData, align_fcm_and_timeseries, build_lagged_design, standardize_timeseries
from .fcm import FCMGraph, EdgeEstimate

logger = logging.getLogger(__name__)

ECONML_IMPORT_ERROR: Optional[Exception]
try:
    from econml.dml import LinearDML  # type: ignore
    try:
        from econml.dml import CausalForestDML  # type: ignore
    except ImportError:
        from econml.ortho_forest import CausalForestDML  # type: ignore

    try:
        from econml.ortho_forest import OrthoForest  # type: ignore
    except ModuleNotFoundError:
        from econml.orf import DMLOrthoForest as OrthoForest  # type: ignore

    HAS_ECONML = True
    ECONML_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    HAS_ECONML = False
    ECONML_IMPORT_ERROR = exc

_LINEAR_DML_ACCEPTS_MODEL_FINAL: Optional[bool] = None


def _set_linear_dml_accepts_model_final(value: bool) -> None:
    """
    Cache whether LinearDML accepts the model_final parameter.
    """

    global _LINEAR_DML_ACCEPTS_MODEL_FINAL
    _LINEAR_DML_ACCEPTS_MODEL_FINAL = value


def _linear_dml_accepts_model_final() -> bool:
    """
    Determine if the installed econml.LinearDML supports model_final.
    """

    global _LINEAR_DML_ACCEPTS_MODEL_FINAL
    if _LINEAR_DML_ACCEPTS_MODEL_FINAL is None:
        if not HAS_ECONML:
            _LINEAR_DML_ACCEPTS_MODEL_FINAL = False
        else:
            try:
                signature = inspect.signature(LinearDML.__init__)  # type: ignore[arg-type]
                _LINEAR_DML_ACCEPTS_MODEL_FINAL = "model_final" in signature.parameters
            except Exception:  # pragma: no cover - defensive
                _LINEAR_DML_ACCEPTS_MODEL_FINAL = False
    return bool(_LINEAR_DML_ACCEPTS_MODEL_FINAL)


def _make_temporal_folds(T: int, n_folds: int, min_train_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Forward-chaining folds. Each fold trains on earlier observations and tests on a later slice.

    This avoids look-ahead bias: no test point ever leaks into training.
    Train size is enforced to be at least `min_train_size` to keep models stable.
    """

    if T <= min_train_size:
        raise ValueError("Not enough observations for requested min_train_size.")
    fold_sizes = []
    remaining = T - min_train_size
    test_size = max(1, remaining // n_folds)
    start = min_train_size
    for _ in range(n_folds):
        stop = min(T, start + test_size)
        if stop <= start:
            break
        fold_sizes.append((start, stop))
        start = stop
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for start, stop in fold_sizes:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, stop)
        folds.append((train_idx, test_idx))
    if not folds:
        raise ValueError("Failed to build temporal folds; increase data or reduce n_folds.")
    return folds


def _get_ml_model(model_type: str, params: Optional[Dict] = None):
    params = params or {}
    mtype = model_type.lower()
    if mtype == "ridge":
        return Ridge(**params)
    if mtype == "lasso":
        return Lasso(**params)
    if mtype == "linear":
        return LinearRegression(**params)
    if mtype in ("random_forest", "rf"):
        return RandomForestRegressor(**params)
    if mtype in ("gbm", "gbr", "gradient_boosting"):
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _init_linear_dml_estimator(
    outcome_model,
    treatment_model,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    dml_config: DMLConfig,
):
    """
    Instantiate LinearDML, handling version differences around model_final.
    """

    base_kwargs = dict(
        model_y=outcome_model,
        model_t=treatment_model,
        discrete_treatment=False,
        cv=folds,
        random_state=dml_config.random_state,
    )
    if _linear_dml_accepts_model_final():
        base_kwargs["model_final"] = LinearRegression()
    try:
        return LinearDML(**base_kwargs)
    except TypeError as exc:
        if "model_final" in base_kwargs and "model_final" in str(exc):
            logger.debug("LinearDML rejected model_final; retrying without it for compatibility.")
            base_kwargs.pop("model_final", None)
            _set_linear_dml_accepts_model_final(False)
            return LinearDML(**base_kwargs)
        raise


def _init_ortho_forest_estimator(
    outcome_model,
    treatment_model,
    dml_config: DMLConfig,
):
    """
    Instantiate OrthoForest, handling signature differences between versions.
    """

    shared_kwargs = dict(
        discrete_treatment=False,
        random_state=dml_config.random_state,
    )

    def _build_kwargs(upper: bool) -> Dict[str, Any]:
        if upper:
            return dict(shared_kwargs, model_Y=outcome_model, model_T=treatment_model)
        return dict(shared_kwargs, model_y=outcome_model, model_t=treatment_model)

    try:
        return OrthoForest(**_build_kwargs(True))
    except TypeError as exc:
        if "model_Y" in str(exc) or "model_T" in str(exc):
            logger.debug("OrthoForest rejected model_Y/model_T; retrying with lowercase aliases.")
            return OrthoForest(**_build_kwargs(False))
        raise


def _prepare_matrices(
    ts: TimeSeriesData, fcm: FCMGraph, source: str, target: str, dml_config: DMLConfig
) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Align, build lags, and return Y, T, W matrices.

    - Align ensures concept columns match FCM ordering.
    - Lagging trims early rows if drop_initial_na is True.
    - Y: target series, T: source series (possibly lagged), W: controls (lagged covariates).
    - Self lags of the target can be dropped if requested to avoid leakage.
    - If treatment_lag > 0, uses T_{t-k} to predict Y_t (temporal precedence).
    """

    aligned = align_fcm_and_timeseries(fcm, ts)
    X_lagged, trimmed_ts, _ = build_lagged_design(aligned, dml_config.lag_config)
    # Align Y and T to trimmed index
    Y = trimmed_ts.data[target].astype(float)
    T = trimmed_ts.data[source].astype(float)
    
    # Apply treatment lag if specified (e.g., T_{t-1} -> Y_t)
    # shift(k) moves each value forward by k positions, so at row t
    # we see the value from row t-k, i.e. T_{t-k}.
    if dml_config.treatment_lag > 0:
        T = T.shift(dml_config.treatment_lag)

    W = X_lagged.copy()

    # Filter controls based on selection strategy
    if dml_config.controls_selection == "connected":
        # Only include lags of concepts that are parents of the target (or the target itself)
        # Note: The source is a parent, so its lags are included.
        # We also typically include the target's own lags unless include_self_lags is False.
        
        # Find parents of target in the FCM graph
        parents = {e.source for e in fcm.edges if e.target == target}
        # Always include the source (it is a parent by definition of the edge we are estimating)
        parents.add(source)
        # Include target itself (for self-lags)
        parents.add(target)

        # Keep only columns that start with "{parent}_lag"
        keep_cols = []
        for col in W.columns:
            for p in parents:
                if col.startswith(f"{p}_lag"):
                    keep_cols.append(col)
                    break
        W = W[keep_cols]

    if not dml_config.lag_config.include_self_lags:
        # Drop lags of the target to avoid leakage of its past into controls if requested.
        drop_cols = [c for c in W.columns if c.startswith(f"{target}_lag")]
        W = W.drop(columns=drop_cols)
    
    # If treatment was lagged, remove rows with NaN in treatment
    if dml_config.treatment_lag > 0:
        valid_mask = ~T.isna()
        Y = Y[valid_mask]
        T = T[valid_mask]
        W = W[valid_mask]
    
    return Y, T, W


def _estimate_edge_dml_econml(
    Y: pd.Series,
    T: pd.Series,
    W: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    dml_config: DMLConfig,
) -> float:
    """
    Use econml estimators to estimate tau_raw with forward folds.

    Supported:
      - linear_dml (default)
      - causal_forest (CausalForestDML)
      - ortho_forest (OrthoForest)
    """

    if not HAS_ECONML:
        hint = ""
        if ECONML_IMPORT_ERROR:
            hint = f" (import error: {ECONML_IMPORT_ERROR})"
        raise ImportError(
            "econml could not be imported; install econml or disable use_econml" + hint
        ) from ECONML_IMPORT_ERROR

    # econml wants sklearn models; reuse our getter
    outcome_model = _get_ml_model(dml_config.outcome_model.model_type, dml_config.outcome_model.params)
    treatment_model = _get_ml_model(dml_config.treatment_model.model_type, dml_config.treatment_model.params)

    est_type = dml_config.econml_estimator.lower()
    if est_type == "linear_dml":
        est = _init_linear_dml_estimator(outcome_model, treatment_model, folds, dml_config)
    elif est_type == "causal_forest":
        est = CausalForestDML(
            model_y=outcome_model,
            model_t=treatment_model,
            discrete_treatment=False,
            cv=folds,
            random_state=dml_config.random_state,
        )
    elif est_type == "ortho_forest":
        est = _init_ortho_forest_estimator(outcome_model, treatment_model, dml_config)
    else:
        raise ValueError(f"Unsupported econml_estimator: {dml_config.econml_estimator}")

    est.fit(Y, T, X=W)
    tau_raw = float(np.mean(est.effect(X=W, T0=0, T1=1)))
    return tau_raw


def estimate_tau_from_matrices(
    Y: pd.Series,
    T: pd.Series,
    W: pd.DataFrame,
    dml_config: DMLConfig,
) -> float:
    """
    Estimate tau_raw from prepared matrices Y, T, W.
    """
    T_len = len(Y)
    folds = _make_temporal_folds(T_len, dml_config.n_folds, dml_config.min_train_size)

    if dml_config.use_econml:
        return _estimate_edge_dml_econml(Y, T, W, folds, dml_config)

    outcome_model = _get_ml_model(dml_config.outcome_model.model_type, dml_config.outcome_model.params)
    treatment_model = _get_ml_model(dml_config.treatment_model.model_type, dml_config.treatment_model.params)

    numer = 0.0
    denom = 0.0

    for train_idx, test_idx in folds:
        W_train, W_test = W.iloc[train_idx], W.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        T_train, T_test = T.iloc[train_idx], T.iloc[test_idx]

        outcome_model.fit(W_train, Y_train)
        treatment_model.fit(W_train, T_train)

        m_hat = outcome_model.predict(W_test)
        g_hat = treatment_model.predict(W_test)

        Y_tilde = Y_test - m_hat
        T_tilde = T_test - g_hat

        numer += float(np.sum(Y_tilde * T_tilde))
        denom += float(np.sum(T_tilde ** 2))

    if denom == 0:
        raise ValueError("Denominator zero in tau computation; T_tilde variance is zero.")

    return numer / denom


def estimate_edge_dml(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    source: str,
    target: str,
    dml_config: DMLConfig,
) -> EdgeEstimate:
    """
    Estimate causal effect for a single edge using time-series DML.

    Algorithm (per edge):
      1) Build lagged controls W, outcomes Y, treatments T.
      2) Create forward-chaining folds.
      3) For each fold:
           - Fit outcome model: Y ~ W (train).
           - Fit treatment model: T ~ W (train).
           - Predict on test; compute residuals Y_tilde, T_tilde.
           - Accumulate tau numerator/denominator.
      4) tau_raw = sum(Y_tilde*T_tilde) / sum(T_tilde^2).
      5) scaled_weight = tanh(alpha_scale * tau_raw) to bound in [-1, +1].
      6) Stamp metadata (n_obs, lag_used, computed_at).
    """

    est = EdgeEstimate(source=source, target=target, status="pending", method="dml")
    try:
        Y, T, W = _prepare_matrices(ts, fcm, source, target, dml_config)
        tau_raw = estimate_tau_from_matrices(Y, T, W, dml_config)
        
        est.tau_raw = tau_raw
        est.n_obs = len(Y)
        est.lag_used = dml_config.lag_config.max_lag
        est.scaled_weight = float(np.tanh(dml_config.alpha_scale * tau_raw))
        est.status = "ok"
        est.computed_at = datetime.now(timezone.utc).isoformat()
        if dml_config.use_econml:
            est.method = f"dml-econml-{dml_config.econml_estimator}"

    except Exception as exc:  # noqa: BLE001
        est.status = "failed"
        est.error_message = str(exc)
        logger.exception("Failed to estimate edge %s->%s", source, target)
    return est


def estimate_all_edges_dml(
    ts: TimeSeriesData,
    fcm: FCMGraph,
    dml_config: DMLConfig,
    bootstrap_config: Optional["BootstrapConfig"] = None,
) -> FCMGraph:
    """
    Estimate all edges with DML, optionally followed by bootstrap for uncertainty.

    First pass computes point estimates (and scaling). If bootstrap_config is
    provided, a second pass augments estimates with SE/CI/sign-stability.
    """

    from .bootstrap import bootstrap_all_edges  # local import to avoid cycle

    results: Dict[Tuple[str, str], EdgeEstimate] = {}
    for edge in fcm.edges:
        est = estimate_edge_dml(ts, fcm, edge.source, edge.target, dml_config)
        results[(edge.source, edge.target)] = est
    fcm.estimates.update(results)

    if bootstrap_config:
        fcm = bootstrap_all_edges(ts, fcm, dml_config, bootstrap_config)
    return fcm
