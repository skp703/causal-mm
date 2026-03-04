from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LagConfig:
    """
    Lag configuration for time-series design matrices.
    """

    max_lag: int = 3
    include_self_lags: bool = True
    drop_initial_na: bool = True


@dataclass
class MLModelConfig:
    """
    Wrapper for ML model selection and hyperparameters.
    """

    model_type: str = "ridge"  # ridge | random_forest | gbm | lasso | linear
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DMLConfig:
    """
    Configuration for Double Machine Learning estimation.
    """

    lag_config: LagConfig
    outcome_model: MLModelConfig
    treatment_model: MLModelConfig
    n_folds: int = 3
    min_train_size: int = 10
    random_state: Optional[int] = 123
    alpha_scale: float = 1.0  # scaling factor in tanh(alpha * tau)
    use_econml: bool = False   # if True, use econml estimators
    econml_estimator: str = "linear_dml"  # linear_dml | causal_forest | ortho_forest
    controls_selection: str = "all"  # "all" | "connected"
    treatment_lag: int = 0  # 0 = contemporaneous (T_t -> Y_t), 1 = lagged (T_{t-1} -> Y_t), etc.
    standardize: bool = True  # z-score concept time series before estimation
    detrend: str = "none"  # "none" | "linear" | "first_diff" — remove deterministic trends before estimation


@dataclass
class BootstrapConfig:
    """
    Configuration for block bootstrap.
    """

    n_bootstrap: int = 200
    block_size: int = 5
    random_state: Optional[int] = 123
    n_jobs: int = 1
