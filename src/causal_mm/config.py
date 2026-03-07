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

    def __post_init__(self):
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")


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

    def __post_init__(self):
        if self.n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {self.n_folds}")
        if self.min_train_size < 1:
            raise ValueError(f"min_train_size must be >= 1, got {self.min_train_size}")
        if self.treatment_lag < 0:
            raise ValueError(f"treatment_lag must be >= 0, got {self.treatment_lag}")


@dataclass
class BootstrapConfig:
    """
    Configuration for block bootstrap.
    """

    n_bootstrap: int = 200
    block_size: int = 5
    random_state: Optional[int] = 123
    n_jobs: int = 1
    ci_alpha: float = 0.05

    def __post_init__(self):
        if self.n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {self.n_bootstrap}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not (0.0 < self.ci_alpha < 1.0):
            raise ValueError(f"ci_alpha must be in (0, 1), got {self.ci_alpha}")
