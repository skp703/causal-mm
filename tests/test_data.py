import numpy as np
import pandas as pd

from causal_mm.config import LagConfig
from causal_mm.data import TimeSeriesData, build_lagged_design


def test_build_lagged_design_basic():
    index = np.arange(5)
    df = pd.DataFrame({"1": [1, 2, 3, 4, 5], "2": [5, 4, 3, 2, 1]})
    ts = TimeSeriesData(index=index, data=df)
    X_lagged, trimmed_ts, meta = build_lagged_design(ts, LagConfig(max_lag=1))
    assert "1_lag1" in X_lagged.columns
    assert len(trimmed_ts.data) == len(X_lagged)
    assert meta["1"] == ["1_lag1"]
