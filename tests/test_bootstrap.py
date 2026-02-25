import numpy as np
import pandas as pd

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig, BootstrapConfig
from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph
from causal_mm.bootstrap import bootstrap_edge, _prepare_bootstrap_matrices


def test_bootstrap_edge_runs():
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(8)
    df = pd.DataFrame({"1": np.random.randn(8), "2": np.random.randn(8)})
    ts = TimeSeriesData(index=index, data=df)

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
    )
    bs_config = BootstrapConfig(n_bootstrap=3, block_size=2, random_state=0, n_jobs=1)

    summary = bootstrap_edge(ts, fcm, "1", "2", dml_config, bs_config)
    assert set(summary.keys()) >= {"tau_se", "ci_low", "ci_high", "sign_stability"}


def test_bootstrap_respect_block_boundaries():
    # Create a series with a large gap so the stitched blocks would otherwise leak lags.
    years = np.array(list(range(1990, 1996)) + list(range(2010, 2016)))
    data = pd.DataFrame({"1": np.arange(len(years)), "2": np.arange(len(years))})
    ts = TimeSeriesData(index=years, data=data)

    fcm = FCMGraph(concepts=[Concept(id="1"), Concept(id="2")], edges=[Edge(source="1", target="2")])
    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="linear"),
        treatment_model=MLModelConfig(model_type="linear"),
        n_folds=2,
        min_train_size=3,
    )

    # Use deterministic indices so block boundary is between 1995 and 2010.
    idx = np.arange(len(ts.data))
    Y, T, W = _prepare_bootstrap_matrices(ts, fcm, "1", "2", dml_config, idx)

    # First row (start) and boundary row should be dropped => length reduced by 2.
    assert len(Y) == len(ts.data) - 2
    # Row after the boundary should use 2010 as the previous observation.
    assert W.iloc[5]["1_lag1"] == 6.0
