import numpy as np
import pandas as pd
import pytest

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig, BootstrapConfig
from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph
from causal_mm.bootstrap import bootstrap_edge, bootstrap_all_edges, _prepare_bootstrap_matrices
from causal_mm.dml import _prepare_matrices


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


def test_bootstrap_ci_alpha_changes_interval_width():
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(30)
    df = pd.DataFrame({"1": np.random.randn(30), "2": np.random.randn(30)})
    ts = TimeSeriesData(index=index, data=df)

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=5,
    )
    bs_95 = BootstrapConfig(n_bootstrap=25, block_size=3, random_state=0, n_jobs=1, ci_alpha=0.05)
    bs_80 = BootstrapConfig(n_bootstrap=25, block_size=3, random_state=0, n_jobs=1, ci_alpha=0.20)

    s95 = bootstrap_edge(ts, fcm, "1", "2", dml_config, bs_95)
    s80 = bootstrap_edge(ts, fcm, "1", "2", dml_config, bs_80)

    assert s95["ci_low"] is not None and s95["ci_high"] is not None
    assert s80["ci_low"] is not None and s80["ci_high"] is not None
    # 95% CI (alpha=0.05) should be at least as wide as 80% CI (alpha=0.20)
    assert s95["ci_low"] <= s80["ci_low"]
    assert s95["ci_high"] >= s80["ci_high"]


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


def test_bootstrap_applies_treatment_lag():
    """Verify _prepare_bootstrap_matrices shifts T by treatment_lag, matching _prepare_matrices."""
    np.random.seed(0)
    n = 20
    index = np.arange(n)
    df = pd.DataFrame({"X": np.arange(n, dtype=float), "Y": np.arange(100, 100 + n, dtype=float)})
    ts = TimeSeriesData(index=index, data=df)

    concepts = [Concept(id="X"), Concept(id="Y")]
    edges = [Edge(source="X", target="Y")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1, include_self_lags=False, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
        treatment_lag=1,
    )

    # Identity resampling (no shuffling) so bootstrap output matches point-estimate path
    identity_idx = np.arange(n)

    Y_bs, T_bs, W_bs = _prepare_bootstrap_matrices(ts, fcm, "X", "Y", dml_config, identity_idx)
    Y_pt, T_pt, W_pt = _prepare_matrices(ts, fcm, "X", "Y", dml_config)

    # Both paths should produce the same T values (past treatment, not future)
    np.testing.assert_array_equal(T_bs.values, T_pt.values)
    np.testing.assert_array_equal(Y_bs.values, Y_pt.values)

    # T should contain past values: at the first valid row, T should be less than Y
    # (X values are 0..19, Y values are 100..119, so T_{t-1} < T_t always)
    assert T_bs.iloc[0] < T_bs.iloc[1], "T should be in ascending order (past values)"


def test_bootstrap_applies_controls_selection():
    """Verify _prepare_bootstrap_matrices filters controls when controls_selection='connected'."""
    np.random.seed(0)
    n = 20
    index = np.arange(n)
    df = pd.DataFrame({
        "X": np.random.randn(n),
        "Y": np.random.randn(n),
        "Z": np.random.randn(n),  # Z is not connected to Y in the FCM
    })
    ts = TimeSeriesData(index=index, data=df)

    concepts = [Concept(id="X"), Concept(id="Y"), Concept(id="Z")]
    edges = [Edge(source="X", target="Y")]  # Z has no edge to Y
    fcm = FCMGraph(concepts=concepts, edges=edges)

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1, include_self_lags=False, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
        controls_selection="connected",
    )

    identity_idx = np.arange(n)

    Y_bs, T_bs, W_bs = _prepare_bootstrap_matrices(ts, fcm, "X", "Y", dml_config, identity_idx)
    Y_pt, T_pt, W_pt = _prepare_matrices(ts, fcm, "X", "Y", dml_config)

    # Both paths should have the same control columns (Z lags excluded)
    assert set(W_bs.columns) == set(W_pt.columns)
    # Z_lag1 should NOT be in controls since Z is not a parent of Y
    assert "Z_lag1" not in W_bs.columns
    # X_lag1 SHOULD be present (X is source/parent of Y)
    assert "X_lag1" in W_bs.columns


def test_bootstrap_all_reps_failed_sets_error_status():
    """Regression test: when all bootstrap reps fail, edge status must be 'failed' with error message."""
    concepts = [Concept(id="X"), Concept(id="Y")]
    edges = [Edge(source="X", target="Y")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    # Tiny dataset: 3 rows with max_lag=2 and block_size=3 means after lag trimming
    # and block boundary masking, resampled matrices will be empty.
    index = np.arange(3)
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0]})
    ts = TimeSeriesData(index=index, data=df)

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=2, include_self_lags=False, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
    )
    bs_config = BootstrapConfig(n_bootstrap=5, block_size=3, random_state=0, n_jobs=1)

    # bootstrap_edge should return error when all reps fail
    summary = bootstrap_edge(ts, fcm, "X", "Y", dml_config, bs_config)
    assert summary.get("error") is not None
    assert summary["tau_se"] is None
    assert summary["ci_low"] is None

    # bootstrap_all_edges should set status='failed' on the estimate
    fcm2 = FCMGraph(concepts=concepts, edges=edges)
    fcm2 = bootstrap_all_edges(ts, fcm2, dml_config, bs_config)
    est = fcm2.estimates.get(("X", "Y"))
    assert est is not None
    assert est.status == "failed"
    assert est.error_message is not None
    assert est.ci_alpha is None


def test_bootstrap_all_edges_sets_ci_alpha():
    concepts = [Concept(id="X"), Concept(id="Y")]
    edges = [Edge(source="X", target="Y")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(30)
    df = pd.DataFrame({"X": np.random.randn(30), "Y": np.random.randn(30)})
    ts = TimeSeriesData(index=index, data=df)
    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1, include_self_lags=False, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=5,
    )
    bs_config = BootstrapConfig(n_bootstrap=20, block_size=3, random_state=0, n_jobs=1, ci_alpha=0.10)
    fcm = bootstrap_all_edges(ts, fcm, dml_config, bs_config)
    est = fcm.estimates.get(("X", "Y"))
    assert est is not None
    assert est.ci_low is not None and est.ci_high is not None
    assert est.ci_alpha == pytest.approx(0.10)


def test_nonpositive_block_size_rejected():
    """Regression test: BootstrapConfig must reject non-positive block_size values."""
    with pytest.raises(ValueError, match="block_size must be >= 1"):
        BootstrapConfig(n_bootstrap=10, block_size=0)


def test_nonpositive_n_bootstrap_rejected():
    """Regression test: BootstrapConfig must reject non-positive n_bootstrap values."""
    with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
        BootstrapConfig(n_bootstrap=0, block_size=5)


def test_invalid_ci_alpha_rejected():
    """Regression test: BootstrapConfig must reject ci_alpha outside (0,1)."""
    with pytest.raises(ValueError, match="ci_alpha must be in \\(0, 1\\)"):
        BootstrapConfig(n_bootstrap=10, block_size=5, ci_alpha=0.0)
    with pytest.raises(ValueError, match="ci_alpha must be in \\(0, 1\\)"):
        BootstrapConfig(n_bootstrap=10, block_size=5, ci_alpha=1.0)
