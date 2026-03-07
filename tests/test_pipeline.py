from pathlib import Path
import json

import numpy as np
import pandas as pd

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig, BootstrapConfig
from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph, EdgeEstimate
from causal_mm.io import save_project, load_project
from causal_mm.pipeline import run_estimation, recompute_adjacency, _get_detrend_plan
from causal_mm.data import detrend_timeseries


def test_pipeline_run(tmp_path: Path):
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(8)
    df = pd.DataFrame({"1": np.random.randn(8), "2": np.random.randn(8)})
    ts = TimeSeriesData(index=index, data=df)

    project_path = tmp_path / "project.json"
    save_project(project_path, fcm, ts, settings={}, meta={"project_id": "pipeline"})

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
    )

    output_path = tmp_path / "out.json"
    run_estimation(project_path, output_path, dml_config, bootstrap_config=None)

    loaded_fcm, _, _, meta = load_project(output_path)
    assert meta.get("weights_computed_at") is not None
    assert ("1", "2") in loaded_fcm.estimates
    with output_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    adj_result = raw.get("results", {}).get("adjacency_matrix")
    assert adj_result is not None
    assert adj_result["concept_ids"] == ["1", "2"]
    assert len(adj_result["matrix"]) == 2
    assert len(adj_result["matrix"][0]) == 2


def test_recompute_adjacency_overwrites_results(tmp_path: Path):
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.8)

    index = np.arange(4)
    df = pd.DataFrame({"1": np.arange(4), "2": np.arange(4)})
    ts = TimeSeriesData(index=index, data=df)

    project_path = tmp_path / "project.json"
    save_project(project_path, fcm, ts, settings={}, meta={"project_id": "adj"}, results={"other": {"foo": 1}})

    result = recompute_adjacency(project_path)
    with project_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    assert raw["results"]["adjacency_matrix"] == result
    assert raw["results"]["other"] == {"foo": 1}
    assert result["concept_ids"] == ["1", "2"]
    assert result["matrix"][0][1] == 0.8


def test_mixed_detrend_order_invariance():
    """Regression test: mixed linear + first_diff detrend must not depend on concept order.

    The pipeline should apply non-row-dropping methods (linear) before first_diff,
    so that linear detrending operates on the full series regardless of concept ordering.
    """
    np.random.seed(42)
    n = 50
    t = np.arange(n, dtype=float)
    # X has a linear trend, Y has a linear trend
    X = 0.5 * t + np.random.randn(n) * 0.1
    Y = 0.3 * t + np.random.randn(n) * 0.1

    df = pd.DataFrame({"X": X, "Y": Y})
    ts = TimeSeriesData(index=np.arange(n), data=df)

    # Concept X gets linear detrend, concept Y gets first_diff
    concepts_a = [
        Concept(id="X", metadata={"detrend": "linear"}),
        Concept(id="Y", metadata={"detrend": "first_diff"}),
    ]
    # Reversed concept order
    concepts_b = [
        Concept(id="Y", metadata={"detrend": "first_diff"}),
        Concept(id="X", metadata={"detrend": "linear"}),
    ]
    edges = [Edge(source="X", target="Y")]

    fcm_a = FCMGraph(concepts=concepts_a, edges=edges)
    fcm_b = FCMGraph(concepts=concepts_b, edges=edges)

    plan_a = _get_detrend_plan(fcm_a, "linear", ts)
    plan_b = _get_detrend_plan(fcm_b, "linear", ts)

    # Both plans should have the same methods and columns
    assert plan_a == plan_b

    # Apply detrend in pipeline order (non-first_diff first, then first_diff)
    ts_a = ts
    for method in sorted(plan_a, key=lambda m: m == "first_diff"):
        ts_a = detrend_timeseries(ts_a, method=method, columns=plan_a[method])

    ts_b = ts
    for method in sorted(plan_b, key=lambda m: m == "first_diff"):
        ts_b = detrend_timeseries(ts_b, method=method, columns=plan_b[method])

    # Results should be identical regardless of concept ordering
    np.testing.assert_array_almost_equal(ts_a.data["X"].values, ts_b.data["X"].values)
    np.testing.assert_array_almost_equal(ts_a.data["Y"].values, ts_b.data["Y"].values)


def test_pipeline_bootstrap_snapshot_includes_ci_alpha(tmp_path: Path):
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(20)
    df = pd.DataFrame({"1": np.random.randn(20), "2": np.random.randn(20)})
    ts = TimeSeriesData(index=index, data=df)

    project_path = tmp_path / "project.json"
    save_project(project_path, fcm, ts, settings={}, meta={"project_id": "pipeline_boot"})

    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=1),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=5,
    )
    bs_config = BootstrapConfig(n_bootstrap=10, block_size=3, random_state=0, ci_alpha=0.10)

    output_path = tmp_path / "out_boot.json"
    run_estimation(project_path, output_path, dml_config, bootstrap_config=bs_config)

    _, _, _, meta = load_project(output_path)
    snap = meta.get("estimation_config", {})
    assert snap.get("bootstrap", {}).get("ci_alpha") == 0.10
