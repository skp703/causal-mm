from pathlib import Path
import json

import numpy as np
import pandas as pd

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig
from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph, EdgeEstimate
from causal_mm.io import save_project, load_project
from causal_mm.pipeline import run_estimation, recompute_adjacency


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
