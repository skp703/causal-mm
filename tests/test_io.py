from pathlib import Path
import json

import numpy as np
import pandas as pd

from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph, EdgeEstimate
from causal_mm.io import load_project, save_project


def test_save_and_load_roundtrip(tmp_path: Path):
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", tau_raw=0.5, scaled_weight=0.4)

    index = np.arange(5)
    df = pd.DataFrame({"1": [1, 2, 3, 4, 5], "2": [5, 4, 3, 2, 1]})
    ts = TimeSeriesData(index=index, data=df)
    settings = {"lag_config": {"max_lag": 1}}
    meta = {"project_id": "test"}

    path = tmp_path / "project.json"
    results = {
        "adjacency_matrix": {
            "concept_ids": ["1", "2"],
            "matrix": [[0.0, 0.0], [0.4, 0.0]],
            "weight_type": "scaled_weight",
        }
    }
    save_project(path, fcm, ts, settings, meta, results=results)
    loaded_fcm, loaded_ts, loaded_settings, loaded_meta = load_project(path)
    assert loaded_settings == settings
    assert loaded_meta["project_id"] == "test"
    assert loaded_ts.data.shape == ts.data.shape
    assert ("1", "2") in loaded_fcm.estimates
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw.get("results") == results
