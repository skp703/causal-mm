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
    fcm.estimates[("1", "2")] = EdgeEstimate(
        source="1",
        target="2",
        tau_raw=0.5,
        scaled_weight=0.4,
        ci_low=0.1,
        ci_high=0.8,
        ci_alpha=0.10,
    )

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
    np.testing.assert_array_equal(loaded_ts.index, index)
    assert ("1", "2") in loaded_fcm.estimates
    assert loaded_fcm.estimates[("1", "2")].ci_alpha == 0.10
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["timeseries"]["index"] == index.tolist()
    assert raw["estimates"]["1->2"]["ci_alpha"] == 0.10
    assert raw.get("results") == results


def test_load_project_coerces_legacy_numeric_string_index(tmp_path: Path):
    path = tmp_path / "legacy_project.json"
    raw = {
        "meta": {"project_id": "legacy"},
        "model": {
            "concepts": [{"id": 1}, {"id": 2}],
            "edges": [{"source": 1, "target": 2}],
        },
        "timeseries": {
            "index": ["1990", "1991", "1993"],
            "data": {"1": [1.0, 2.0, 3.0], "2": [4.0, 5.0, 6.0]},
        },
        "settings": {},
        "estimates": {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f)

    _, ts, _, _ = load_project(path)
    np.testing.assert_array_equal(ts.index, np.array([1990, 1991, 1993]))


def test_load_project_preserves_nonfinite_numeric_tokens_as_strings(tmp_path: Path):
    path = tmp_path / "legacy_nonfinite_index.json"
    raw = {
        "meta": {"project_id": "legacy_nonfinite"},
        "model": {
            "concepts": [{"id": 1}, {"id": 2}],
            "edges": [{"source": 1, "target": 2}],
        },
        "timeseries": {
            "index": ["inf", "2"],
            "data": {"1": [1.0, 2.0], "2": [3.0, 4.0]},
        },
        "settings": {},
        "estimates": {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f)

    _, ts, _, _ = load_project(path)
    np.testing.assert_array_equal(ts.index, np.array(["inf", "2"], dtype=object))


def test_load_project_parses_datetime_like_legacy_strings(tmp_path: Path):
    path = tmp_path / "legacy_datetime_index.json"
    raw = {
        "meta": {"project_id": "legacy_dt"},
        "model": {
            "concepts": [{"id": 1}, {"id": 2}],
            "edges": [{"source": 1, "target": 2}],
        },
        "timeseries": {
            "index": ["2020-01-01", "2020-01-02"],
            "data": {"1": [1.0, 2.0], "2": [3.0, 4.0]},
        },
        "settings": {},
        "estimates": {},
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f)

    _, ts, _, _ = load_project(path)
    assert np.issubdtype(np.asarray(ts.index).dtype, np.datetime64)
