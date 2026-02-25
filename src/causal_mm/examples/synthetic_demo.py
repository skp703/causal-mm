"""
Minimal synthetic demo to exercise causal_mm pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig
from causal_mm.fcm import Concept, Edge, FCMGraph
from causal_mm.data import TimeSeriesData
from causal_mm.pipeline import run_estimation
from causal_mm.io import save_project


def build_toy_project(path: Path) -> None:
    concepts = [Concept(id="1", label="A"), Concept(id="2", label="B")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(2000, 2030)
    data = pd.DataFrame({"1": np.random.randn(len(index)), "2": np.random.randn(len(index))})
    ts = TimeSeriesData(index=index, data=data)

    project = {
        "meta": {"project_id": "toy"},
        "model": {"concepts": [{"id": 1}, {"id": 2}], "edges": [{"source": 1, "target": 2}]},
        "timeseries": {"index": index.tolist(), "data": {"1": data["1"].tolist(), "2": data["2"].tolist()}},
        "settings": {},
        "estimates": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_project(path, fcm, ts, settings={}, meta={"project_id": "toy"})


def main():
    root = Path("demo_output")
    input_path = root / "toy.fcm_project.json"
    build_toy_project(input_path)

    lag_config = LagConfig(max_lag=1)
    outcome_model = MLModelConfig(model_type="ridge", params={"alpha": 1.0})
    treatment_model = MLModelConfig(model_type="ridge", params={"alpha": 1.0})
    dml_config = DMLConfig(
        lag_config=lag_config,
        outcome_model=outcome_model,
        treatment_model=treatment_model,
        n_folds=2,
        alpha_scale=1.0,
    )
    output_path = root / "toy_with_estimates.fcm_project.json"
    run_estimation(input_path=input_path, output_path=output_path, dml_config=dml_config, bootstrap_config=None)
    print(f"Demo complete. Output at {output_path}")


if __name__ == "__main__":
    main()
