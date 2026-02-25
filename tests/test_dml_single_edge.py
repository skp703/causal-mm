import numpy as np
import pandas as pd

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig
from causal_mm.data import TimeSeriesData
from causal_mm.fcm import Concept, Edge, FCMGraph
from causal_mm.dml import estimate_edge_dml
import causal_mm.dml as dml_module


def test_estimate_edge_runs():
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2")]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = np.arange(10)
    df = pd.DataFrame({"1": np.linspace(0, 1, 10), "2": np.linspace(1, 0, 10)})
    ts = TimeSeriesData(index=index, data=df)

    lag_config = LagConfig(max_lag=1, drop_initial_na=True)
    dml_config = DMLConfig(
        lag_config=lag_config,
        outcome_model=MLModelConfig(model_type="ridge", params={"alpha": 1.0}),
        treatment_model=MLModelConfig(model_type="ridge", params={"alpha": 1.0}),
        n_folds=2,
        min_train_size=3,
        alpha_scale=1.0,
    )

    est = estimate_edge_dml(ts, fcm, "1", "2", dml_config)
    assert est.status in ("ok", "failed")


def _basic_dml_config():
    lag_config = LagConfig(max_lag=1, drop_initial_na=True)
    return DMLConfig(
        lag_config=lag_config,
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=2,
        min_train_size=3,
        alpha_scale=1.0,
    )


def test_init_linear_dml_includes_model_final_when_supported(monkeypatch):
    prev_flag = dml_module._LINEAR_DML_ACCEPTS_MODEL_FINAL
    prev_has = dml_module.HAS_ECONML
    try:
        dml_module._set_linear_dml_accepts_model_final(True)
        dml_module.HAS_ECONML = True

        captured_kwargs = {}

        class DummyLinearDML:
            def __init__(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(dml_module, "LinearDML", DummyLinearDML, raising=False)
        dml_module._init_linear_dml_estimator(
            outcome_model="y_model",
            treatment_model="t_model",
            folds=[(np.array([0]), np.array([1]))],
            dml_config=_basic_dml_config(),
        )

        assert "model_final" in captured_kwargs
    finally:
        dml_module._LINEAR_DML_ACCEPTS_MODEL_FINAL = prev_flag
        dml_module.HAS_ECONML = prev_has


def test_init_linear_dml_falls_back_when_model_final_rejected(monkeypatch):
    prev_flag = dml_module._LINEAR_DML_ACCEPTS_MODEL_FINAL
    prev_has = dml_module.HAS_ECONML
    try:
        dml_module._set_linear_dml_accepts_model_final(True)
        dml_module.HAS_ECONML = True

        init_calls = []

        class RejectingLinearDML:
            def __init__(self, **kwargs):
                init_calls.append(kwargs)
                if "model_final" in kwargs:
                    raise TypeError("unexpected keyword argument 'model_final'")

        monkeypatch.setattr(dml_module, "LinearDML", RejectingLinearDML, raising=False)
        dml_module._init_linear_dml_estimator(
            outcome_model="y_model",
            treatment_model="t_model",
            folds=[(np.array([0]), np.array([1]))],
            dml_config=_basic_dml_config(),
        )

        assert len(init_calls) == 2
        assert "model_final" in init_calls[0]
        assert "model_final" not in init_calls[1]
        assert dml_module._linear_dml_accepts_model_final() is False
    finally:
        dml_module._LINEAR_DML_ACCEPTS_MODEL_FINAL = prev_flag
        dml_module.HAS_ECONML = prev_has


def test_init_ortho_forest_prefers_uppercase(monkeypatch):
    captured_kwargs = {}

    class DummyOrthoForest:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(dml_module, "OrthoForest", DummyOrthoForest, raising=False)
    dml_module._init_ortho_forest_estimator(
        outcome_model="y_model",
        treatment_model="t_model",
        dml_config=_basic_dml_config(),
    )
    assert "model_Y" in captured_kwargs
    assert "model_T" in captured_kwargs


def test_init_ortho_forest_falls_back_to_lowercase(monkeypatch):
    init_calls = []

    class RejectingOrthoForest:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            if "model_Y" in kwargs:
                raise TypeError("unexpected keyword argument 'model_Y'")

    monkeypatch.setattr(dml_module, "OrthoForest", RejectingOrthoForest, raising=False)
    dml_module._init_ortho_forest_estimator(
        outcome_model="y_model",
        treatment_model="t_model",
        dml_config=_basic_dml_config(),
    )
    assert len(init_calls) == 2
    assert "model_Y" in init_calls[0]
    assert "model_y" in init_calls[1]
