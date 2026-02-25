"""
Placebo tests: Estimate effects for edges that shouldn't exist.

These should produce small/insignificant weights if the method is working correctly.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from causal_mm.config import LagConfig, MLModelConfig, DMLConfig
from causal_mm.io import load_project
from causal_mm.dml import estimate_edge_dml
from causal_mm.fcm import Edge


def test_reverse_causality_placebo(project_path="data/models/a.fcm_project.json"):
    """
    Test reverse direction of a known causal relationship.
    
    Example: If Temperature -> Pecan_Area makes sense,
             then Pecan_Area -> Temperature should be weak/zero.
    """
    if not Path(project_path).exists():
        return  # Skip if test data not available
    
    fcm, ts, settings, meta = load_project(Path(project_path))
    
    dml_config = DMLConfig(
        lag_config=LagConfig(max_lag=3, include_self_lags=False, drop_initial_na=True),
        outcome_model=MLModelConfig(model_type="ridge"),
        treatment_model=MLModelConfig(model_type="ridge"),
        n_folds=3,
        min_train_size=10,
        alpha_scale=1.0,
    )
    
    # Find a strong edge in the model
    # Then test the reverse
    # Example: If you have "temp" -> "pecan", test "pecan" -> "temp"
    
    # This is a template - customize with your actual concept IDs
    # source, target = "your_treatment_concept", "your_outcome_concept"
    # est_forward = estimate_edge_dml(ts, fcm, source, target, dml_config)
    # est_reverse = estimate_edge_dml(ts, fcm, target, source, dml_config)
    
    # print(f"Forward ({source} -> {target}): {est_forward.tau_raw}")
    # print(f"Reverse ({target} -> {source}): {est_reverse.tau_raw}")
    
    # Reverse should be weaker (in absolute value)
    # assert abs(est_reverse.tau_raw) < abs(est_forward.tau_raw), \
    #     "Reverse causality should be weaker"


def test_temporally_impossible_relationship(project_path="data/models/a.fcm_project.json"):
    """
    Test an edge between concepts that occur at very different time scales
    or have no plausible mechanism.
    
    Example: "Historical precipitation 10 years ago" -> "Current price"
    (unless there's a strong storage/memory mechanism, this should be weak)
    """
    pass  # Implement based on your domain knowledge


if __name__ == "__main__":
    print("Running placebo tests...")
    test_reverse_causality_placebo()
    print("✅ Placebo tests completed")
