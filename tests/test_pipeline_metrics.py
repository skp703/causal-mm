
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from causal_mm.pipeline import recompute_adjacency
from causal_mm.fcm import FCMGraph, Concept, EdgeEstimate
from causal_mm.io import save_project
from causal_mm.data import TimeSeriesData

def test_pipeline_metrics_integration():
    # Create a dummy project
    c1 = Concept(id="1", label="C1")
    c2 = Concept(id="2", label="C2")
    
    fcm = FCMGraph(concepts=[c1, c2], edges=[])
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.8)
    
    # Dummy timeseries (needed for save_project but not used for recompute_adjacency)
    ts = TimeSeriesData(
        data=pd.DataFrame({"1": np.random.randn(10), "2": np.random.randn(10)}),
        index=pd.date_range("2020-01-01", periods=10)
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test_project.json"
        
        # Save initial project
        save_project(p, fcm, ts, settings={}, meta={})
        
        # Run recompute_adjacency
        recompute_adjacency(p)
        
        # Load and check results
        with open(p, "r") as f:
            data = json.load(f)
            
        results = data.get("results", {})
        
        # Check Graph Complexity
        assert "graph_complexity" in results
        comp = results["graph_complexity"]
        assert comp["N"] == 2
        assert comp["C"] == 1
        assert abs(comp["Density"] - 0.25) < 1e-6
        
        # Check Concept Centrality
        assert "concept_centrality" in results
        cent = results["concept_centrality"]
        assert isinstance(cent, list)
        assert len(cent) == 2
        
        # Find concept 1
        c1_metrics = next(item for item in cent if item["concept"] == "1")
        assert abs(c1_metrics["out_degree"] - 0.8) < 1e-6
        assert abs(c1_metrics["in_degree"] - 0.0) < 1e-6
        
        print("Pipeline metrics integration test passed!")

if __name__ == "__main__":
    test_pipeline_metrics_integration()
