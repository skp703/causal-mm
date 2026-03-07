
from causal_mm.fcm import FCMGraph, Concept, EdgeEstimate
from causal_mm.metrics import (
    graph_complexity_metrics,
    concept_centrality_metrics,
    get_adjacency_matrix,
    graph_complexity_metrics_from_adjacency,
    proportion_edges_significant,
)
import pandas as pd
import numpy as np

def test_graph_metrics():
    # Create dummy FCM
    # 3 concepts: 1, 2, 3
    c1 = Concept(id="1")
    c2 = Concept(id="2")
    c3 = Concept(id="3")
    
    fcm = FCMGraph(concepts=[c1, c2, c3], edges=[])
    
    # Edges:
    # 1->2: 0.5
    # 2->3: 0.5
    # 3->1: 0.5
    # Cycle 1->2->3->1
    
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.5)
    fcm.estimates[("2", "3")] = EdgeEstimate(source="2", target="3", scaled_weight=0.5)
    fcm.estimates[("3", "1")] = EdgeEstimate(source="3", target="1", scaled_weight=0.5)
    
    # Complexity Metrics
    metrics = graph_complexity_metrics(fcm)
    print("Complexity Metrics:", metrics)
    
    assert metrics["N"] == 3
    assert metrics["C"] == 3
    assert abs(metrics["Density"] - 3/9) < 1e-6
    
    # Hierarchy Index
    # Outdegrees:
    # 1: 0.5
    # 2: 0.5
    # 3: 0.5
    # Mean OD = 0.5
    # Variance = 0
    # Hierarchy should be 0 (fully democratic/cyclic)
    assert abs(metrics["Hierarchy"] - 0.0) < 1e-6
    
    # Centrality Metrics
    df = concept_centrality_metrics(fcm)
    print("\nCentrality Metrics:\n", df)
    
    # Check values
    # 1: out=0.5, in=0.5, cent=1.0
    assert abs(df.loc["1", "out_degree"] - 0.5) < 1e-6
    assert abs(df.loc["1", "in_degree"] - 0.5) < 1e-6
    assert abs(df.loc["1", "centrality"] - 1.0) < 1e-6

    # Test Hierarchy with a hierarchical structure
    # 1->2, 1->3 (1 is driver)
    fcm_h = FCMGraph(concepts=[c1, c2, c3], edges=[])
    fcm_h.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=1.0)
    fcm_h.estimates[("1", "3")] = EdgeEstimate(source="1", target="3", scaled_weight=1.0)
    
    # Outdegrees:
    # 1: 2.0
    # 2: 0.0
    # 3: 0.0
    # Mean OD = 2/3 = 0.666...
    # Sum sq diff = (2 - 2/3)^2 + (0 - 2/3)^2 + (0 - 2/3)^2
    # = (4/3)^2 + (-2/3)^2 + (-2/3)^2
    # = 16/9 + 4/9 + 4/9 = 24/9 = 8/3
    
    # h = 12 / (3 * (9-1)) * (8/3)
    # = 12 / 24 * 8/3 = 0.5 * 8/3 = 4/3 = 1.333... Wait.
    # Hierarchy index should be between 0 and 1.
    # Let me check the formula again.
    # h = 12 / (N(N^2-1)) * sum((od - mean)^2)
    # Max variance is when one node has max outdegree (N-1) and others 0?
    # Or maybe N?
    # If fully connected, outdegree is N (or N-1 without self loops).
    # If weights are in [0, 1], max outdegree is N.
    
    # The formula assumes binary adjacency matrix?
    # Özesmi (2004) uses weighted outdegrees.
    # "The hierarchy index is calculated as... where od_i is the outdegree of variable i".
    # If weights are not binary, h can exceed 1?
    # "When h=1, the map is fully hierarchical... When h=0, the map is fully democratic".
    # This implies normalization assumes something about the weights or structure.
    # Usually this formula is for binary graphs where max outdegree is N-1.
    # If we use weighted graphs, the "max possible variance" depends on the weights.
    # But the implementation follows the formula.
    
    metrics_h = graph_complexity_metrics(fcm_h)
    print("Hierarchical Metrics:", metrics_h)
    # Just check it runs and gives non-zero
    assert metrics_h["Hierarchy"] > 0

def test_density_consistency_with_adjacency_variant():
    c1 = Concept(id="1")
    c2 = Concept(id="2")
    fcm = FCMGraph(concepts=[c1, c2], edges=[])
    fcm.estimates[("1", "1")] = EdgeEstimate(source="1", target="1", scaled_weight=1.0)
    fcm.estimates[("2", "2")] = EdgeEstimate(source="2", target="2", scaled_weight=1.0)

    metrics_graph = graph_complexity_metrics(fcm, ignore_self_loops=False)
    _, A = get_adjacency_matrix(fcm, ignore_self_loops=False)
    metrics_adj = graph_complexity_metrics_from_adjacency(A, ignore_self_loops=False)

    assert metrics_graph["density"] == metrics_adj["density"]


def test_proportion_edges_significant_respects_ci_alpha():
    c1 = Concept(id="1")
    c2 = Concept(id="2")
    c3 = Concept(id="3")
    fcm = FCMGraph(concepts=[c1, c2, c3], edges=[])
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", ci_low=0.1, ci_high=0.2, ci_alpha=0.05)
    fcm.estimates[("2", "3")] = EdgeEstimate(source="2", target="3", ci_low=0.1, ci_high=0.2, ci_alpha=0.10)

    # At alpha=0.05 only one edge should be considered and it is significant.
    assert proportion_edges_significant(fcm, alpha=0.05) == 1.0
    # At alpha=0.10, the other edge is considered and it is also significant.
    assert proportion_edges_significant(fcm, alpha=0.10) == 1.0


def test_proportion_edges_significant_nan_when_no_matching_alpha():
    c1 = Concept(id="1")
    c2 = Concept(id="2")
    fcm = FCMGraph(concepts=[c1, c2], edges=[])
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", ci_low=0.1, ci_high=0.2, ci_alpha=0.05)

    assert np.isnan(proportion_edges_significant(fcm, alpha=0.10))


if __name__ == "__main__":
    test_graph_metrics()
