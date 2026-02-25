import numpy as np

from causal_mm.fcm import Concept, Edge, EdgeEstimate, FCMGraph


def test_adjacency_uses_estimates():
    concepts = [Concept(id="1"), Concept(id="2")]
    edges = [Edge(source="1", target="2", stakeholder_weight=0.5)]
    fcm = FCMGraph(concepts=concepts, edges=edges)
    fcm.estimates[("1", "2")] = EdgeEstimate(source="1", target="2", scaled_weight=0.8)
    A = fcm.adjacency_matrix(use_estimates=True)
    assert np.isclose(A[0, 1], 0.8)
