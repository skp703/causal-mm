from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Concept:
    id: str  # stringified concept ID
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    stakeholder_weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeEstimate:
    source: str
    target: str
    tau_raw: Optional[float] = None
    tau_se: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_alpha: Optional[float] = None
    sign_stability: Optional[float] = None
    scaled_weight: Optional[float] = None
    n_obs: Optional[int] = None
    lag_used: Optional[int] = None
    status: str = "pending"
    error_message: Optional[str] = None
    computed_at: Optional[str] = None
    method: str = "dml"


@dataclass
class FCMGraph:
    concepts: List[Concept]
    edges: List[Edge]
    estimates: Dict[Tuple[str, str], EdgeEstimate] = field(default_factory=dict)

    def get_concept_ids(self) -> List[str]:
        return [str(c.id) for c in self.concepts]

    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        for edge in self.edges:
            if edge.source == source and edge.target == target:
                return edge
        return None

    def adjacency_matrix(self, use_estimates: bool = False) -> np.ndarray:
        """
        Build adjacency matrix A of shape (n_concepts, n_concepts).

        If use_estimates is False: use stakeholder_weight.
        If True: use scaled_weight from estimates (bounded in [-1, +1]).
        """

        concepts = self.get_concept_ids()
        n = len(concepts)
        idx = {cid: i for i, cid in enumerate(concepts)}
        A = np.zeros((n, n))
        for edge in self.edges:
            i = idx[edge.source]
            j = idx[edge.target]
            if use_estimates:
                est = self.estimates.get((edge.source, edge.target))
                val = est.scaled_weight if est and est.scaled_weight is not None else 0.0
            else:
                val = edge.stakeholder_weight if edge.stakeholder_weight is not None else 0.0
            A[i, j] = val
        return A
