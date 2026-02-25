"""
causal_mm package initialization.

Exports core configs, data structures, IO, DML, bootstrap, metrics, and pipeline helpers.
"""

from .config import LagConfig, MLModelConfig, DMLConfig, BootstrapConfig
from .data import TimeSeriesData, build_lagged_design, align_fcm_and_timeseries, standardize_timeseries
from .fcm import Concept, Edge, EdgeEstimate, FCMGraph
from .dml import estimate_edge_dml, estimate_all_edges_dml
from .bootstrap import bootstrap_edge, bootstrap_all_edges
from .io import load_project, save_project
from .pipeline import run_estimation, recompute_adjacency
from .metrics import (
    proportion_edges_significant,
    average_sign_stability,
    score_models,
    best_model,
    compare_weights,
    weight_distance,
    generalized_distance_ratio,
    graph_complexity_metrics,
    concept_centrality_metrics,
    get_adjacency_matrix,
    get_adjacency_matrix_labeled,
    load_id_to_label_from_output_json,
    graph_complexity_metrics_from_adjacency,
)

__all__ = [
    "LagConfig",
    "MLModelConfig",
    "DMLConfig",
    "BootstrapConfig",
    "TimeSeriesData",
    "build_lagged_design",
    "align_fcm_and_timeseries",
    "standardize_timeseries",
    "Concept",
    "Edge",
    "EdgeEstimate",
    "FCMGraph",
    "estimate_edge_dml",
    "estimate_all_edges_dml",
    "bootstrap_edge",
    "bootstrap_all_edges",
    "load_project",
    "save_project",
    "run_estimation",
    "recompute_adjacency",
    "proportion_edges_significant",
    "average_sign_stability",
    "score_models",
    "best_model",
    "compare_weights",
    "weight_distance",
    "generalized_distance_ratio",
    "graph_complexity_metrics",
    "concept_centrality_metrics",
    "get_adjacency_matrix",
    "get_adjacency_matrix_labeled",
    "load_id_to_label_from_output_json",
    "graph_complexity_metrics_from_adjacency",
]
