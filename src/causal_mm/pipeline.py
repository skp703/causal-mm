from __future__ import annotations

import argparse
import logging
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from .bootstrap import bootstrap_all_edges
from .config import BootstrapConfig, DMLConfig, LagConfig, MLModelConfig
from .data import standardize_timeseries
from .dml import estimate_all_edges_dml
from .io import load_project, save_project
from .fcm import FCMGraph
from .metrics import graph_complexity_metrics, concept_centrality_metrics

logger = logging.getLogger(__name__)


def _build_adjacency_result(fcm: FCMGraph) -> Dict[str, Any]:
    """
    Construct adjacency matrix results payload from the current estimates.
    """

    adjacency_matrix = fcm.adjacency_matrix(use_estimates=True).tolist()
    return {
        "concept_ids": fcm.get_concept_ids(),
        "matrix": adjacency_matrix,
        "weight_type": "scaled_weight",
    }


def _compute_graph_metrics(fcm: FCMGraph) -> Dict[str, Any]:
    """
    Compute complexity and centrality metrics for the results block.
    """
    complexity = graph_complexity_metrics(fcm)
    centrality_df = concept_centrality_metrics(fcm)
    
    # Convert DataFrame to list of dicts for JSON
    centrality_list = centrality_df.reset_index().to_dict(orient="records")
    
    return {
        "graph_complexity": complexity,
        "concept_centrality": centrality_list
    }


def run_estimation(
    input_path: Path,
    output_path: Path,
    dml_config: DMLConfig,
    bootstrap_config: Optional[BootstrapConfig] = None,
) -> None:
    """
    Load fcm_project.json, estimate all edges (DML + optional bootstrap),
    normalize weights to [-1, +1], stamp metadata, and save to output_path.

    This is the high-level entry point to go from raw project JSON to an
    enriched JSON with estimates and computation metadata.
    """

    fcm, ts, settings, meta = load_project(Path(input_path))
    meta_out = deepcopy(meta)

    # Standardize once; both DML and bootstrap use the same transformed data.
    if dml_config.standardize:
        ts = standardize_timeseries(ts)

    fcm = estimate_all_edges_dml(ts=ts, fcm=fcm, dml_config=dml_config, bootstrap_config=None)
    if bootstrap_config:
        fcm = bootstrap_all_edges(ts, fcm, dml_config, bootstrap_config)
        meta_out["weights_method"] = "bootstrap-dml"
    else:
        meta_out["weights_method"] = "dml"

    meta_out["weights_computed_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    adjacency_result = _build_adjacency_result(fcm)
    metrics_result = _compute_graph_metrics(fcm)
    
    results = {
        "adjacency_matrix": adjacency_result,
        **metrics_result
    }
    save_project(output_path, fcm, ts, settings, meta_out, results=results)


def recompute_adjacency(input_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Rebuild the adjacency matrix results block from an existing project JSON.

    If output_path is provided, write to that file; otherwise overwrite the input.
    Returns the adjacency result payload.
    """

    source_path = Path(input_path)
    target_path = Path(output_path) if output_path else source_path
    fcm, ts, settings, meta = load_project(source_path)
    
    adjacency_result = _build_adjacency_result(fcm)
    metrics_result = _compute_graph_metrics(fcm)

    with source_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    existing_results = raw.get("results")
    if not isinstance(existing_results, dict):
        existing_results = {}
        
    existing_results["adjacency_matrix"] = adjacency_result
    existing_results.update(metrics_result)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_project(target_path, fcm, ts, settings, meta, results=existing_results)
    return adjacency_result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run causal_mm estimation on an fcm_project.json")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to input fcm_project.json")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Path to write output fcm_project.json")
    parser.add_argument("--max-lag", type=int, default=3, help="Maximum lag to include (default: 3)")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of forward CV folds (default: 3)")
    parser.add_argument("--min-train-size", type=int, default=10, help="Minimum train size per fold (default: 10)")
    parser.add_argument("--alpha-scale", type=float, default=1.0, help="Scaling factor for tanh mapping of tau")
    parser.add_argument("--outcome-model", type=str, default="ridge", help="Outcome model type (ridge, rf, gbm, lasso, linear)")
    parser.add_argument("--treatment-model", type=str, default="ridge", help="Treatment model type (ridge, rf, gbm, lasso, linear)")
    parser.add_argument("--use-econml", action="store_true", help="Use econml estimators instead of built-in DML")
    parser.add_argument(
        "--econml-estimator",
        type=str,
        default="linear_dml",
        choices=["linear_dml", "causal_forest", "ortho_forest"],
        help="econml estimator to use when --use-econml is set",
    )
    parser.add_argument(
        "--controls-selection",
        type=str,
        default="all",
        choices=["all", "connected"],
        help="Strategy for selecting control variables: 'all' (default) uses all concepts, 'connected' uses only parents of the target.",
    )
    parser.add_argument(
        "--treatment-lag",
        type=int,
        default=0,
        help="Lag for treatment variable: 0 (default) = contemporaneous T_t -> Y_t, 1 = lagged T_{t-1} -> Y_t, etc.",
    )
    parser.add_argument("--no-standardize", action="store_true", help="Skip z-score standardization of concept time series (default: standardize)")
    parser.add_argument("--bootstrap", action="store_true", help="Enable block bootstrap for uncertainty")
    parser.add_argument("--n-bootstrap", type=int, default=200, help="Number of bootstrap replications")
    parser.add_argument("--block-size", type=int, default=5, help="Block size for bootstrap")
    parser.add_argument("--bs-n-jobs", type=int, default=1, help="Parallel jobs for bootstrap (joblib style)")
    parser.add_argument("--random-state", type=int, default=123, help="Random seed for folds/bootstrap")
    return parser


def main():
    """
    CLI entrypoint: parse args, construct configs, and run estimation.
    """

    args = _build_arg_parser().parse_args()

    lag_config = LagConfig(max_lag=args.max_lag)
    outcome_model = MLModelConfig(model_type=args.outcome_model)
    treatment_model = MLModelConfig(model_type=args.treatment_model)
    dml_config = DMLConfig(
        lag_config=lag_config,
        outcome_model=outcome_model,
        treatment_model=treatment_model,
        n_folds=args.n_folds,
        min_train_size=args.min_train_size,
        random_state=args.random_state,
        alpha_scale=args.alpha_scale,
        use_econml=bool(args.use_econml),
        econml_estimator=args.econml_estimator,
        controls_selection=args.controls_selection,
        treatment_lag=args.treatment_lag,
        standardize=not args.no_standardize,
    )

    bs_config: Optional[BootstrapConfig] = None
    if args.bootstrap:
        bs_config = BootstrapConfig(
            n_bootstrap=args.n_bootstrap,
            block_size=args.block_size,
            random_state=args.random_state,
            n_jobs=args.bs_n_jobs,
        )

    run_estimation(
        input_path=args.input,
        output_path=args.output,
        dml_config=dml_config,
        bootstrap_config=bs_config,
    )


if __name__ == "__main__":
    main()
