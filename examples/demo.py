#!/usr/bin/env python
"""
Quick-start demo for causal-mm.

Reads a small 3-concept FCM project, runs DML estimation (with optional
bootstrap), and prints the results.  No external data required — the
demo JSON ships with the package.

Usage
-----
    python examples/demo.py                   # point estimates only
    python examples/demo.py --bootstrap       # with uncertainty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_mm.config import BootstrapConfig, DMLConfig, LagConfig, MLModelConfig
from causal_mm.pipeline import run_estimation


DEMO_INPUT = Path(__file__).with_name("demo_project.fcm_project.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="causal-mm quick-start demo")
    parser.add_argument(
        "--bootstrap", action="store_true", help="Enable block bootstrap for uncertainty"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_output.fcm_project.json"),
        help="Where to write the enriched project file (default: demo_output.fcm_project.json)",
    )
    args = parser.parse_args()

    print(f"Input : {DEMO_INPUT}")
    print(f"Output: {args.output}")

    dml_cfg = DMLConfig(
        lag_config=LagConfig(max_lag=2),
        outcome_model=MLModelConfig("ridge", {"alpha": 1.0}),
        treatment_model=MLModelConfig("ridge", {"alpha": 1.0}),
        n_folds=3,
        min_train_size=10,
        alpha_scale=1.0,
        standardize=True,
    )

    bs_cfg = None
    if args.bootstrap:
        bs_cfg = BootstrapConfig(n_bootstrap=200, block_size=5, random_state=123, n_jobs=1)

    run_estimation(
        input_path=DEMO_INPUT,
        output_path=args.output,
        dml_config=dml_cfg,
        bootstrap_config=bs_cfg,
    )

    # Pretty-print estimates
    with open(args.output) as f:
        proj = json.load(f)

    print("\n--- Estimated edge weights ---")
    for edge_key, est in proj.get("estimates", {}).items():
        line = f"  {edge_key}:  scaled_weight={est.get('scaled_weight', '?'):>7.4f}"
        if est.get("ci_low") is not None:
            line += f"  CI=[{est['ci_low']:.4f}, {est['ci_high']:.4f}]"
        if est.get("sign_stability") is not None:
            line += f"  sign_stability={est['sign_stability']:.2f}"
        line += f"  status={est.get('status', '?')}"
        print(line)

    adj = proj.get("results", {}).get("adjacency_matrix", {})
    if adj:
        print("\n--- Adjacency matrix ---")
        ids = adj["concept_ids"]
        print(f"  Concepts: {ids}")
        for row_id, row in zip(ids, adj["matrix"]):
            print(f"  {row_id}: {[round(v, 4) for v in row]}")

    print(f"\nDone. Full output written to {args.output}")


if __name__ == "__main__":
    main()
