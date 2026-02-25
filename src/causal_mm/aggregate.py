# src/causal_mm/aggregate.py
"""
Aggregate multiple stakeholder FCM adjacency matrices using scaled_weight.

Supports two input modes:
1) JSON project outputs from causal-mm (e.g., marisol_output.json)
   -> uses causal_mm.io.load_project, extracts FCMGraph at index 0, builds weighted adjacency
2) CSV adjacency matrices you already exported (e.g., marisol_output_matrix.csv)
   -> reads labeled square matrix (header row + row labels)

Aggregation outputs:
- aggregated weight matrix (mean/median)
- optional support matrix (how many maps have a non-zero edge)
- optional sign-agreement matrix

Designed to be imported as a module OR run as a CLI:
    python -m causal_mm.aggregate --inputs ... --output aggregated.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np

from causal_mm.io import load_project
from causal_mm.fcm import FCMGraph


@dataclass
class LabeledMatrix:
    labels: List[str]
    matrix: np.ndarray  # shape (n, n)


# ----------------------------
# IO helpers
# ----------------------------

def read_labeled_matrix_csv(path: str) -> LabeledMatrix:
    """
    Read a square labeled adjacency matrix CSV:
      - first row contains column labels (first cell blank)
      - first column contains row labels
    """
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col_labels = header[1:]
        row_labels: List[str] = []
        rows: List[List[float]] = []
        for line in r:
            row_labels.append(line[0])
            rows.append([float(x) for x in line[1:]])

    if row_labels != col_labels:
        raise ValueError(f"Row/col labels mismatch in {path}")

    M = np.asarray(rows, dtype=float)
    return LabeledMatrix(labels=row_labels, matrix=M)


def write_labeled_matrix_csv(path: str, labels: Sequence[str], matrix: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(labels))
        for i, lab in enumerate(labels):
            w.writerow([lab] + [float(x) for x in matrix[i, :]])


def load_fcm_from_json_project(path: str) -> FCMGraph:
    """
    Your confirmed structure: load_project(path) returns a tuple where index 0 is FCMGraph.
    """
    proj = load_project(path)
    if isinstance(proj, tuple) and len(proj) > 0 and isinstance(proj[0], FCMGraph):
        return proj[0]
    if isinstance(proj, FCMGraph):
        return proj
    raise TypeError(f"Could not extract FCMGraph from {path}. Got {type(proj)}")


# ----------------------------
# Building matrices from FCMGraph (scaled_weight)
# ----------------------------

def adjacency_from_fcm_scaled_weight(
    fcm: FCMGraph,
    *,
    ignore_self_loops: bool = True,
    include_zero_edges: bool = False,
) -> LabeledMatrix:
    """
    Build weighted adjacency matrix from fcm.estimates using est.scaled_weight.

    - ignore_self_loops=True removes i->i edges.
    - include_zero_edges=False means we only place non-zero weights; otherwise zeros are still zeros.
    """
    labels = list(fcm.get_concept_ids())
    idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    A = np.zeros((n, n), dtype=float)

    for (src, tgt), est in fcm.estimates.items():
        if src not in idx or tgt not in idx:
            continue
        i, j = idx[src], idx[tgt]
        if ignore_self_loops and i == j:
            continue
        w = est.scaled_weight if est.scaled_weight is not None else 0.0
        w = float(w)
        if not include_zero_edges and abs(w) < 1e-12:
            continue
        A[i, j] = w

    return LabeledMatrix(labels=labels, matrix=A)


# ----------------------------
# Alignment (union of nodes)
# ----------------------------

def align_to_universe(mats: Sequence[LabeledMatrix]) -> Tuple[List[str], List[np.ndarray], List[List[bool]]]:
    """
    Align a list of labeled matrices to a common universe of labels.
    Missing rows/cols are filled with 0.0.
    Returns:
      universe_labels,
      expanded_matrices,
      present_flags_per_matrix (bool list per universe label)
    """
    universe = sorted(set().union(*(set(m.labels) for m in mats)))
    u_idx = {lab: i for i, lab in enumerate(universe)}
    nU = len(universe)

    expanded: List[np.ndarray] = []
    present_flags: List[List[bool]] = []

    for m in mats:
        E = np.zeros((nU, nU), dtype=float)
        local_idx = {lab: i for i, lab in enumerate(m.labels)}
        for a in m.labels:
            ia = u_idx[a]
            la = local_idx[a]
            for b in m.labels:
                ib = u_idx[b]
                lb = local_idx[b]
                E[ia, ib] = float(m.matrix[la, lb])
        expanded.append(E)
        present_flags.append([lab in local_idx for lab in universe])

    return universe, expanded, present_flags


# ----------------------------
# Aggregation
# ----------------------------

def aggregate_matrices(
    matrices: Sequence[np.ndarray],
    *,
    method: str = "mean",
    ignore_self_loops: bool = True,
) -> np.ndarray:
    """
    Aggregate aligned matrices (same shape) into one.
    method:
      - mean: arithmetic mean
      - median: elementwise median
    """
    if not matrices:
        raise ValueError("No matrices provided to aggregate_matrices().")

    stack = np.stack(matrices, axis=0)  # (k, n, n)

    if method == "mean":
        agg = np.mean(stack, axis=0)
    elif method == "median":
        agg = np.median(stack, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    if ignore_self_loops:
        np.fill_diagonal(agg, 0.0)

    return agg


def support_matrix(
    matrices: Sequence[np.ndarray],
    *,
    threshold: float = 1e-12,
    ignore_self_loops: bool = True,
) -> np.ndarray:
    """
    Count how many input matrices have a non-zero edge at each (i,j).
    """
    stack = np.stack(matrices, axis=0)
    sup = np.sum(np.abs(stack) > threshold, axis=0).astype(int)
    if ignore_self_loops:
        np.fill_diagonal(sup, 0)
    return sup


def sign_agreement_matrix(
    matrices: Sequence[np.ndarray],
    *,
    threshold: float = 1e-12,
    ignore_self_loops: bool = True,
) -> np.ndarray:
    """
    For each edge (i,j), compute sign agreement among non-zero edges.
    Returns values in [0,1]:
      1.0 means all non-zero edges share the same sign,
      0.5 means split evenly (or low agreement),
      0.0 means perfect disagreement (rare with >2 maps).
    If no non-zero edges exist for (i,j), returns 0.0.
    """
    stack = np.stack(matrices, axis=0)
    signs = np.sign(stack)  # -1,0,1
    mask = np.abs(stack) > threshold

    n = stack.shape[1]
    out = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if ignore_self_loops and i == j:
                continue
            s = signs[:, i, j]
            m = mask[:, i, j]
            if not np.any(m):
                out[i, j] = 0.0
                continue
            s_nz = s[m]
            pos = np.sum(s_nz > 0)
            neg = np.sum(s_nz < 0)
            total = pos + neg
            out[i, j] = (max(pos, neg) / total) if total > 0 else 0.0

    return out


# ----------------------------
# CLI
# ----------------------------

def _infer_input_kind(path: str) -> str:
    p = path.lower()
    if p.endswith(".json"):
        return "json"
    if p.endswith(".csv"):
        return "csv"
    raise ValueError(f"Unsupported input extension: {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate FCM adjacency matrices using scaled_weight.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input files: either JSON causal-mm outputs or labeled adjacency CSVs.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output CSV path for aggregated weighted adjacency matrix.",
    )
    ap.add_argument(
        "--method",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation method.",
    )
    ap.add_argument(
        "--ignore_self_loops",
        action="store_true",
        help="Force diagonal to 0 in outputs (recommended).",
    )
    ap.add_argument(
        "--support_output",
        default=None,
        help="Optional output CSV for support counts (non-zero edge count across maps).",
    )
    ap.add_argument(
        "--sign_output",
        default=None,
        help="Optional output CSV for sign agreement (0..1).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=1e-12,
        help="Threshold for considering an edge present/non-zero.",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    # Load inputs into LabeledMatrix objects
    loaded: List[LabeledMatrix] = []
    kinds = {_infer_input_kind(p) for p in args.inputs}
    if len(kinds) != 1:
        raise SystemExit("Please provide all inputs as JSON or all as CSV (do not mix).")

    kind = next(iter(kinds))

    if kind == "json":
        for p in args.inputs:
            fcm = load_fcm_from_json_project(p)
            loaded.append(adjacency_from_fcm_scaled_weight(fcm, ignore_self_loops=args.ignore_self_loops))
    else:  # csv
        for p in args.inputs:
            loaded.append(read_labeled_matrix_csv(p))

    # Align and aggregate
    universe, expanded, _present = align_to_universe(loaded)
    agg = aggregate_matrices(expanded, method=args.method, ignore_self_loops=args.ignore_self_loops)

    write_labeled_matrix_csv(args.output, universe, agg)

    if args.support_output:
        sup = support_matrix(expanded, threshold=args.threshold, ignore_self_loops=args.ignore_self_loops)
        write_labeled_matrix_csv(args.support_output, universe, sup)

    if args.sign_output:
        sgn = sign_agreement_matrix(expanded, threshold=args.threshold, ignore_self_loops=args.ignore_self_loops)
        write_labeled_matrix_csv(args.sign_output, universe, sgn)

    print(f"Wrote aggregated matrix to {args.output}")
    if args.support_output:
        print(f"Wrote support matrix to {args.support_output}")
    if args.sign_output:
        print(f"Wrote sign agreement matrix to {args.sign_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
