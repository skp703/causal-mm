from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

from .fcm import FCMGraph


def proportion_edges_significant(fcm: FCMGraph, alpha: float = 0.05) -> float:
    """
    Fraction of edges whose CI excludes zero at the requested alpha.

    If an estimate carries ``ci_alpha`` metadata and it does not match the
    requested ``alpha``, that edge is skipped.

    Returns NaN when no compatible confidence intervals are available.
    """

    count = 0
    total = 0
    for est in fcm.estimates.values():
        if est.ci_low is None or est.ci_high is None:
            continue
        if est.ci_alpha is not None and not np.isclose(est.ci_alpha, alpha):
            continue
        total += 1
        if est.ci_low > 0 or est.ci_high < 0:
            count += 1
    return count / total if total else float("nan")


def average_sign_stability(fcm: FCMGraph) -> float:
    """
    Average sign_stability across edges with estimates.
    """

    vals = [est.sign_stability for est in fcm.estimates.values() if est.sign_stability is not None]
    return float(np.mean(vals)) if vals else 0.0


def score_models(X: pd.DataFrame, y: pd.Series, models: Dict[str, object], n_splits: int = 3) -> List[Dict]:
    """
    Score candidate regressors using forward-chaining CV on provided data.

    Returns a list of records with mean R²/MAE and per-fold details so callers
    can inspect stability, not just aggregate metrics.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    records: List[Dict] = []
    for name, model in models.items():
        fold_scores: List[Dict] = []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            fold_scores.append(
                {
                    "r2": r2_score(y.iloc[test_idx], preds),
                    "mae": mean_absolute_error(y.iloc[test_idx], preds),
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )
        # Aggregate
        r2_vals = [fs["r2"] for fs in fold_scores]
        mae_vals = [fs["mae"] for fs in fold_scores]
        records.append(
            {
                "model": name,
                "r2_mean": float(np.mean(r2_vals)),
                "mae_mean": float(np.mean(mae_vals)),
                "folds": fold_scores,
            }
        )
    return records


def best_model(models_scores: List[Dict], metric: str = "r2") -> Optional[str]:
    """
    Pick best model name based on metric ('r2' higher is better, 'mae' lower is better).
    """

    if not models_scores:
        return None
    if metric == "r2":
        return max(models_scores, key=lambda r: r.get("r2_mean", -np.inf)).get("model")
    if metric == "mae":
        return min(models_scores, key=lambda r: r.get("mae_mean", np.inf)).get("model")
    raise ValueError(f"Unsupported metric: {metric}")


def compare_weights(fcm_list: Iterable[FCMGraph]) -> pd.DataFrame:
    """
    Combine scaled weights across multiple FCM graphs for comparison.

    Useful for comparing mental models: rows are edge-level weights tagged by model index.
    """

    records: List[Dict] = []
    for idx, fcm in enumerate(fcm_list):
        for (src, tgt), est in fcm.estimates.items():
            records.append(
                {
                    "model": idx,
                    "edge": f"{src}->{tgt}",
                    "scaled_weight": est.scaled_weight,
                    "tau_raw": est.tau_raw,
                }
            )
    return pd.DataFrame(records)


def weight_distance(fcm_a: FCMGraph, fcm_b: FCMGraph, norm: str = "l2") -> float:
    """
    Compute distance between two graphs' scaled weights on common edges.

    Returns NaN if there are no overlapping edges.
    """

    keys_a = set(fcm_a.estimates.keys())
    keys_b = set(fcm_b.estimates.keys())
    common = keys_a & keys_b
    if not common:
        return float("nan")
    diffs = []
    for key in common:
        wa = fcm_a.estimates[key].scaled_weight or 0.0
        wb = fcm_b.estimates[key].scaled_weight or 0.0
        diffs.append(wa - wb)
    arr = np.asarray(diffs)
    if norm == "l2":
        return float(np.linalg.norm(arr))
    if norm == "l1":
        return float(np.sum(np.abs(arr)))
    raise ValueError(f"Unsupported norm: {norm}")


def generalized_distance_ratio(
    fcm_a: FCMGraph,
    fcm_b: FCMGraph,
    *,
    alpha: int = 1,          # 1=ignore self-loops (diagonal), 0=include
    beta: float = 1.0,       # max connection strength scale (keep 1.0 if scaled_weight in [-1,1])
    epsilon: float = 2.0,    # number of polarities (typically 2)
    delta: float = 0.0,      # extra penalty when signs differ
    gamma: int = 2,          # missing-node mode: 0/1/2 (as in your table)
    gamma_prime: float = 1.0 # penalty weight for unique concepts term
) -> float:
    """
    GDR/DR computed from FCMGraph scaled_weight values, following your table/notebook semantics.

    diff(i,j):
      (i) 0 if i=j and alpha=1
      (ii) missing node logic controlled by gamma
      (iii) opposite signs: |a-b| + delta
      (iv) otherwise: |a-b|
    """

    # 1) concept sets
    ids_a = set(fcm_a.get_concept_ids())
    ids_b = set(fcm_b.get_concept_ids())

    common = ids_a & ids_b
    unique_a = ids_a - ids_b
    unique_b = ids_b - ids_a
    union = sorted(ids_a | ids_b)

    p_c = len(common)
    p_uA = len(unique_a)
    p_uB = len(unique_b)

    # presence flags for missing-node logic
    present_a = {x: (x in ids_a) for x in union}
    present_b = {x: (x in ids_b) for x in union}

    # 2) weight lookups from scaled_weight (missing edge => 0.0)
    weights_a = {}
    for (src, tgt), est in fcm_a.estimates.items():
        w = est.scaled_weight if est.scaled_weight is not None else 0.0
        weights_a[(src, tgt)] = float(w)

    weights_b = {}
    for (src, tgt), est in fcm_b.estimates.items():
        w = est.scaled_weight if est.scaled_weight is not None else 0.0
        weights_b[(src, tgt)] = float(w)

    # 3) numerator
    numerator = 0.0
    for i in union:
        for j in union:
            # (i) ignore diagonal if alpha=1
            if alpha == 1 and i == j:
                continue

            a = weights_a.get((i, j), 0.0)
            b = weights_b.get((i, j), 0.0)

            # (ii) missing node logic
            node_missing = (not present_a[i]) or (not present_a[j]) or (not present_b[i]) or (not present_b[j])
            if node_missing:
                if gamma == 0:
                    d = 0.0
                elif gamma == 1:
                    d = 0.0 if (a == 0.0 and b == 0.0) else 1.0
                else:  # gamma == 2
                    d = 1.0
                numerator += d
                continue

            # (iii) opposite signs
            if a * b < 0:
                numerator += abs(a - b) + delta
            else:
                # (iv) default
                numerator += abs(a - b)

    # 4) denominator (exact structure from your table)
    denom = ((epsilon * beta) + delta) * (p_c ** 2)
    denom += gamma_prime * (2 * p_uA * p_uB + 2 * p_c * (p_uA + p_uB) + p_uA**2 + p_uB**2)
    denom -= alpha * (((epsilon * beta) + delta) * p_c + gamma_prime * (p_uA + p_uB))

    return 0.0 if denom == 0 else numerator / denom


def get_adjacency_matrix(fcm: FCMGraph, ignore_self_loops: bool = False) -> Tuple[List[str], np.ndarray]:
    """
    Convert FCM estimates to an adjacency matrix.

    Parameters
    ----------
    fcm : FCMGraph
        The cognitive map to convert.
    ignore_self_loops : bool
        If True, do not include self-edges (i->i) in the returned adjacency.
        This ensures graph metrics (density, degrees, etc.) do not count self-loops.

    Returns
    -------
    (concepts, adj) :
        concepts: list of concept IDs (order of rows/cols)
        adj: NxN numpy array of weights (self-loop entries set to 0 when ignored)
    """
    concepts = fcm.get_concept_ids()
    n = len(concepts)
    idx_map = {c: i for i, c in enumerate(concepts)}
    adj = np.zeros((n, n))

    for (src, tgt), est in fcm.estimates.items():
        if est.scaled_weight is None:
            continue
        if src in idx_map and tgt in idx_map:
            # Drop ONLY the specific self-loop "1->1" if it exists, or all self-loops
            if ignore_self_loops and src == tgt:
                continue
            adj[idx_map[src], idx_map[tgt]] = est.scaled_weight
    return concepts, adj


def load_id_to_label_from_output_json(path: Path) -> Dict[str, str]:
    """
    Read causal-mm-run output JSON and build a mapping: concept_id (as str) -> concept label.
    """
    with open(path, "r") as f:
        data = json.load(f)

    concepts = data["model"]["concepts"]
    # store keys as strings because fcm.get_concept_ids() and estimates keys are typically strings
    return {str(c["id"]): c.get("label", str(c["id"])) for c in concepts}


def get_adjacency_matrix_labeled(
    fcm: FCMGraph,
    id_to_label: Dict[str, str],
    *,
    ignore_self_loops: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Convert FCM estimates (scaled_weight) to adjacency matrix using *concept labels* instead of IDs.

    Returns:
      labels: list of concept labels in the same order as fcm.get_concept_ids()
      adj: NxN numpy array of scaled_weight
    """
    concept_ids = list(fcm.get_concept_ids())
    labels = [id_to_label.get(str(cid), str(cid)) for cid in concept_ids]

    n = len(concept_ids)
    idx_map = {str(cid): i for i, cid in enumerate(concept_ids)}
    adj = np.zeros((n, n), dtype=float)

    for (src, tgt), est in fcm.estimates.items():
        src = str(src)
        tgt = str(tgt)
        if ignore_self_loops and src == tgt:
            continue
        if est.scaled_weight is None:
            continue
        if src in idx_map and tgt in idx_map:
            adj[idx_map[src], idx_map[tgt]] = float(est.scaled_weight)

    return labels, adj


def graph_complexity_metrics(fcm: FCMGraph, ignore_self_loops: bool = False) -> Dict[str, float]:
    """
    Compute graph-theoretic metrics for the FCM structure.
    
    Metrics:
    - num_nodes: Number of variables (concepts)
    - num_connections: Number of connections (non-zero weights)
    - density: C / (N * (N-1))
    - hierarchy_index: Measure of hierarchical structure (0-1)
    - num_transmitters: Nodes with only outgoing edges
    - num_receivers: Nodes with only incoming edges
    - num_ordinary: Nodes with both incoming and outgoing edges
    - num_isolated: Nodes with no edges
    - connections_per_node: C / N
    - complexity: receivers / transmitters
    """
    concepts, A = get_adjacency_matrix(fcm, ignore_self_loops=ignore_self_loops)
    A = np.asarray(A)

    N = len(concepts)
    if N == 0:
        return {
            "num_nodes": 0, "num_connections": 0, "density": 0.0, "hierarchy_index": 0.0,
            "num_transmitters": 0, "num_receivers": 0, "num_ordinary": 0, "num_isolated": 0,
            "connections_per_node": 0.0, "complexity": 0.0
        }

    B = (A != 0).astype(int)

    if ignore_self_loops:
        np.fill_diagonal(B, 0)

    C = int(B.sum())

    if ignore_self_loops:
        density = float(C) / (N * (N - 1)) if N > 1 else 0.0
    else:
        density = float(C) / (N * N) if N > 0 else 0.0

    out_degree = B.sum(axis=1)
    in_degree  = B.sum(axis=0)

    n_transmitters = int(np.sum((out_degree > 0) & (in_degree == 0)))
    n_receivers    = int(np.sum((in_degree > 0) & (out_degree == 0)))
    n_ordinary     = int(np.sum((in_degree > 0) & (out_degree > 0)))
    n_isolated     = int(np.sum((in_degree == 0) & (out_degree == 0)))

    connections_per_node = float(C) / N if N > 0 else 0.0

    # Complexity (FCM definition): receivers / transmitters
    if n_transmitters > 0:
        complexity = n_receivers / n_transmitters
    else:
        complexity = 0.0

    if N > 1:
        mean_od = np.sum(out_degree) / N
        sq_dev = np.sum((out_degree - mean_od) ** 2)
        hierarchy_index = 12.0 * sq_dev / ((N - 1) * N * (N + 1))
    else:
        hierarchy_index = 0.0

    return {
        "num_nodes": N,
        "num_connections": C,
        "density": density,
        "hierarchy_index": hierarchy_index,
        "num_transmitters": n_transmitters,
        "num_receivers": n_receivers,
        "num_ordinary": n_ordinary,
        "num_isolated": n_isolated,
        "connections_per_node": connections_per_node,
        "complexity": complexity,
        # Backward compatibility aliases
        "N": N,
        "C": C,
        "Density": density,
        "Hierarchy": hierarchy_index,
    }


def graph_complexity_metrics_from_adjacency(
    A: np.ndarray,
    *,
    ignore_self_loops: bool = False
) -> dict:
    """
    Compute graph-theoretic metrics directly from a weighted adjacency matrix A.
    Intended for aggregated matrices (CSV-based), not FCMGraph objects.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 0:
        return {}

    if ignore_self_loops:
        A = A.copy()
        np.fill_diagonal(A, 0.0)

    # Binary adjacency
    B = (A != 0).astype(int)

    C = int(B.sum())
    if ignore_self_loops:
        density = float(C) / (n * (n - 1)) if n > 1 else 0.0
    else:
        density = float(C) / (n * n) if n > 0 else 0.0

    out_degree = B.sum(axis=1)
    in_degree  = B.sum(axis=0)

    n_transmitters = int(np.sum((out_degree > 0) & (in_degree == 0)))
    n_receivers    = int(np.sum((in_degree > 0) & (out_degree == 0)))
    n_ordinary     = int(np.sum((in_degree > 0) & (out_degree > 0)))
    n_isolated     = int(np.sum((in_degree == 0) & (out_degree == 0)))

    connections_per_node = float(C) / n if n > 0 else 0.0

    # Complexity (receiver / transmitter)
    complexity = (
        float(n_receivers) / float(n_transmitters)
        if n_transmitters > 0 else 0.0
    )

    # Normalized hierarchy index (Özesmi & Özesmi)
    if n > 1:
        mean_od = np.sum(out_degree) / n
        sq_dev = np.sum((out_degree - mean_od) ** 2)
        hierarchy_index = 12.0 * sq_dev / ((n - 1) * n * (n + 1))
    else:
        hierarchy_index = 0.0

    return {
        "num_nodes": n,
        "num_connections": C,
        "density": density,
        "hierarchy_index": hierarchy_index,
        "num_transmitters": n_transmitters,
        "num_receivers": n_receivers,
        "num_ordinary": n_ordinary,
        "num_isolated": n_isolated,
        "connections_per_node": connections_per_node,
        "complexity": complexity,
        # Backward compatibility aliases
        "N": n,
        "C": C,
        "Density": density,
        "Hierarchy": hierarchy_index,
    }


def concept_centrality_metrics(fcm: FCMGraph) -> pd.DataFrame:
    """
    Compute concept-level centrality metrics.
    
    Metrics:
    - Outdegree: Sum of absolute weights of outgoing edges.
    - Indegree: Sum of absolute weights of incoming edges.
    - Centrality: Outdegree + Indegree.
    """
    concepts, adj = get_adjacency_matrix(fcm)
    if not concepts:
        return pd.DataFrame()
        
    # Weighted degrees (sum of absolute weights)
    out_degree = np.sum(np.abs(adj), axis=1)
    in_degree = np.sum(np.abs(adj), axis=0)
    centrality = out_degree + in_degree
    
    return pd.DataFrame({
        "concept": concepts,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "centrality": centrality
    }).set_index("concept")
