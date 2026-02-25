#!/usr/bin/env python3
"""
Compare multiple mental model outputs using adjacency matrices.

For each pair of provided project JSON files the script computes:
1. A weighted similarity index (1 - normalized L1 distance between adjacency matrices).
2. The Generalized Distance Ratio (GDR) using the existing causal_mm.metrics helper.

Usage:
    python scripts/compare_mental_models.py data/models/a_rf2.json data/models/a_gbm_bs2.json
    python scripts/compare_mental_models.py data/models/a_rf2.json data/models/a_gbm_bs2.json \
        --format csv --output comparisons.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_mm.fcm import Concept, EdgeEstimate, FCMGraph
from causal_mm.metrics import generalized_distance_ratio

EPS = 1e-9


@dataclass
class MentalModel:
    """Container holding adjacency and metadata for a mental model."""

    name: str
    path: Path
    concept_ids: List[str]
    adjacency: np.ndarray
    weight_map: Dict[Tuple[str, str], float]
    fcm: FCMGraph


def _build_weight_map(concept_ids: List[str], adjacency: np.ndarray) -> Dict[Tuple[str, str], float]:
    """
    Convert an adjacency matrix into a sparse weight map keyed by (source, target).
    """

    weights: Dict[Tuple[str, str], float] = {}
    for i, src in enumerate(concept_ids):
        for j, tgt in enumerate(concept_ids):
            value = float(adjacency[i, j])
            if abs(value) > EPS:
                weights[(src, tgt)] = value
    return weights


def _fcm_from_weights(concept_ids: List[str], weights: Dict[Tuple[str, str], float]) -> FCMGraph:
    """
    Build a lightweight FCMGraph instance backed only by the provided adjacency weights.
    """

    concepts = [Concept(id=str(cid)) for cid in concept_ids]
    fcm = FCMGraph(concepts=concepts, edges=[])
    for (src, tgt), weight in weights.items():
        fcm.estimates[(src, tgt)] = EdgeEstimate(source=src, target=tgt, scaled_weight=weight)
    return fcm


def load_model(path: Path) -> MentalModel:
    """
    Load a project JSON and return a MentalModel with adjacency information.
    """

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    adjacency_data = raw.get("results", {}).get("adjacency_matrix")
    if not adjacency_data:
        raise ValueError(f"{path} has no results.adjacency_matrix block")

    concept_ids = [str(cid) for cid in adjacency_data.get("concept_ids", [])]
    matrix = np.asarray(adjacency_data.get("matrix", []), dtype=float)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(concept_ids):
        raise ValueError(f"{path} adjacency matrix shape mismatch")

    weight_map = _build_weight_map(concept_ids, matrix)
    fcm = _fcm_from_weights(concept_ids, weight_map)

    meta = raw.get("meta", {})
    name = meta.get("project_id") or path.stem

    return MentalModel(
        name=name,
        path=path,
        concept_ids=concept_ids,
        adjacency=matrix,
        weight_map=weight_map,
        fcm=fcm,
    )


def similarity_index(model_a: MentalModel, model_b: MentalModel) -> float:
    """
    Weighted overlap similarity between two adjacency matrices.

    The score is defined as:
        1 - (sum_{i,j} |A_ij - B_ij|) / (sum_{i,j} (|A_ij| + |B_ij|))

    Missing concepts are treated as having zero connections.
    """

    union = set(model_a.concept_ids) | set(model_b.concept_ids)
    if not union:
        return 1.0

    total_diff = 0.0
    total_weight = 0.0
    for src in union:
        for tgt in union:
            w_a = model_a.weight_map.get((src, tgt), 0.0)
            w_b = model_b.weight_map.get((src, tgt), 0.0)
            total_diff += abs(w_a - w_b)
            total_weight += abs(w_a) + abs(w_b)

    if total_weight <= EPS:
        return 1.0

    score = 1.0 - (total_diff / total_weight)
    return max(0.0, min(1.0, score))


def compare_models(models: Iterable[MentalModel]) -> List[Dict[str, object]]:
    """Compute pairwise metrics across all models."""

    results: List[Dict[str, object]] = []
    for model_a, model_b in combinations(models, 2):
        sim = similarity_index(model_a, model_b)
        gdr = generalized_distance_ratio(model_a.fcm, model_b.fcm)
        results.append(
            {
                "model_a": model_a.name,
                "model_b": model_b.name,
                "similarity_index": sim,
                "generalized_distance_ratio": gdr,
            }
        )
    return results


def _format_float(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def format_table(rows: List[Dict[str, object]], precision: int = 4) -> str:
    """Return a formatted text table for the provided rows."""

    if not rows:
        return "Provide at least two models to compare."

    sim_vals = [_format_float(row["similarity_index"], precision) for row in rows]
    gdr_vals = [_format_float(row["generalized_distance_ratio"], precision) for row in rows]

    headers = ["Model A", "Model B", "Similarity", "GDR"]
    widths = [
        max(len(headers[0]), *(len(row["model_a"]) for row in rows)),
        max(len(headers[1]), *(len(row["model_b"]) for row in rows)),
        max(len(headers[2]), *(len(val) for val in sim_vals)),
        max(len(headers[3]), *(len(val) for val in gdr_vals)),
    ]

    header_line = f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:>{widths[2]}}  {headers[3]:>{widths[3]}}"
    lines = [header_line, "-" * len(header_line)]
    for row, sim_str, gdr_str in zip(rows, sim_vals, gdr_vals):
        line = (
            f"{row['model_a']:<{widths[0]}}  "
            f"{row['model_b']:<{widths[1]}}  "
            f"{sim_str:>{widths[2]}}  "
            f"{gdr_str:>{widths[3]}}"
        )
        lines.append(line)
    return "\n".join(lines)


def format_csv(rows: List[Dict[str, object]], precision: int = 4) -> str:
    """Return CSV-formatted rows."""

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["model_a", "model_b", "similarity_index", "generalized_distance_ratio"])
    for row in rows:
        writer.writerow(
            [
                row["model_a"],
                row["model_b"],
                _format_float(row["similarity_index"], precision),
                _format_float(row["generalized_distance_ratio"], precision),
            ]
        )
    return output.getvalue().rstrip("\n")


def format_json(rows: List[Dict[str, object]], precision: int = 4) -> str:
    """Return JSON string for rows (rounded to requested precision)."""

    payload = [
        {
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "similarity_index": round(float(row["similarity_index"]), precision),
            "generalized_distance_ratio": round(float(row["generalized_distance_ratio"]), precision),
        }
        for row in rows
    ]
    return json.dumps(payload, indent=2)


def output_results(
    rows: List[Dict[str, object]],
    fmt: str,
    precision: int = 4,
    output_path: Optional[Path] = None,
) -> None:
    """Dispatch to the requested output format and destination."""

    if fmt == "table":
        content = format_table(rows, precision)
    elif fmt == "csv":
        content = format_csv(rows, precision)
    elif fmt == "json":
        content = format_json(rows, precision)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if output_path:
        output_path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
    else:
        print(content)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare mental models using adjacency matrices.")
    parser.add_argument(
        "projects",
        nargs="+",
        help="Path(s) to mental model JSON files with results.adjacency_matrix populated.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision for printed metrics (default: 4).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (table, csv, json). Default prints a human-readable table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination file. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    if len(args.projects) < 2:
        print("Please provide at least two project files.")
        return 1

    models: List[MentalModel] = []
    for raw_path in args.projects:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        models.append(load_model(path))

    rows = compare_models(models)
    output_results(rows, fmt=args.format, precision=args.precision, output_path=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
