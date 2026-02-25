from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

from .data import TimeSeriesData
from .fcm import Concept, Edge, EdgeEstimate, FCMGraph


def _stringify_id(value) -> str:
    return str(value)


def _maybe_int(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def load_project(path: Path) -> Tuple[FCMGraph, TimeSeriesData, Dict, Dict]:
    """
    Load a full fcm_project.json and return (fcm, ts, settings, meta).

    - Concept/edge IDs are stringified for internal consistency.
    - Timeseries keys are coerced to strings to match concept IDs.
    - Existing estimates (if any) are attached.
    """

    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    meta = raw.get("meta", {})
    model = raw.get("model", {})
    ts_raw = raw.get("timeseries", {})
    settings = raw.get("settings", {})
    estimates_raw = raw.get("estimates", {})

    concepts = [
        Concept(id=_stringify_id(c["id"]), label=c.get("label"), metadata=c.get("metadata", {}))
        for c in model.get("concepts", [])
    ]
    edges = [
        Edge(
            source=_stringify_id(e["source"]),
            target=_stringify_id(e["target"]),
            stakeholder_weight=e.get("stakeholder_weight"),
            metadata=e.get("metadata", {}),
        )
        for e in model.get("edges", [])
    ]
    fcm = FCMGraph(concepts=concepts, edges=edges)

    index = ts_raw.get("index", [])
    data_dict = ts_raw.get("data", {})
    ts_df = pd.DataFrame(data_dict)
    ts_df.columns = [str(c) for c in ts_df.columns]
    ts = TimeSeriesData(index=pd.Index(index), data=ts_df)

    # Load estimates if present
    estimates: Dict = {}
    for key, val in estimates_raw.items():
        if "->" in key:
            source, target = key.split("->", 1)
            estimates[(source, target)] = EdgeEstimate(
                source=source,
                target=target,
                tau_raw=val.get("tau_raw"),
                tau_se=val.get("tau_se"),
                ci_low=val.get("ci_low"),
                ci_high=val.get("ci_high"),
                sign_stability=val.get("sign_stability"),
                scaled_weight=val.get("scaled_weight"),
                n_obs=val.get("n_obs"),
                lag_used=val.get("lag_used"),
                status=val.get("status", "pending"),
                error_message=val.get("error_message"),
                computed_at=val.get("computed_at"),
                method=val.get("method", "dml"),
            )
    fcm.estimates = estimates
    return fcm, ts, settings, meta


def _edge_estimates_to_dict(estimates: Dict) -> Dict:
    out = {}
    for (src, tgt), est in estimates.items():
        out[f"{src}->{tgt}"] = {
            "tau_raw": est.tau_raw,
            "tau_se": est.tau_se,
            "ci_low": est.ci_low,
            "ci_high": est.ci_high,
            "sign_stability": est.sign_stability,
            "scaled_weight": est.scaled_weight,
            "n_obs": est.n_obs,
            "lag_used": est.lag_used,
            "status": est.status,
            "error_message": est.error_message,
            "computed_at": est.computed_at,
            "method": est.method,
        }
    return out


def _serialize_concepts(concepts):
    return [
        {"id": _maybe_int(c.id), "label": c.label, "metadata": c.metadata or {}}
        for c in concepts
    ]


def _serialize_edges(edges):
    return [
        {
            "source": _maybe_int(e.source),
            "target": _maybe_int(e.target),
            "stakeholder_weight": e.stakeholder_weight,
            "metadata": e.metadata or {},
        }
        for e in edges
    ]


def save_project(
    path: Path,
    fcm: FCMGraph,
    ts: TimeSeriesData,
    settings: Dict,
    meta: Dict,
    results: Optional[Dict] = None,
) -> None:
    """
    Write a full fcm_project.json. Preserves existing fields and injects estimates.

    Concept/edge IDs are written as ints when possible to match the spec,
    but kept as strings internally during computation.
    """

    out = {
        "meta": meta,
        "model": {
            "concepts": _serialize_concepts(fcm.concepts),
            "edges": _serialize_edges(fcm.edges),
        },
        "timeseries": {
            "index": [x.isoformat() if hasattr(x, "isoformat") else str(x) for x in ts.index],
            "data": {str(k): ts.data[k].tolist() for k in ts.data.columns},
        },
        "settings": settings,
        "estimates": _edge_estimates_to_dict(fcm.estimates),
    }
    if results:
        out["results"] = results
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
