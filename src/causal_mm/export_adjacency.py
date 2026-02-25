from pathlib import Path
import argparse
import pandas as pd

from causal_mm.io import load_project
from causal_mm.metrics import (
    load_id_to_label_from_output_json,
    get_adjacency_matrix_labeled,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export adjacency matrix from causal_mm output JSON (labeled by concept names)"
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--ignore_self_loops", action="store_true", help="Remove diagonal edges (i->i).")
    args = parser.parse_args()

    # Load project (you know this returns tuple; fcm is index 0)
    fcm, ts, settings, meta = load_project(args.input)

    # Build id -> label mapping from JSON
    id_to_label = load_id_to_label_from_output_json(args.input)

    # Compute adjacency with labels
    labels, adj = get_adjacency_matrix_labeled(
        fcm, id_to_label, ignore_self_loops=args.ignore_self_loops
    )

    # Write CSV
    df = pd.DataFrame(adj, index=labels, columns=labels)
    df.index.name = "source"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output)

    print(f"Adjacency matrix written to {args.output}")


if __name__ == "__main__":
    main()